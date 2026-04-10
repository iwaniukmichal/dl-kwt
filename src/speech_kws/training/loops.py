from __future__ import annotations

import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from speech_kws.audio.augment import apply_specaugment
from speech_kws.audio.frontend import AudioFrontend
from speech_kws.config import (
    REPO_ROOT,
    clean_config_for_dump,
    get_run_dir,
    resolve_paths_in_config,
    validate_config,
)
from speech_kws.data.dataset import SpeechCommandsDataset
from speech_kws.data.labels import EVAL_LABELS, RAW_WORD_LABELS, SILENCE_LABEL, TARGET_KEYWORDS
from speech_kws.data.samplers import build_training_sampler
from speech_kws.evaluation.metrics import (
    build_prediction_rows,
    compute_classification_metrics,
    save_confusion_plot,
)
from speech_kws.evaluation.thresholds import apply_rejection, sweep_thresholds
from speech_kws.models.wrappers import build_model
from speech_kws.training.losses import compute_strategy_loss
from speech_kws.training.optim import build_optimizer, build_scheduler
from speech_kws.utils.io import ensure_dir, save_csv, save_json, save_yaml
from speech_kws.utils.reproducibility import (
    build_torch_generator,
    get_git_commit,
    seed_worker,
    set_global_seed,
)


def _device_from_config(experiment_config: dict) -> torch.device:
    requested = experiment_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _progress_enabled() -> bool:
    return sys.stderr.isatty()


def _available_cpu_worker_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass

    cpu_count = os.cpu_count()
    return max(1, int(cpu_count) if cpu_count is not None else 1)


def _resolve_dataloader_settings(config: dict, device: torch.device) -> dict[str, int | bool | str | None]:
    data_config = config.get("data", {})
    requested_num_workers = data_config.get("num_workers", "auto")
    available_cpu_workers = _available_cpu_worker_count()

    if requested_num_workers is None:
        normalized_num_workers: int | str = "auto"
    elif isinstance(requested_num_workers, str):
        normalized_requested = requested_num_workers.strip().lower()
        normalized_num_workers = "auto" if normalized_requested == "auto" else int(normalized_requested)
    else:
        normalized_num_workers = int(requested_num_workers)

    num_workers = available_cpu_workers if normalized_num_workers == "auto" else int(normalized_num_workers)
    if num_workers < 0:
        raise ValueError("data.num_workers must be >= 0 or 'auto'")

    pin_memory = bool(data_config.get("pin_memory", False)) and device.type == "cuda"
    persistent_workers = bool(data_config.get("persistent_workers", True)) if num_workers > 0 else False
    prefetch_factor = int(data_config.get("prefetch_factor", 2)) if num_workers > 0 else None
    return {
        "requested_num_workers": normalized_num_workers,
        "available_cpu_workers": available_cpu_workers,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
    }


def _forward_model(
    frontend: AudioFrontend,
    model: torch.nn.Module,
    batch: dict,
    augment_config: dict,
    training: bool,
) -> dict:
    waveform_device = batch["waveform"].device
    frontend_context = nullcontext()
    if waveform_device.type in {"cpu", "cuda"}:
        frontend_context = torch.autocast(device_type=waveform_device.type, enabled=False)

    with frontend_context:
        features = frontend(batch["waveform"].float())
    specaugment_cfg = augment_config.get("specaugment", {})
    if training and specaugment_cfg.get("enabled", False):
        features = apply_specaugment(
            features,
            time_masks=int(specaugment_cfg.get("time_masks", 1)),
            time_mask_width=int(specaugment_cfg.get("time_mask_width", 10)),
            freq_masks=int(specaugment_cfg.get("freq_masks", 1)),
            freq_mask_width=int(specaugment_cfg.get("freq_mask_width", 5)),
        )
    return model(features)


def _predictions_strategy_a(logits: torch.Tensor) -> tuple[list[int], list[float]]:
    probs = torch.softmax(logits, dim=-1)
    predicted = probs.argmax(dim=-1).tolist()
    confidences = probs.max(dim=-1).values.tolist()
    train_labels = RAW_WORD_LABELS + [SILENCE_LABEL]

    eval_predictions = []
    for label_idx in predicted:
        label_name = train_labels[int(label_idx)]
        if label_name in TARGET_KEYWORDS:
            eval_predictions.append(EVAL_LABELS.index(label_name))
        elif label_name == SILENCE_LABEL:
            eval_predictions.append(EVAL_LABELS.index(SILENCE_LABEL))
        else:
            eval_predictions.append(EVAL_LABELS.index("unknown"))
    return eval_predictions, [float(value) for value in confidences]


def _predictions_strategy_b(logits: torch.Tensor) -> tuple[list[int], list[float]]:
    probs = torch.softmax(logits, dim=-1)
    return probs.argmax(dim=-1).tolist(), [float(value) for value in probs.max(dim=-1).values.tolist()]


def _predictions_strategy_c(head1_logits: torch.Tensor, head2_logits: torch.Tensor) -> tuple[list[int], list[float]]:
    head1_probs = torch.softmax(head1_logits, dim=-1)
    head2_probs = torch.softmax(head2_logits, dim=-1)
    head1_pred = head1_probs.argmax(dim=-1).tolist()
    head2_pred = head2_probs.argmax(dim=-1).tolist()

    predictions = []
    confidences = []
    for decision, keyword_idx, probs1, probs2 in zip(
        head1_pred,
        head2_pred,
        head1_probs,
        head2_probs,
    ):
        if decision == 0:
            predictions.append(EVAL_LABELS.index("silence"))
            confidences.append(float(probs1[0].item()))
        elif decision == 1:
            predictions.append(EVAL_LABELS.index("unknown"))
            confidences.append(float(probs1[1].item()))
        else:
            predictions.append(keyword_idx)
            confidences.append(float((probs1[2] * probs2.max()).item()))
    return predictions, confidences


def _evaluate_model(
    model: torch.nn.Module,
    frontend: AudioFrontend,
    data_loader: DataLoader,
    strategy: str,
    device: torch.device,
    threshold_config: dict | None = None,
    selected_tau: float | None = None,
) -> dict:
    model.eval()
    frontend.eval()

    y_true: list[int] = []
    sample_ids: list[str] = []
    relpaths: list[str] = []
    raw_labels: list[str] = []
    logits_buffer: list[torch.Tensor] = []
    head1_buffer: list[torch.Tensor] = []
    head2_buffer: list[torch.Tensor] = []

    with torch.inference_mode():
        for batch in data_loader:
            batch = _move_batch_to_device(batch, device)
            outputs = _forward_model(frontend, model, batch, augment_config={}, training=False)
            y_true.extend(batch["eval_target"].detach().cpu().tolist())
            sample_ids.extend(batch["sample_id"])
            relpaths.extend(batch["relpath"])
            raw_labels.extend(batch["raw_label"])

            if strategy == "c":
                head1_buffer.append(outputs["head1_logits"].detach().cpu())
                head2_buffer.append(outputs["head2_logits"].detach().cpu())
            else:
                logits_buffer.append(outputs["logits"].detach().cpu())

    if strategy == "a":
        logits = torch.cat(logits_buffer, dim=0)
        y_pred, confidences = _predictions_strategy_a(logits)
        metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred, label_names=EVAL_LABELS)
        threshold_rows = None
        selected_tau_value = None
    elif strategy == "b":
        logits = torch.cat(logits_buffer, dim=0)
        y_pred, confidences = _predictions_strategy_b(logits)
        metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred, label_names=EVAL_LABELS)
        threshold_rows = None
        selected_tau_value = None
    elif strategy == "c":
        head1_logits = torch.cat(head1_buffer, dim=0)
        head2_logits = torch.cat(head2_buffer, dim=0)
        y_pred, confidences = _predictions_strategy_c(head1_logits, head2_logits)
        metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred, label_names=EVAL_LABELS)
        threshold_rows = None
        selected_tau_value = None
    elif strategy == "d":
        logits = torch.cat(logits_buffer, dim=0)
        probs = torch.softmax(logits, dim=-1).numpy()
        if selected_tau is None:
            sweep = sweep_thresholds(probs=probs, y_true=y_true, threshold_config=threshold_config or {})
            y_pred = sweep["predictions"]
            confidences = sweep["confidences"]
            metrics = sweep["best_metrics"]
            threshold_rows = sweep["rows"]
            selected_tau_value = sweep["best_tau"]
        else:
            y_pred, confidences = apply_rejection(probs, selected_tau)
            metrics = compute_classification_metrics(y_true=y_true, y_pred=y_pred, label_names=EVAL_LABELS)
            threshold_rows = None
            selected_tau_value = selected_tau
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    prediction_rows = build_prediction_rows(
        sample_ids=sample_ids,
        relpaths=relpaths,
        raw_labels=raw_labels,
        y_true=y_true,
        y_pred=y_pred,
        label_names=EVAL_LABELS,
        confidences=confidences,
    )

    payload = dict(metrics)
    if selected_tau_value is not None:
        payload["selected_tau"] = float(selected_tau_value)

    return {
        "metrics": payload,
        "prediction_rows": prediction_rows,
        "threshold_rows": threshold_rows,
        "selected_tau": selected_tau_value,
    }


def _save_evaluation_artifacts(run_dir: Path, split_name: str, evaluation_bundle: dict) -> None:
    save_json(run_dir / f"{split_name}_metrics.json", evaluation_bundle["metrics"])
    save_csv(
        run_dir / f"predictions_{split_name}.csv",
        ["sample_id", "relpath", "raw_label", "true_label", "predicted_label", "confidence"],
        evaluation_bundle["prediction_rows"],
    )

    per_class_path = run_dir / ("per_class_metrics.csv" if split_name == "test" else f"per_class_metrics_{split_name}.csv")
    save_csv(
        per_class_path,
        ["label", "precision", "recall", "f1", "support"],
        evaluation_bundle["metrics"]["per_class"],
    )

    if split_name == "test":
        plots_dir = ensure_dir(run_dir / "plots")
        save_confusion_plot(
            evaluation_bundle["metrics"]["confusion_matrix_raw"],
            EVAL_LABELS,
            plots_dir / "confusion_raw.png",
            title="Test confusion matrix",
            value_format=".0f",
        )
        save_confusion_plot(
            evaluation_bundle["metrics"]["confusion_matrix_row_normalized"],
            EVAL_LABELS,
            plots_dir / "confusion_row_normalized.png",
            title="Test row-normalized confusion matrix",
            value_format=".2f",
        )

    if evaluation_bundle["threshold_rows"] is not None:
        save_csv(
            run_dir / "threshold_sweep.csv",
            ["tau", "accuracy", "macro_f1", "keyword_macro_f1"],
            evaluation_bundle["threshold_rows"],
        )


def _print_evaluation_info(
    *,
    split_name: str,
    metrics: dict,
    num_samples: int,
    epoch: int | None = None,
    step: int | None = None,
) -> None:
    parts = [f"split={split_name}", f"samples={num_samples}"]
    if epoch is not None:
        parts.append(f"epoch={epoch}")
    if step is not None:
        parts.append(f"step={step}")
    parts.extend(
        [
            f"accuracy={float(metrics['accuracy']):.4f}",
            f"macro_f1={float(metrics['macro_f1']):.4f}",
            f"keyword_macro_f1={float(metrics['keyword_macro_f1']):.4f}",
        ]
    )
    selected_tau = metrics.get("selected_tau")
    if selected_tau is not None:
        parts.append(f"selected_tau={float(selected_tau):.4f}")
    print(f"Evaluation info: {', '.join(parts)}", flush=True)


def _save_learning_curve(history_rows: list[dict], destination: Path) -> None:
    if not history_rows:
        return

    import matplotlib.pyplot as plt

    ensure_dir(destination.parent)
    x_key = "step" if any(row.get("step") is not None for row in history_rows) else "epoch"
    x_values = [row.get(x_key) for row in history_rows]
    train_loss = [row.get("train_loss") for row in history_rows]
    val_macro_f1 = [row.get("val_macro_f1") for row in history_rows]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(x_values, train_loss, color="tab:blue", label="train_loss")
    ax1.set_xlabel(x_key)
    ax1.set_ylabel("train loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(x_values, val_macro_f1, color="tab:orange", label="val_macro_f1")
    ax2.set_ylabel("val macro-F1", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    best_metric: float,
    selected_tau: float | None,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "selected_tau": selected_tau,
    }
    ensure_dir(path.parent)
    torch.save(checkpoint, path)


def _load_checkpoint(path: Path, model: torch.nn.Module, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def _build_dataloaders(
    config: dict,
    seed: int,
    device: torch.device,
    loader_settings: dict[str, int | bool | str | None] | None = None,
):
    batch_size = int(config["train"]["batch_size"])
    loader_settings = _resolve_dataloader_settings(config, device) if loader_settings is None else loader_settings
    num_workers = int(loader_settings["num_workers"])
    pin_memory = bool(loader_settings["pin_memory"])
    persistent_workers = bool(loader_settings["persistent_workers"])
    prefetch_factor = loader_settings["prefetch_factor"]
    clip_duration_sec = float(config["task"].get("clip_duration_sec", 1.0))
    sample_rate = int(config["frontend"].get("sample_rate", 16000))
    prepared_root = config["paths"]["prepared_root"]
    strategy = config["task"]["unknown_strategy"]

    train_dataset = SpeechCommandsDataset(
        prepared_root=prepared_root,
        split="train",
        strategy=strategy,
        augment_config=config.get("augment", {}),
        sample_rate=sample_rate,
        clip_duration_sec=clip_duration_sec,
        training=True,
    )
    val_dataset = SpeechCommandsDataset(
        prepared_root=prepared_root,
        split="validation",
        strategy=strategy,
        augment_config={},
        sample_rate=sample_rate,
        clip_duration_sec=clip_duration_sec,
        training=False,
    )
    test_dataset = SpeechCommandsDataset(
        prepared_root=prepared_root,
        split="test",
        strategy=strategy,
        augment_config={},
        sample_rate=sample_rate,
        clip_duration_sec=clip_duration_sec,
        training=False,
    )

    sampler_bundle = build_training_sampler(train_dataset, strategy=strategy, batch_size=batch_size, seed=seed)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker if num_workers > 0 else None,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    if sampler_bundle.batch_sampler is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler_bundle.batch_sampler,
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler_bundle.sampler,
            shuffle=sampler_bundle.shuffle,
            generator=build_torch_generator(seed),
            **loader_kwargs,
        )

    eval_batch_size = int(config.get("evaluation", {}).get("batch_size", batch_size))
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def _loss_component_summary(loss_dict: dict) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key, value in loss_dict.items():
        if torch.is_tensor(value):
            summary[key] = float(value.detach().item())
        else:
            summary[key] = float(value)
    return summary


def _optimizer_step(
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
) -> None:
    if getattr(scaler, "is_enabled", lambda: False)():
        scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    if scheduler is not None:
        scheduler.step()


def run_experiment(config: dict) -> dict:
    config = resolve_paths_in_config(config)
    validate_config(config)

    run_dir = get_run_dir(config)
    on_exists = config.get("outputs", {}).get("on_exists", "skip")
    if (run_dir / "test_metrics.json").exists():
        if on_exists == "skip":
            return {"run_dir": str(run_dir), "status": "skipped"}
        if on_exists == "error":
            raise FileExistsError(f"Run directory already contains test metrics: {run_dir}")

    ensure_dir(run_dir)
    ensure_dir(run_dir / "plots")

    seed = int(config["experiment"]["seed"])
    set_global_seed(seed, deterministic=bool(config["experiment"].get("deterministic", False)))
    device = _device_from_config(config["experiment"])
    amp_enabled = bool(config["experiment"].get("amp", True)) and device.type == "cuda"

    loader_settings = _resolve_dataloader_settings(config, device)
    loader_info_parts = [
        f"requested_workers={loader_settings['requested_num_workers']}",
        f"available_cpu_workers={loader_settings['available_cpu_workers']}",
        f"using_workers={loader_settings['num_workers']}",
        f"pin_memory={loader_settings['pin_memory']}",
        f"persistent_workers={loader_settings['persistent_workers']}",
    ]
    if loader_settings["prefetch_factor"] is not None:
        loader_info_parts.append(f"prefetch_factor={loader_settings['prefetch_factor']}")
    print(f"DataLoader info: {', '.join(loader_info_parts)}", flush=True)

    train_loader, val_loader, test_loader = _build_dataloaders(
        config,
        seed,
        device=device,
        loader_settings=loader_settings,
    )
    frontend = AudioFrontend(config["frontend"]).to(device)
    model = build_model(
        model_config=config["model"],
        strategy=config["task"]["unknown_strategy"],
        frontend_feature_dim=frontend.feature_dim,
    ).to(device)

    optimizer = build_optimizer(model.parameters(), config["train"]["optimizer"])
    grad_accum_steps = int(config["train"].get("grad_accumulation_steps", 1))
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum_steps))
    if config["train"]["mode"] == "epochs":
        total_optimizer_steps = int(config["train"]["max_epochs"]) * steps_per_epoch
    else:
        total_optimizer_steps = int(config["train"]["max_steps"])
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_config=config["train"].get("scheduler"),
        total_steps=total_optimizer_steps,
        steps_per_epoch=steps_per_epoch,
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    save_yaml(run_dir / "resolved_config.yaml", clean_config_for_dump(config))
    save_json(
        run_dir / "run_info.json",
        {
            "config_source": config.get("_config_path"),
            "device": str(device),
            "amp_enabled": amp_enabled,
            "seed": seed,
            "dataloader": loader_settings,
            "git_commit": get_git_commit(REPO_ROOT),
        },
    )

    strategy = config["task"]["unknown_strategy"].lower()
    strategy_config = config.get("strategy", {})
    history_rows: list[dict] = []
    best_metric = float("-inf")
    best_val_bundle = None
    best_selected_tau = None
    global_step = 0
    epoch = 0

    optimizer.zero_grad(set_to_none=True)
    show_progress = _progress_enabled()
    experiment_id = str(config["experiment"]["id"])

    if config["train"]["mode"] == "epochs":
        max_epochs = int(config["train"]["max_epochs"])
        for epoch in range(1, max_epochs + 1):
            model.train()
            frontend.train()
            loss_sums: dict[str, float] = {}
            num_batches = 0
            progress = tqdm(
                train_loader,
                desc=f"{experiment_id} epoch {epoch}/{max_epochs}",
                leave=False,
                dynamic_ncols=True,
                disable=not show_progress,
            )

            for batch_index, batch in enumerate(progress, start=1):
                batch = _move_batch_to_device(batch, device)
                with _autocast_context(device, amp_enabled):
                    outputs = _forward_model(
                        frontend,
                        model,
                        batch,
                        augment_config=config.get("augment", {}),
                        training=True,
                    )
                    loss_dict = compute_strategy_loss(strategy, outputs, batch, strategy_config)
                    total_loss = loss_dict["loss"]
                    scaled_loss = total_loss / grad_accum_steps

                metric_values = _loss_component_summary(loss_dict)
                scaler.scale(scaled_loss).backward()
                for key, value in metric_values.items():
                    loss_sums[key] = loss_sums.get(key, 0.0) + value

                if batch_index % grad_accum_steps == 0 or batch_index == len(train_loader):
                    _optimizer_step(
                        optimizer,
                        scheduler,
                        scaler,
                    )
                    global_step += 1
                if show_progress:
                    average_loss = loss_sums.get("loss", 0.0) / max(1, num_batches + 1)
                    progress.set_postfix(
                        loss=f"{average_loss:.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    )
                num_batches += 1

            progress.close()

            val_bundle = _evaluate_model(
                model=model,
                frontend=frontend,
                data_loader=val_loader,
                strategy=strategy,
                device=device,
                threshold_config=strategy_config.get("tau_sweep"),
            )
            _print_evaluation_info(
                split_name="validation",
                metrics=val_bundle["metrics"],
                num_samples=len(val_bundle["prediction_rows"]),
                epoch=epoch,
                step=global_step,
            )
            current_metric = float(val_bundle["metrics"]["macro_f1"])
            if current_metric > best_metric:
                best_metric = current_metric
                best_val_bundle = val_bundle
                best_selected_tau = val_bundle.get("selected_tau")
                _save_checkpoint(
                    run_dir / "best.ckpt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    best_metric=best_metric,
                    selected_tau=best_selected_tau,
                )

            _save_checkpoint(
                run_dir / "last.ckpt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                selected_tau=best_selected_tau,
            )

            history_rows.append(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "train_loss": loss_sums.get("loss", 0.0) / max(1, num_batches),
                    "val_macro_f1": val_bundle["metrics"]["macro_f1"],
                    "val_accuracy": val_bundle["metrics"]["accuracy"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
    else:
        max_steps = int(config["train"]["max_steps"])
        eval_interval_steps = int(config["train"].get("eval_interval_steps", 1000))
        data_iterator = iter(train_loader)
        accumulation_index = 0
        loss_sums: dict[str, float] = {}
        window_batches = 0
        pseudo_epoch = 0
        progress = tqdm(
            total=max_steps,
            desc=f"{experiment_id} steps",
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        )

        while global_step < max_steps:
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(train_loader)
                batch = next(data_iterator)
                pseudo_epoch += 1

            model.train()
            frontend.train()
            batch = _move_batch_to_device(batch, device)
            with _autocast_context(device, amp_enabled):
                outputs = _forward_model(
                    frontend,
                    model,
                    batch,
                    augment_config=config.get("augment", {}),
                    training=True,
                )
                loss_dict = compute_strategy_loss(strategy, outputs, batch, strategy_config)
                total_loss = loss_dict["loss"]
                scaled_loss = total_loss / grad_accum_steps

            metric_values = _loss_component_summary(loss_dict)
            scaler.scale(scaled_loss).backward()
            accumulation_index += 1
            for key, value in metric_values.items():
                loss_sums[key] = loss_sums.get(key, 0.0) + value
            window_batches += 1

            if accumulation_index >= grad_accum_steps:
                _optimizer_step(
                    optimizer,
                    scheduler,
                    scaler,
                )
                global_step += 1
                accumulation_index = 0
                progress.update(1)
                if show_progress:
                    running_loss = loss_sums.get("loss", 0.0) / max(1, window_batches)
                    progress.set_postfix(
                        loss=f"{running_loss:.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    )

                if global_step % eval_interval_steps == 0 or global_step == max_steps:
                    val_bundle = _evaluate_model(
                        model=model,
                        frontend=frontend,
                        data_loader=val_loader,
                        strategy=strategy,
                        device=device,
                        threshold_config=strategy_config.get("tau_sweep"),
                    )
                    _print_evaluation_info(
                        split_name="validation",
                        metrics=val_bundle["metrics"],
                        num_samples=len(val_bundle["prediction_rows"]),
                        epoch=pseudo_epoch,
                        step=global_step,
                    )
                    current_metric = float(val_bundle["metrics"]["macro_f1"])
                    if current_metric > best_metric:
                        best_metric = current_metric
                        best_val_bundle = val_bundle
                        best_selected_tau = val_bundle.get("selected_tau")
                        _save_checkpoint(
                            run_dir / "best.ckpt",
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=pseudo_epoch,
                            global_step=global_step,
                            best_metric=best_metric,
                            selected_tau=best_selected_tau,
                        )

                    _save_checkpoint(
                        run_dir / "last.ckpt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=pseudo_epoch,
                        global_step=global_step,
                        best_metric=best_metric,
                        selected_tau=best_selected_tau,
                    )

                    history_rows.append(
                        {
                            "epoch": pseudo_epoch,
                            "step": global_step,
                            "train_loss": loss_sums.get("loss", 0.0) / max(1, window_batches),
                            "val_macro_f1": val_bundle["metrics"]["macro_f1"],
                            "val_accuracy": val_bundle["metrics"]["accuracy"],
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        }
                    )
                    loss_sums = {}
                    window_batches = 0

        epoch = pseudo_epoch
        progress.close()

    if best_val_bundle is None:
        raise RuntimeError("Training finished without a validation evaluation.")

    checkpoint = _load_checkpoint(run_dir / "best.ckpt", model=model, device=device)
    best_selected_tau = checkpoint.get("selected_tau")
    _save_evaluation_artifacts(run_dir, "val", best_val_bundle)

    test_bundle = _evaluate_model(
        model=model,
        frontend=frontend,
        data_loader=test_loader,
        strategy=strategy,
        device=device,
        threshold_config=strategy_config.get("tau_sweep"),
        selected_tau=best_selected_tau,
    )
    _print_evaluation_info(
        split_name="test",
        metrics=test_bundle["metrics"],
        num_samples=len(test_bundle["prediction_rows"]),
        epoch=epoch,
        step=global_step,
    )
    _save_evaluation_artifacts(run_dir, "test", test_bundle)

    save_csv(
        run_dir / "train_history.csv",
        ["epoch", "step", "train_loss", "val_macro_f1", "val_accuracy", "learning_rate"],
        history_rows,
    )
    _save_learning_curve(history_rows, run_dir / "plots" / "learning_curve.png")

    return {
        "run_dir": str(run_dir),
        "best_metric": best_metric,
        "selected_tau": best_selected_tau,
    }
