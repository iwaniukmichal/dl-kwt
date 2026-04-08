from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_kws.audio.frontend import AudioFrontend  # noqa: E402
from speech_kws.config import load_config, resolve_paths_in_config  # noqa: E402
from speech_kws.models.wrappers import build_model  # noqa: E402
from speech_kws.training.losses import compute_strategy_loss  # noqa: E402
from speech_kws.training.loops import (  # noqa: E402
    _autocast_context,
    _build_dataloaders,
    _device_from_config,
    _forward_model,
    _move_batch_to_device,
)
from speech_kws.training.optim import build_optimizer  # noqa: E402
from speech_kws.utils.reproducibility import set_global_seed  # noqa: E402


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _next_batch(iterator, loader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def _loader_profile(train_loader, num_batches: int) -> dict[str, float]:
    iterator = iter(train_loader)
    total_wait = 0.0
    total_samples = 0

    for _ in range(num_batches):
        start = time.perf_counter()
        batch, iterator = _next_batch(iterator, train_loader)
        total_wait += time.perf_counter() - start
        total_samples += int(batch["waveform"].shape[0])

    return {
        "num_batches": num_batches,
        "num_samples": total_samples,
        "elapsed_sec": round(total_wait, 6),
        "samples_per_sec": round(total_samples / max(total_wait, 1e-12), 2),
    }


def _train_step_profile(config: dict, train_loader, device: torch.device, num_batches: int) -> dict[str, float]:
    amp_enabled = bool(config["experiment"].get("amp", True)) and device.type == "cuda"
    strategy = config["task"]["unknown_strategy"]
    strategy_config = config.get("strategy", {})

    frontend = AudioFrontend(config["frontend"]).to(device)
    model = build_model(
        model_config=config["model"],
        strategy=strategy,
        frontend_feature_dim=frontend.feature_dim,
    ).to(device)
    optimizer = build_optimizer(model.parameters(), config["train"]["optimizer"])
    grad_accum_steps = int(config["train"].get("grad_accumulation_steps", 1))

    frontend.train()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    iterator = iter(train_loader)
    total_wait = 0.0
    total_compute = 0.0
    total_loss = 0.0
    total_samples = 0
    accumulation_index = 0

    for _ in range(num_batches):
        start_wait = time.perf_counter()
        batch, iterator = _next_batch(iterator, train_loader)
        total_wait += time.perf_counter() - start_wait

        _sync_device(device)
        start_compute = time.perf_counter()
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
            loss = loss_dict["loss"] / grad_accum_steps

        loss.backward()
        accumulation_index += 1
        total_loss += float(loss_dict["loss"].detach().item())
        total_samples += int(batch["waveform"].shape[0])

        if accumulation_index >= grad_accum_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation_index = 0

        _sync_device(device)
        total_compute += time.perf_counter() - start_compute

    if accumulation_index > 0:
        _sync_device(device)
        start_compute = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _sync_device(device)
        total_compute += time.perf_counter() - start_compute

    total_elapsed = total_wait + total_compute
    return {
        "num_batches": num_batches,
        "num_samples": total_samples,
        "avg_loss": round(total_loss / max(num_batches, 1), 6),
        "data_wait_sec": round(total_wait, 6),
        "step_compute_sec": round(total_compute, 6),
        "total_sec": round(total_elapsed, 6),
        "samples_per_sec": round(total_samples / max(total_elapsed, 1e-12), 2),
        "data_wait_fraction": round(total_wait / max(total_elapsed, 1e-12), 4),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile dataloader wait time and end-to-end train-step throughput")
    parser.add_argument("--config", required=True, help="Experiment YAML to profile")
    parser.add_argument("--num-batches", type=int, default=20, help="Number of train batches to profile")
    parser.add_argument("--num-workers", type=int, help="Override data.num_workers for the profiling run")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = resolve_paths_in_config(load_config(args.config))
    if args.num_workers is not None:
        config.setdefault("data", {})
        config["data"]["num_workers"] = args.num_workers

    seed = int(config["experiment"]["seed"])
    set_global_seed(seed, deterministic=bool(config["experiment"].get("deterministic", False)))
    device = _device_from_config(config["experiment"])
    train_loader, _, _ = _build_dataloaders(config, seed=seed, device=device)

    report = {
        "config_path": str(Path(args.config).resolve()),
        "model": config["model"]["name"],
        "strategy": config["task"]["unknown_strategy"],
        "device": str(device),
        "batch_size": int(config["train"]["batch_size"]),
        "grad_accumulation_steps": int(config["train"].get("grad_accumulation_steps", 1)),
        "num_workers": int(config.get("data", {}).get("num_workers", 0)),
        "loader_only": _loader_profile(train_loader, args.num_batches),
        "train_step": _train_step_profile(config, train_loader, device, args.num_batches),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
