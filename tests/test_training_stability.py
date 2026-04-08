from __future__ import annotations

from pathlib import Path

import pytest
import torch

from speech_kws.audio.frontend import AudioFrontend
from speech_kws.config import load_config
from speech_kws.models.wrappers import build_model
from speech_kws.training.losses import compute_strategy_loss, uniform_target_loss
from speech_kws.training.loops import _build_dataloaders, run_experiment


STAGE1_CONFIG_NAMES = sorted(path.name for path in Path("configs/experiments/stage1").glob("*.yaml"))


def _config_for_tiny_dataset(config_name: str, dataset_paths: dict[str, Path]) -> dict:
    config = load_config(Path("configs/experiments/stage1") / config_name)
    config["experiment"]["device"] = "cpu"
    config["experiment"]["amp"] = False
    config["paths"] = {
        "audio_root": str(dataset_paths["audio_root"]),
        "validation_list": str(dataset_paths["validation_list"]),
        "testing_list": str(dataset_paths["testing_list"]),
        "prepared_root": str(dataset_paths["prepared_root"]),
        "output_root": str(dataset_paths["output_root"]),
    }
    config["data"] = {
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 2,
    }
    config["outputs"] = {"on_exists": "error"}
    return config


def _smoke_run_config(dataset_paths: dict[str, Path]) -> dict:
    config = _config_for_tiny_dataset("bcresnet_strategy_b.yaml", dataset_paths)
    config["augment"] = {
        "time_shift": {"enabled": False, "max_shift_ms": 100.0},
        "background_noise": {"enabled": False, "probability": 0.8, "gain_min": 0.0, "gain_max": 0.3},
        "specaugment": {"enabled": False, "time_masks": 1, "time_mask_width": 10, "freq_masks": 1, "freq_mask_width": 5},
    }
    config["train"]["max_epochs"] = 1
    config["train"]["batch_size"] = 4
    config["evaluation"] = {"batch_size": 4}
    return config


@pytest.mark.parametrize("config_name", STAGE1_CONFIG_NAMES)
def test_stage1_configs_produce_finite_features_losses_and_gradients(
    config_name: str,
    tiny_prepared_dataset: dict[str, Path],
) -> None:
    config = _config_for_tiny_dataset(config_name, tiny_prepared_dataset)
    train_loader, _, _ = _build_dataloaders(config, seed=int(config["experiment"]["seed"]), device=torch.device("cpu"))
    batch = next(iter(train_loader))

    frontend = AudioFrontend(config["frontend"])
    model = build_model(
        model_config=config["model"],
        strategy=config["task"]["unknown_strategy"],
        frontend_feature_dim=frontend.feature_dim,
    )
    frontend.train()
    model.train()

    features = frontend(batch["waveform"])
    outputs = model(features)
    loss_dict = compute_strategy_loss(
        config["task"]["unknown_strategy"],
        outputs,
        batch,
        config.get("strategy"),
    )
    loss = loss_dict["loss"]

    assert bool(torch.isfinite(features).all().item())
    for value in outputs.values():
        if torch.is_tensor(value):
            assert bool(torch.isfinite(value).all().item())
    assert bool(torch.isfinite(loss).all().item())
    for key, value in loss_dict.items():
        if key == "loss":
            continue
        assert bool(torch.isfinite(torch.tensor(float(value))).item())

    loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert any(grad is not None for grad in grads)
    assert all(grad is None or bool(torch.isfinite(grad).all().item()) for grad in grads)


def test_strategy_c_loss_is_finite_without_keyword_samples() -> None:
    outputs = {
        "head1_logits": torch.randn(4, 3, requires_grad=True),
        "head2_logits": torch.randn(4, 10, requires_grad=True),
    }
    batch = {
        "head1_target": torch.tensor([0, 1, 1, 0], dtype=torch.long),
        "head2_target": torch.full((4,), -100, dtype=torch.long),
    }

    loss_dict = compute_strategy_loss("c", outputs, batch, {"lambda_head2": 1.5})
    loss_dict["loss"].backward()

    assert bool(torch.isfinite(loss_dict["loss"]).all().item())
    assert loss_dict["head2_loss"] == 0.0
    assert outputs["head1_logits"].grad is not None
    assert outputs["head2_logits"].grad is None


def test_strategy_d_loss_is_finite_without_known_samples() -> None:
    outputs = {"logits": torch.randn(4, 11, requires_grad=True)}
    batch = {
        "known_target": torch.full((4,), -100, dtype=torch.long),
        "is_outlier": torch.tensor([True, True, True, True]),
    }

    loss_dict = compute_strategy_loss("d", outputs, batch, {"oe_weight": 0.5})
    loss_dict["loss"].backward()

    assert bool(torch.isfinite(loss_dict["loss"]).all().item())
    assert loss_dict["known_loss"] == 0.0
    assert outputs["logits"].grad is not None
    assert bool(torch.isfinite(outputs["logits"].grad).all().item())


def test_strategy_d_loss_is_finite_without_outlier_samples() -> None:
    outputs = {"logits": torch.randn(4, 11, requires_grad=True)}
    batch = {
        "known_target": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "is_outlier": torch.tensor([False, False, False, False]),
    }

    loss_dict = compute_strategy_loss("d", outputs, batch, {"oe_weight": 0.5})
    loss_dict["loss"].backward()

    assert bool(torch.isfinite(loss_dict["loss"]).all().item())
    assert loss_dict["oe_loss"] == 0.0
    assert outputs["logits"].grad is not None
    assert bool(torch.isfinite(outputs["logits"].grad).all().item())


def test_uniform_target_loss_is_finite_for_extreme_logits() -> None:
    logits = torch.tensor(
        [[1000.0, -1000.0, 0.0], [-500.0, 500.0, 10.0]],
        requires_grad=True,
    )

    loss = uniform_target_loss(logits)
    loss.backward()

    assert bool(torch.isfinite(loss).all().item())
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all().item())


def test_run_experiment_raises_on_nonfinite_loss(
    tiny_prepared_dataset: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from speech_kws.training import loops

    config = _smoke_run_config(tiny_prepared_dataset)

    def bad_loss(*args, **kwargs):
        logits = kwargs["outputs"]["logits"] if "outputs" in kwargs else args[1]["logits"]
        loss = logits.sum() * torch.tensor(float("nan"))
        return {"loss": loss, "classification_loss": float("nan")}

    monkeypatch.setattr(loops, "compute_strategy_loss", bad_loss)

    with pytest.raises(RuntimeError, match="non-finite loss"):
        run_experiment(config)


def test_run_experiment_raises_on_nonfinite_gradients(
    tiny_prepared_dataset: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from speech_kws.training import loops

    config = _smoke_run_config(tiny_prepared_dataset)

    def bad_gradient_loss(*args, **kwargs):
        logits = kwargs["outputs"]["logits"] if "outputs" in kwargs else args[1]["logits"]
        logits.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        loss = logits.mean()
        return {"loss": loss, "classification_loss": float(loss.detach().item())}

    monkeypatch.setattr(loops, "compute_strategy_loss", bad_gradient_loss)

    with pytest.raises(RuntimeError, match="non-finite gradients"):
        run_experiment(config)
