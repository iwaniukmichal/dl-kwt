from __future__ import annotations

import csv
import math
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from speech_kws.config import load_config
from speech_kws.data.labels import TARGET_KEYWORDS
from speech_kws.data.prepare import prepare_dataset
from speech_kws.models.wrappers import build_model
from speech_kws.utils.io import load_json, save_yaml


def _write_wav(path: Path, sample_rate: int = 16000, duration_sec: float = 1.0, amplitude: int = 1000) -> None:
    num_frames = int(sample_rate * duration_sec)
    frames = bytearray()
    for index in range(num_frames):
        value = int(amplitude * math.sin((2.0 * math.pi * index) / 80.0))
        frames.extend(int(value).to_bytes(2, "little", signed=True))

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))


def test_shared_backbone_configs_match_expected_shapes_and_parameter_ranges() -> None:
    expectations = {
        "bcresnet_strategy_b.yaml": {"feature_dim": 40, "frames": 98, "logits": 12, "min_params": 52000, "max_params": 57000},
        "kwt_strategy_b.yaml": {"feature_dim": 40, "frames": 98, "logits": 12, "min_params": 600000, "max_params": 615000},
        "matchboxnet_strategy_b.yaml": {"feature_dim": 64, "frames": 128, "logits": 12, "min_params": 64000, "max_params": 70000},
    }

    for config_name, expected in expectations.items():
        config = load_config(Path("configs/experiments/stage1") / config_name)
        model = build_model(
            model_config=config["model"],
            strategy=config["task"]["unknown_strategy"],
            frontend_feature_dim=expected["feature_dim"],
        )
        total_params = sum(parameter.numel() for parameter in model.parameters())
        outputs = model(torch.randn(2, expected["feature_dim"], expected["frames"]))

        assert expected["min_params"] <= total_params <= expected["max_params"]
        assert outputs["logits"].shape == (2, expected["logits"])


def test_build_stage2_manifest_generates_24_configs(tmp_path: Path, monkeypatch) -> None:
    from speech_kws.cli import build_stage2_manifest

    repo_root = tmp_path / "repo"
    outputs_root = repo_root / "outputs"
    winners_dir = outputs_root / "summaries" / "stage1"
    winners_dir.mkdir(parents=True, exist_ok=True)

    configs_dir = repo_root / "seed_configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    base_payloads = [
        {"experiment": {"id": "bcresnet_strategy_b", "stage": "stage1"}, "task": {"unknown_strategy": "b"}, "model": {"name": "bcresnet"}, "augment": {"time_shift": {"enabled": True}, "background_noise": {"enabled": True}, "specaugment": {"enabled": False}}},
        {"experiment": {"id": "kwt_strategy_c", "stage": "stage1"}, "task": {"unknown_strategy": "c"}, "model": {"name": "kwt"}, "augment": {"time_shift": {"enabled": True}, "background_noise": {"enabled": True}, "specaugment": {"enabled": False}}},
        {"experiment": {"id": "matchboxnet_strategy_d", "stage": "stage1"}, "task": {"unknown_strategy": "d"}, "model": {"name": "matchboxnet"}, "augment": {"time_shift": {"enabled": True}, "background_noise": {"enabled": True}, "specaugment": {"enabled": False}}},
    ]

    resolved_paths: list[str] = []
    for index, payload in enumerate(base_payloads):
        path = configs_dir / f"winner_{index}.yaml"
        save_yaml(path, payload)
        resolved_paths.append(str(path))

    winners_path = winners_dir / "winners.csv"
    with winners_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["resolved_config_path"])
        writer.writeheader()
        for path in resolved_paths:
            writer.writerow({"resolved_config_path": path})

    monkeypatch.setattr("speech_kws.cli.REPO_ROOT", repo_root)
    result = build_stage2_manifest(SimpleNamespace(output_root=str(outputs_root)))

    generated_configs = sorted((repo_root / "configs" / "experiments" / "stage2").glob("*.yaml"))
    manifest_entries = (repo_root / "configs" / "manifests" / "stage2.txt").read_text(encoding="utf-8").splitlines()

    assert Path(result["manifest"]) == repo_root / "configs" / "manifests" / "stage2.txt"
    assert len(generated_configs) == 24
    assert len(manifest_entries) == 24


def test_run_experiment_smoke_writes_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("torchaudio")
    from speech_kws.training.loops import run_experiment

    data_root = tmp_path / "data"
    audio_root = data_root / "audio"
    background_root = audio_root / "_background_noise_"
    _write_wav(background_root / "noise.wav", duration_sec=30.0)

    validation_entries: list[str] = []
    testing_entries: list[str] = []
    for label in TARGET_KEYWORDS:
        _write_wav(audio_root / label / f"{label}_train_nohash_0.wav")
        _write_wav(audio_root / label / f"{label}_val_nohash_0.wav")
        _write_wav(audio_root / label / f"{label}_test_nohash_0.wav")
        validation_entries.append(f"{label}/{label}_val_nohash_0.wav")
        testing_entries.append(f"{label}/{label}_test_nohash_0.wav")

    _write_wav(audio_root / "bed" / "bed_train_nohash_0.wav")

    validation_list = data_root / "validation_list.txt"
    testing_list = data_root / "testing_list.txt"
    validation_list.write_text("\n".join(validation_entries) + "\n", encoding="utf-8")
    testing_list.write_text("\n".join(testing_entries) + "\n", encoding="utf-8")

    prepared_root = data_root / "prepared"
    prepare_dataset(
        audio_root=audio_root,
        validation_list=validation_list,
        testing_list=testing_list,
        prepared_root=prepared_root,
    )

    output_root = tmp_path / "outputs"
    config = {
        "experiment": {"id": "smoke_bcresnet_strategy_b", "stage": "stage1", "seed": 7, "device": "cpu", "amp": False, "deterministic": False},
        "paths": {
            "audio_root": str(audio_root),
            "validation_list": str(validation_list),
            "testing_list": str(testing_list),
            "prepared_root": str(prepared_root),
            "output_root": str(output_root),
        },
        "task": {"unknown_strategy": "b", "clip_duration_sec": 1.0, "target_keywords": TARGET_KEYWORDS},
        "frontend": {"kind": "log_mel", "sample_rate": 16000, "n_mels": 40, "window_ms": 30.0, "hop_ms": 10.0, "expected_frames": 98},
        "augment": {
            "time_shift": {"enabled": False, "max_shift_ms": 100.0},
            "background_noise": {"enabled": False, "probability": 0.8, "gain_min": 0.0, "gain_max": 0.3},
            "specaugment": {"enabled": False, "time_masks": 1, "time_mask_width": 10, "freq_masks": 1, "freq_mask_width": 5},
        },
        "model": {"name": "bcresnet", "variant": "bcresnet-3", "base_channels": 24, "stage_blocks": [2, 2, 4, 4], "stride_stages": [1, 2], "spec_groups": 5, "dropout": 0.1},
        "train": {
            "mode": "epochs",
            "max_epochs": 1,
            "batch_size": 4,
            "grad_accumulation_steps": 1,
            "optimizer": {"name": "sgd", "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0},
            "scheduler": {"name": "none"},
        },
        "data": {"num_workers": 0, "pin_memory": False, "persistent_workers": False, "prefetch_factor": 2},
        "evaluation": {"batch_size": 4},
        "strategy": {},
        "outputs": {"on_exists": "skip"},
    }

    result = run_experiment(config)
    run_dir = Path(result["run_dir"])

    assert (run_dir / "train_history.csv").exists()
    assert (run_dir / "resolved_config.yaml").exists()
    assert (run_dir / "best.ckpt").exists()
    assert (run_dir / "last.ckpt").exists()
    assert (run_dir / "val_metrics.json").exists()
    assert (run_dir / "test_metrics.json").exists()
    assert (run_dir / "plots" / "learning_curve.png").exists()
    assert load_json(run_dir / "test_metrics.json")["accuracy"] >= 0.0
