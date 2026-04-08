from __future__ import annotations

import warnings
from pathlib import Path

import torch

from speech_kws.audio.frontend import AudioFrontend
from speech_kws.data import dataset as dataset_module
from speech_kws.data.dataset import SpeechCommandsDataset
from speech_kws.data.labels import SILENCE_LABEL


def test_matchboxnet_mfcc_frontend_is_warning_free_and_finite() -> None:
    config = {
        "kind": "mfcc",
        "sample_rate": 16000,
        "n_mfcc": 64,
        "window_ms": 25.0,
        "overlap_ms": 10.0,
        "expected_frames": 128,
    }
    waveforms = torch.randn(2, 16000)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frontend = AudioFrontend(config)
        features = frontend(waveforms)

    assert bool(torch.isfinite(features).all().item())
    assert features.shape == (2, 64, 128)
    assert not [warning for warning in caught if "mel filterbank" in str(warning.message)]


def test_load_audio_crop_opens_wave_file_once(
    tiny_prepared_dataset: dict[str, Path],
    monkeypatch,
) -> None:
    dataset = SpeechCommandsDataset(
        prepared_root=tiny_prepared_dataset["prepared_root"],
        split="train",
        strategy="b",
        augment_config={},
        training=False,
    )
    original_open = dataset_module.wave.open
    open_count = 0

    def counted_open(*args, **kwargs):
        nonlocal open_count
        open_count += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(dataset_module.wave, "open", counted_open)

    _ = dataset[0]

    assert open_count == 1


def test_background_augmentation_opens_one_foreground_and_one_noise_crop(
    tiny_prepared_dataset: dict[str, Path],
    monkeypatch,
) -> None:
    dataset = SpeechCommandsDataset(
        prepared_root=tiny_prepared_dataset["prepared_root"],
        split="train",
        strategy="b",
        augment_config={
            "time_shift": {"enabled": False, "max_shift_ms": 100.0},
            "background_noise": {"enabled": True, "probability": 1.0, "gain_min": 0.0, "gain_max": 0.3},
        },
        training=True,
    )
    sample_index = next(index for index, record in enumerate(dataset.records) if record["raw_label"] != SILENCE_LABEL)
    original_open = dataset_module.wave.open
    open_count = 0

    def counted_open(*args, **kwargs):
        nonlocal open_count
        open_count += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(dataset_module.wave, "open", counted_open)
    monkeypatch.setattr(dataset_module.random, "choice", lambda pool: pool[0])

    _ = dataset[sample_index]

    assert open_count == 2
