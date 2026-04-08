from __future__ import annotations

import math
import wave
from pathlib import Path

from speech_kws.data.labels import TARGET_KEYWORDS
from speech_kws.data.prepare import prepare_dataset
from speech_kws.utils.io import load_csv, load_json


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


def test_prepare_dataset_builds_manifests(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    audio_root = data_root / "audio"
    background_root = audio_root / "_background_noise_"
    _write_wav(background_root / "noise.wav", duration_sec=30.0)

    validation_entries = []
    testing_entries = []
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
    summary = prepare_dataset(
        audio_root=audio_root,
        validation_list=validation_list,
        testing_list=testing_list,
        prepared_root=prepared_root,
    )

    spoken_rows = load_csv(prepared_root / "spoken_manifest.csv")
    silence_rows = load_csv(prepared_root / "silence_manifest.csv")
    dataset_summary = load_json(prepared_root / "dataset_summary.json")

    assert summary["num_spoken_samples"] == 31
    assert dataset_summary["silence_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert len(silence_rows) == 3
    assert {row["split"] for row in silence_rows} == {"train", "validation", "test"}
    assert any(row["raw_label"] == "bed" and row["eval_label"] == "unknown" for row in spoken_rows)
