from __future__ import annotations

import math
import wave
from pathlib import Path

import pytest

from speech_kws.data.labels import TARGET_KEYWORDS
from speech_kws.data.prepare import prepare_dataset


def write_wav(path: Path, sample_rate: int = 16000, duration_sec: float = 1.0, amplitude: int = 1000) -> None:
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


@pytest.fixture
def tiny_prepared_dataset(tmp_path: Path) -> dict[str, Path]:
    data_root = tmp_path / "data"
    audio_root = data_root / "audio"
    background_root = audio_root / "_background_noise_"
    write_wav(background_root / "noise.wav", duration_sec=30.0)

    validation_entries: list[str] = []
    testing_entries: list[str] = []
    for label in TARGET_KEYWORDS:
        write_wav(audio_root / label / f"{label}_train_nohash_0.wav")
        write_wav(audio_root / label / f"{label}_val_nohash_0.wav")
        write_wav(audio_root / label / f"{label}_test_nohash_0.wav")
        validation_entries.append(f"{label}/{label}_val_nohash_0.wav")
        testing_entries.append(f"{label}/{label}_test_nohash_0.wav")

    for unknown_label in ["bed", "cat"]:
        write_wav(audio_root / unknown_label / f"{unknown_label}_train_nohash_0.wav")
        write_wav(audio_root / unknown_label / f"{unknown_label}_val_nohash_0.wav")
        write_wav(audio_root / unknown_label / f"{unknown_label}_test_nohash_0.wav")
        validation_entries.append(f"{unknown_label}/{unknown_label}_val_nohash_0.wav")
        testing_entries.append(f"{unknown_label}/{unknown_label}_test_nohash_0.wav")

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

    return {
        "data_root": data_root,
        "audio_root": audio_root,
        "validation_list": validation_list,
        "testing_list": testing_list,
        "prepared_root": prepared_root,
        "output_root": tmp_path / "outputs",
    }
