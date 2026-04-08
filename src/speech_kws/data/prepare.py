from __future__ import annotations

import math
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from speech_kws.data.labels import (
    SILENCE_LABEL,
    TARGET_KEYWORDS,
    eval_label_from_raw,
    sample_kind,
    speaker_id_from_relpath,
)
from speech_kws.utils.io import ensure_dir, save_csv, save_json


SPOKEN_MANIFEST = "spoken_manifest.csv"
SILENCE_MANIFEST = "silence_manifest.csv"
BACKGROUND_MANIFEST = "background_manifest.csv"
SUMMARY_JSON = "dataset_summary.json"


@dataclass(frozen=True)
class BackgroundRange:
    abs_path: str
    relpath: str
    duration_sec: float
    split: str
    start_sec: float
    end_sec: float

    @property
    def usable_duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def read_split_file(path: str | Path) -> set[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def get_wav_duration_sec(path: str | Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
    return frames / float(rate)


def _determine_split(relpath: str, validation_set: set[str], testing_set: set[str]) -> str:
    if relpath in validation_set:
        return "validation"
    if relpath in testing_set:
        return "test"
    return "train"


def _scan_spoken_records(
    audio_root: Path,
    validation_set: set[str],
    testing_set: set[str],
) -> list[dict]:
    records: list[dict] = []
    for label_dir in sorted(audio_root.iterdir()):
        if not label_dir.is_dir():
            continue
        if label_dir.name.startswith(".") or label_dir.name == "_background_noise_":
            continue

        for wav_path in sorted(label_dir.glob("*.wav")):
            relpath = wav_path.relative_to(audio_root).as_posix()
            raw_label = label_dir.name
            split = _determine_split(relpath, validation_set, testing_set)
            records.append(
                {
                    "sample_id": f"spoken::{relpath}",
                    "split": split,
                    "sample_type": "spoken",
                    "raw_label": raw_label,
                    "eval_label": eval_label_from_raw(raw_label),
                    "sample_kind": sample_kind(raw_label),
                    "speaker_id": speaker_id_from_relpath(relpath),
                    "relpath": relpath,
                    "abs_path": str(wav_path.resolve()),
                    "start_sec": 0.0,
                    "duration_sec": get_wav_duration_sec(wav_path),
                }
            )
    return records


def _average_target_count(records: Iterable[dict], split: str) -> int:
    counts = Counter()
    for record in records:
        if record["split"] == split and record["raw_label"] in TARGET_KEYWORDS:
            counts[record["raw_label"]] += 1

    present_counts = [counts[label] for label in TARGET_KEYWORDS if counts[label] > 0]
    if not present_counts:
        return 0
    return int(round(sum(present_counts) / len(present_counts)))


def _reserve_background_ranges(audio_root: Path) -> list[BackgroundRange]:
    background_dir = audio_root / "_background_noise_"
    if not background_dir.is_dir():
        raise FileNotFoundError(f"Missing background-noise directory: {background_dir}")

    ranges: list[BackgroundRange] = []
    for wav_path in sorted(background_dir.glob("*.wav")):
        duration_sec = get_wav_duration_sec(wav_path)
        train_end = duration_sec * 0.8
        validation_end = duration_sec * 0.9
        relpath = wav_path.relative_to(audio_root).as_posix()
        ranges.extend(
            [
                BackgroundRange(
                    abs_path=str(wav_path.resolve()),
                    relpath=relpath,
                    duration_sec=duration_sec,
                    split="train",
                    start_sec=0.0,
                    end_sec=train_end,
                ),
                BackgroundRange(
                    abs_path=str(wav_path.resolve()),
                    relpath=relpath,
                    duration_sec=duration_sec,
                    split="validation",
                    start_sec=train_end,
                    end_sec=validation_end,
                ),
                BackgroundRange(
                    abs_path=str(wav_path.resolve()),
                    relpath=relpath,
                    duration_sec=duration_sec,
                    split="test",
                    start_sec=validation_end,
                    end_sec=duration_sec,
                ),
            ]
        )
    return ranges


def _allocate_counts(weights: list[float], total_count: int) -> list[int]:
    if total_count <= 0:
        return [0 for _ in weights]
    if not weights or sum(weights) <= 0:
        raise ValueError("Cannot allocate silence counts without usable background ranges.")

    raw = [weight / sum(weights) * total_count for weight in weights]
    allocated = [int(math.floor(value)) for value in raw]
    remainder = total_count - sum(allocated)
    fractional_order = sorted(
        range(len(weights)),
        key=lambda idx: raw[idx] - allocated[idx],
        reverse=True,
    )
    for idx in fractional_order[:remainder]:
        allocated[idx] += 1
    return allocated


def _make_silence_rows(
    ranges: list[BackgroundRange],
    split: str,
    count: int,
    clip_duration_sec: float,
) -> list[dict]:
    split_ranges = [entry for entry in ranges if entry.split == split and entry.usable_duration >= clip_duration_sec]
    if count == 0:
        return []
    if not split_ranges:
        raise ValueError(f"No usable background ranges available for split '{split}'.")

    weights = [entry.usable_duration for entry in split_ranges]
    per_range_counts = _allocate_counts(weights, count)
    rows: list[dict] = []
    sample_index = 0
    for range_entry, range_count in zip(split_ranges, per_range_counts):
        if range_count == 0:
            continue
        max_start = max(range_entry.start_sec, range_entry.end_sec - clip_duration_sec)
        if range_count == 1 or max_start <= range_entry.start_sec:
            start_positions = [range_entry.start_sec]
        else:
            span = max_start - range_entry.start_sec
            start_positions = [
                range_entry.start_sec + (span * index / (range_count - 1))
                for index in range(range_count)
            ]

        for start_sec in start_positions:
            rows.append(
                {
                    "sample_id": f"silence::{split}::{sample_index:06d}",
                    "split": split,
                    "sample_type": SILENCE_LABEL,
                    "raw_label": SILENCE_LABEL,
                    "eval_label": SILENCE_LABEL,
                    "sample_kind": SILENCE_LABEL,
                    "speaker_id": "",
                    "relpath": range_entry.relpath,
                    "abs_path": range_entry.abs_path,
                    "start_sec": round(float(start_sec), 6),
                    "duration_sec": clip_duration_sec,
                }
            )
            sample_index += 1
    return rows


def prepare_dataset(
    audio_root: str | Path,
    validation_list: str | Path,
    testing_list: str | Path,
    prepared_root: str | Path,
    clip_duration_sec: float = 1.0,
) -> dict:
    audio_root = Path(audio_root).resolve()
    validation_list = Path(validation_list).resolve()
    testing_list = Path(testing_list).resolve()
    prepared_root = ensure_dir(Path(prepared_root).resolve())

    validation_set = read_split_file(validation_list)
    testing_set = read_split_file(testing_list)
    overlap = validation_set.intersection(testing_set)
    if overlap:
        raise ValueError("Validation and testing split files overlap.")

    spoken_records = _scan_spoken_records(audio_root, validation_set, testing_set)
    silence_counts = {
        split: _average_target_count(spoken_records, split)
        for split in ["train", "validation", "test"]
    }

    background_ranges = _reserve_background_ranges(audio_root)
    silence_records: list[dict] = []
    for split, count in silence_counts.items():
        silence_records.extend(
            _make_silence_rows(
                ranges=background_ranges,
                split=split,
                count=count,
                clip_duration_sec=clip_duration_sec,
            )
        )

    spoken_manifest_path = prepared_root / SPOKEN_MANIFEST
    save_csv(
        spoken_manifest_path,
        [
            "sample_id",
            "split",
            "sample_type",
            "raw_label",
            "eval_label",
            "sample_kind",
            "speaker_id",
            "relpath",
            "abs_path",
            "start_sec",
            "duration_sec",
        ],
        spoken_records,
    )

    silence_manifest_path = prepared_root / SILENCE_MANIFEST
    save_csv(
        silence_manifest_path,
        [
            "sample_id",
            "split",
            "sample_type",
            "raw_label",
            "eval_label",
            "sample_kind",
            "speaker_id",
            "relpath",
            "abs_path",
            "start_sec",
            "duration_sec",
        ],
        silence_records,
    )

    background_manifest_path = prepared_root / BACKGROUND_MANIFEST
    save_csv(
        background_manifest_path,
        ["abs_path", "relpath", "duration_sec", "split", "start_sec", "end_sec"],
        [
            {
                "abs_path": entry.abs_path,
                "relpath": entry.relpath,
                "duration_sec": round(entry.duration_sec, 6),
                "split": entry.split,
                "start_sec": round(entry.start_sec, 6),
                "end_sec": round(entry.end_sec, 6),
            }
            for entry in background_ranges
        ],
    )

    split_counts = defaultdict(Counter)
    for record in spoken_records + silence_records:
        split_counts[record["split"]]["total"] += 1
        split_counts[record["split"]][record["eval_label"]] += 1

    summary = {
        "audio_root": str(audio_root),
        "validation_list": str(validation_list),
        "testing_list": str(testing_list),
        "prepared_root": str(prepared_root),
        "spoken_manifest": str(spoken_manifest_path),
        "silence_manifest": str(silence_manifest_path),
        "background_manifest": str(background_manifest_path),
        "num_spoken_samples": len(spoken_records),
        "num_silence_samples": len(silence_records),
        "silence_counts": silence_counts,
        "split_counts": {split: dict(counter) for split, counter in split_counts.items()},
    }
    save_json(prepared_root / SUMMARY_JSON, summary)
    return summary
