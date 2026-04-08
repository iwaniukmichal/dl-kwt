from __future__ import annotations

import random
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from speech_kws.audio.augment import (
    mix_background,
    pad_or_trim_waveform,
    random_time_shift,
)
from speech_kws.data.labels import (
    EVAL_LABELS,
    HEAD1_LABELS,
    KEYWORD_SUPERCLASS,
    KNOWN_LABELS_STRATEGY_D,
    SILENCE_LABEL,
    TARGET_KEYWORDS,
    UNKNOWN_LABEL,
    build_index,
    sample_kind,
    strategy_label_spec,
)
from speech_kws.utils.io import load_csv, load_json


@dataclass(frozen=True)
class ManifestPaths:
    spoken_manifest: Path
    silence_manifest: Path
    summary_json: Path


def prepared_paths(prepared_root: str | Path) -> ManifestPaths:
    root = Path(prepared_root).resolve()
    return ManifestPaths(
        spoken_manifest=root / "spoken_manifest.csv",
        silence_manifest=root / "silence_manifest.csv",
        summary_json=root / "dataset_summary.json",
    )


def _load_records(prepared_root: str | Path, split: str) -> list[dict]:
    paths = prepared_paths(prepared_root)
    if not paths.spoken_manifest.exists():
        raise FileNotFoundError(f"Missing prepared spoken manifest: {paths.spoken_manifest}")
    if not paths.silence_manifest.exists():
        raise FileNotFoundError(f"Missing prepared silence manifest: {paths.silence_manifest}")

    spoken_records = [row for row in load_csv(paths.spoken_manifest) if row["split"] == split]
    silence_records = [row for row in load_csv(paths.silence_manifest) if row["split"] == split]
    return spoken_records + silence_records


def load_dataset_summary(prepared_root: str | Path) -> dict:
    paths = prepared_paths(prepared_root)
    if not paths.summary_json.exists():
        raise FileNotFoundError(f"Missing dataset summary: {paths.summary_json}")
    return load_json(paths.summary_json)


class SpeechCommandsDataset(Dataset):
    def __init__(
        self,
        prepared_root: str | Path,
        split: str,
        strategy: str,
        augment_config: dict | None = None,
        sample_rate: int = 16000,
        clip_duration_sec: float = 1.0,
        training: bool = False,
    ) -> None:
        self.prepared_root = Path(prepared_root).resolve()
        self.split = split
        self.training = training
        self.sample_rate = sample_rate
        self.clip_num_samples = int(sample_rate * clip_duration_sec)
        self.strategy = strategy.lower()
        self.strategy_spec = strategy_label_spec(self.strategy)
        self.records = _load_records(self.prepared_root, split)
        self.summary = load_dataset_summary(self.prepared_root)
        self.augment_config = augment_config or {}

        self.eval_index = build_index(EVAL_LABELS)
        self.train_index = build_index(self.strategy_spec.train_labels)
        self.head1_index = build_index(HEAD1_LABELS)
        self.known_index = build_index(KNOWN_LABELS_STRATEGY_D)
        self.keyword_index = build_index(TARGET_KEYWORDS)

        self.train_background_pool: list[dict] = []
        if training and self.augment_config.get("background_noise", {}).get("enabled", False):
            self.train_background_pool = [
                row
                for row in load_csv(prepared_paths(self.prepared_root).silence_manifest)
                if row["split"] == "train"
            ]

        self.indices_by_kind: Dict[str, list[int]] = {
            SILENCE_LABEL: [],
            UNKNOWN_LABEL: [],
            KEYWORD_SUPERCLASS: [],
        }
        self.indices_by_unknown_raw_label: Dict[str, list[int]] = {}
        for index, record in enumerate(self.records):
            group = sample_kind(record["raw_label"])
            self.indices_by_kind[group].append(index)
            if group == UNKNOWN_LABEL:
                self.indices_by_unknown_raw_label.setdefault(record["raw_label"], []).append(index)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def target_counts(self) -> Dict[str, int]:
        counts = {label: 0 for label in TARGET_KEYWORDS}
        for record in self.records:
            if record["raw_label"] in counts:
                counts[record["raw_label"]] += 1
        return counts

    @property
    def average_target_count(self) -> int:
        counts = [value for value in self.target_counts.values() if value > 0]
        if not counts:
            return 0
        return int(round(sum(counts) / len(counts)))

    @property
    def silence_count(self) -> int:
        return len(self.indices_by_kind[SILENCE_LABEL])

    @property
    def keyword_count(self) -> int:
        return len(self.indices_by_kind[KEYWORD_SUPERCLASS])

    def _read_wav_frames(self, path: str | Path, frame_offset: int, num_frames: int) -> tuple[torch.Tensor, int]:
        with wave.open(str(path), "rb") as handle:
            num_channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            total_frames = handle.getnframes()

            start_frame = min(max(0, frame_offset), total_frames)
            handle.setpos(start_frame)
            raw_frames = handle.readframes(max(0, num_frames))

        if not raw_frames:
            return torch.zeros((1, 0), dtype=torch.float32), sample_rate

        if sample_width == 1:
            array = np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32)
            array = (array - 128.0) / 128.0
        elif sample_width == 2:
            array = np.frombuffer(raw_frames, dtype="<i2").astype(np.float32) / 32768.0
        elif sample_width == 4:
            array = np.frombuffer(raw_frames, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

        waveform = torch.from_numpy(array).view(-1, num_channels).transpose(0, 1).contiguous()
        return waveform, sample_rate

    def _resample_waveform(self, waveform: torch.Tensor, source_rate: int) -> torch.Tensor:
        if source_rate == self.sample_rate:
            return pad_or_trim_waveform(waveform, self.clip_num_samples)

        target_duration_sec = self.clip_num_samples / float(self.sample_rate)
        expected_source_samples = int(round(target_duration_sec * source_rate))
        waveform = pad_or_trim_waveform(waveform, expected_source_samples)
        resampled = F.interpolate(
            waveform.unsqueeze(0),
            size=self.clip_num_samples,
            mode="linear",
            align_corners=False,
        )
        return resampled.squeeze(0)

    def _load_audio_crop(self, record: dict) -> torch.Tensor:
        target_duration_sec = self.clip_num_samples / float(self.sample_rate)
        with wave.open(str(record["abs_path"]), "rb") as handle:
            file_sample_rate = handle.getframerate()
        frame_offset = int(round(float(record["start_sec"]) * file_sample_rate))
        source_num_frames = int(round(target_duration_sec * file_sample_rate))
        waveform, file_sample_rate = self._read_wav_frames(
            record["abs_path"],
            frame_offset=frame_offset,
            num_frames=source_num_frames,
        )
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return self._resample_waveform(waveform, source_rate=file_sample_rate)

    def _maybe_apply_waveform_augmentations(self, waveform: torch.Tensor, record: dict) -> torch.Tensor:
        if not self.training:
            return waveform

        time_shift_cfg = self.augment_config.get("time_shift", {})
        if time_shift_cfg.get("enabled", False) and record["raw_label"] != SILENCE_LABEL:
            max_shift_ms = float(time_shift_cfg.get("max_shift_ms", 100.0))
            waveform = random_time_shift(waveform, int(self.sample_rate * max_shift_ms / 1000.0))

        background_cfg = self.augment_config.get("background_noise", {})
        if (
            background_cfg.get("enabled", False)
            and record["raw_label"] != SILENCE_LABEL
            and self.train_background_pool
            and torch.rand(1).item() < float(background_cfg.get("probability", 0.8))
        ):
            noise_record = random.choice(self.train_background_pool)
            noise = self._load_audio_crop(noise_record)
            min_gain = float(background_cfg.get("gain_min", 0.0))
            max_gain = float(background_cfg.get("gain_max", 0.3))
            gain = torch.empty(1).uniform_(min_gain, max_gain).item()
            waveform = mix_background(waveform, noise, gain)

        return waveform

    def _strategy_targets(self, record: dict) -> dict:
        raw_label = record["raw_label"]
        eval_label = record["eval_label"]
        targets = {
            "eval_target": self.eval_index[eval_label],
            "train_target": -100,
            "head1_target": -100,
            "head2_target": -100,
            "known_target": -100,
            "is_outlier": eval_label == UNKNOWN_LABEL,
        }

        if self.strategy == "a":
            targets["train_target"] = self.train_index[raw_label]
        elif self.strategy == "b":
            targets["train_target"] = self.train_index[eval_label]
        elif self.strategy == "c":
            superclass = sample_kind(raw_label)
            targets["head1_target"] = self.head1_index[superclass]
            if raw_label in TARGET_KEYWORDS:
                targets["head2_target"] = self.keyword_index[raw_label]
        elif self.strategy == "d":
            if eval_label != UNKNOWN_LABEL:
                targets["known_target"] = self.known_index[eval_label]
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

        return targets

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        waveform = self._load_audio_crop(record)
        waveform = self._maybe_apply_waveform_augmentations(waveform, record)
        targets = self._strategy_targets(record)
        return {
            "waveform": waveform,
            "sample_id": record["sample_id"],
            "relpath": record["relpath"],
            "raw_label": record["raw_label"],
            "eval_label_name": record["eval_label"],
            "sample_kind": sample_kind(record["raw_label"]),
            **targets,
        }
