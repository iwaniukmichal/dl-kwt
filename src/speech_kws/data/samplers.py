from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
from torch.utils.data import BatchSampler, WeightedRandomSampler

from speech_kws.data.labels import KEYWORD_SUPERCLASS, SILENCE_LABEL, UNKNOWN_LABEL
from speech_kws.utils.reproducibility import build_torch_generator


@dataclass
class SamplerBundle:
    sampler: Optional[WeightedRandomSampler] = None
    batch_sampler: Optional[BatchSampler] = None
    shuffle: bool = False


class _CyclingGroup:
    def __init__(self, indices: list[int], rng: random.Random) -> None:
        if not indices:
            raise ValueError("Balanced batch sampling requires non-empty groups.")
        self.indices = list(indices)
        self.rng = rng
        self.buffer: list[int] = []

    def draw(self, count: int) -> list[int]:
        batch: list[int] = []
        while len(batch) < count:
            if not self.buffer:
                self.buffer = self.indices.copy()
                self.rng.shuffle(self.buffer)
            batch.append(self.buffer.pop())
        return batch


class SuperclassBalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size: int, seed: int) -> None:
        if batch_size < 3:
            raise ValueError("Strategy C batch size must be at least 3.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        group_cyclers = {
            SILENCE_LABEL: _CyclingGroup(self.dataset.indices_by_kind[SILENCE_LABEL], rng),
            UNKNOWN_LABEL: _CyclingGroup(self.dataset.indices_by_kind[UNKNOWN_LABEL], rng),
            KEYWORD_SUPERCLASS: _CyclingGroup(self.dataset.indices_by_kind[KEYWORD_SUPERCLASS], rng),
        }

        base = self.batch_size // 3
        composition = {
            SILENCE_LABEL: base,
            UNKNOWN_LABEL: base,
            KEYWORD_SUPERCLASS: self.batch_size - (2 * base),
        }

        max_group = max(len(indices) for indices in self.dataset.indices_by_kind.values())
        num_batches = max(1, math.ceil((3 * max_group) / self.batch_size))

        for _ in range(num_batches):
            batch = []
            for group_name, count in composition.items():
                batch.extend(group_cyclers[group_name].draw(count))
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        max_group = max(len(indices) for indices in self.dataset.indices_by_kind.values())
        return max(1, math.ceil((3 * max_group) / self.batch_size))


def _weighted_sampler_for_unknown_balancing(dataset, seed: int) -> WeightedRandomSampler:
    weights = torch.zeros(len(dataset), dtype=torch.double)

    target_total = dataset.keyword_count
    silence_total = dataset.silence_count
    unknown_target_total = max(1, dataset.average_target_count)

    for index in dataset.indices_by_kind[KEYWORD_SUPERCLASS]:
        weights[index] = 1.0
    for index in dataset.indices_by_kind[SILENCE_LABEL]:
        weights[index] = 1.0

    unknown_groups = dataset.indices_by_unknown_raw_label
    if unknown_groups:
        unknown_class_budget = unknown_target_total / len(unknown_groups)
        for indices in unknown_groups.values():
            for index in indices:
                weights[index] = unknown_class_budget / max(1, len(indices))

    num_samples = max(1, target_total + silence_total + unknown_target_total)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True,
        generator=build_torch_generator(seed),
    )


def build_training_sampler(dataset, strategy: str, batch_size: int, seed: int) -> SamplerBundle:
    normalized = strategy.lower()
    if normalized in {"b", "d"}:
        return SamplerBundle(
            sampler=_weighted_sampler_for_unknown_balancing(dataset, seed),
            batch_sampler=None,
            shuffle=False,
        )
    if normalized == "c":
        return SamplerBundle(
            sampler=None,
            batch_sampler=SuperclassBalancedBatchSampler(dataset, batch_size=batch_size, seed=seed),
            shuffle=False,
        )
    return SamplerBundle(sampler=None, batch_sampler=None, shuffle=True)
