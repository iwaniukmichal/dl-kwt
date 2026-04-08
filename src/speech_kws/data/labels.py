from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


TARGET_KEYWORDS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
]

RAW_WORD_LABELS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "bed",
    "bird",
    "cat",
    "dog",
    "eight",
    "five",
    "four",
    "happy",
    "house",
    "marvin",
    "nine",
    "one",
    "seven",
    "sheila",
    "six",
    "three",
    "tree",
    "two",
    "wow",
    "zero",
]

UNKNOWN_RAW_LABELS = [label for label in RAW_WORD_LABELS if label not in TARGET_KEYWORDS]
EVAL_LABELS = TARGET_KEYWORDS + ["unknown", "silence"]
KNOWN_LABELS_STRATEGY_D = TARGET_KEYWORDS + ["silence"]
HEAD1_LABELS = ["silence", "unknown", "keyword"]
SILENCE_LABEL = "silence"
UNKNOWN_LABEL = "unknown"
KEYWORD_SUPERCLASS = "keyword"


@dataclass(frozen=True)
class StrategyLabelSpec:
    strategy: str
    train_labels: List[str]
    eval_labels: List[str]


def speaker_id_from_relpath(relpath: str) -> str:
    stem = Path(relpath).stem
    return stem.split("_nohash_")[0]


def eval_label_from_raw(raw_label: str) -> str:
    if raw_label == SILENCE_LABEL:
        return SILENCE_LABEL
    if raw_label in TARGET_KEYWORDS:
        return raw_label
    return UNKNOWN_LABEL


def strategy_label_spec(strategy: str) -> StrategyLabelSpec:
    normalized = strategy.lower()
    if normalized == "a":
        train_labels = RAW_WORD_LABELS + [SILENCE_LABEL]
    elif normalized == "b":
        train_labels = EVAL_LABELS
    elif normalized == "c":
        train_labels = HEAD1_LABELS
    elif normalized == "d":
        train_labels = KNOWN_LABELS_STRATEGY_D
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    return StrategyLabelSpec(strategy=normalized, train_labels=train_labels, eval_labels=EVAL_LABELS)


def build_index(label_names: List[str]) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(label_names)}


def is_target_label(raw_label: str) -> bool:
    return raw_label in TARGET_KEYWORDS


def sample_kind(raw_label: str) -> str:
    if raw_label == SILENCE_LABEL:
        return SILENCE_LABEL
    if raw_label in TARGET_KEYWORDS:
        return KEYWORD_SUPERCLASS
    return UNKNOWN_LABEL
