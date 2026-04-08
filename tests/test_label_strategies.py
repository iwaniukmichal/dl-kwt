from __future__ import annotations

from speech_kws.data.labels import (
    EVAL_LABELS,
    HEAD1_LABELS,
    KNOWN_LABELS_STRATEGY_D,
    RAW_WORD_LABELS,
    eval_label_from_raw,
    sample_kind,
    strategy_label_spec,
)


def test_eval_label_mapping() -> None:
    assert eval_label_from_raw("yes") == "yes"
    assert eval_label_from_raw("bed") == "unknown"
    assert eval_label_from_raw("silence") == "silence"


def test_sample_kind_mapping() -> None:
    assert sample_kind("yes") == "keyword"
    assert sample_kind("bed") == "unknown"
    assert sample_kind("silence") == "silence"


def test_strategy_label_specs() -> None:
    assert strategy_label_spec("a").train_labels == RAW_WORD_LABELS + ["silence"]
    assert strategy_label_spec("b").train_labels == EVAL_LABELS
    assert strategy_label_spec("c").train_labels == HEAD1_LABELS
    assert strategy_label_spec("d").train_labels == KNOWN_LABELS_STRATEGY_D
