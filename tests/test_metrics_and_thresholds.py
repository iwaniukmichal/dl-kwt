from __future__ import annotations

import numpy as np

from speech_kws.data.labels import EVAL_LABELS
from speech_kws.evaluation.metrics import compute_classification_metrics
from speech_kws.evaluation.thresholds import sweep_thresholds


def test_compute_classification_metrics() -> None:
    metrics = compute_classification_metrics(
        y_true=[0, 1, 10, 11],
        y_pred=[0, 1, 10, 11],
        label_names=EVAL_LABELS,
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_threshold_sweep_selects_rejection_threshold() -> None:
    probs = np.array(
        [
            [0.95] + [0.005] * 10,
            [0.40] + [0.06] * 10,
        ],
        dtype=np.float64,
    )
    result = sweep_thresholds(
        probs=probs,
        y_true=[0, 10],
        threshold_config={"start": 0.0, "stop": 0.9, "step": 0.3},
    )
    assert 0.0 <= result["best_tau"] <= 0.9
    assert len(result["rows"]) == 4
