from __future__ import annotations

from typing import Sequence

import numpy as np

from speech_kws.data.labels import EVAL_LABELS, KNOWN_LABELS_STRATEGY_D
from speech_kws.evaluation.metrics import compute_classification_metrics


KNOWN_TO_EVAL_INDEX = [EVAL_LABELS.index(label) for label in KNOWN_LABELS_STRATEGY_D]


def threshold_values_from_config(config: dict) -> list[float]:
    if "values" in config:
        return [float(value) for value in config["values"]]

    start = float(config.get("start", 0.0))
    stop = float(config.get("stop", 1.0))
    step = float(config.get("step", 0.01))
    values = []
    current = start
    while current <= stop + (step / 2):
        values.append(round(current, 6))
        current += step
    return values


def apply_rejection(probs: np.ndarray, threshold: float) -> tuple[list[int], list[float]]:
    confidences = probs.max(axis=1)
    known_predictions = probs.argmax(axis=1)
    predicted_eval = []
    for known_idx, confidence in zip(known_predictions, confidences):
        if confidence < threshold:
            predicted_eval.append(EVAL_LABELS.index("unknown"))
        else:
            predicted_eval.append(KNOWN_TO_EVAL_INDEX[int(known_idx)])
    return predicted_eval, confidences.astype(float).tolist()


def sweep_thresholds(
    probs: np.ndarray,
    y_true: Sequence[int],
    threshold_config: dict,
) -> dict:
    threshold_values = threshold_values_from_config(threshold_config)
    rows = []
    best = None

    for tau in threshold_values:
        predicted_eval, confidences = apply_rejection(probs, tau)
        metrics = compute_classification_metrics(y_true=y_true, y_pred=predicted_eval, label_names=EVAL_LABELS)
        row = {
            "tau": tau,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "keyword_macro_f1": metrics["keyword_macro_f1"],
        }
        rows.append(row)
        if best is None or row["macro_f1"] > best["row"]["macro_f1"]:
            best = {
                "tau": tau,
                "metrics": metrics,
                "row": row,
                "predictions": predicted_eval,
                "confidences": confidences,
            }

    if best is None:
        raise ValueError("Threshold sweep received no threshold values.")

    return {
        "best_tau": float(best["tau"]),
        "best_metrics": best["metrics"],
        "rows": rows,
        "predictions": best["predictions"],
        "confidences": best["confidences"],
    }
