from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from speech_kws.data.labels import TARGET_KEYWORDS
from speech_kws.utils.io import ensure_dir


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Sequence[str],
) -> dict:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    label_indices = list(range(len(label_names)))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=label_indices,
        zero_division=0,
    )
    raw_confusion = confusion_matrix(y_true_arr, y_pred_arr, labels=label_indices)
    normalized_confusion = raw_confusion.astype(np.float64)
    row_sums = normalized_confusion.sum(axis=1, keepdims=True)
    normalized_confusion = np.divide(
        normalized_confusion,
        row_sums,
        out=np.zeros_like(normalized_confusion, dtype=np.float64),
        where=row_sums != 0,
    )

    keyword_indices = [label_names.index(label) for label in TARGET_KEYWORDS if label in label_names]
    keyword_mask = np.isin(y_true_arr, keyword_indices)
    if keyword_mask.any():
        keyword_macro_f1 = float(
            f1_score(
                y_true_arr[keyword_mask],
                y_pred_arr[keyword_mask],
                labels=keyword_indices,
                average="macro",
                zero_division=0,
            )
        )
    else:
        keyword_macro_f1 = 0.0

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "keyword_macro_f1": keyword_macro_f1,
        "per_class": [
            {
                "label": label_names[index],
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in label_indices
        ],
        "confusion_matrix_raw": raw_confusion.astype(int).tolist(),
        "confusion_matrix_row_normalized": normalized_confusion.tolist(),
    }


def save_confusion_plot(
    matrix: Sequence[Sequence[float]],
    label_names: Sequence[str],
    destination: str | Path,
    title: str,
    value_format: str = ".2f",
) -> None:
    import matplotlib.pyplot as plt

    destination = Path(destination)
    ensure_dir(destination.parent)
    matrix_arr = np.asarray(matrix)

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix_arr, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    threshold = matrix_arr.max() / 2.0 if matrix_arr.size else 0.0
    for row_idx in range(matrix_arr.shape[0]):
        for col_idx in range(matrix_arr.shape[1]):
            value = matrix_arr[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                format(value, value_format),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def build_prediction_rows(
    sample_ids: Sequence[str],
    relpaths: Sequence[str],
    raw_labels: Sequence[str],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Sequence[str],
    confidences: Sequence[float],
) -> list[dict]:
    rows = []
    for sample_id, relpath, raw_label, true_idx, pred_idx, confidence in zip(
        sample_ids,
        relpaths,
        raw_labels,
        y_true,
        y_pred,
        confidences,
    ):
        rows.append(
            {
                "sample_id": sample_id,
                "relpath": relpath,
                "raw_label": raw_label,
                "true_label": label_names[int(true_idx)],
                "predicted_label": label_names[int(pred_idx)],
                "confidence": float(confidence),
            }
        )
    return rows
