from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from speech_kws.utils.io import ensure_dir, load_json, load_yaml


STRATEGY_TIEBREAK = {"b": 0, "c": 1, "d": 2, "a": 3}


def _collect_run_rows(runs_root: Path) -> list[dict]:
    rows = []
    for metrics_path in runs_root.rglob("test_metrics.json"):
        run_dir = metrics_path.parent
        val_metrics_path = run_dir / "val_metrics.json"
        config_path = run_dir / "resolved_config.yaml"
        if not val_metrics_path.exists() or not config_path.exists():
            continue

        test_metrics = load_json(metrics_path)
        val_metrics = load_json(val_metrics_path)
        config = load_yaml(config_path)

        rows.append(
            {
                "stage": config["experiment"]["stage"],
                "experiment_id": config["experiment"]["id"],
                "seed": config["experiment"]["seed"],
                "backbone": config["model"]["name"],
                "strategy": config["task"]["unknown_strategy"],
                "resolved_config_path": str(config_path),
                "run_dir": str(run_dir),
                "val_accuracy": val_metrics.get("accuracy", 0.0),
                "val_macro_f1": val_metrics.get("macro_f1", 0.0),
                "val_keyword_macro_f1": val_metrics.get("keyword_macro_f1", 0.0),
                "test_accuracy": test_metrics.get("accuracy", 0.0),
                "test_macro_f1": test_metrics.get("macro_f1", 0.0),
                "test_keyword_macro_f1": test_metrics.get("keyword_macro_f1", 0.0),
                "selected_tau": val_metrics.get("selected_tau"),
            }
        )
    return rows


def _save_bar_plot(frame: pd.DataFrame, destination: Path, metric_key: str, title: str) -> None:
    if frame.empty:
        return
    ensure_dir(destination.parent)
    plot_frame = frame.sort_values(metric_key, ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(plot_frame["experiment_id"], plot_frame[metric_key])
    ax.set_title(title)
    ax.set_ylabel(metric_key)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _stage_summary(frame: pd.DataFrame, stage: str, output_root: Path) -> pd.DataFrame:
    stage_frame = frame[frame["stage"] == stage].copy()
    if stage_frame.empty:
        return stage_frame

    stage_dir = ensure_dir(output_root / "summaries" / stage)
    stage_frame.to_csv(stage_dir / "summary.csv", index=False)
    _save_bar_plot(
        stage_frame,
        stage_dir / "val_macro_f1.png",
        metric_key="val_macro_f1",
        title=f"{stage} validation macro-F1",
    )
    _save_bar_plot(
        stage_frame,
        stage_dir / "test_macro_f1.png",
        metric_key="test_macro_f1",
        title=f"{stage} test macro-F1",
    )

    if stage == "stage1":
        winners = (
            stage_frame.sort_values(
                by=["backbone", "val_macro_f1", "strategy"],
                ascending=[True, False, True],
            )
            .assign(strategy_rank=lambda df: df["strategy"].map(STRATEGY_TIEBREAK))
            .sort_values(by=["backbone", "val_macro_f1", "strategy_rank"], ascending=[True, False, True])
            .groupby("backbone", as_index=False)
            .first()
            .drop(columns=["strategy_rank"])
        )
        winners.to_csv(stage_dir / "winners.csv", index=False)

    return stage_frame


def aggregate_runs(output_root: str | Path) -> dict:
    output_root = Path(output_root).resolve()
    runs_root = output_root / "runs"
    rows = _collect_run_rows(runs_root)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"num_runs": 0, "stages": []}

    ensure_dir(output_root / "summaries")
    frame.to_csv(output_root / "summaries" / "all_runs.csv", index=False)
    stages = sorted(frame["stage"].dropna().unique().tolist())
    for stage in stages:
        _stage_summary(frame, stage, output_root)

    return {"num_runs": int(len(frame)), "stages": stages}
