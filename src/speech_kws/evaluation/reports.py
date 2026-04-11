from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from speech_kws.utils.io import ensure_dir, load_json, load_yaml


STRATEGY_TIEBREAK = {"b": 0, "c": 1, "d": 2, "a": 3}
AUGMENTATION_FACTORS = ("time_shift", "background_noise", "specaugment")


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


def _decode_augmentation_name(config: dict) -> str:
    augment_config = config.get("augment", {})
    enabled_factors = [
        factor
        for factor in AUGMENTATION_FACTORS
        if bool(augment_config.get(factor, {}).get("enabled", False))
    ]
    if not enabled_factors:
        return "none"
    return "+".join(enabled_factors)


def _load_augmentation_name(config_path: str | Path) -> str:
    return _decode_augmentation_name(load_yaml(Path(config_path)))


def _build_plot_frame(frame: pd.DataFrame, stage: str) -> tuple[pd.DataFrame, str]:
    plot_frame = frame.copy()
    plot_frame["plot_group"] = plot_frame["backbone"].astype(str)
    plot_frame["plot_seed"] = pd.to_numeric(plot_frame["seed"], errors="coerce").fillna(float("inf"))

    if stage == "stage1":
        plot_frame["plot_series"] = plot_frame["strategy"].astype(str).str.upper()
        legend_title = "Strategy"
    elif stage == "stage2":
        plot_frame["plot_series"] = plot_frame["resolved_config_path"].apply(_load_augmentation_name)
        legend_title = "Augmentation"
    else:
        plot_frame["plot_series"] = plot_frame["experiment_id"].astype(str)
        legend_title = "Experiment"

    return plot_frame, legend_title


def _series_color_map(series_labels: list[str]) -> dict[str, tuple]:
    sorted_labels = sorted({str(label) for label in series_labels if pd.notna(label)})
    if not sorted_labels:
        return {}

    cmap = plt.get_cmap("tab20")
    palette = getattr(cmap, "colors", None)
    if palette is None:
        denominator = max(1, len(sorted_labels) - 1)
        palette = [cmap(index / denominator) for index in range(len(sorted_labels))]

    return {label: palette[index % len(palette)] for index, label in enumerate(sorted_labels)}


def _save_grouped_bar_plot(
    frame: pd.DataFrame,
    destination: Path,
    metric_key: str,
    title: str,
    group_key: str,
    series_key: str,
    sort_keys: list[str],
    legend_title: str,
) -> None:
    if frame.empty:
        return

    ensure_dir(destination.parent)
    plot_frame = frame.sort_values(
        by=[group_key, *sort_keys],
        ascending=[True] * (1 + len(sort_keys)),
        kind="mergesort",
    ).copy()
    colors_by_series = _series_color_map(plot_frame[series_key].tolist())

    x_positions: list[float] = []
    values: list[float] = []
    bar_colors: list[tuple] = []
    tick_positions: list[float] = []
    tick_labels: list[str] = []

    current_x = 0.0
    group_gap = 1.0
    for group_name, group_rows in plot_frame.groupby(group_key, sort=False):
        group_positions: list[float] = []
        for _, row in group_rows.iterrows():
            group_positions.append(current_x)
            x_positions.append(current_x)
            values.append(float(row[metric_key]))
            bar_colors.append(colors_by_series[str(row[series_key])])
            current_x += 1.0
        if group_positions:
            tick_positions.append(sum(group_positions) / len(group_positions))
            tick_labels.append(str(group_name))
            current_x += group_gap

    fig_width = max(12.0, 0.75 * len(values) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    bars = ax.bar(x_positions, values, width=0.8, color=bar_colors, edgecolor="black", linewidth=0.6)

    max_value = max(values, default=0.0)
    label_padding = max(0.005, max_value * 0.01)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + (bar.get_width() / 2.0),
            value + label_padding,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title(title)
    ax.set_ylabel(metric_key)
    ax.set_xlabel("backbone")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(0.0, max_value + max(0.03, label_padding * 4.0))
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)

    legend_handles = [
        Patch(facecolor=color, edgecolor="black", label=label)
        for label, color in colors_by_series.items()
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, title=legend_title, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _stage_summary(frame: pd.DataFrame, stage: str, output_root: Path) -> pd.DataFrame:
    stage_frame = frame[frame["stage"] == stage].copy()
    if stage_frame.empty:
        return stage_frame

    stage_dir = ensure_dir(output_root / "summaries" / stage)
    stage_frame.to_csv(stage_dir / "summary.csv", index=False)
    plot_frame, legend_title = _build_plot_frame(stage_frame, stage)
    _save_grouped_bar_plot(
        plot_frame,
        stage_dir / "val_macro_f1.png",
        metric_key="val_macro_f1",
        title=f"{stage} validation macro-F1",
        group_key="plot_group",
        series_key="plot_series",
        sort_keys=["plot_series", "plot_seed", "experiment_id"],
        legend_title=legend_title,
    )
    _save_grouped_bar_plot(
        plot_frame,
        stage_dir / "test_macro_f1.png",
        metric_key="test_macro_f1",
        title=f"{stage} test macro-F1",
        group_key="plot_group",
        series_key="plot_series",
        sort_keys=["plot_series", "plot_seed", "experiment_id"],
        legend_title=legend_title,
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
