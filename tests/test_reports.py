from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")

from speech_kws.evaluation.reports import (
    _build_plot_frame,
    _decode_augmentation_name,
    _save_grouped_bar_plot,
    _series_color_map,
    _stage_summary,
)


SUMMARY_COLUMNS = [
    "stage",
    "experiment_id",
    "seed",
    "backbone",
    "strategy",
    "resolved_config_path",
    "run_dir",
    "val_accuracy",
    "val_macro_f1",
    "val_keyword_macro_f1",
    "test_accuracy",
    "test_macro_f1",
    "test_keyword_macro_f1",
    "selected_tau",
]


def _write_config(
    path: Path,
    *,
    time_shift: bool,
    background_noise: bool,
    specaugment: bool,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "augment": {
                    "time_shift": {"enabled": time_shift},
                    "background_noise": {"enabled": background_noise},
                    "specaugment": {"enabled": specaugment},
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _row(
    *,
    stage: str,
    experiment_id: str,
    seed: int,
    backbone: str,
    strategy: str,
    resolved_config_path: Path,
    val_macro_f1: float,
    test_macro_f1: float,
) -> dict:
    return {
        "stage": stage,
        "experiment_id": experiment_id,
        "seed": seed,
        "backbone": backbone,
        "strategy": strategy,
        "resolved_config_path": str(resolved_config_path),
        "run_dir": str(resolved_config_path.parent),
        "val_accuracy": val_macro_f1,
        "val_macro_f1": val_macro_f1,
        "val_keyword_macro_f1": val_macro_f1,
        "test_accuracy": test_macro_f1,
        "test_macro_f1": test_macro_f1,
        "test_keyword_macro_f1": test_macro_f1,
        "selected_tau": None,
    }


def test_decode_augmentation_name() -> None:
    assert (
        _decode_augmentation_name(
            {
                "augment": {
                    "time_shift": {"enabled": False},
                    "background_noise": {"enabled": False},
                    "specaugment": {"enabled": False},
                }
            }
        )
        == "none"
    )
    assert (
        _decode_augmentation_name(
            {
                "augment": {
                    "time_shift": {"enabled": False},
                    "background_noise": {"enabled": True},
                    "specaugment": {"enabled": True},
                }
            }
        )
        == "background_noise+specaugment"
    )


def test_build_plot_frame_orders_stage1_rows_by_backbone_strategy_seed(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "configs" / "baseline.yaml",
        time_shift=False,
        background_noise=False,
        specaugment=False,
    )
    frame = pd.DataFrame(
        [
            _row(
                stage="stage1",
                experiment_id="kwt_strategy_b_seed_1338",
                seed=1338,
                backbone="kwt",
                strategy="b",
                resolved_config_path=config_path,
                val_macro_f1=0.91,
                test_macro_f1=0.90,
            ),
            _row(
                stage="stage1",
                experiment_id="bcresnet_strategy_d_seed_1337",
                seed=1337,
                backbone="bcresnet",
                strategy="d",
                resolved_config_path=config_path,
                val_macro_f1=0.93,
                test_macro_f1=0.92,
            ),
            _row(
                stage="stage1",
                experiment_id="bcresnet_strategy_a_seed_1338",
                seed=1338,
                backbone="bcresnet",
                strategy="a",
                resolved_config_path=config_path,
                val_macro_f1=0.95,
                test_macro_f1=0.94,
            ),
            _row(
                stage="stage1",
                experiment_id="kwt_strategy_a_seed_1337",
                seed=1337,
                backbone="kwt",
                strategy="a",
                resolved_config_path=config_path,
                val_macro_f1=0.96,
                test_macro_f1=0.95,
            ),
            _row(
                stage="stage1",
                experiment_id="bcresnet_strategy_a_seed_1337",
                seed=1337,
                backbone="bcresnet",
                strategy="a",
                resolved_config_path=config_path,
                val_macro_f1=0.94,
                test_macro_f1=0.93,
            ),
        ]
    )

    plot_frame, legend_title = _build_plot_frame(frame, "stage1")
    sorted_frame = plot_frame.sort_values(
        by=["plot_group", "plot_series", "plot_seed", "experiment_id"],
        kind="mergesort",
    )

    assert legend_title == "Strategy"
    assert list(sorted_frame["experiment_id"]) == [
        "bcresnet_strategy_a_seed_1337",
        "bcresnet_strategy_a_seed_1338",
        "bcresnet_strategy_d_seed_1337",
        "kwt_strategy_a_seed_1337",
        "kwt_strategy_b_seed_1338",
    ]


def test_build_plot_frame_orders_stage2_rows_by_augmentation_name(tmp_path: Path) -> None:
    none_path = _write_config(
        tmp_path / "configs" / "none.yaml",
        time_shift=False,
        background_noise=False,
        specaugment=False,
    )
    bg_spec_path = _write_config(
        tmp_path / "configs" / "bg_spec.yaml",
        time_shift=False,
        background_noise=True,
        specaugment=True,
    )
    time_shift_path = _write_config(
        tmp_path / "configs" / "time.yaml",
        time_shift=True,
        background_noise=False,
        specaugment=False,
    )
    frame = pd.DataFrame(
        [
            _row(
                stage="stage2",
                experiment_id="kwt_strategy_a_aug_101",
                seed=1338,
                backbone="kwt",
                strategy="a",
                resolved_config_path=bg_spec_path,
                val_macro_f1=0.96,
                test_macro_f1=0.95,
            ),
            _row(
                stage="stage2",
                experiment_id="bcresnet_strategy_a_aug_100",
                seed=1338,
                backbone="bcresnet",
                strategy="a",
                resolved_config_path=time_shift_path,
                val_macro_f1=0.94,
                test_macro_f1=0.93,
            ),
            _row(
                stage="stage2",
                experiment_id="bcresnet_strategy_a_aug_000",
                seed=1337,
                backbone="bcresnet",
                strategy="a",
                resolved_config_path=none_path,
                val_macro_f1=0.92,
                test_macro_f1=0.91,
            ),
            _row(
                stage="stage2",
                experiment_id="kwt_strategy_a_aug_000",
                seed=1337,
                backbone="kwt",
                strategy="a",
                resolved_config_path=none_path,
                val_macro_f1=0.91,
                test_macro_f1=0.90,
            ),
        ]
    )

    plot_frame, legend_title = _build_plot_frame(frame, "stage2")
    sorted_frame = plot_frame.sort_values(
        by=["plot_group", "plot_series", "plot_seed", "experiment_id"],
        kind="mergesort",
    )

    assert legend_title == "Augmentation"
    assert list(sorted_frame["plot_series"]) == [
        "none",
        "time_shift",
        "background_noise+specaugment",
        "none",
    ]
    assert list(sorted_frame["experiment_id"]) == [
        "bcresnet_strategy_a_aug_000",
        "bcresnet_strategy_a_aug_100",
        "kwt_strategy_a_aug_101",
        "kwt_strategy_a_aug_000",
    ]


def test_series_color_map_uses_sorted_series_names() -> None:
    color_map = _series_color_map(["C", "A", "B", "A"])
    assert list(color_map.keys()) == ["A", "B", "C"]


def test_stage_summary_creates_grouped_plot_outputs(tmp_path: Path) -> None:
    baseline_path = _write_config(
        tmp_path / "configs" / "baseline.yaml",
        time_shift=False,
        background_noise=False,
        specaugment=False,
    )
    spec_path = _write_config(
        tmp_path / "configs" / "spec.yaml",
        time_shift=False,
        background_noise=False,
        specaugment=True,
    )
    frame = pd.DataFrame(
        [
            _row(
                stage="stage1",
                experiment_id="bcresnet_strategy_a_seed_1337",
                seed=1337,
                backbone="bcresnet",
                strategy="a",
                resolved_config_path=baseline_path,
                val_macro_f1=0.95,
                test_macro_f1=0.94,
            ),
            _row(
                stage="stage1",
                experiment_id="kwt_strategy_b_seed_1338",
                seed=1338,
                backbone="kwt",
                strategy="b",
                resolved_config_path=baseline_path,
                val_macro_f1=0.92,
                test_macro_f1=0.91,
            ),
            _row(
                stage="stage2",
                experiment_id="bcresnet_strategy_a_aug_000",
                seed=1337,
                backbone="bcresnet",
                strategy="a",
                resolved_config_path=baseline_path,
                val_macro_f1=0.96,
                test_macro_f1=0.95,
            ),
            _row(
                stage="stage2",
                experiment_id="kwt_strategy_a_aug_001",
                seed=1338,
                backbone="kwt",
                strategy="a",
                resolved_config_path=spec_path,
                val_macro_f1=0.97,
                test_macro_f1=0.96,
            ),
        ],
        columns=SUMMARY_COLUMNS,
    )

    _stage_summary(frame, "stage1", tmp_path)
    _stage_summary(frame, "stage2", tmp_path)

    stage1_dir = tmp_path / "summaries" / "stage1"
    stage2_dir = tmp_path / "summaries" / "stage2"
    assert (stage1_dir / "summary.csv").exists()
    assert (stage1_dir / "winners.csv").exists()
    assert (stage1_dir / "val_macro_f1.png").exists()
    assert (stage1_dir / "test_macro_f1.png").exists()
    assert (stage2_dir / "summary.csv").exists()
    assert (stage2_dir / "val_macro_f1.png").exists()
    assert (stage2_dir / "test_macro_f1.png").exists()
    assert list(pd.read_csv(stage1_dir / "summary.csv").columns) == SUMMARY_COLUMNS
    assert list(pd.read_csv(stage2_dir / "summary.csv").columns) == SUMMARY_COLUMNS


def test_save_grouped_bar_plot_skips_empty_frame(tmp_path: Path) -> None:
    destination = tmp_path / "empty.png"
    _save_grouped_bar_plot(
        pd.DataFrame(columns=["plot_group", "plot_series", "plot_seed", "experiment_id", "val_macro_f1"]),
        destination=destination,
        metric_key="val_macro_f1",
        title="empty",
        group_key="plot_group",
        series_key="plot_series",
        sort_keys=["plot_series", "plot_seed", "experiment_id"],
        legend_title="Series",
    )
    assert not destination.exists()
