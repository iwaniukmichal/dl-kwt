from __future__ import annotations

import argparse
from pathlib import Path

from speech_kws.config import REPO_ROOT, clean_config_for_dump, load_config
from speech_kws.data.prepare import prepare_dataset
from speech_kws.evaluation.reports import aggregate_runs
from speech_kws.utils.io import ensure_dir, load_csv, save_yaml


def prepare_command(args) -> dict:
    return prepare_dataset(
        audio_root=args.audio_root,
        validation_list=args.validation_list,
        testing_list=args.testing_list,
        prepared_root=args.prepared_root,
        clip_duration_sec=args.clip_duration_sec,
    )


def run_command(args) -> dict:
    from speech_kws.training.loops import run_experiment

    config = load_config(args.config)
    return run_experiment(config)


def run_manifest(manifest_path: str | Path) -> list[dict]:
    from speech_kws.training.loops import run_experiment

    manifest_path = Path(manifest_path).resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        config_paths = [line.strip() for line in handle if line.strip()]

    results = []
    for config_path in config_paths:
        if not Path(config_path).is_absolute():
            config_path = str((REPO_ROOT / config_path).resolve())
        results.append(run_experiment(load_config(config_path)))
    return results


def run_manifest_command(args) -> list[dict]:
    return run_manifest(args.manifest)


def build_stage2_manifest(args) -> dict:
    output_root = Path(args.output_root).resolve()
    summary_dir = output_root / "summaries" / "stage1"
    winners_path = summary_dir / "winners.csv"
    if not winners_path.exists():
        aggregate_runs(output_root)
    if not winners_path.exists():
        raise FileNotFoundError(
            "Missing Stage 1 winners summary. Run Stage 1 experiments and aggregate results first."
        )

    winners = load_csv(winners_path)
    stage2_dir = ensure_dir(REPO_ROOT / "configs" / "experiments" / "stage2")
    manifest_entries = []
    generated_configs = []

    for winner in winners:
        resolved_config_path = Path(winner["resolved_config_path"]).resolve()
        base_config = load_config(resolved_config_path)
        strategy = base_config["task"]["unknown_strategy"].lower()
        backbone = base_config["model"]["name"].lower()

        for time_shift_enabled in [False, True]:
            for background_enabled in [False, True]:
                for specaugment_enabled in [False, True]:
                    bits = "".join(
                        [
                            "1" if time_shift_enabled else "0",
                            "1" if background_enabled else "0",
                            "1" if specaugment_enabled else "0",
                        ]
                    )
                    experiment_id = f"{backbone}_strategy_{strategy}_aug_{bits}"
                    config = clean_config_for_dump(base_config)
                    config["experiment"]["stage"] = "stage2"
                    config["experiment"]["id"] = experiment_id
                    config["augment"]["time_shift"]["enabled"] = time_shift_enabled
                    config["augment"]["background_noise"]["enabled"] = background_enabled
                    config["augment"]["specaugment"]["enabled"] = specaugment_enabled

                    destination = stage2_dir / f"{experiment_id}.yaml"
                    save_yaml(destination, config)
                    manifest_entries.append(destination.relative_to(REPO_ROOT).as_posix())
                    generated_configs.append(str(destination))

    manifest_path = REPO_ROOT / "configs" / "manifests" / "stage2.txt"
    ensure_dir(manifest_path.parent)
    manifest_path.write_text("\n".join(sorted(manifest_entries)) + "\n", encoding="utf-8")
    return {"manifest": str(manifest_path), "configs": generated_configs}


def aggregate_command(args) -> dict:
    return aggregate_runs(args.output_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Speech Commands keyword spotting experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Prepare dataset manifests and silence metadata")
    prepare_parser.add_argument("--audio-root", default=str(REPO_ROOT / "data" / "audio"))
    prepare_parser.add_argument("--validation-list", default=str(REPO_ROOT / "data" / "validation_list.txt"))
    prepare_parser.add_argument("--testing-list", default=str(REPO_ROOT / "data" / "testing_list.txt"))
    prepare_parser.add_argument("--prepared-root", default=str(REPO_ROOT / "data" / "prepared"))
    prepare_parser.add_argument("--clip-duration-sec", type=float, default=1.0)
    prepare_parser.set_defaults(func=prepare_command)

    run_parser = subparsers.add_parser("run", help="Run one experiment config")
    run_parser.add_argument("--config", required=True)
    run_parser.set_defaults(func=run_command)

    manifest_parser = subparsers.add_parser("run-manifest", help="Run a manifest of experiment configs")
    manifest_parser.add_argument("--manifest", required=True)
    manifest_parser.set_defaults(func=run_manifest_command)

    build_stage2_parser = subparsers.add_parser("build-stage2", help="Create Stage 2 configs from Stage 1 winners")
    build_stage2_parser.add_argument("--output-root", default=str(REPO_ROOT / "outputs"))
    build_stage2_parser.set_defaults(func=build_stage2_manifest)

    aggregate_parser = subparsers.add_parser("aggregate", help="Aggregate completed run outputs")
    aggregate_parser.add_argument("--output-root", default=str(REPO_ROOT / "outputs"))
    aggregate_parser.set_defaults(func=aggregate_command)

    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
