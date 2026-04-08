from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return payload


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    raw = _load_yaml(path)

    merged: Dict[str, Any] = {}
    for base_entry in raw.pop("base_configs", []):
        base_path = Path(base_entry)
        if not base_path.is_absolute():
            base_path = (path.parent / base_path).resolve()
        merged = deep_merge(merged, load_config(base_path))

    merged = deep_merge(merged, raw)
    merged["_config_path"] = str(path)
    return merged


def get_required(config: Dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Missing config key: {dotted_key}")
        current = current[part]
    return current


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def resolve_paths_in_config(config: Dict[str, Any]) -> Dict[str, Any]:
    resolved = copy.deepcopy(config)
    paths = resolved.setdefault("paths", {})
    for key in ["audio_root", "validation_list", "testing_list", "prepared_root", "output_root"]:
        if key in paths:
            paths[key] = str(resolve_repo_path(paths[key]))
    return resolved


def validate_config(config: Dict[str, Any]) -> None:
    for key in [
        "experiment.id",
        "experiment.stage",
        "experiment.seed",
        "paths.audio_root",
        "paths.validation_list",
        "paths.testing_list",
        "paths.prepared_root",
        "paths.output_root",
        "task.unknown_strategy",
        "model.name",
        "frontend.kind",
        "train.batch_size",
    ]:
        get_required(config, key)


def get_run_dir(config: Dict[str, Any]) -> Path:
    resolved = resolve_paths_in_config(config)
    output_root = Path(get_required(resolved, "paths.output_root"))
    stage = str(get_required(resolved, "experiment.stage"))
    experiment_id = str(get_required(resolved, "experiment.id"))
    seed = int(get_required(resolved, "experiment.seed"))
    return output_root / "runs" / stage / experiment_id / f"seed_{seed}"


def clean_config_for_dump(config: Dict[str, Any]) -> Dict[str, Any]:
    payload = copy.deepcopy(config)
    payload.pop("_config_path", None)
    return payload
