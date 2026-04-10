from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_kws.audio.frontend import AudioFrontend  # noqa: E402
from speech_kws.config import load_config, resolve_paths_in_config  # noqa: E402
from speech_kws.models.wrappers import build_model  # noqa: E402
from speech_kws.training.loops import _device_from_config  # noqa: E402
from speech_kws.utils.io import ensure_dir  # noqa: E402


@dataclass
class ModuleTiming:
    name: str
    class_name: str
    param_count: int
    total_sec: float = 0.0
    calls: int = 0


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_device(config: dict, requested_device: str) -> torch.device:
    if requested_device == "auto":
        return _device_from_config(config.get("experiment", {}))
    return _device_from_config({"device": requested_device})


def _parameter_count(module: torch.nn.Module, recurse: bool = True) -> int:
    return sum(parameter.numel() for parameter in module.parameters(recurse=recurse))


def _parameter_summary(model: torch.nn.Module) -> dict[str, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = _parameter_count(model)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }


def _build_synthetic_input(config: dict, batch_size: int, device: torch.device) -> tuple[torch.Tensor, AudioFrontend]:
    frontend = AudioFrontend(config["frontend"])
    inputs = torch.randn(batch_size, frontend.feature_dim, frontend.expected_frames, device=device)
    return inputs, frontend


def _forward_benchmark(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    warmup_iters: int,
    iters: int,
) -> dict[str, float]:
    timings_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup_iters):
            model(inputs)
        _sync_device(device)

        for _ in range(iters):
            _sync_device(device)
            start = time.perf_counter()
            model(inputs)
            _sync_device(device)
            timings_ms.append((time.perf_counter() - start) * 1000.0)

    batch_size = int(inputs.shape[0])
    mean_ms = statistics.mean(timings_ms)
    return {
        "mean_ms": mean_ms,
        "median_ms": statistics.median(timings_ms),
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
        "std_ms": statistics.pstdev(timings_ms) if len(timings_ms) > 1 else 0.0,
        "samples_per_sec": (batch_size * 1000.0) / max(mean_ms, 1e-12),
    }


def _leaf_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if name and not any(module.children())
    ]


def _module_timing(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    module_iters: int,
) -> list[ModuleTiming]:
    timings_by_id: dict[int, ModuleTiming] = {}
    starts_by_id: dict[int, list[float]] = {}
    handles = []

    for name, module in _leaf_modules(model):
        module_id = id(module)
        timings_by_id[module_id] = ModuleTiming(
            name=name,
            class_name=module.__class__.__name__,
            param_count=_parameter_count(module, recurse=False),
        )
        starts_by_id[module_id] = []

        def pre_hook(current_module, _inputs, current_module_id=module_id):
            _sync_device(device)
            starts_by_id[current_module_id].append(time.perf_counter())

        def post_hook(current_module, _inputs, _outputs, current_module_id=module_id):
            _sync_device(device)
            start = starts_by_id[current_module_id].pop()
            timing = timings_by_id[current_module_id]
            timing.total_sec += time.perf_counter() - start
            timing.calls += 1

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))

    try:
        with torch.inference_mode():
            for _ in range(module_iters):
                model(inputs)
        _sync_device(device)
    finally:
        for handle in handles:
            handle.remove()

    return sorted(timings_by_id.values(), key=lambda timing: timing.total_sec, reverse=True)


def _format_count(value: int) -> str:
    return f"{value:,}"


def _format_ms(value: float) -> str:
    return f"{value:.4f}"


def _append_config_report(
    lines: list[str],
    config_path: Path,
    config: dict,
    args: argparse.Namespace,
) -> None:
    device = _resolve_device(config, args.device)
    frontend_input, frontend = _build_synthetic_input(config, batch_size=args.batch_size, device=device)
    model = build_model(
        model_config=config["model"],
        strategy=config["task"]["unknown_strategy"],
        frontend_feature_dim=frontend.feature_dim,
    ).to(device)
    model.eval()

    parameter_counts = _parameter_summary(model)
    forward_stats = _forward_benchmark(
        model=model,
        inputs=frontend_input,
        device=device,
        warmup_iters=args.warmup_iters,
        iters=args.iters,
    )
    module_timings = _module_timing(
        model=model,
        inputs=frontend_input,
        device=device,
        module_iters=args.module_iters,
    )
    captured_module_time = sum(timing.total_sec for timing in module_timings)

    lines.extend(
        [
            "",
            "=" * 100,
            f"Config: {config_path.resolve()}",
            f"Model: {config['model']['name']}",
            f"Strategy: {config['task']['unknown_strategy']}",
            f"Device: {device}",
            f"Input shape: {tuple(frontend_input.shape)}",
            f"Input dtype: {frontend_input.dtype}",
            "Benchmark settings: "
            f"batch_size={args.batch_size}, iters={args.iters}, "
            f"warmup_iters={args.warmup_iters}, module_iters={args.module_iters}, top_k={args.top_k}",
            "",
            "Parameter counts",
            f"Total parameters: {_format_count(parameter_counts['total'])}",
            f"Trainable parameters: {_format_count(parameter_counts['trainable'])}",
            f"Non-trainable parameters: {_format_count(parameter_counts['non_trainable'])}",
            "",
            "Architecture",
            str(model),
            "",
            "Forward benchmark",
            f"Mean ms/forward: {_format_ms(forward_stats['mean_ms'])}",
            f"Median ms/forward: {_format_ms(forward_stats['median_ms'])}",
            f"Min ms/forward: {_format_ms(forward_stats['min_ms'])}",
            f"Max ms/forward: {_format_ms(forward_stats['max_ms'])}",
            f"Std ms/forward: {_format_ms(forward_stats['std_ms'])}",
            f"Samples/sec: {forward_stats['samples_per_sec']:.2f}",
            "",
            "Module timing",
            "Note: timings are captured from leaf-module forward hooks and percentages are relative to captured leaf-module time.",
            "| Module | Class | Parameters | Calls/iter | Avg ms/iter | % captured |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for timing in module_timings[: args.top_k]:
        calls_per_iter = timing.calls / max(args.module_iters, 1)
        avg_ms_per_iter = (timing.total_sec * 1000.0) / max(args.module_iters, 1)
        percent = (timing.total_sec / captured_module_time * 100.0) if captured_module_time > 0 else 0.0
        lines.append(
            f"| `{timing.name}` | {timing.class_name} | {_format_count(timing.param_count)} | "
            f"{calls_per_iter:.2f} | {_format_ms(avg_ms_per_iter)} | {percent:.2f}% |"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark configured KWS model architecture forward passes")
    parser.add_argument("--config", nargs="+", required=True, help="One or more experiment YAML configs to benchmark")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "benchmark" / "bench.txt"), help="Benchmark report path")
    parser.add_argument("--batch-size", type=int, default=16, help="Synthetic feature batch size")
    parser.add_argument("--iters", type=int, default=20, help="Timed forward iterations")
    parser.add_argument("--warmup-iters", type=int, default=5, help="Untimed warmup forward iterations")
    parser.add_argument("--module-iters", type=int, default=10, help="Forward iterations with module timing hooks")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Benchmark device")
    parser.add_argument("--top-k", type=int, default=30, help="Number of timed module rows to show")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.iters <= 0:
        parser.error("--iters must be > 0")
    if args.warmup_iters < 0:
        parser.error("--warmup-iters must be >= 0")
    if args.module_iters <= 0:
        parser.error("--module-iters must be > 0")
    if args.top_k <= 0:
        parser.error("--top-k must be > 0")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    lines = [
        "Model Benchmark Report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
    ]
    for config_path_value in args.config:
        config_path = Path(config_path_value)
        config = resolve_paths_in_config(load_config(config_path))
        _append_config_report(lines, config_path, config, args)

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote benchmark report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
