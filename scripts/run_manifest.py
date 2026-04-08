from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_kws.cli import build_parser, main  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one config or a manifest of configs")
    parser.add_argument("--config")
    parser.add_argument("--manifest")
    args = parser.parse_args()

    if bool(args.config) == bool(args.manifest):
        raise SystemExit("Provide exactly one of --config or --manifest.")

    if args.config:
        main(["run", "--config", args.config])
    else:
        main(["run-manifest", "--manifest", args.manifest])
