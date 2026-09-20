#!/usr/bin/env python3
"""Write the current runtime manifest as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_testing.reproducibility import collect_runtime_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/runtime_manifest.json")
    )
    args = parser.parse_args()

    manifest = collect_runtime_manifest(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

