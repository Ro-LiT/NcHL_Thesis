#!/usr/bin/env python3
"""Thin fixed-order orchestration of the existing thesis experiment scripts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from thesis_testing.protocol import load_scientific_protocol, scientific_identity


REPOSITORY = Path(__file__).resolve().parents[1]
STAGES = (
    "preflight", "nominal_training", "nominal_analysis",
)


def _run(arguments: list[str]) -> None:
    print("+ " + " ".join(arguments), flush=True)
    subprocess.run([sys.executable, *arguments], cwd=REPOSITORY, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--final", action="store_true")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    phase = "pilot" if args.pilot else "final"
    protocol = load_scientific_protocol()
    if args.final and not args.dry_run and protocol.status != "frozen":
        parser.error("final runs are blocked until scientific_protocol.yaml status is frozen")
    seeds = len(protocol.calibration_seeds if args.pilot else protocol.training_seeds)
    root = args.results_root
    if root.exists() and not root.is_dir():
        parser.error("--results-root exists and is not a directory")
    root.mkdir(parents=True, exist_ok=True)
    environment_digest = hashlib.sha256(
        (REPOSITORY / "environment.yml").read_bytes()
    ).hexdigest()
    accounting = {
        "task_morphology_strata": len(protocol.conditions),
        "controller_conditions": len(protocol.conditions) * len(protocol.controllers),
        "nominal_training_runs": (
            len(protocol.conditions) * len(protocol.controllers) * seeds
        ),
        "population_size": protocol.population_size,
        "generations": 10 if phase == "pilot" else protocol.generations,
        "candidate_training_episodes": (
            len(protocol.conditions) * seeds
            * (10 if phase == "pilot" else protocol.generations)
            * protocol.population_size
            * sum(
                protocol.episodes_per_candidate.for_controller(controller)
                for controller in protocol.controllers
            )
        ),
        "heldout_episodes": (
            len(protocol.conditions) * len(protocol.controllers) * seeds
            * protocol.heldout_episodes_per_run
        ),
    }
    reproduction = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "dry_run": args.dry_run,
        "resume": args.resume,
        "workers": args.workers,
        "results_root": str(root),
        "stages": list(STAGES),
        "accounting": accounting,
        "conda_environment_sha256": environment_digest,
        "command": sys.argv,
        **scientific_identity(),
    }
    (root / "reproduction_manifest.json").write_text(
        json.dumps(reproduction, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(accounting, indent=2, sort_keys=True), flush=True)

    _run(["scripts/capture_versions.py", "--output", str(root / "runtime_manifest.json")])
    nominal_command = [
        "scripts/run_nominal_suite.py", f"--{phase}", "--output", str(root / "nominal"),
        "--workers", str(args.workers),
    ]
    if args.resume:
        nominal_command.append("--resume")
    if args.dry_run:
        nominal_command.append("--dry-run")
    _run(nominal_command)

    if args.dry_run:
        print(json.dumps({"status": "dry_run_complete", "stages_enumerated": list(STAGES)}, sort_keys=True))
        return

    _run([
        "scripts/analyze_nominal.py", str(root / "nominal"),
        "--output", str(root / "nominal_analysis"),
    ])
    print(json.dumps({"status": "official_nominal_pipeline_complete"}, sort_keys=True))


if __name__ == "__main__":
    main()
