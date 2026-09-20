#!/usr/bin/env python3
"""Train and validate a tiny matched Static/NcHL integration fixture."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

from thesis_testing.artifacts import validate_training_run
from thesis_testing.config.loader import load_experiment_config
from thesis_testing.config.resolve import resolve_experiment
from thesis_testing.nominal_analysis import assert_fair_pair
from thesis_testing.training import train


REPOSITORY = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    root = args.output or Path(tempfile.mkdtemp(prefix="thesis-e2e-"))
    if root.exists() and any(root.iterdir()):
        parser.error(f"fast verification output must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    resolved_pair = []
    outputs = []
    summaries = []
    for controller in ("static", "nchl"):
        base = load_experiment_config(
            REPOSITORY / f"configs/nominal/walker_{controller}.yaml"
        )
        config = replace(
            base,
            name=f"e2e_walker_{controller}",
            seed=91_001,
            evolution=replace(base.evolution, population_size=2, generations=1),
            runtime=replace(
                base.runtime,
                workers=args.workers,
                torch_threads_per_worker=1,
            ),
        )
        resolved = resolve_experiment(config)
        output = root / controller
        summary = train(resolved, output, checkpoint_every=1)
        validation = validate_training_run(output, expected=resolved)
        if not validation.valid:
            detail = "; ".join(issue.message for issue in validation.issues)
            raise RuntimeError(f"invalid {controller} smoke artifact: {detail}")
        resolved_pair.append(resolved)
        outputs.append(str(output))
        summaries.append(summary)

    assert_fair_pair(resolved_pair[0], resolved_pair[1])
    manifest = {
        "status": "passed",
        "scope": "tiny_training_and_artifact_integration_not_thesis_evidence",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "Walker-v0",
        "morphology": "biped",
        "controllers": ["static", "nchl"],
        "training_seed": 91_001,
        "training_runs": 2,
        "population_size": 2,
        "generations": 1,
        "workers": args.workers,
        "best_fitness": [summary.best_fitness for summary in summaries],
        "output_directories": outputs,
    }
    (root / "end_to_end_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
