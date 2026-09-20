#!/usr/bin/env python3
"""Time a short RQ1 training rehearsal and project the complete training matrix."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time


for variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

from thesis_testing.artifacts import validate_training_run
from thesis_testing.config.loader import load_experiment_config
from thesis_testing.config.resolve import resolve_experiment
from thesis_testing.matrix_timing import (
    MatrixTimingObservation,
    estimate_full_matrix,
)
from thesis_testing.nominal_analysis import assert_fair_pair, read_metrics
from thesis_testing.nominal_suite import CONTROLLERS, NOMINAL_CONDITIONS
from thesis_testing.protocol import load_scientific_protocol
from thesis_testing.reproducibility import collect_runtime_manifest
from thesis_testing.training import train


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_WORKERS = 40
TIMING_SEED = 9_001
TIMING_GENERATIONS = 10
CONTINGENCY = 0.20


def _slug(value: str) -> str:
    return value.lower().replace("-", "_")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(report: dict[str, object]) -> str:
    estimate = report["full_training_projection"]
    assert isinstance(estimate, dict)
    return "\n".join(
        [
            "# RQ1 training timing rehearsal",
            "",
            "Status: **passed**",
            "",
            "This is a timing/integration rehearsal, not thesis evidence.",
            "",
            "## Measured rehearsal",
            "",
            "- Task/morphology strata: `Walker-v0/biped, Carrier-v0/biped, Jumper-v0/biped, Thrower-v0/biped, Thrower-v0/thrower_arm`",
            f"- Controllers: `{', '.join(CONTROLLERS)}`",
            f"- Controller conditions: `{report['condition_count']}`",
            f"- One disjoint timing seed, population `{report['timing_population']}`, generations `{TIMING_GENERATIONS}`",
            f"- Workers: `{report['worker_processes']}` processes with one numerical-library thread each",
            f"- Candidate episodes executed: `{report['timing_candidate_episodes']:,}`",
            f"- Training wall time: **{report['timing_wall_seconds'] / 60.0:.2f} minutes**",
            "",
            "## Complete RQ1 training projection",
            "",
            f"- Independent training runs: `{estimate['total_training_runs']}`",
            f"- Runs per controller condition: `{estimate['runs_per_condition']}`",
            f"- Population: `{estimate['population_size']}`",
            f"- Generations: `{estimate['generations']}`",
            f"- Candidate episodes: `{estimate['total_candidate_episodes']:,}`",
            f"- Measured projection: **{estimate['measured_projection_hours']:.2f} hours ({estimate['measured_projection_days']:.2f} days)**",
            f"- With 20% planning contingency: **{estimate['planning_with_contingency_hours']:.2f} hours ({estimate['planning_with_contingency_days']:.2f} days)**",
            "",
            "## Interpretation",
            "",
            "- The projection covers RQ1 Static-ES versus NcHL-ES training only.",
            "- Each task/morphology/controller condition has one ten-generation timing observation.",
            "- The 20% figure is a scheduling allowance, not a statistical confidence bound.",
            "- Held-out, weight-trace, freeze, statistics, and figure stages are not timed or projected.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/rq1_timing_40_workers"),
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if (os.cpu_count() or 0) < args.workers:
        parser.error(f"requires at least {args.workers} logical CPUs")
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error(f"--output must be empty or absent: {root}")
    root.mkdir(parents=True, exist_ok=True)
    protocol = load_scientific_protocol()
    (root / "runtime_manifest.json").write_text(
        json.dumps(collect_runtime_manifest(TIMING_SEED), indent=2, sort_keys=True) + "\n"
    )

    resolved_by_condition = {}
    for condition in NOMINAL_CONDITIONS:
        pair = []
        for controller in CONTROLLERS:
            config_path = (
                REPOSITORY / "configs" / "nominal"
                / f"{condition.config_stem}_{controller}.yaml"
            )
            base = load_experiment_config(config_path)
            config = replace(
                base,
                name=f"timing_{condition.config_stem}_{controller}",
                seed=TIMING_SEED,
                evolution=replace(
                    base.evolution,
                    population_size=protocol.population_size,
                    generations=TIMING_GENERATIONS,
                ),
                runtime=replace(
                    base.runtime, workers=args.workers, torch_threads_per_worker=1,
                ),
            )
            resolved = resolve_experiment(config)
            pair.append(resolved)
            key = (condition.task, condition.morphology, controller)
            resolved_by_condition[key] = resolved
        assert_fair_pair(pair[0], pair[1])

    conditions = list(resolved_by_condition)
    random.Random(TIMING_SEED).shuffle(conditions)
    timing_rows: list[dict[str, object]] = []
    observations: list[MatrixTimingObservation] = []
    matrix_started = time.perf_counter()
    for run_order, condition in enumerate(conditions, start=1):
        task, morphology, controller = condition
        resolved = resolved_by_condition[condition]
        output = (
            root / "runs" / _slug(task) / morphology / controller
            / f"seed_{TIMING_SEED}"
        )
        print(
            f"[{run_order:02d}/{len(conditions)}] {task} {morphology} {controller}",
            flush=True,
        )
        started = time.perf_counter()
        summary = train(resolved, output, checkpoint_every=1)
        wall_seconds = time.perf_counter() - started
        metrics = read_metrics(output / "metrics.csv")
        steady_seconds = sum(row["elapsed_seconds"] for row in metrics)
        fixed_overhead_seconds = max(0.0, wall_seconds - steady_seconds)
        validation = validate_training_run(output, expected=resolved)
        if not validation.valid:
            messages = "; ".join(issue.message for issue in validation.issues)
            raise RuntimeError(f"invalid timing artifacts for {condition}: {messages}")
        candidate_episodes = (
            protocol.population_size * TIMING_GENERATIONS
            * protocol.episodes_per_candidate.for_controller(controller)
        )
        observation = MatrixTimingObservation(
            task=task,
            morphology=morphology,
            controller=controller,
            candidate_episodes=candidate_episodes,
            steady_seconds=steady_seconds,
            fixed_overhead_seconds=fixed_overhead_seconds,
        )
        observations.append(observation)
        row = {
            "run_order": run_order,
            **asdict(observation),
            "episodes_per_second": observation.episodes_per_second,
            "wall_seconds": wall_seconds,
            "observation_dim": resolved.env_contract.observation_dim,
            "action_dim": resolved.env_contract.action_dim,
            "genome_size": (
                resolved.network.num_weights
                if controller == "static"
                else resolved.network.num_neurons * 4
            ),
            "best_fitness": summary.best_fitness,
            "artifact_validation": "passed",
            "output_dir": str(output),
        }
        timing_rows.append(row)
        _write_csv(root / "timing_runs.csv", timing_rows)
        print(json.dumps(row, sort_keys=True), flush=True)

    timing_wall_seconds = time.perf_counter() - matrix_started
    estimate = estimate_full_matrix(
        observations,
        runs_per_condition=len(protocol.training_seeds),
        generations=protocol.generations,
        population_size=protocol.population_size,
        episodes_per_candidate={
            "static": protocol.episodes_per_candidate.static,
            "nchl": protocol.episodes_per_candidate.nchl,
        },
        contingency=CONTINGENCY,
    )
    _write_csv(root / "full_training_projection.csv", estimate["conditions"])
    report = {
        "schema_version": 1,
        "status": "passed",
        "scope": "rq1_training_timing_rehearsal_not_thesis_evidence",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_morphologies": [
            {"task": item.task, "morphology": item.morphology}
            for item in NOMINAL_CONDITIONS
        ],
        "controllers": list(CONTROLLERS),
        "condition_count": len(conditions),
        "timing_seed": TIMING_SEED,
        "timing_generations": TIMING_GENERATIONS,
        "timing_population": protocol.population_size,
        "worker_processes": args.workers,
        "torch_threads_per_worker": 1,
        "timing_candidate_episodes": sum(
            observation.candidate_episodes for observation in observations
        ),
        "timing_wall_seconds": timing_wall_seconds,
        "full_training_projection": estimate,
    }
    (root / "timing_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (root / "timing_report.md").write_text(_markdown(report))
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
