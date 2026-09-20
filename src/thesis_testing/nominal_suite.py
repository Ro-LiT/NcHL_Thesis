"""Frozen nominal-suite enumeration and small manifest/index helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Sequence

from thesis_testing.config.loader import load_experiment_config
from thesis_testing.config.resolve import resolve_experiment
from thesis_testing.artifacts import validate_training_run
from thesis_testing.protocol import load_scientific_protocol, scientific_identity
from thesis_testing.nominal_analysis import assert_fair_pair
from thesis_testing.specs import ResolvedExperiment


@dataclass(frozen=True)
class NominalCondition:
    config_stem: str
    task: str
    morphology: str


NOMINAL_CONDITIONS = (
    NominalCondition("walker", "Walker-v0", "biped"),
    NominalCondition("carrier", "Carrier-v0", "biped"),
    NominalCondition("jumper", "Jumper-v0", "biped"),
    NominalCondition("thrower_biped", "Thrower-v0", "biped"),
    NominalCondition("thrower_arm", "Thrower-v0", "thrower_arm"),
)
CONTROLLERS = ("static", "nchl")


@dataclass(frozen=True)
class NominalRunSpec:
    run_id: str
    phase: str
    config_path: Path
    output_dir: Path
    resolved: ResolvedExperiment


def nominal_run_dir(root: Path, resolved: ResolvedExperiment) -> Path:
    task = resolved.task.env_id.split("-")[0].lower()
    return (
        Path(root)
        / task
        / resolved.morphology.name
        / resolved.config.controller.kind
        / f"seed_{resolved.config.seed:03d}"
    )


def enumerate_nominal_runs(
    config_dir: Path,
    output_root: Path,
    *,
    phase: str,
    seeds: Sequence[int],
    workers: int | None = None,
) -> tuple[NominalRunSpec, ...]:
    if phase not in ("pilot", "final"):
        raise ValueError("phase must be 'pilot' or 'final'")
    protocol = load_scientific_protocol()
    expected_conditions = {
        (condition.task, condition.morphology)
        for condition in protocol.conditions
    }
    enumerated_conditions = {
        (condition.task, condition.morphology)
        for condition in NOMINAL_CONDITIONS
    }
    if enumerated_conditions != expected_conditions:
        raise RuntimeError("nominal condition registry differs from scientific protocol")
    generations = 10 if phase == "pilot" else protocol.generations
    specs: list[NominalRunSpec] = []
    for condition in NOMINAL_CONDITIONS:
        pair_by_seed: dict[int, list[ResolvedExperiment]] = {
            seed: [] for seed in seeds
        }
        for controller in CONTROLLERS:
            path = Path(config_dir) / f"{condition.config_stem}_{controller}.yaml"
            base = load_experiment_config(path)
            if (base.task, base.morphology) != (condition.task, condition.morphology):
                raise ValueError(f"nominal config does not match registry: {path}")
            expected_episodes = protocol.episodes_per_candidate.for_controller(controller)
            if (
                base.evolution.population_size != protocol.population_size
                or base.evolution.sigma != protocol.sigma
                or base.evaluation.episodes_per_candidate != expected_episodes
            ):
                raise ValueError(f"nominal config budget differs from protocol: {path}")
            for seed in seeds:
                runtime = (
                    base.runtime
                    if workers is None
                    else replace(base.runtime, workers=workers)
                )
                config = replace(
                    base,
                    seed=seed,
                    name=f"{base.name}_seed_{seed:03d}",
                    evolution=replace(base.evolution, generations=generations),
                    runtime=runtime,
                )
                resolved = resolve_experiment(config)
                pair_by_seed[seed].append(resolved)
                task = resolved.task.env_id.split("-")[0].lower()
                run_id = (
                    f"{task}__{resolved.morphology.name}__"
                    f"{controller}__seed_{seed:03d}"
                )
                specs.append(
                    NominalRunSpec(
                        run_id=run_id,
                        phase=phase,
                        config_path=path,
                        output_dir=nominal_run_dir(output_root, resolved),
                        resolved=resolved,
                    )
                )
        for pair in pair_by_seed.values():
            assert_fair_pair(pair[0], pair[1])
    run_ids = [item.run_id for item in specs]
    if len(run_ids) != len(set(run_ids)):
        raise RuntimeError("nominal matrix produced duplicate run IDs")
    return tuple(specs)


def completed_run_summary(spec: NominalRunSpec) -> tuple[bool, int, float | None]:
    validation = validate_training_run(spec.output_dir, expected=spec.resolved)
    return (
        validation.valid,
        validation.completed_generations,
        validation.best_fitness,
    )


def run_index_row(
    spec: NominalRunSpec,
    *,
    status: str,
    completed_generations: int = 0,
    best_fitness: float | None = None,
    started_from_checkpoint: bool = False,
) -> dict[str, object]:
    if status not in ("planned", "running", "complete", "failed"):
        raise ValueError("invalid nominal run status")
    return {
        "run_id": spec.run_id,
        "phase": spec.phase,
        "task": spec.resolved.task.env_id,
        "morphology": spec.resolved.morphology.name,
        "controller": spec.resolved.config.controller.kind,
        "evolution": spec.resolved.config.evolution.kind,
        "training_seed": spec.resolved.config.seed,
        "config_path": str(spec.output_dir / "config.yaml"),
        "resolved_config_digest": spec.resolved.config_digest,
        "output_dir": str(spec.output_dir),
        "status": status,
        "completed_generations": completed_generations,
        "best_fitness": "" if best_fitness is None else best_fitness,
        "started_from_checkpoint": started_from_checkpoint,
    }


def write_run_index(path: Path, rows: Sequence[dict[str, object]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(path).with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_suite_manifest(
    path: Path,
    specs: Sequence[NominalRunSpec],
    *,
    command: str,
) -> None:
    first = specs[0]
    protocol = load_scientific_protocol()
    data = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": _git_sha(),
        "phase": first.phase,
        "expected_runs": len(specs),
        "conditions": [
            {
                "task": condition.task,
                "morphology": condition.morphology,
            }
            for condition in NOMINAL_CONDITIONS
        ],
        "tasks": list(dict.fromkeys(condition.task for condition in NOMINAL_CONDITIONS)),
        "controllers": list(CONTROLLERS),
        "training_seeds": sorted({item.resolved.config.seed for item in specs}),
        "scientific_settings": {
            "task_morphologies": [
                {
                    "task": condition.task,
                    "morphology": condition.morphology,
                }
                for condition in NOMINAL_CONDITIONS
            ],
            "evolution": {
                "kind": first.resolved.config.evolution.kind,
                "population_size": first.resolved.config.evolution.population_size,
                "generations": first.resolved.config.evolution.generations,
                "sigma": first.resolved.config.evolution.sigma,
            },
            "episodes_per_candidate": {
                "static": protocol.episodes_per_candidate.static,
                "nchl": protocol.episodes_per_candidate.nchl,
            },
            "fitness_aggregation": first.resolved.config.evaluation.fitness_aggregation,
        },
        "command": command,
        "run_index": "run_index.csv",
        **scientific_identity(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
