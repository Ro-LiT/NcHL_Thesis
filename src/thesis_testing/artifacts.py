"""Validation of completed training result data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from thesis_testing.config.loader import load_experiment_config
from thesis_testing.config.resolve import resolve_experiment
from thesis_testing.evaluation import make_controller
from thesis_testing.specs import ResolvedExperiment


METRIC_FIELDS = (
    "generation",
    "best_fitness",
    "best_so_far_fitness",
    "mean_fitness",
    "median_fitness",
    "std_fitness",
    "evaluation_count",
    "elapsed_seconds",
)
FINAL_ARTIFACTS = (
    "config.yaml",
    "metrics.csv",
    "best_genome.npy",
    "summary.json",
)


@dataclass(frozen=True)
class ArtifactIssue:
    code: str
    message: str


@dataclass(frozen=True)
class TrainingRunValidation:
    valid: bool
    completed_generations: int
    best_fitness: float | None
    issues: tuple[ArtifactIssue, ...]


def _finite_float(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def validate_training_run(
    output_dir: Path,
    *,
    expected: ResolvedExperiment | None = None,
) -> TrainingRunValidation:
    """Validate the configuration, metrics, best genome, and summary of one run."""

    output_dir = Path(output_dir)
    issues: list[ArtifactIssue] = []

    def issue(code: str, message: str) -> None:
        issues.append(ArtifactIssue(code, message))

    missing = [name for name in FINAL_ARTIFACTS if not (output_dir / name).is_file()]
    if missing:
        issue("missing_artifact", "missing artifacts: " + ", ".join(missing))
        return TrainingRunValidation(False, 0, None, tuple(issues))

    try:
        config = load_experiment_config(output_dir / "config.yaml")
        resolved = resolve_experiment(config)
    except Exception as error:
        issue("invalid_config", f"configuration cannot be resolved: {error}")
        return TrainingRunValidation(False, 0, None, tuple(issues))
    if expected is not None and resolved.config_digest != expected.config_digest:
        issue("config_digest", "resolved configuration differs from expected run")

    metrics: list[dict[str, str]] = []
    try:
        with (output_dir / "metrics.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != METRIC_FIELDS:
                raise ValueError("metrics header does not match the exact schema")
            metrics = list(reader)
        if not metrics:
            raise ValueError("metrics contains no generations")
    except Exception as error:
        issue("invalid_metrics", str(error))

    completed = len(metrics)
    best_from_metrics: float | None = None
    if metrics:
        try:
            generations = [int(row["generation"]) for row in metrics]
            if generations != list(range(len(metrics))):
                raise ValueError("metrics generations are not contiguous from zero")
            best_so_far = []
            for generation, row in enumerate(metrics):
                numeric = {
                    name: _finite_float(row[name], f"metrics[{generation}].{name}")
                    for name in METRIC_FIELDS
                    if name not in ("generation", "evaluation_count")
                }
                evaluation_count = int(row["evaluation_count"])
                expected_count = (
                    (generation + 1)
                    * config.evolution.population_size
                    * config.evaluation.episodes_per_candidate
                )
                if evaluation_count != expected_count:
                    raise ValueError(
                        f"generation {generation} evaluation_count is {evaluation_count}, expected {expected_count}"
                    )
                if numeric["elapsed_seconds"] < 0.0:
                    raise ValueError("elapsed_seconds must be non-negative")
                if numeric["best_so_far_fitness"] < numeric["best_fitness"]:
                    raise ValueError("best_so_far_fitness is below generation best")
                best_so_far.append(numeric["best_so_far_fitness"])
            if any(right < left for left, right in zip(best_so_far, best_so_far[1:])):
                raise ValueError("best_so_far_fitness is not monotonic")
            best_from_metrics = best_so_far[-1]
        except Exception as error:
            issue("invalid_metrics", str(error))

    try:
        summary = json.loads((output_dir / "summary.json").read_text())
        summary_completed = int(summary["completed_generations"])
        summary_best = _finite_float(summary["best_fitness"], "summary.best_fitness")
        if summary_completed != config.evolution.generations:
            issue("summary_generations", "summary does not contain configured generations")
        if summary_completed != completed:
            issue("summary_metrics", "summary generation count differs from metrics")
        if best_from_metrics is not None and not math.isclose(
            summary_best, best_from_metrics, rel_tol=1e-12, abs_tol=1e-12
        ):
            issue("summary_best", "summary best fitness differs from metrics")
    except Exception as error:
        summary_completed, summary_best = completed, None
        issue("invalid_summary", f"summary cannot be read: {error}")

    try:
        genome = np.load(output_dir / "best_genome.npy", allow_pickle=False)
        expected_size = make_controller(resolved).genome_size
        if (
            genome.shape != (expected_size,)
            or genome.dtype != np.float64
            or not np.all(np.isfinite(genome))
        ):
            raise ValueError(
                f"best genome must be finite float64 shape ({expected_size},), got {genome.dtype} {genome.shape}"
            )
    except Exception as error:
        genome = None
        issue("invalid_genome", str(error))

    return TrainingRunValidation(
        not issues,
        summary_completed,
        summary_best,
        tuple(issues),
    )
