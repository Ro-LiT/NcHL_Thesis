"""Pure projection helpers for a measured nominal matrix rehearsal."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class MatrixTimingObservation:
    task: str
    morphology: str
    controller: str
    candidate_episodes: int
    steady_seconds: float
    fixed_overhead_seconds: float

    @property
    def episodes_per_second(self) -> float:
        return self.candidate_episodes / self.steady_seconds


def estimate_full_matrix(
    observations: Sequence[MatrixTimingObservation],
    *,
    runs_per_condition: int,
    generations: int,
    population_size: int,
    episodes_per_candidate: Mapping[str, int],
    contingency: float = 0.20,
) -> dict[str, object]:
    """Scale one measured rehearsal of each condition to a full training matrix.

    ``episodes_per_candidate`` maps each controller name to its candidate-episode
    budget, so Static and NcHL conditions scale by different factors.
    """

    integer_inputs = (runs_per_condition, generations, population_size)
    if any(type(value) is not int or value < 1 for value in integer_inputs):
        raise ValueError("full-matrix integer inputs must be positive")
    if any(type(value) is not int or value < 1 for value in episodes_per_candidate.values()):
        raise ValueError("episodes_per_candidate values must be positive integers")
    if not math.isfinite(contingency) or not 0.0 <= contingency <= 2.0:
        raise ValueError("contingency must be finite and in [0, 2]")
    if not observations:
        raise ValueError("at least one timing observation is required")

    keys: set[tuple[str, str, str]] = set()
    condition_rows: list[dict[str, object]] = []
    total_full_episodes = 0
    for observation in observations:
        key = (
            observation.task, observation.morphology, observation.controller,
        )
        if key in keys:
            raise ValueError(f"duplicate timing condition: {key}")
        keys.add(key)
        if observation.controller not in episodes_per_candidate:
            raise ValueError(
                f"missing episodes_per_candidate for controller: "
                f"{observation.controller}"
            )
        numeric = (
            observation.steady_seconds, observation.fixed_overhead_seconds,
        )
        if (
            observation.candidate_episodes < 1
            or not all(math.isfinite(value) for value in numeric)
            or observation.steady_seconds <= 0.0
            or observation.fixed_overhead_seconds < 0.0
        ):
            raise ValueError(f"invalid timing observation: {observation}")
        rate = observation.episodes_per_second
        full_episodes_per_condition = (
            runs_per_condition * generations * population_size
            * episodes_per_candidate[observation.controller]
        )
        total_full_episodes += full_episodes_per_condition
        steady = full_episodes_per_condition / rate
        overhead = runs_per_condition * observation.fixed_overhead_seconds
        projected = steady + overhead
        condition_rows.append(
            {
                **asdict(observation),
                "episodes_per_second": rate,
                "full_candidate_episodes": full_episodes_per_condition,
                "projected_wall_seconds": projected,
                "projected_wall_hours": projected / 3600.0,
            }
        )

    total_seconds = sum(float(row["projected_wall_seconds"]) for row in condition_rows)
    contingency_seconds = total_seconds * (1.0 + contingency)
    return {
        "condition_count": len(condition_rows),
        "runs_per_condition": runs_per_condition,
        "generations": generations,
        "population_size": population_size,
        "episodes_per_candidate": dict(episodes_per_candidate),
        "total_training_runs": len(condition_rows) * runs_per_condition,
        "total_candidate_episodes": total_full_episodes,
        "contingency_fraction": contingency,
        "measured_projection_hours": total_seconds / 3600.0,
        "measured_projection_days": total_seconds / 86400.0,
        "planning_with_contingency_hours": contingency_seconds / 3600.0,
        "planning_with_contingency_days": contingency_seconds / 86400.0,
        "conditions": condition_rows,
    }


def estimate_serial_evaluation_from_parallel_rate(
    observations: Sequence[MatrixTimingObservation],
    *,
    runs_per_condition: int,
    episodes_per_run: int,
    measured_worker_processes: int,
) -> dict[str, object]:
    """Approximate a serial evaluation pass from measured parallel throughput.

    Multiplying wall time per episode by the measured worker count estimates the
    single-worker episode cost. This is a planning approximation, not a benchmark
    confidence interval.
    """

    integer_inputs = (
        runs_per_condition, episodes_per_run, measured_worker_processes,
    )
    if any(type(value) is not int or value < 1 for value in integer_inputs):
        raise ValueError("serial-evaluation inputs must be positive integers")
    if not observations:
        raise ValueError("at least one timing observation is required")
    keys: set[tuple[str, str, str]] = set()
    condition_rows: list[dict[str, object]] = []
    for observation in observations:
        key = (
            observation.task, observation.morphology, observation.controller,
        )
        if key in keys:
            raise ValueError(f"duplicate timing condition: {key}")
        keys.add(key)
        if (
            observation.candidate_episodes < 1
            or not math.isfinite(observation.steady_seconds)
            or observation.steady_seconds <= 0.0
        ):
            raise ValueError(f"invalid timing observation: {observation}")
        episodes = runs_per_condition * episodes_per_run
        seconds = episodes * measured_worker_processes / observation.episodes_per_second
        condition_rows.append(
            {
                "task": observation.task,
                "morphology": observation.morphology,
                "controller": observation.controller,
                "episodes": episodes,
                "projected_wall_seconds": seconds,
                "projected_wall_hours": seconds / 3600.0,
            }
        )
    total_seconds = sum(float(row["projected_wall_seconds"]) for row in condition_rows)
    return {
        "method": "parallel_rate_times_measured_worker_count",
        "runs_per_condition": runs_per_condition,
        "episodes_per_run": episodes_per_run,
        "measured_worker_processes": measured_worker_processes,
        "total_episodes": sum(int(row["episodes"]) for row in condition_rows),
        "projected_wall_seconds": total_seconds,
        "projected_wall_hours": total_seconds / 3600.0,
        "conditions": condition_rows,
    }
