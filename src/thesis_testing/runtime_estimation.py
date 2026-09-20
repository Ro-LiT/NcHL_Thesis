"""Pure workload accounting and calibrated runtime estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TimingObservation:
    task: str
    controller: str
    network_width: int
    workers: int
    repetition: int
    candidate_episodes: int
    worker_waves: float
    startup_seconds: float
    steady_seconds: float
    mean_episode_steps: float
    early_terminations: int

    @property
    def episodes_per_second(self) -> float:
        return self.candidate_episodes / self.steady_seconds


@dataclass(frozen=True)
class WorkloadStratum:
    stage: str
    task: str
    controller: str
    network_width: int
    runs: int
    generations: int
    population: int
    episodes_per_candidate: int
    reused_runs: int = 0

    @property
    def new_runs(self) -> int:
        return self.runs - self.reused_runs

    @property
    def candidate_episodes(self) -> int:
        return (
            self.new_runs * self.generations * self.population
            * self.episodes_per_candidate
        )


def validate_timing_observations(
    observations: Sequence[TimingObservation],
    *,
    required_strata: Sequence[tuple[str, str, int]],
    target_workers: int,
    minimum_repetitions: int = 3,
    minimum_worker_waves: float = 2.0,
) -> None:
    """Reject a calibration that cannot support the requested extrapolation."""

    if type(target_workers) is not int or target_workers < 1:
        raise ValueError("target_workers must be positive")
    for item in observations:
        numeric = (
            item.candidate_episodes, item.worker_waves, item.startup_seconds,
            item.steady_seconds, item.mean_episode_steps,
        )
        if (
            item.workers < 1 or item.repetition < 0
            or not all(math.isfinite(float(value)) for value in numeric)
            or item.candidate_episodes < 1 or item.worker_waves <= 0.0
            or item.startup_seconds < 0.0 or item.steady_seconds <= 0.0
            or item.mean_episode_steps <= 0.0 or item.early_terminations < 0
        ):
            raise ValueError(f"invalid timing observation: {item}")
    for task, controller, width in required_strata:
        selected = [
            item for item in observations
            if item.task == task and item.controller == controller
            and item.network_width == width and item.workers == target_workers
        ]
        repetitions = {item.repetition for item in selected}
        if len(repetitions) < minimum_repetitions:
            raise ValueError(
                f"{task}/{controller}/width={width} has {len(repetitions)} "
                f"target-worker repetitions; requires {minimum_repetitions}"
            )
        if any(item.worker_waves < minimum_worker_waves for item in selected):
            raise ValueError(
                f"{task}/{controller}/width={width} has fewer than "
                f"{minimum_worker_waves:g} worker waves"
            )


def estimate_workload(
    observations: Sequence[TimingObservation],
    workload: Sequence[WorkloadStratum],
    *,
    target_workers: int = 25,
    contingency: float = 0.20,
    enforce_full_calibration: bool = True,
) -> dict[str, object]:
    """Estimate sequential-run wall time using measured population throughput."""

    if not 0.0 <= contingency <= 2.0 or not math.isfinite(contingency):
        raise ValueError("contingency must be finite and in [0, 2]")
    required = sorted({(item.task, item.controller, item.network_width) for item in workload})
    validate_timing_observations(
        observations,
        required_strata=required,
        target_workers=target_workers,
        minimum_repetitions=3 if enforce_full_calibration else 1,
        minimum_worker_waves=2.0,
    )
    rows: list[dict[str, object]] = []
    for item in workload:
        if item.runs < 0 or item.reused_runs < 0 or item.reused_runs > item.runs:
            raise ValueError(f"invalid run/reuse counts: {item}")
        if min(item.generations, item.population, item.episodes_per_candidate) < 1:
            raise ValueError(f"invalid workload multiplier: {item}")
        samples = [
            sample for sample in observations
            if sample.task == item.task and sample.controller == item.controller
            and sample.network_width == item.network_width
            and sample.workers == target_workers
        ]
        rates = np.asarray([sample.episodes_per_second for sample in samples], dtype=np.float64)
        startups = np.asarray([sample.startup_seconds for sample in samples], dtype=np.float64)
        central_rate = float(np.median(rates))
        conservative_rate = float(np.min(rates))
        central_seconds = item.candidate_episodes / central_rate + item.new_runs * float(np.median(startups))
        conservative_seconds = (
            item.candidate_episodes / conservative_rate
            + item.new_runs * float(np.max(startups))
        ) * (1.0 + contingency)
        rows.append(
            {
                **asdict(item), "new_runs": item.new_runs,
                "candidate_episodes": item.candidate_episodes,
                "median_episodes_per_second": central_rate,
                "slow_bound_episodes_per_second": conservative_rate,
                "median_startup_seconds": float(np.median(startups)),
                "central_wall_hours": central_seconds / 3600.0,
                "conservative_wall_hours": conservative_seconds / 3600.0,
                "central_cpu_hours": central_seconds * target_workers / 3600.0,
                "conservative_cpu_hours": conservative_seconds * target_workers / 3600.0,
            }
        )
    central = float(sum(float(row["central_wall_hours"]) for row in rows))
    conservative = float(sum(float(row["conservative_wall_hours"]) for row in rows))
    return {
        "target_workers": target_workers,
        "contingency_fraction": contingency,
        "orchestration": "training runs sequential; candidates parallel within generation",
        "strata": rows,
        "total_candidate_episodes": int(sum(item.candidate_episodes for item in workload)),
        "central_wall_hours": central,
        "conservative_wall_hours": conservative,
        "central_wall_days": central / 24.0,
        "conservative_wall_days": conservative / 24.0,
        "central_cpu_hours": central * target_workers,
        "conservative_cpu_hours": conservative * target_workers,
    }


def speed_rows(observations: Sequence[TimingObservation]) -> list[dict[str, object]]:
    """Summarize measured throughput, speedup, and efficiency by stratum."""

    groups: dict[tuple[str, str, int, int], list[float]] = {}
    for item in observations:
        groups.setdefault(
            (item.task, item.controller, item.network_width, item.workers), []
        ).append(item.episodes_per_second)
    baseline = {
        key[:3]: float(np.median(values))
        for key, values in groups.items() if key[3] == 1
    }
    output = []
    for (task, controller, width, workers), values in sorted(groups.items()):
        median = float(np.median(values))
        one = baseline.get((task, controller, width))
        speedup = median / one if one is not None else math.nan
        output.append(
            {
                "task": task, "controller": controller, "network_width": width,
                "workers": workers, "repetitions": len(values),
                "median_episodes_per_second": median,
                "min_episodes_per_second": float(np.min(values)),
                "max_episodes_per_second": float(np.max(values)),
                "speedup_vs_one_worker": speedup,
                "parallel_efficiency": speedup / workers,
            }
        )
    return output
