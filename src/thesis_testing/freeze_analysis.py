"""Evaluate stored NcHL weights as fixed controllers, then summarize by ES run.

This module never trains a controller or changes a source artifact. Each snapshot
starts a fresh environment with its source trace's reset seed.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from thesis_testing.config import load_experiment_config, resolve_experiment
from thesis_testing.controllers import StaticMLP
from thesis_testing.evaluation import RolloutResult, rollout
from thesis_testing.evogym_contract import make_env
from thesis_testing.nominal_analysis import WeightTrace, genome_sha256, load_weight_trace
from thesis_testing.specs import ResolvedExperiment


SNAPSHOT_NAMES = ("W0", "W25", "W50", "W75", "Wfinal")
SNAPSHOT_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
FREEZE_FITNESS_COLUMNS = (
    "run_id", "source_run", "config_digest", "genome_sha256", "training_seed",
    "task", "morphology", "analysis_index", "source_environment_seed",
    "source_controller_seed", "snapshot", "snapshot_fraction", "snapshot_state",
    "frozen_fitness", "episode_length", "terminated", "truncated",
    "source_plastic_fitness", "trace_path",
)
FREEZE_RUN_COLUMNS = (
    "run_id", "task", "morphology", "training_seed", "snapshot", "snapshot_fraction",
    "n_analysis_traces", "mean_frozen_fitness", "median_frozen_fitness",
    "sd_frozen_fitness", "min_frozen_fitness", "max_frozen_fitness",
)
FREEZE_CHANGE_COLUMNS = (
    "run_id", "task", "morphology", "training_seed",
    *(f"{name}_mean_fitness" for name in SNAPSHOT_NAMES),
    *(f"delta_{name}_from_W0" for name in SNAPSHOT_NAMES[1:]),
    "best_snapshot", "best_snapshot_fitness",
)
FREEZE_SUMMARY_COLUMNS = (
    "task", "morphology", "snapshot", "snapshot_fraction", "independent_runs",
    "mean", "sd", "median", "q1", "q3", "min", "max",
)


def evaluate_frozen_snapshot(
    resolved: ResolvedExperiment,
    snapshot: np.ndarray,
    *,
    environment_seed: int,
    controller_seed: int,
) -> RolloutResult:
    """Load concrete weights into StaticMLP and run a fresh, non-plastic episode."""

    if not isinstance(snapshot, np.ndarray):
        raise TypeError("snapshot must be a NumPy array")
    if snapshot.shape != (resolved.network.num_weights,):
        raise ValueError(f"snapshot must have shape ({resolved.network.num_weights},)")
    if snapshot.dtype.kind not in "fi" or not np.all(np.isfinite(snapshot)):
        raise ValueError("snapshot must contain finite real weights")
    frozen_controller = StaticMLP(resolved.network)
    frozen_controller.set_genome(snapshot.astype(np.float64, copy=True))
    environment = make_env(resolved.task.env_id, resolved.morphology)
    try:
        return rollout(
            environment, frozen_controller,
            environment_seed=environment_seed,
            controller_seed=controller_seed,
        )
    finally:
        environment.close()


def evaluate_frozen_trace(
    resolved: ResolvedExperiment,
    trace: WeightTrace,
    *,
    environment_seed: int,
    controller_seed: int,
) -> list[dict[str, object]]:
    """Evaluate exactly the five stored states; repeated states in short traces are valid."""

    if resolved.config.controller.kind != "nchl":
        raise ValueError("freeze analysis requires an NcHL source run")
    if trace.snapshots.shape != (5, resolved.network.num_weights):
        raise ValueError("trace must contain five snapshots matching the network")
    if trace.rewards.ndim != 1 or not trace.rewards.size or not np.all(np.isfinite(trace.rewards)):
        raise ValueError("source trace rewards must be a nonempty finite vector")
    expected_states = np.asarray([round(fraction * trace.rewards.size) for fraction in SNAPSHOT_FRACTIONS])
    if not np.array_equal(trace.snapshot_states, expected_states):
        raise ValueError("snapshot states do not match source episode fractions")
    expected_offsets = np.asarray([0, *np.cumsum([out * inp for out, inp in resolved.network.weight_shapes])])
    if not np.array_equal(trace.layer_offsets, expected_offsets):
        raise ValueError("trace layer offsets do not match the network")
    rows = []
    for name, fraction, state, snapshot in zip(
        SNAPSHOT_NAMES, SNAPSHOT_FRACTIONS, trace.snapshot_states, trace.snapshots, strict=True,
    ):
        result = evaluate_frozen_snapshot(
            resolved, snapshot,
            environment_seed=environment_seed, controller_seed=controller_seed,
        )
        rows.append({
            "snapshot": name, "snapshot_fraction": fraction, "snapshot_state": int(state),
            "frozen_fitness": result.fitness, "episode_length": result.episode_length,
            "terminated": result.terminated, "truncated": result.truncated,
            "source_plastic_fitness": float(np.sum(trace.rewards, dtype=np.float64)),
        })
    return rows


def evaluate_freeze_records(
    trace_records: Sequence[tuple[dict[str, object], Path]],
) -> list[dict[str, object]]:
    """Consume the weight stage's trace records; skip Static sources without re-tracing."""

    rows = []
    resolved_runs: dict[Path, tuple[ResolvedExperiment, str]] = {}
    seen_traces: set[tuple[str, int]] = set()
    for metadata, trace_path in trace_records:
        if metadata["controller"] != "nchl":
            continue
        run_dir = Path(str(metadata["source_run"]))
        if run_dir not in resolved_runs:
            resolved_runs[run_dir] = (
                resolve_experiment(load_experiment_config(run_dir / "config.yaml")),
                genome_sha256(run_dir / "best_genome.npy"),
            )
        resolved, genome_digest = resolved_runs[run_dir]
        trace, stored_metadata = load_weight_trace(trace_path)
        expected_metadata = {
            "config_digest": resolved.config_digest,
            "genome_sha256": genome_digest,
            "controller": "nchl", "task": resolved.task.env_id,
            "morphology": resolved.morphology.name, "training_seed": resolved.config.seed,
        }
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError(f"trace does not match its source run: {trace_path}")
        for key in (*expected_metadata, "run_id", "analysis_index", "environment_seed", "controller_seed"):
            if stored_metadata.get(key) != metadata.get(key):
                raise ValueError(f"trace metadata mismatch for {key}: {trace_path}")
        trace_key = (str(metadata["run_id"]), int(metadata["analysis_index"]))
        if trace_key in seen_traces:
            raise ValueError(f"duplicate freeze source trace: {trace_key}")
        seen_traces.add(trace_key)
        frozen_rows = evaluate_frozen_trace(
            resolved, trace,
            environment_seed=int(metadata["environment_seed"]),
            controller_seed=int(metadata["controller_seed"]),
        )
        for frozen_row in frozen_rows:
            rows.append({
                **{key: metadata[key] for key in (
                    "run_id", "source_run", "config_digest", "genome_sha256",
                    "training_seed", "task", "morphology", "analysis_index",
                )},
                "source_environment_seed": metadata["environment_seed"],
                "source_controller_seed": metadata["controller_seed"],
                **frozen_row, "trace_path": str(trace_path),
            })
    return rows


def summarize_freeze_results(
    episode_rows: Sequence[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Average nested traces first, then describe equally weighted independent runs.

    Returns run scores, changes from W0, and task/morphology summaries. SD uses
    the sample convention (ddof=1); a singleton receives 0, as in nominal reports.
    """

    runs: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in episode_rows:
        runs[str(row["run_id"])].append(row)
    run_scores, change_rows = [], []
    for run_id, members in sorted(runs.items()):
        first = members[0]
        identity = {key: first[key] for key in ("run_id", "task", "morphology", "training_seed")}
        by_snapshot: dict[str, dict[int, float]] = {name: {} for name in SNAPSHOT_NAMES}
        for row in members:
            if any(row[key] != value for key, value in identity.items()):
                raise ValueError(f"inconsistent freeze run identity: {run_id}")
            name = str(row["snapshot"])
            if name not in by_snapshot:
                raise ValueError(f"unknown snapshot: {name}")
            trace_index = int(row["analysis_index"])
            if trace_index in by_snapshot[name]:
                raise ValueError(f"duplicate snapshot score: {run_id}/{trace_index}/{name}")
            fitness = float(row["frozen_fitness"])
            if not np.isfinite(fitness):
                raise ValueError("frozen fitness must be finite")
            by_snapshot[name][trace_index] = fitness
        trace_indices = set(by_snapshot["W0"])
        if not trace_indices or any(set(scores) != trace_indices for scores in by_snapshot.values()):
            raise ValueError(f"each source trace must have all five snapshots: {run_id}")
        means = []
        for name, fraction in zip(SNAPSHOT_NAMES, SNAPSHOT_FRACTIONS):
            values = np.asarray([by_snapshot[name][i] for i in sorted(trace_indices)])
            mean = float(np.mean(values))
            means.append(mean)
            run_scores.append({
                **identity, "snapshot": name, "snapshot_fraction": fraction,
                "n_analysis_traces": len(values), "mean_frozen_fitness": mean,
                "median_frozen_fitness": float(np.median(values)),
                "sd_frozen_fitness": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min_frozen_fitness": float(np.min(values)), "max_frozen_fitness": float(np.max(values)),
            })
        best_index = int(np.argmax(means))  # Ties select the earliest lifetime stage.
        change_rows.append({
            **identity,
            **{f"{name}_mean_fitness": mean for name, mean in zip(SNAPSHOT_NAMES, means)},
            **{f"delta_{name}_from_W0": mean - means[0] for name, mean in zip(SNAPSHOT_NAMES[1:], means[1:])},
            "best_snapshot": SNAPSHOT_NAMES[best_index], "best_snapshot_fitness": means[best_index],
        })

    strata: dict[tuple[str, str, float, str], list[float]] = defaultdict(list)
    for row in run_scores:
        key = (str(row["task"]), str(row["morphology"]), float(row["snapshot_fraction"]), str(row["snapshot"]))
        strata[key].append(float(row["mean_frozen_fitness"]))
    summaries = []
    for (task, morphology, fraction, name), scores in sorted(strata.items()):
        values = np.asarray(scores)
        summaries.append({
            "task": task, "morphology": morphology, "snapshot": name, "snapshot_fraction": fraction,
            "independent_runs": len(values), "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)), "q1": float(np.quantile(values, 0.25)),
            "q3": float(np.quantile(values, 0.75)), "min": float(np.min(values)), "max": float(np.max(values)),
        })
    return run_scores, change_rows, summaries
