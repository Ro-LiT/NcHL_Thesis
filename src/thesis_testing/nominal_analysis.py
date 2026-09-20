"""Focused nominal comparison, held-out evaluation, and weight tracing."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from thesis_testing.artifacts import validate_training_run
from thesis_testing.config.loader import load_experiment_config
from thesis_testing.config.resolve import resolve_experiment
from thesis_testing.config.schema import NcHLControllerConfig, StaticControllerConfig
from thesis_testing.evaluation import make_controller, rollout
from thesis_testing.evogym_contract import make_env, map_tanh_to_actuator_target
from thesis_testing.specs import ResolvedExperiment, derive_seed


WEIGHT_STAT_NAMES = ("mean", "std", "mean_abs", "rms", "min", "max")


@dataclass(frozen=True)
class ConvergenceMetrics:
    final_best_fitness: float
    normalized_auc: float
    generation_to_90_percent: int
    wall_seconds: float


@dataclass(frozen=True)
class WeightTrace:
    rewards: np.ndarray
    weight_summary: np.ndarray
    update_rms: np.ndarray
    update_mean_abs: np.ndarray
    snapshot_states: np.ndarray
    snapshots: np.ndarray
    layer_offsets: np.ndarray


def assert_fair_pair(
    static: ResolvedExperiment,
    nchl: ResolvedExperiment,
) -> tuple[int, int]:
    """Fail before training when a Static/NcHL comparison is not matched."""

    if type(static.config.controller) is not StaticControllerConfig:
        raise ValueError("first comparison member must use the static controller")
    if type(nchl.config.controller) is not NcHLControllerConfig:
        raise ValueError("second comparison member must use the nchl controller")
    checks = {
        "task": static.task == nchl.task,
        "morphology name": static.morphology.name == nchl.morphology.name,
        "morphology body": np.array_equal(
            static.morphology.body, nchl.morphology.body
        ),
        "morphology connections": np.array_equal(
            static.morphology.connections, nchl.morphology.connections
        ),
        "network": static.network == nchl.network,
        "network config": static.config.network == nchl.config.network,
        "observation contract": (
            static.env_contract.observation_dim == nchl.env_contract.observation_dim
        ),
        "action contract": (
            static.env_contract.action_dim == nchl.env_contract.action_dim
            and np.array_equal(
                static.env_contract.action_low, nchl.env_contract.action_low
            )
            and np.array_equal(
                static.env_contract.action_high, nchl.env_contract.action_high
            )
        ),
        "rollout horizon": (
            static.env_contract.max_episode_steps
            == nchl.env_contract.max_episode_steps
        ),
        "evolution": static.config.evolution == nchl.config.evolution,
        "evaluation protocol": (
            static.config.evaluation.protocol == nchl.config.evaluation.protocol
        ),
        "fitness aggregation": (
            static.config.evaluation.fitness_aggregation
            == nchl.config.evaluation.fitness_aggregation
        ),
        "record trace": (
            static.config.evaluation.record_trace == nchl.config.evaluation.record_trace
        ),
        "run seed": static.config.seed == nchl.config.seed,
    }
    mismatches = [name for name, matched in checks.items() if not matched]
    if mismatches:
        raise ValueError("unfair Static/NcHL comparison: " + ", ".join(mismatches))
    return make_controller(static).genome_size, make_controller(nchl).genome_size


def read_metrics(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with Path(path).open(newline="") as stream:
        for raw in csv.DictReader(stream):
            rows.append({key: float(value) for key, value in raw.items()})
    if not rows:
        raise ValueError(f"training metrics are empty: {path}")
    return rows


def convergence_metrics(rows: Sequence[dict[str, float]]) -> ConvergenceMetrics:
    generations = np.asarray([row["generation"] for row in rows], dtype=np.float64)
    best = np.asarray([row["best_so_far_fitness"] for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(generations)) or not np.all(np.isfinite(best)):
        raise ValueError("training metrics contain non-finite values")
    if np.any(np.diff(generations) <= 0) or np.any(np.diff(best) < -1e-12):
        raise ValueError("generation must increase and best-so-far must not decrease")
    if len(rows) == 1:
        auc = float(best[0])
    else:
        auc = float(np.trapz(best, generations) / (generations[-1] - generations[0]))
    threshold = best[0] + 0.9 * (best[-1] - best[0])
    reached = int(np.flatnonzero(best >= threshold)[0])
    return ConvergenceMetrics(
        final_best_fitness=float(best[-1]),
        normalized_auc=auc,
        generation_to_90_percent=int(generations[reached]),
        wall_seconds=float(math.fsum(row["elapsed_seconds"] for row in rows)),
    )


def heldout_episode_seeds(
    run_seed: int,
    count: int,
    *,
    namespace: str = "nominal_test",
) -> tuple[tuple[int, int], ...]:
    if type(count) is not int or count < 1:
        raise ValueError("held-out episode count must be positive")
    return tuple(
        (
            derive_seed(run_seed, f"{namespace}_environment", index),
            derive_seed(run_seed, f"{namespace}_initial_weight", index),
        )
        for index in range(count)
    )


def evaluate_heldout_run(
    run_dir: Path,
    episode_count: int = 10,
) -> list[dict[str, object]]:
    """Evaluate a selected best genome without modifying training artifacts."""

    run_dir = Path(run_dir)
    resolved = resolve_experiment(load_experiment_config(run_dir / "config.yaml"))
    genome = np.load(run_dir / "best_genome.npy")
    if genome.dtype != np.float64:
        genome = genome.astype(np.float64)
    controller = make_controller(resolved)
    controller.set_genome(genome)
    seeds = heldout_episode_seeds(resolved.config.seed, episode_count)
    rows: list[dict[str, object]] = []
    run_id = (
        f"{resolved.task.env_id.split('-')[0].lower()}__{resolved.morphology.name}__"
        f"{resolved.config.controller.kind}__seed_{resolved.config.seed:03d}"
    )
    env = make_env(resolved.task.env_id, resolved.morphology)
    try:
        for index, (environment_seed, controller_seed) in enumerate(seeds):
            result = rollout(
                env,
                controller,
                environment_seed=environment_seed,
                controller_seed=controller_seed,
            )
            rows.append(
                {
                    "run_id": run_id,
                    "source_run": str(run_dir),
                    "config_digest": resolved.config_digest,
                    "training_seed": resolved.config.seed,
                    "heldout_index": index,
                    "environment_seed": environment_seed,
                    "controller_seed": controller_seed,
                    "task": resolved.task.env_id,
                    "morphology": resolved.morphology.name,
                    "controller": resolved.config.controller.kind,
                    "hidden_sizes": json.dumps(resolved.network.hidden_dims),
                    "fitness": result.fitness,
                    "episode_length": result.episode_length,
                    "terminated": result.terminated,
                    "truncated": result.truncated,
                }
            )
    finally:
        env.close()
    return rows


def _weight_vectors(controller: object) -> tuple[np.ndarray, ...]:
    weights = getattr(controller, "weights", None)
    if weights is None:
        raise TypeError("weight analysis requires a concrete network controller")
    return tuple(
        weight.detach().cpu().numpy().astype(np.float64, copy=True).reshape(-1)
        for weight in weights
    )


def _summaries(vectors: tuple[np.ndarray, ...]) -> np.ndarray:
    combined = np.concatenate(vectors)
    rows = [combined, *vectors]
    return np.asarray(
        [
            (
                float(np.mean(values)),
                float(np.std(values)),
                float(np.mean(np.abs(values))),
                float(np.sqrt(np.mean(np.square(values)))),
                float(np.min(values)),
                float(np.max(values)),
            )
            for values in rows
        ],
        dtype=np.float64,
    )


def _rms_delta(
    before: tuple[np.ndarray, ...], after: tuple[np.ndarray, ...]
) -> np.ndarray:
    deltas = tuple(new - old for old, new in zip(before, after, strict=True))
    combined = np.concatenate(deltas)
    return np.asarray(
        [
            np.sqrt(np.mean(np.square(values)))
            for values in (combined, *deltas)
        ],
        dtype=np.float64,
    )


def _mean_abs_delta(
    before: tuple[np.ndarray, ...], after: tuple[np.ndarray, ...]
) -> np.ndarray:
    deltas = tuple(new - old for old, new in zip(before, after, strict=True))
    combined = np.concatenate(deltas)
    return np.asarray(
        [np.mean(np.abs(values)) for values in (combined, *deltas)],
        dtype=np.float64,
    )


def trace_weights(
    resolved: ResolvedExperiment,
    genome: np.ndarray,
    *,
    environment_seed: int,
    controller_seed: int,
) -> WeightTrace:
    """Rerun one best solution, saving summaries and five full snapshots only."""

    controller = make_controller(resolved)
    controller.set_genome(genome)
    sizing_env = make_env(resolved.task.env_id, resolved.morphology)
    try:
        episode_length = rollout(
            sizing_env,
            controller,
            environment_seed=environment_seed,
            controller_seed=controller_seed,
        ).episode_length
    finally:
        sizing_env.close()

    target_states = np.asarray(
        [
            0,
            round(0.25 * episode_length),
            round(0.50 * episode_length),
            round(0.75 * episode_length),
            episode_length,
        ],
        dtype=np.int64,
    )
    env = make_env(resolved.task.env_id, resolved.morphology)
    controller.set_genome(genome)
    controller.reset_episode(controller_seed)
    observation, _ = env.reset(seed=environment_seed)
    summaries = [_summaries(_weight_vectors(controller))]
    updates: list[np.ndarray] = []
    update_mean_abs: list[np.ndarray] = []
    snapshots: dict[int, np.ndarray] = {
        0: np.concatenate(_weight_vectors(controller))
    }
    rewards: list[float] = []
    terminated = truncated = False
    try:
        while not (terminated or truncated):
            before = _weight_vectors(controller)
            with torch.inference_mode():
                raw = controller.forward(
                    torch.as_tensor(observation, dtype=torch.float32, device="cpu")
                )
                controller.post_forward_update()
            after = _weight_vectors(controller)
            updates.append(_rms_delta(before, after))
            update_mean_abs.append(_mean_abs_delta(before, after))
            summaries.append(_summaries(after))
            state = len(updates)
            if state in target_states:
                snapshots[state] = np.concatenate(after)
            action = map_tanh_to_actuator_target(raw, env.action_space)
            observation, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
    finally:
        env.close()
    if len(rewards) != episode_length:
        raise RuntimeError("deterministic weight-trace rerun changed episode length")
    if set(snapshots) != set(target_states.tolist()):
        raise RuntimeError("weight trace did not capture all requested snapshots")
    layer_sizes = [vector.size for vector in _weight_vectors(controller)]
    return WeightTrace(
        rewards=np.asarray(rewards, dtype=np.float64),
        weight_summary=np.asarray(summaries, dtype=np.float64),
        update_rms=np.asarray(updates, dtype=np.float64),
        update_mean_abs=np.asarray(update_mean_abs, dtype=np.float64),
        snapshot_states=target_states,
        snapshots=np.asarray([snapshots[int(state)] for state in target_states]),
        layer_offsets=np.asarray([0, *np.cumsum(layer_sizes)], dtype=np.int64),
    )


def save_weight_trace(path: Path, trace: WeightTrace, metadata: dict[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rewards=trace.rewards,
        weight_summary=trace.weight_summary,
        update_rms=trace.update_rms,
        update_mean_abs=trace.update_mean_abs,
        snapshot_states=trace.snapshot_states,
        snapshots=trace.snapshots,
        layer_offsets=trace.layer_offsets,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def load_weight_trace(path: Path) -> tuple[WeightTrace, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        trace = WeightTrace(
            rewards=data["rewards"].copy(),
            weight_summary=data["weight_summary"].copy(),
            update_rms=data["update_rms"].copy(),
            update_mean_abs=(
                data["update_mean_abs"].copy()
                if "update_mean_abs" in data
                else np.zeros_like(data["update_rms"])
            ),
            snapshot_states=data["snapshot_states"].copy(),
            snapshots=data["snapshots"].copy(),
            layer_offsets=data["layer_offsets"].copy(),
        )
    return trace, metadata


def genome_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plasticity_summary(trace: WeightTrace) -> dict[str, float]:
    initial = trace.snapshots[0]
    final = trace.snapshots[-1]
    peak_index = int(np.argmax(trace.update_rms[:, 0])) if trace.update_rms.size else 0
    denominator = max(1, trace.update_rms.shape[0] - 1)
    return {
        "weight_drift_rms": float(np.sqrt(np.mean(np.square(final - initial)))),
        "plasticity_path_length": float(np.sum(trace.update_rms[:, 0])),
        "mean_update_rms": float(np.mean(trace.update_rms[:, 0])),
        "peak_update_rms": float(np.max(trace.update_rms[:, 0])),
        "peak_update_normalized_time": peak_index / denominator,
        "initial_weight_rms": float(np.sqrt(np.mean(np.square(initial)))),
        "final_weight_rms": float(np.sqrt(np.mean(np.square(final)))),
    }


def layer_snapshot_rows(trace: WeightTrace) -> list[dict[str, object]]:
    labels = ("W0", "W25", "W50", "W75", "Wfinal")
    rows: list[dict[str, object]] = []
    for snapshot_index, (label, state) in enumerate(
        zip(labels, trace.snapshot_states, strict=True)
    ):
        for layer in range(trace.layer_offsets.size - 1):
            start = int(trace.layer_offsets[layer])
            stop = int(trace.layer_offsets[layer + 1])
            values = trace.snapshots[snapshot_index, start:stop]
            rows.append(
                {
                    "snapshot": label,
                    "state": int(state),
                    "layer": layer,
                    "number_of_weights": values.size,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "mean_abs": float(np.mean(np.abs(values))),
                    "rms": float(np.sqrt(np.mean(np.square(values)))),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return rows


def aggregate_run_scores(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["run_id"]), []).append(row)
    output = []
    for run_id, members in sorted(grouped.items()):
        values = np.asarray([float(row["fitness"]) for row in members])
        first = members[0]
        output.append(
            {
                "run_id": run_id,
                "source_run": first["source_run"],
                "config_digest": first["config_digest"],
                "task": first["task"],
                "morphology": first["morphology"],
                "controller": first["controller"],
                "training_seed": first["training_seed"],
                "n_heldout_episodes": len(members),
                "mean_fitness": float(np.mean(values)),
                "median_fitness": float(np.median(values)),
                "std_fitness": float(np.std(values)),
                "min_fitness": float(np.min(values)),
                "max_fitness": float(np.max(values)),
            }
        )
    return output


def completed_training_runs(root: Path) -> tuple[Path, ...]:
    index = Path(root) / "run_index.csv"
    if index.is_file():
        with index.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        complete = []
        for row in rows:
            if row.get("status") != "complete":
                continue
            run_dir = Path(row["output_dir"])
            validation = validate_training_run(run_dir)
            if not validation.valid:
                messages = "; ".join(issue.message for issue in validation.issues)
                raise ValueError(f"indexed complete run is invalid: {run_dir}: {messages}")
            complete.append(run_dir)
        return tuple(complete)
    complete = []
    for run_dir in discover_training_runs(root):
        if validate_training_run(run_dir).valid:
            complete.append(run_dir)
    return tuple(complete)


def cliffs_delta(first: Sequence[float], second: Sequence[float]) -> float:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        raise ValueError("Cliff's delta requires two non-empty samples")
    differences = x[:, None] - y[None, :]
    return float((np.sum(differences > 0) - np.sum(differences < 0)) / differences.size)


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(tuple(p_values), dtype=np.float64)
    if values.ndim != 1 or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must lie in [0,1]")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = values.size
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def discover_training_runs(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path.parent
            for path in Path(root).rglob("metrics.csv")
            if (path.parent / "config.yaml").is_file()
            and (path.parent / "best_genome.npy").is_file()
        )
    )
