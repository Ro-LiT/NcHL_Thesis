"""Short, explicit ES training loop with local atomic checkpoints."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np
import yaml

from thesis_testing.config.resolve import experiment_to_dict
from thesis_testing.evaluation import make_controller
from thesis_testing.evolution import ES1, EvolutionStrategy
from thesis_testing.parallel import PopulationEvaluator
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


@dataclass(frozen=True)
class TrainingSummary:
    completed_generations: int
    configured_generations: int
    best_fitness: float
    best_genome_path: str
    resumed: bool


def make_evolution_strategy(
    resolved: ResolvedExperiment,
    genome_size: int,
) -> EvolutionStrategy:
    """Construct the protocol's ES1 optimizer."""

    config = resolved.config.evolution
    common = dict(
        genome_size=genome_size,
        population_size=config.population_size,
        sigma=config.sigma,
        seed=resolved.seeds.optimizer_seed,
    )
    return ES1(**common)


def _atomic_pickle(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _initialize_outputs(resolved: ResolvedExperiment, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    config_data = experiment_to_dict(resolved.config)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config_data, sort_keys=False)
    )
    with (output_dir / "metrics.csv").open("w", newline="") as stream:
        csv.DictWriter(stream, fieldnames=METRIC_FIELDS).writeheader()


def _load_checkpoint(output_dir: Path, digest: str) -> dict[str, object]:
    path = output_dir / "checkpoint.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {path}")
    with path.open("rb") as stream:
        state = pickle.load(stream)
    if not isinstance(state, dict) or state.get("config_digest") != digest:
        raise ValueError("checkpoint configuration digest does not match")
    return state


def _reconcile_resume_outputs(output_dir: Path, state: dict[str, object]) -> None:
    """Roll mutable artifacts back to the last committed checkpoint boundary."""

    next_generation = int(state["next_generation"])
    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != METRIC_FIELDS:
            raise ValueError("metrics schema does not match checkpointed training")
        rows = list(reader)
    committed = [row for row in rows if int(row["generation"]) < next_generation]
    if [int(row["generation"]) for row in committed] != list(range(next_generation)):
        raise ValueError("metrics before checkpoint are missing or non-contiguous")
    temporary = metrics_path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(committed)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, metrics_path)
    best_genome = np.asarray(state["best_genome"], dtype=np.float64)
    temporary_genome = output_dir / "best_genome.tmp.npy"
    np.save(temporary_genome, best_genome)
    os.replace(temporary_genome, output_dir / "best_genome.npy")
    for stale in (output_dir / "summary.json",):
        if stale.exists():
            stale.unlink()


def train(
    resolved: ResolvedExperiment,
    output_dir: Path,
    *,
    checkpoint_every: int = 25,
    resume: bool = False,
    stop_after_generation: int | None = None,
) -> TrainingSummary:
    """Run the visible ask/evaluate/tell loop and save minimal artifacts.

    ``stop_after_generation`` is a bounded-run/resume aid; it does not alter the
    experiment configuration or its digest.
    """

    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError("checkpoint_every must be a positive integer")
    generations = resolved.config.evolution.generations
    if stop_after_generation is not None and (
        type(stop_after_generation) is not int
        or stop_after_generation < 1
        or stop_after_generation > generations
    ):
        raise ValueError("stop_after_generation must lie inside the configured run")
    target_generation = stop_after_generation or generations
    output_dir = Path(output_dir)
    controller = make_controller(resolved)

    if resume:
        state = _load_checkpoint(output_dir, resolved.config_digest)
        _reconcile_resume_outputs(output_dir, state)
        start_generation = int(state["next_generation"])
        es = state["es"]
        best_genome = state["best_genome"]
        best_fitness = float(state["best_fitness"])
        if not isinstance(es, ES1):
            raise TypeError("checkpoint contains an unsupported ES object")
    else:
        _initialize_outputs(resolved, output_dir)
        start_generation = 0
        es = make_evolution_strategy(resolved, controller.genome_size)
        best_genome = None
        best_fitness = -math.inf

    if start_generation > target_generation:
        raise ValueError("checkpoint is already beyond requested stop generation")

    with PopulationEvaluator(resolved) as evaluator:
        for generation in range(start_generation, target_generation):
            started = perf_counter()
            population = es.ask()
            fitness = evaluator.evaluate(population)

            generation_best_index = int(np.argmax(fitness))
            generation_best = float(fitness[generation_best_index])
            if generation_best > best_fitness:
                best_fitness = generation_best
                best_genome = population[generation_best_index].copy()
                np.save(output_dir / "best_genome.npy", best_genome)

            row = {
                "generation": generation,
                "best_fitness": generation_best,
                "best_so_far_fitness": best_fitness,
                "mean_fitness": float(np.mean(fitness)),
                "median_fitness": float(np.median(fitness)),
                "std_fitness": float(np.std(fitness)),
                "evaluation_count": (
                    (generation + 1)
                    * resolved.config.evolution.population_size
                    * resolved.config.evaluation.episodes_per_candidate
                ),
                "elapsed_seconds": perf_counter() - started,
            }
            with (output_dir / "metrics.csv").open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
                writer.writerow(row)

            es.tell(population, fitness)
            next_generation = generation + 1
            if next_generation % checkpoint_every == 0 or next_generation == target_generation:
                _atomic_pickle(
                    output_dir / "checkpoint.pkl",
                    {
                        "config_digest": resolved.config_digest,
                        "next_generation": next_generation,
                        "es": es,
                        "best_genome": best_genome,
                        "best_fitness": best_fitness,
                    },
                )

    if best_genome is None or not math.isfinite(best_fitness):
        raise RuntimeError("training completed without a finite best genome")
    summary = TrainingSummary(
        completed_generations=target_generation,
        configured_generations=generations,
        best_fitness=best_fitness,
        best_genome_path=str(output_dir / "best_genome.npy"),
        resumed=resume,
    )
    _write_json(output_dir / "summary.json", summary.__dict__)
    if target_generation == generations:
        (output_dir / "checkpoint.pkl").unlink()
    return summary
