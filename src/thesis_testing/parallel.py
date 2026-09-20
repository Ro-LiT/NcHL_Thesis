"""Persistent process-level population evaluation."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.pool import Pool

import numpy as np
import torch

from thesis_testing.evaluation import (
    evaluate_genome,
    evaluation_episode_seeds,
    make_controller,
)
from thesis_testing.evogym_contract import make_env
from thesis_testing.specs import ResolvedExperiment


_WORKER_ENV = None
_WORKER_CONTROLLER = None
_WORKER_EPISODE_SEEDS: tuple[tuple[int, int], ...] | None = None


def _initialize_worker(resolved: ResolvedExperiment) -> None:
    global _WORKER_ENV, _WORKER_CONTROLLER, _WORKER_EPISODE_SEEDS
    torch.set_num_threads(resolved.config.runtime.torch_threads_per_worker)
    _WORKER_ENV = make_env(resolved.task.env_id, resolved.morphology)
    _WORKER_CONTROLLER = make_controller(resolved)
    _WORKER_EPISODE_SEEDS = evaluation_episode_seeds(resolved)


def _evaluate_worker(genome: np.ndarray) -> float:
    if (
        _WORKER_ENV is None
        or _WORKER_CONTROLLER is None
        or _WORKER_EPISODE_SEEDS is None
    ):
        raise RuntimeError("population worker was not initialized")
    fitness, _ = evaluate_genome(
        _WORKER_ENV,
        _WORKER_CONTROLLER,
        genome,
        _WORKER_EPISODE_SEEDS,
    )
    return fitness


class PopulationEvaluator:
    """Own one reusable environment/controller locally or per spawned worker."""

    def __init__(self, resolved: ResolvedExperiment) -> None:
        self.resolved = resolved
        self.genome_size = make_controller(resolved).genome_size
        self._closed = False
        self._env = None
        self._controller = None
        self._episode_seeds = evaluation_episode_seeds(resolved)
        self._pool: Pool | None = None

        if resolved.config.runtime.workers == 1:
            torch.set_num_threads(resolved.config.runtime.torch_threads_per_worker)
            self._env = make_env(resolved.task.env_id, resolved.morphology)
            self._controller = make_controller(resolved)
        else:
            context = mp.get_context("spawn")
            self._pool = context.Pool(
                processes=resolved.config.runtime.workers,
                initializer=_initialize_worker,
                initargs=(resolved,),
            )

    def evaluate(self, population: np.ndarray) -> np.ndarray:
        if self._closed:
            raise RuntimeError("population evaluator is closed")
        if not isinstance(population, np.ndarray) or population.ndim != 2:
            raise ValueError("population must be a two-dimensional NumPy array")
        if population.shape[1] != self.genome_size:
            raise ValueError("population genome size does not match the controller")
        if population.dtype != np.float64 or not np.all(np.isfinite(population)):
            raise ValueError("population must contain finite np.float64 values")

        if self._pool is not None:
            values = self._pool.map(_evaluate_worker, list(population))
        else:
            values = [
                evaluate_genome(
                    self._env,
                    self._controller,
                    genome,
                    self._episode_seeds,
                )[0]
                for genome in population
            ]
        fitness = np.asarray(values, dtype=np.float64)
        if fitness.shape != (population.shape[0],) or not np.all(np.isfinite(fitness)):
            raise RuntimeError("population evaluation returned invalid fitness")
        return fitness

    def close(self) -> None:
        if self._closed:
            return
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
        if self._env is not None:
            self._env.close()
        self._closed = True

    def __enter__(self) -> "PopulationEvaluator":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
