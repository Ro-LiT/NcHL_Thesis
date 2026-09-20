"""Direct implementation of the pinned parent-mean ES1 algorithm."""

from __future__ import annotations

import math

import numpy as np


class ES1:
    """Uniform first generation, Gaussian parent-mean mutation, and elitism."""

    def __init__(
        self,
        genome_size: int,
        population_size: int,
        sigma: float,
        seed: int,
    ) -> None:
        if type(genome_size) is not int or genome_size < 1:
            raise ValueError("genome_size must be a positive integer")
        if type(population_size) is not int or population_size < 1:
            raise ValueError("population_size must be a positive integer")
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma must be positive and finite")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")

        self._genome_size = genome_size
        self._population_size = population_size
        self.sigma = float(sigma)
        self._rng = np.random.default_rng(seed)
        self._parents: np.ndarray | None = None
        self._elite: np.ndarray | None = None
        self._elite_fitness: float | None = None
        self._asked_population: np.ndarray | None = None

    @property
    def population_size(self) -> int:
        return self._population_size

    @property
    def genome_size(self) -> int:
        return self._genome_size

    def ask(self) -> np.ndarray:
        if self._asked_population is not None:
            raise RuntimeError("tell() is required before the next ask()")
        if self._elite is None:
            population = self._rng.uniform(
                -1.0,
                1.0,
                (self.population_size, self.genome_size),
            )
        else:
            assert self._parents is not None
            center = np.mean(self._parents, axis=0)
            children = [
                center
                + self._rng.standard_normal(size=self.genome_size) * self.sigma
                for _ in range(self.population_size - 1)
            ]
            population = np.asarray([*children, self._elite], dtype=np.float64)
        self._asked_population = np.asarray(population, dtype=np.float64)
        return self._asked_population.copy()

    def tell(self, population: np.ndarray, fitness: np.ndarray) -> None:
        population, fitness = self._validate_tell(population, fitness)
        order = np.argsort(-fitness, kind="stable")
        best_index = int(order[0])
        best_fitness = float(fitness[best_index])
        if self._elite_fitness is None or best_fitness > self._elite_fitness:
            self._elite_fitness = best_fitness
            self._elite = population[best_index].copy()
        parent_count = min(10, self.population_size)
        self._parents = population[order[:parent_count]].copy()
        self._asked_population = None

    def _validate_tell(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._asked_population is None:
            raise RuntimeError("ask() is required before tell()")
        if not isinstance(population, np.ndarray) or population.shape != (
            self.population_size,
            self.genome_size,
        ):
            raise ValueError("population has the wrong shape")
        if population.dtype != np.float64 or not np.all(np.isfinite(population)):
            raise ValueError("population must contain finite np.float64 values")
        if not np.array_equal(population, self._asked_population):
            raise ValueError("population must be the most recent ask() result")
        if not isinstance(fitness, np.ndarray) or fitness.shape != (
            self.population_size,
        ):
            raise ValueError("fitness has the wrong shape")
        if fitness.dtype != np.float64 or not np.all(np.isfinite(fitness)):
            raise ValueError("fitness must contain finite np.float64 values")
        return population, fitness
