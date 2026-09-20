"""Optimizer-independent NumPy boundary for future Evolution Strategies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EvolutionStrategy(Protocol):
    """Maximizing ask/tell interface over float64 `(P,G)` populations."""

    @property
    def population_size(self) -> int: ...

    @property
    def genome_size(self) -> int: ...

    def ask(self) -> np.ndarray: ...

    def tell(self, population: np.ndarray, fitness: np.ndarray) -> None: ...

