"""Runtime interface shared by future static and plastic controllers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class Controller(Protocol):
    """CPU controller lifecycle with post-forward plasticity separated."""

    @property
    def genome_size(self) -> int: ...

    def set_genome(self, genome: np.ndarray) -> None: ...

    def reset_episode(self, seed: int) -> None: ...

    def forward(self, observation: torch.Tensor) -> torch.Tensor: ...

    def post_forward_update(self) -> None: ...

