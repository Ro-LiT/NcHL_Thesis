"""Production static, bias-free PyTorch controller phenotype."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from thesis_testing.controllers.genome import decode_weight_genome
from thesis_testing.specs import NetworkSpec


class StaticMLP(nn.Module):
    """A fixed CPU float32 tanh MLP loaded from a canonical float64 genome."""

    def __init__(self, network: NetworkSpec) -> None:
        super().__init__()
        self.network = network
        self.weights = nn.ParameterList(
            nn.Parameter(
                torch.zeros(shape, dtype=torch.float32, device="cpu"),
                requires_grad=False,
            )
            for shape in network.weight_shapes
        )
        self._genome_loaded = False
        self.eval()

    @property
    def genome_size(self) -> int:
        """Return the exact number of evolved static connection weights."""

        return self.network.num_weights

    def set_genome(self, genome: np.ndarray) -> None:
        """Load a validated NumPy float64 genome as CPU float32 weights."""

        matrices = decode_weight_genome(genome, self.network)
        converted: list[torch.Tensor] = []
        for layer_index, matrix in enumerate(matrices):
            tensor = torch.from_numpy(matrix).to(device="cpu", dtype=torch.float32)
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(
                    f"genome layer {layer_index} overflows finite torch.float32"
                )
            converted.append(tensor)

        with torch.no_grad():
            for weight, tensor in zip(self.weights, converted):
                weight.copy_(tensor)
        self._genome_loaded = True

    def reset_episode(self, seed: int) -> None:
        """Validate lifecycle state while leaving static weights unchanged."""

        self._require_genome()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the raw tanh action without applying EvoGym action mapping."""

        self._require_genome()
        if not isinstance(observation, torch.Tensor):
            raise TypeError("observation must be a torch.Tensor")
        if observation.dtype != torch.float32:
            raise TypeError("observation dtype must be torch.float32")
        if observation.device.type != "cpu":
            raise ValueError("observation must be on the CPU")
        if observation.shape != (self.network.input_dim,):
            raise ValueError(
                f"observation must have shape {(self.network.input_dim,)}, "
                f"got {tuple(observation.shape)}"
            )
        if not bool(torch.isfinite(observation).all()):
            raise ValueError("observation contains NaN or infinity")

        with torch.no_grad():
            output = observation
            for weight in self.weights:
                output = torch.tanh(F.linear(output, weight, bias=None))

        return output

    def post_forward_update(self) -> None:
        """Leave static weights unchanged after a forward pass."""

        return None

    def _require_genome(self) -> None:
        if not self._genome_loaded:
            raise RuntimeError("a valid genome must be loaded before controller use")
