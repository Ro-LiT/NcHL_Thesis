"""Paper-faithful neuron-centric Hebbian learning controller."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from thesis_testing.specs import NetworkSpec


def nchl_delta(
    pre_activation: torch.Tensor,
    post_activation: torch.Tensor,
    pre_params: torch.Tensor,
    post_params: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    """Calculate published Equation 3 for one `(post, pre)` weight matrix."""

    pre_a = pre_activation[None, :]
    post_a = post_activation[:, None]
    a_term = pre_params[:, 0][None, :] * pre_a
    b_term = post_params[:, 1][:, None] * post_a
    c_term = (
        post_params[:, 2][:, None]
        * pre_params[:, 2][None, :]
        * post_a
        * pre_a
    )
    d_term = post_params[:, 3][:, None] * pre_params[:, 3][None, :]
    delta = eta * (a_term + b_term + c_term + d_term)
    if not bool(torch.isfinite(delta).all()):
        raise ValueError("NcHL Equation-3 update is non-finite")
    return delta


class NcHLController(nn.Module):
    """Bias-free tanh MLP whose runtime weights follow published Equation 3."""

    def __init__(
        self,
        network: NetworkSpec,
        initial_weight_low: float,
        initial_weight_high: float,
        eta: float,
    ) -> None:
        super().__init__()
        if not math.isfinite(initial_weight_low) or not math.isfinite(
            initial_weight_high
        ):
            raise ValueError("initial weight bounds must be finite")
        if initial_weight_low >= initial_weight_high:
            raise ValueError("initial_weight_low must be less than initial_weight_high")
        if not math.isfinite(eta) or eta <= 0.0:
            raise ValueError("eta must be a positive finite float")

        self.network = network
        self.initial_weight_low = float(initial_weight_low)
        self.initial_weight_high = float(initial_weight_high)
        self.eta = float(eta)
        self.weights = nn.ParameterList(
            nn.Parameter(torch.zeros(shape, dtype=torch.float32), requires_grad=False)
            for shape in network.weight_shapes
        )
        self.hebbian_parameters = nn.ParameterList(
            nn.Parameter(
                torch.zeros((width, 4), dtype=torch.float32),
                requires_grad=False,
            )
            for width in network.layer_sizes
        )
        self._rule_loaded = False
        self._episode_ready = False
        self._cached_activations: tuple[torch.Tensor, ...] | None = None
        self.eval()

    @property
    def genome_size(self) -> int:
        return 4 * self.network.num_neurons

    def set_genome(self, genome: np.ndarray) -> None:
        """Load `[A,B,C,D]` for every neuron in forward layer order."""

        if not isinstance(genome, np.ndarray):
            raise TypeError("genome must be a NumPy array")
        if genome.shape != (self.genome_size,):
            raise ValueError(
                f"genome must have shape {(self.genome_size,)}, got {genome.shape}"
            )
        if genome.dtype != np.float64:
            raise TypeError("genome dtype must be np.float64")
        if not np.all(np.isfinite(genome)):
            raise ValueError("genome contains NaN or infinity")

        values = torch.from_numpy(genome.reshape(self.network.num_neurons, 4)).to(
            dtype=torch.float32, device="cpu"
        )
        if not bool(torch.isfinite(values).all()):
            raise ValueError("genome overflows finite torch.float32")

        offset = 0
        with torch.no_grad():
            for parameters, width in zip(
                self.hebbian_parameters, self.network.layer_sizes
            ):
                parameters.copy_(values[offset : offset + width])
                offset += width
        self._rule_loaded = True
        self._episode_ready = False
        self._cached_activations = None

    def reset_episode(self, seed: int) -> None:
        """Sample reproducible runtime weights without changing global RNG state."""

        if not self._rule_loaded:
            raise RuntimeError("a valid NcHL rule genome must be loaded before reset")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")

        rng = np.random.default_rng(seed)
        initial_weights = [
            torch.from_numpy(
                rng.uniform(
                    self.initial_weight_low,
                    self.initial_weight_high,
                    size=tuple(weight.shape),
                ).astype(np.float32)
            )
            for weight in self.weights
        ]
        with torch.no_grad():
            for weight, initial in zip(self.weights, initial_weights):
                weight.copy_(initial)
        self._cached_activations = None
        self._episode_ready = True

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Compute the raw action with current weights and cache all activations."""

        if not self._episode_ready:
            raise RuntimeError("set_genome() and reset_episode() are required")
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
            activation = observation
            activations = [observation.detach().clone()]
            for weight in self.weights:
                activation = torch.tanh(F.linear(activation, weight, bias=None))
                activations.append(activation)
        self._cached_activations = tuple(activations)
        return activation

    def post_forward_update(self) -> None:
        """Apply Equation 3 after the action has been computed with current weights."""

        if not self._episode_ready:
            raise RuntimeError("set_genome() and reset_episode() are required")
        if self._cached_activations is None:
            raise RuntimeError("forward() is required before an NcHL update")

        updated_weights: list[torch.Tensor] = []
        for layer_index, weight in enumerate(self.weights):
            delta = nchl_delta(
                self._cached_activations[layer_index],
                self._cached_activations[layer_index + 1],
                self.hebbian_parameters[layer_index],
                self.hebbian_parameters[layer_index + 1],
                self.eta,
            )
            updated = weight + delta
            if not bool(torch.isfinite(updated).all()):
                raise ValueError("NcHL runtime weight update is non-finite")
            updated_weights.append(updated)

        with torch.no_grad():
            for weight, updated in zip(self.weights, updated_weights):
                weight.copy_(updated)
        self._cached_activations = None
