"""Resolved, immutable architecture and experiment specifications."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from thesis_testing.evogym_contract import EnvContract
from thesis_testing.morphology import MorphologySpec
from thesis_testing.task_catalog import TaskSpec

if TYPE_CHECKING:
    from thesis_testing.config.schema import ExperimentConfig, NetworkConfig


@dataclass(frozen=True)
class NetworkSpec:
    input_dim: int
    hidden_dims: tuple[int, ...]
    output_dim: int
    layer_sizes: tuple[int, ...]
    weight_shapes: tuple[tuple[int, int], ...]
    num_neurons: int
    num_weights: int


@dataclass(frozen=True)
class SeedPlan:
    run_seed: int
    optimizer_seed: int
    initial_weight_seed: int
    evaluation_seed: int


@dataclass(frozen=True)
class ResolvedExperiment:
    config: ExperimentConfig
    task: TaskSpec
    morphology: MorphologySpec
    env_contract: EnvContract
    network: NetworkSpec
    seeds: SeedPlan
    config_digest: str


def resolve_network(
    network_config: NetworkConfig, env_contract: EnvContract
) -> NetworkSpec:
    """Resolve symbolic widths and calculate architecture-only N/W counts."""

    hidden_dims = tuple(
        env_contract.observation_dim if width == "input" else width
        for width in network_config.hidden_sizes
    )
    layer_sizes = (
        env_contract.observation_dim,
        *hidden_dims,
        env_contract.action_dim,
    )
    weight_shapes = tuple(
        (output_width, input_width)
        for input_width, output_width in zip(layer_sizes, layer_sizes[1:])
    )
    return NetworkSpec(
        input_dim=env_contract.observation_dim,
        hidden_dims=hidden_dims,
        output_dim=env_contract.action_dim,
        layer_sizes=layer_sizes,
        weight_shapes=weight_shapes,
        num_neurons=sum(layer_sizes),
        num_weights=sum(out_width * in_width for out_width, in_width in weight_shapes),
    )


def derive_seed(run_seed: int, namespace: str, *indices: int) -> int:
    """Derive a stable non-negative 63-bit seed with SHA-256."""

    if type(run_seed) is not int or run_seed < 0:
        raise ValueError("run_seed must be a non-negative integer")
    if type(namespace) is not str or not namespace:
        raise ValueError("namespace must be a non-empty string")
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("seed indices must be non-negative integers")
    payload = json.dumps(
        {"indices": list(indices), "namespace": namespace, "run_seed": run_seed},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def derive_seed_plan(run_seed: int) -> SeedPlan:
    """Create deterministic run-level subordinate seed streams."""

    return SeedPlan(
        run_seed=run_seed,
        optimizer_seed=derive_seed(run_seed, "optimizer"),
        initial_weight_seed=derive_seed(run_seed, "initial_weight"),
        evaluation_seed=derive_seed(run_seed, "evaluation"),
    )
