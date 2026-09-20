"""Deterministic flat-genome encoding for bias-free network weights."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from thesis_testing.specs import NetworkSpec


def validate_weight_genome(genome: np.ndarray, network: NetworkSpec) -> None:
    """Require one finite NumPy float64 value per resolved network weight."""

    if not isinstance(genome, np.ndarray):
        raise TypeError("genome must be a NumPy array")
    if genome.shape != (network.num_weights,):
        raise ValueError(
            f"genome must have shape {(network.num_weights,)}, got {genome.shape}"
        )
    if genome.dtype != np.float64:
        raise TypeError("genome dtype must be np.float64")
    if not np.all(np.isfinite(genome)):
        raise ValueError("genome contains NaN or infinity")


def decode_weight_genome(
    genome: np.ndarray,
    network: NetworkSpec,
) -> tuple[np.ndarray, ...]:
    """Slice a flat genome into forward-ordered C/row-major `(out, in)` matrices."""

    validate_weight_genome(genome, network)
    matrices: list[np.ndarray] = []
    start = 0
    for shape in network.weight_shapes:
        stop = start + shape[0] * shape[1]
        matrices.append(
            np.ascontiguousarray(genome[start:stop].reshape(shape), dtype=np.float64)
        )
        start = stop
    return tuple(matrices)


def encode_weight_matrices(
    matrices: Sequence[np.ndarray],
    network: NetworkSpec,
) -> np.ndarray:
    """Flatten forward-ordered `(out, in)` matrices into NumPy float64 data."""

    if len(matrices) != len(network.weight_shapes):
        raise ValueError(
            f"expected {len(network.weight_shapes)} matrices, got {len(matrices)}"
        )

    genome = np.empty(network.num_weights, dtype=np.float64)
    start = 0
    for layer_index, (matrix, shape) in enumerate(
        zip(matrices, network.weight_shapes)
    ):
        values = np.asarray(matrix)
        if values.shape != shape:
            raise ValueError(
                f"matrix {layer_index} must have shape {shape}, got {values.shape}"
            )
        if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
            raise TypeError(f"matrix {layer_index} must contain real numbers")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"matrix {layer_index} contains NaN or infinity")
        stop = start + values.size
        genome[start:stop] = values.reshape(-1)
        start = stop
    return genome
