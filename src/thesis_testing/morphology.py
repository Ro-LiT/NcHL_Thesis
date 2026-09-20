"""Fixed, validated morphologies for the frozen thesis experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


THROWER_ARM_BODY = np.array(
    [
        [0, 0, 1, 1, 3, 3],
        [0, 1, 2, 3, 4, 3],
        [1, 4, 3, 2, 4, 1],
        [4, 3, 4, 3, 4, 0],
        [4, 0, 4, 0, 4, 0],
        [4, 0, 4, 0, 4, 0],
    ],
    dtype=np.int8,
)


@dataclass(frozen=True)
class MorphologySpec:
    """Immutable EvoGym body and pairwise connectivity arrays."""

    name: str
    body: np.ndarray
    connections: np.ndarray

    def __post_init__(self) -> None:
        body = np.array(self.body, dtype=np.int64, copy=True)
        connections = np.array(self.connections, dtype=np.int64, copy=True)
        body.setflags(write=False)
        connections.setflags(write=False)
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "connections", connections)


def validate_morphology(morphology: MorphologySpec) -> None:
    """Reject invalid voxel values, topology, actuators, or connections."""

    from evogym import get_full_connectivity, has_actuator, is_connected

    body = morphology.body
    connections = morphology.connections

    if body.ndim != 2 or body.size == 0:
        raise ValueError("body must be a non-empty two-dimensional array")
    if not np.issubdtype(body.dtype, np.integer):
        raise ValueError("body must use integer voxel codes")
    if np.any((body < 0) | (body > 4)):
        raise ValueError("body voxel codes must be in the EvoGym range 0..4")
    if not is_connected(body):
        raise ValueError("morphology body is disconnected")
    if not has_actuator(body):
        raise ValueError("morphology must contain at least one actuator voxel")

    if connections.ndim != 2 or connections.shape[0] != 2:
        raise ValueError("connections must have shape (2, E)")
    if not np.issubdtype(connections.dtype, np.integer):
        raise ValueError("connections must contain integer voxel indices")
    if connections.size:
        if np.any(connections < 0) or np.any(connections >= body.size):
            raise ValueError("connections contain an out-of-range voxel index")
        if np.any(connections[0] == connections[1]):
            raise ValueError("connections may not contain self-edges")
        edges = {tuple(sorted(edge)) for edge in connections.T.tolist()}
        if len(edges) != connections.shape[1]:
            raise ValueError("connections contain duplicate edges")

    expected = get_full_connectivity(body).astype(np.int64, copy=False)
    actual_edges = {tuple(sorted(edge)) for edge in connections.T.tolist()}
    expected_edges = {tuple(sorted(edge)) for edge in expected.T.tolist()}
    if actual_edges != expected_edges:
        raise ValueError("connections do not match EvoGym full connectivity")


def _make_morphology(name: str, body: np.ndarray) -> MorphologySpec:
    from evogym import get_full_connectivity

    morphology = MorphologySpec(name, body, get_full_connectivity(body))
    validate_morphology(morphology)
    return morphology


def reference_morphologies() -> tuple[MorphologySpec, ...]:
    """Return every explicitly supplied, fixed thesis morphology."""

    biped = np.array(
        [
            [3, 4, 3, 4],
            [3, 1, 1, 4],
            [3, 0, 0, 4],
        ],
        dtype=np.int64,
    )
    return (
        _make_morphology("biped", biped),
        _make_morphology("thrower_arm", THROWER_ARM_BODY),
    )


def get_reference_morphology(name: str) -> MorphologySpec:
    """Look up one of the fixed thesis morphologies by name."""

    for morphology in reference_morphologies():
        if morphology.name == name:
            return morphology
    choices = ", ".join(item.name for item in reference_morphologies())
    raise KeyError(f"unknown morphology {name!r}; choose one of: {choices}")
