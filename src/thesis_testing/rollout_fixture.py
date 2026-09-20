"""Replay and compare committed real-EvoGym rollout regression fixtures."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn

from thesis_testing.evogym_contract import inspect_env, make_env
from thesis_testing.morphology import get_reference_morphology
from thesis_testing.smoke_rollout import RolloutResult, build_smoke_policy, run_rollout


def array_sha256(value: np.ndarray) -> str:
    """Hash dtype, shape, and C-order bytes so array identity is unambiguous."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def smoke_policy_weights(policy: nn.Module) -> NDArray[np.float32]:
    """Return forward-ordered smoke-policy weights as an auditable flat array."""

    values = [
        layer.weight.detach().cpu().numpy().astype(np.float32, copy=True).ravel()
        for layer in policy.modules()
        if isinstance(layer, nn.Linear)
    ]
    if not values:
        raise ValueError("smoke policy contains no linear weights")
    return np.concatenate(values).astype(np.float32, copy=False)


def _scalar(fixture: Mapping[str, np.ndarray], key: str) -> object:
    if key not in fixture:
        raise AssertionError(f"rollout fixture is missing {key}")
    return fixture[key].item()


def assert_rollout_matches_fixture(
    result: RolloutResult,
    fixture: Mapping[str, np.ndarray],
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> None:
    """Compare every scientific rollout field, not timing/diagnostic noise."""

    assert result.episode_length == int(_scalar(fixture, "episode_length"))
    assert result.terminated is bool(_scalar(fixture, "terminated"))
    assert result.truncated is bool(_scalar(fixture, "truncated"))
    assert result.simulation_instability is bool(
        _scalar(fixture, "simulation_instability")
    )
    np.testing.assert_allclose(
        result.total_reward,
        float(_scalar(fixture, "total_reward")),
        rtol=rtol,
        atol=atol,
    )
    for field in (
        "initial_observation",
        "final_observation",
        "reward_trace",
        "observation_trace",
        "action_trace",
    ):
        if field not in fixture:
            raise AssertionError(f"rollout fixture is missing {field}")
        np.testing.assert_allclose(
            getattr(result, field), fixture[field], rtol=rtol, atol=atol
        )


def replay_rollout_fixture(path: Path) -> RolloutResult:
    """Reconstruct independent fixture inputs and execute a fresh real rollout."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as stored:
        fixture = {key: stored[key].copy() for key in stored.files}
    env_id = str(_scalar(fixture, "env_id"))
    morphology = get_reference_morphology(str(_scalar(fixture, "morphology")))
    if str(_scalar(fixture, "body_sha256")) != array_sha256(morphology.body):
        raise AssertionError("rollout fixture body hash does not match morphology")
    if str(_scalar(fixture, "connections_sha256")) != array_sha256(
        morphology.connections
    ):
        raise AssertionError("rollout fixture connection hash does not match morphology")
    env = make_env(env_id, morphology)
    contract = inspect_env(env)
    if int(_scalar(fixture, "observation_dim")) != contract.observation_dim:
        env.close()
        raise AssertionError("rollout fixture observation dimension changed")
    if int(_scalar(fixture, "action_dim")) != contract.action_dim:
        env.close()
        raise AssertionError("rollout fixture action dimension changed")
    if int(_scalar(fixture, "max_episode_steps")) != contract.max_episode_steps:
        env.close()
        raise AssertionError("rollout fixture horizon changed")
    policy_seed = int(_scalar(fixture, "policy_seed"))
    policy = build_smoke_policy(contract.observation_dim, contract.action_dim, policy_seed)
    np.testing.assert_array_equal(
        smoke_policy_weights(policy), fixture["policy_weights"]
    )
    result = run_rollout(
        env,
        policy,
        int(_scalar(fixture, "environment_seed")),
        record_trace=True,
    )
    assert_rollout_matches_fixture(result, fixture)
    return result
