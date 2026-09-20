"""Functional adapter around the supported Gymnasium/EvoGym boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

from thesis_testing.morphology import MorphologySpec


@dataclass(frozen=True)
class EnvContract:
    """Observed dimensions and actuator semantics for one environment instance."""

    env_id: str
    observation_dim: int
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    actuator_indices: np.ndarray
    max_episode_steps: int

    def __post_init__(self) -> None:
        low = np.array(self.action_low, dtype=np.float64, copy=True)
        high = np.array(self.action_high, dtype=np.float64, copy=True)
        actuator_indices = np.array(self.actuator_indices, dtype=np.int64, copy=True)
        low.setflags(write=False)
        high.setflags(write=False)
        actuator_indices.setflags(write=False)
        object.__setattr__(self, "action_low", low)
        object.__setattr__(self, "action_high", high)
        object.__setattr__(self, "actuator_indices", actuator_indices)


def make_env(
    env_id: str,
    morphology: MorphologySpec,
    *,
    render_mode: str | None = None,
) -> gym.Env:
    """Construct an EvoGym benchmark, headless unless a render mode is requested."""

    import evogym.envs  # noqa: F401 - imports register EvoGym task IDs

    return gym.make(
        env_id,
        body=morphology.body.copy(),
        connections=morphology.connections.copy(),
        render_mode=render_mode,
    )


def inspect_env(env: gym.Env) -> EnvContract:
    """Derive a contract from spaces and the instantiated robot."""

    if not isinstance(env.observation_space, Box):
        raise TypeError("EvoGym observation_space must be a Gymnasium Box")
    if not isinstance(env.action_space, Box):
        raise TypeError("EvoGym action_space must be a Gymnasium Box")
    if len(env.observation_space.shape) != 1:
        raise ValueError("observation_space must be one-dimensional")
    if len(env.action_space.shape) != 1:
        raise ValueError("action_space must be one-dimensional")

    env_id = env.spec.id if env.spec is not None else type(env.unwrapped).__name__
    max_steps = env.spec.max_episode_steps if env.spec is not None else None
    if max_steps is None or max_steps <= 0:
        raise ValueError("environment must declare a positive TimeLimit horizon")

    try:
        actuator_indices = env.unwrapped.get_actuator_indices("robot")
    except (AttributeError, KeyError) as error:
        raise ValueError("environment does not expose the expected 'robot' actuators") from error

    contract = EnvContract(
        env_id=env_id,
        observation_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.shape[0]),
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        actuator_indices=actuator_indices,
        max_episode_steps=int(max_steps),
    )
    validate_env_contract(env, contract)
    return contract


def validate_env_contract(env: gym.Env, contract: EnvContract) -> None:
    """Ensure a stored contract still agrees with an environment instance."""

    if contract.observation_dim <= 0:
        raise ValueError("observation dimension must be positive")
    if contract.action_dim <= 0:
        raise ValueError("zero-length action spaces are not valid")
    if env.observation_space.shape != (contract.observation_dim,):
        raise ValueError("observation dimension changed unexpectedly")
    if env.action_space.shape != (contract.action_dim,):
        raise ValueError("action dimension changed unexpectedly")
    if contract.actuator_indices.shape != (contract.action_dim,):
        raise ValueError("actuator count does not match action dimension")
    if contract.action_low.shape != (contract.action_dim,):
        raise ValueError("action_low has the wrong shape")
    if contract.action_high.shape != (contract.action_dim,):
        raise ValueError("action_high has the wrong shape")
    if not np.all(np.isfinite(contract.action_low)):
        raise ValueError("action lower bounds must be finite")
    if not np.all(np.isfinite(contract.action_high)):
        raise ValueError("action upper bounds must be finite")
    if np.any(contract.action_low >= contract.action_high):
        raise ValueError("every action lower bound must be below its upper bound")
    if np.any(contract.action_low > 1.0) or np.any(contract.action_high < 1.0):
        raise ValueError("EvoGym neutral actuator target 1.0 must be in every bound")
    if not np.array_equal(contract.action_low, env.action_space.low):
        raise ValueError("action lower bounds changed unexpectedly")
    if not np.array_equal(contract.action_high, env.action_space.high):
        raise ValueError("action upper bounds changed unexpectedly")


def map_tanh_to_actuator_target(
    raw_action: Any,
    action_space: Box,
    *,
    neutral_target: float = 1.0,
) -> np.ndarray:
    """Map `[-1, 1]` monotonically to Box bounds with zero at neutral."""

    if not isinstance(action_space, Box) or len(action_space.shape) != 1:
        raise TypeError("action_space must be a one-dimensional Gymnasium Box")

    if hasattr(raw_action, "detach"):
        raw_action = raw_action.detach().cpu().numpy()
    raw = np.asarray(raw_action, dtype=np.float64)
    if raw.shape != action_space.shape:
        raise ValueError(f"raw action shape {raw.shape} does not match {action_space.shape}")
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw action contains NaN or infinity")
    if np.any(raw < -1.0) or np.any(raw > 1.0):
        raise ValueError("raw tanh action must lie in [-1, 1]")

    low = np.asarray(action_space.low, dtype=np.float64)
    high = np.asarray(action_space.high, dtype=np.float64)
    if low.shape != raw.shape or high.shape != raw.shape:
        raise ValueError("action bounds and raw action shapes differ")
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError("action bounds must be finite")
    if np.any(low > neutral_target) or np.any(high < neutral_target):
        raise ValueError("neutral target must lie inside every action bound")

    mapped = np.where(
        raw < 0.0,
        neutral_target + (neutral_target - low) * raw,
        neutral_target + (high - neutral_target) * raw,
    )
    return mapped.astype(action_space.dtype, copy=False)
