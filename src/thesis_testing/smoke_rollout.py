"""Deterministic, single-process PyTorch-to-EvoGym smoke rollouts."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import math
from time import perf_counter
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from thesis_testing.evogym_contract import inspect_env, map_tanh_to_actuator_target
from thesis_testing.reproducibility import seed_everything


Policy = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class RolloutResult:
    """Outcome, trace, termination reason, and baseline timing measurements."""

    total_reward: float
    episode_length: int
    terminated: bool
    truncated: bool
    simulation_instability: bool
    initial_observation: np.ndarray
    final_observation: np.ndarray
    reward_trace: np.ndarray
    observation_trace: np.ndarray
    action_trace: np.ndarray
    reset_seconds: float
    rollout_seconds: float
    env_step_seconds: float
    policy_seconds: float
    step_messages: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "initial_observation",
            "final_observation",
            "reward_trace",
            "observation_trace",
            "action_trace",
        ):
            array = np.array(getattr(self, field_name), copy=True)
            array.setflags(write=False)
            object.__setattr__(self, field_name, array)

    @property
    def average_env_step_seconds(self) -> float:
        return self.env_step_seconds / self.episode_length

    @property
    def average_policy_seconds(self) -> float:
        return self.policy_seconds / self.episode_length

    @property
    def steps_per_second(self) -> float:
        return self.episode_length / self.rollout_seconds


def build_smoke_policy(obs_dim: int, action_dim: int, seed: int) -> nn.Sequential:
    """Build the fixed `O -> O -> A` tanh policy used only for smoke tests."""

    if obs_dim <= 0 or action_dim <= 0:
        raise ValueError("policy dimensions must be positive")
    seed_everything(seed)
    policy = nn.Sequential(
        nn.Linear(obs_dim, obs_dim, bias=False),
        nn.Tanh(),
        nn.Linear(obs_dim, action_dim, bias=False),
        nn.Tanh(),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for layer in policy:
        if isinstance(layer, nn.Linear):
            nn.init.uniform_(layer.weight, -0.1, 0.1, generator=generator)
    policy.eval()
    return policy


def build_neutral_policy(action_dim: int) -> Policy:
    """Return a stateless policy whose mapped action is exactly all ones."""

    if action_dim <= 0:
        raise ValueError("action dimension must be positive")

    def policy(_: torch.Tensor) -> torch.Tensor:
        return torch.zeros(action_dim, dtype=torch.float32)

    return policy


def _validate_observation(observation: object, observation_dim: int) -> np.ndarray:
    obs = np.asarray(observation)
    if obs.shape != (observation_dim,):
        raise ValueError(
            f"observation shape changed unexpectedly: {obs.shape} != {(observation_dim,)}"
        )
    if not np.all(np.isfinite(obs)):
        raise ValueError("observation contains NaN or infinity")
    return obs


def run_rollout(
    env: gym.Env,
    policy: Policy,
    seed: int,
    record_trace: bool = False,
) -> RolloutResult:
    """Run and close one fresh environment, stopping on termination or truncation."""

    contract = inspect_env(env)
    seed_everything(seed)
    rewards: list[float] = []
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    messages: list[str] = []
    terminated = False
    truncated = False
    unstable = False
    env_step_seconds = 0.0
    policy_seconds = 0.0

    try:
        reset_start = perf_counter()
        reset_result = env.reset(seed=seed)
        reset_seconds = perf_counter() - reset_start
        if not isinstance(reset_result, tuple) or len(reset_result) != 2:
            raise TypeError("Gymnasium reset must return (observation, info)")
        observation, info = reset_result
        if not isinstance(info, dict):
            raise TypeError("Gymnasium reset info must be a dictionary")
        obs = _validate_observation(observation, contract.observation_dim)
        initial_observation = obs.copy()
        if record_trace:
            observations.append(obs.copy())

        rollout_start = perf_counter()
        while not (terminated or truncated):
            tensor_obs = torch.as_tensor(obs, dtype=torch.float32)
            policy_start = perf_counter()
            with torch.inference_mode():
                raw_action = policy(tensor_obs)
            policy_seconds += perf_counter() - policy_start
            action = map_tanh_to_actuator_target(raw_action, env.action_space)

            step_output = io.StringIO()
            step_start = perf_counter()
            with redirect_stdout(step_output):
                transition = env.step(action)
            env_step_seconds += perf_counter() - step_start
            output = step_output.getvalue().strip()
            if output:
                messages.append(output)
                unstable = unstable or "SIMULATION UNSTABLE" in output.upper()

            if not isinstance(transition, tuple) or len(transition) != 5:
                raise TypeError("Gymnasium step must return a five-tuple")
            observation, reward, terminated_value, truncated_value, info = transition
            if not isinstance(info, dict):
                raise TypeError("Gymnasium step info must be a dictionary")
            terminated = bool(terminated_value)
            truncated = bool(truncated_value)
            reward_value = float(reward)
            if not math.isfinite(reward_value):
                raise ValueError("environment reward is NaN or infinity")
            obs = _validate_observation(observation, contract.observation_dim)

            rewards.append(reward_value)
            if record_trace:
                observations.append(obs.copy())
                actions.append(action.copy())

            if len(rewards) > contract.max_episode_steps:
                raise RuntimeError("rollout exceeded the registered TimeLimit horizon")

        rollout_seconds = perf_counter() - rollout_start
        reward_trace = np.asarray(rewards, dtype=np.float64)
        if record_trace:
            observation_trace = np.asarray(observations)
            action_trace = np.asarray(actions)
        else:
            observation_trace = np.empty((0, contract.observation_dim), dtype=np.float64)
            action_trace = np.empty((0, contract.action_dim), dtype=env.action_space.dtype)

        return RolloutResult(
            total_reward=math.fsum(rewards),
            episode_length=len(rewards),
            terminated=terminated,
            truncated=truncated,
            simulation_instability=unstable,
            initial_observation=initial_observation,
            final_observation=obs,
            reward_trace=reward_trace,
            observation_trace=observation_trace,
            action_trace=action_trace,
            reset_seconds=reset_seconds,
            rollout_seconds=rollout_seconds,
            env_step_seconds=env_step_seconds,
            policy_seconds=policy_seconds,
            step_messages=tuple(messages),
        )
    finally:
        env.close()
