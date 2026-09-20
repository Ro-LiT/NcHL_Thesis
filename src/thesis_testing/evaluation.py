"""Production controller-to-EvoGym rollout and deterministic fitness evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import gymnasium as gym
import numpy as np
import torch

from thesis_testing.config.schema import NcHLControllerConfig, StaticControllerConfig
from thesis_testing.controllers import Controller, NcHLController, StaticMLP
from thesis_testing.evogym_contract import inspect_env, map_tanh_to_actuator_target
from thesis_testing.specs import ResolvedExperiment, derive_seed


@dataclass(frozen=True)
class RolloutResult:
    fitness: float
    episode_length: int
    terminated: bool
    truncated: bool
    rewards: tuple[float, ...] | None = None


def make_controller(resolved: ResolvedExperiment) -> Controller:
    config = resolved.config.controller
    if type(config) is StaticControllerConfig:
        return StaticMLP(resolved.network)
    if type(config) is NcHLControllerConfig:
        return NcHLController(
            resolved.network,
            config.initial_weight_low,
            config.initial_weight_high,
            config.eta,
        )


def evaluation_episode_seeds(
    resolved: ResolvedExperiment,
) -> tuple[tuple[int, int], ...]:
    """Return the fixed common-random-number seed pairs for one ES run."""

    count = resolved.config.evaluation.episodes_per_candidate
    pairs = [
        (resolved.seeds.evaluation_seed, resolved.seeds.initial_weight_seed)
    ]
    pairs.extend(
        (
            derive_seed(resolved.seeds.run_seed, "evaluation", index),
            derive_seed(resolved.seeds.run_seed, "initial_weight", index),
        )
        for index in range(1, count)
    )
    return tuple(pairs)


def _observation(value: object, expected_dim: int) -> np.ndarray:
    observation = np.asarray(value)
    if observation.shape != (expected_dim,):
        raise ValueError(
            f"observation shape changed unexpectedly: {observation.shape} != "
            f"{(expected_dim,)}"
        )
    if not np.all(np.isfinite(observation)):
        raise ValueError("observation contains NaN or infinity")
    return observation


def rollout(
    env: gym.Env,
    controller: Controller,
    *,
    environment_seed: int,
    controller_seed: int,
    record_rewards: bool = False,
) -> RolloutResult:
    """Run one reset episode; the caller owns and may reuse the environment."""

    contract = inspect_env(env)
    reset = env.reset(seed=environment_seed)
    if not isinstance(reset, tuple) or len(reset) != 2:
        raise TypeError("Gymnasium reset must return (observation, info)")
    observation, info = reset
    if not isinstance(info, dict):
        raise TypeError("Gymnasium reset info must be a dictionary")
    obs = _observation(observation, contract.observation_dim)
    controller.reset_episode(controller_seed)

    rewards: list[float] = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        tensor_obs = torch.as_tensor(obs, dtype=torch.float32, device="cpu")
        with torch.inference_mode():
            raw_action = controller.forward(tensor_obs)
            controller.post_forward_update()
        action = map_tanh_to_actuator_target(raw_action, env.action_space)

        transition = env.step(action)
        if not isinstance(transition, tuple) or len(transition) != 5:
            raise TypeError("Gymnasium step must return a five-tuple")
        observation, reward, terminated_value, truncated_value, info = transition
        if not isinstance(info, dict):
            raise TypeError("Gymnasium step info must be a dictionary")
        reward_value = float(reward)
        if not math.isfinite(reward_value):
            raise ValueError("environment reward is NaN or infinity")
        rewards.append(reward_value)
        terminated = bool(terminated_value)
        truncated = bool(truncated_value)
        obs = _observation(observation, contract.observation_dim)
        if len(rewards) > contract.max_episode_steps:
            raise RuntimeError("rollout exceeded the registered TimeLimit horizon")

    return RolloutResult(
        fitness=math.fsum(rewards),
        episode_length=len(rewards),
        terminated=terminated,
        truncated=truncated,
        rewards=tuple(rewards) if record_rewards else None,
    )


def evaluate_genome(
    env: gym.Env,
    controller: Controller,
    genome: np.ndarray,
    episode_seeds: tuple[tuple[int, int], ...],
    *,
    record_rewards: bool = False,
) -> tuple[float, tuple[RolloutResult, ...]]:
    """Evaluate one genome using the mean official return over fixed episodes."""

    if not episode_seeds:
        raise ValueError("at least one episode seed pair is required")
    controller.set_genome(genome)
    results = tuple(
        rollout(
            env,
            controller,
            environment_seed=environment_seed,
            controller_seed=controller_seed,
            record_rewards=record_rewards,
        )
        for environment_seed, controller_seed in episode_seeds
    )
    fitness = float(np.mean([result.fitness for result in results], dtype=np.float64))
    if not math.isfinite(fitness):
        raise ValueError("candidate fitness is NaN or infinity")
    return fitness, results
