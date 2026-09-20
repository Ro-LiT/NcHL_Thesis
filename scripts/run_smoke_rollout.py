#!/usr/bin/env python3
"""Run neutral and fixed PyTorch smoke policies in fresh EvoGym tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from thesis_testing.evogym_contract import inspect_env, make_env, map_tanh_to_actuator_target
from thesis_testing.morphology import get_reference_morphology
from thesis_testing.reproducibility import collect_runtime_manifest
from thesis_testing.smoke_rollout import (
    RolloutResult,
    build_neutral_policy,
    build_smoke_policy,
    run_rollout,
)
from thesis_testing.task_catalog import selected_tasks


def _construct(env_id: str, morphology: Any) -> tuple[Any, float]:
    start = perf_counter()
    env = make_env(env_id, morphology)
    return env, perf_counter() - start


def _summary(result: RolloutResult, construction_seconds: float) -> dict[str, Any]:
    return {
        "construction_seconds": construction_seconds,
        "reset_seconds": result.reset_seconds,
        "rollout_seconds": result.rollout_seconds,
        "average_env_step_seconds": result.average_env_step_seconds,
        "average_policy_seconds": result.average_policy_seconds,
        "steps_per_second": result.steps_per_second,
        "episode_length": result.episode_length,
        "total_reward": result.total_reward,
        "terminated": result.terminated,
        "truncated": result.truncated,
        "simulation_instability": result.simulation_instability,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default=selected_tasks()[0].env_id)
    parser.add_argument(
        "--morphology", choices=("biped", "thrower_arm"), default="biped"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/runtime_manifest.json")
    )
    args = parser.parse_args()

    morphology = get_reference_morphology(args.morphology)

    env, neutral_construction = _construct(args.env_id, morphology)
    contract = inspect_env(env)
    neutral_policy = build_neutral_policy(contract.action_dim)
    neutral_raw = neutral_policy(torch.zeros(contract.observation_dim))
    neutral_action = map_tanh_to_actuator_target(neutral_raw, env.action_space)
    if not np.array_equal(neutral_action, np.ones(contract.action_dim)):
        env.close()
        raise AssertionError("neutral policy did not map exactly to actuator target 1.0")
    neutral_result = run_rollout(env, neutral_policy, args.seed, record_trace=False)

    env, smoke_construction = _construct(args.env_id, morphology)
    smoke_policy = build_smoke_policy(contract.observation_dim, contract.action_dim, args.seed)
    smoke_result = run_rollout(env, smoke_policy, args.seed, record_trace=False)

    output = collect_runtime_manifest(args.seed)
    output.update(
        {
            "env_id": args.env_id,
            "morphology": args.morphology,
            "contract": {
                "observation_dim": contract.observation_dim,
                "action_dim": contract.action_dim,
                "action_low": contract.action_low.tolist(),
                "action_high": contract.action_high.tolist(),
                "actuator_indices": contract.actuator_indices.tolist(),
                "max_episode_steps": contract.max_episode_steps,
            },
            "neutral_rollout": _summary(neutral_result, neutral_construction),
            "smoke_rollout": _summary(smoke_result, smoke_construction),
        }
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
