#!/usr/bin/env python3
"""Generate independent morphology, upstream ES1, and rollout fixtures."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np

from thesis_testing.evogym_contract import inspect_env, make_env
from thesis_testing.morphology import get_reference_morphology
from thesis_testing.protocol import load_scientific_protocol
from thesis_testing.reproducibility import collect_runtime_manifest
from thesis_testing.rollout_fixture import array_sha256, smoke_policy_weights
from thesis_testing.smoke_rollout import build_smoke_policy, run_rollout


FIXTURE_ROOT = Path("tests/fixtures")
OPTIMIZER_SHA256 = "88ae893db3f4c50dcabc2ca11e2870147f9bfcfb98d809198e6be3a6198a323f"


def _load_upstream_optimizer(source: Path) -> types.ModuleType:
    optimizer_path = source / "optimizer.py"
    digest = hashlib.sha256(optimizer_path.read_bytes()).hexdigest()
    if digest != OPTIMIZER_SHA256:
        raise ValueError("optimizer.py content does not match the audited upstream file")

    # `cma` is imported at module scope but ES1 does not use it. Avoid adding the
    # upstream repository's unrelated optional dependency to this project.
    cma_stub = types.ModuleType("cma")
    cma_stub.CMAEvolutionStrategy = object
    previous = sys.modules.get("cma")
    sys.modules["cma"] = cma_stub
    try:
        spec = importlib.util.spec_from_file_location("pinned_nchl_optimizer", optimizer_path)
        if spec is None or spec.loader is None:
            raise ImportError("could not load pinned optimizer.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("cma", None)
        else:
            sys.modules["cma"] = previous
    return module


def generate_es1_fixture(source: Path) -> None:
    es1_class = _load_upstream_optimizer(source).ES1
    seed = 123
    genome_length = 4
    population_size = 6
    sigma = 0.35
    fitness = np.array([0.25, -1.0, 3.0, 2.0, 0.0, 1.0], dtype=np.float64)

    optimizer = es1_class(seed, genome_length, population_size, sigma)
    ask_generation_0 = np.asarray(optimizer.ask(), dtype=np.float64)
    optimizer.tell(ask_generation_0, fitness)
    ask_generation_1 = np.asarray(optimizer.ask(), dtype=np.float64)

    np.savez_compressed(
        FIXTURE_ROOT / "es1_reference_fixture.npz",
        optimizer_sha256=np.array(OPTIMIZER_SHA256),
        seed=np.array(seed, dtype=np.int64),
        genome_length=np.array(genome_length, dtype=np.int64),
        population_size=np.array(population_size, dtype=np.int64),
        sigma=np.array(sigma, dtype=np.float64),
        artificial_fitness=fitness,
        ask_generation_0=ask_generation_0,
        ask_generation_1=ask_generation_1,
    )


def generate_morphology_fixture() -> None:
    morphology = get_reference_morphology("biped")
    np.savez_compressed(
        FIXTURE_ROOT / "reference_morphology.npz",
        name=np.array(morphology.name),
        body=morphology.body,
        connections=morphology.connections,
    )


def generate_rollout_fixtures(seed: int) -> None:
    output = FIXTURE_ROOT / "reference_rollouts"
    output.mkdir(parents=True, exist_ok=True)
    manifest = collect_runtime_manifest(seed)
    metadata = json.dumps(
        {
            key: manifest[key]
            for key in (
                "python_version",
                "torch_version",
                "numpy_version",
                "gymnasium_version",
                "evogym_version",
            )
        },
        sort_keys=True,
    )
    for condition in load_scientific_protocol().conditions:
        env_id = condition.task
        morphology = get_reference_morphology(condition.morphology)
        env = make_env(env_id, morphology)
        contract = inspect_env(env)
        policy = build_smoke_policy(contract.observation_dim, contract.action_dim, seed)
        result = run_rollout(env, policy, seed, record_trace=True)
        filename = f"{env_id.lower().replace('-', '_')}__{morphology.name}.npz"
        np.savez_compressed(
            output / filename,
            env_id=np.array(env_id),
            morphology=np.array(morphology.name),
            seed=np.array(seed, dtype=np.int64),
            environment_seed=np.array(seed, dtype=np.int64),
            policy_seed=np.array(seed, dtype=np.int64),
            policy_weights=smoke_policy_weights(policy),
            body_sha256=np.array(array_sha256(morphology.body)),
            connections_sha256=np.array(array_sha256(morphology.connections)),
            observation_dim=np.array(contract.observation_dim, dtype=np.int64),
            action_dim=np.array(contract.action_dim, dtype=np.int64),
            max_episode_steps=np.array(contract.max_episode_steps, dtype=np.int64),
            metadata_json=np.array(metadata),
            episode_length=np.array(result.episode_length, dtype=np.int64),
            total_reward=np.array(result.total_reward, dtype=np.float64),
            terminated=np.array(result.terminated),
            truncated=np.array(result.truncated),
            simulation_instability=np.array(result.simulation_instability),
            initial_observation=result.initial_observation,
            reward_trace=result.reward_trace,
            final_observation=result.final_observation,
            observation_trace=result.observation_trace,
            action_trace=result.action_trace,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nchl-source", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-rollouts", action="store_true")
    parser.add_argument("--rollouts-only", action="store_true")
    args = parser.parse_args()

    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    generate_morphology_fixture()
    if not args.rollouts_only:
        if args.nchl_source is None:
            parser.error("--nchl-source is required unless --rollouts-only is used")
        generate_es1_fixture(args.nchl_source.resolve())
    if not args.skip_rollouts:
        generate_rollout_fixtures(args.seed)
    print(f"wrote reference fixtures under {FIXTURE_ROOT}")


if __name__ == "__main__":
    main()
