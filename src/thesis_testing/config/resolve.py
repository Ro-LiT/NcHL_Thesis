"""Resolve validated configuration into environment-derived specifications."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

from thesis_testing.config.loader import ConfigError
from thesis_testing.config.schema import (
    ExperimentConfig,
    NcHLControllerConfig,
    StaticControllerConfig,
    validate_experiment_config,
)
from thesis_testing.evogym_contract import inspect_env, make_env
from thesis_testing.morphology import get_reference_morphology
from thesis_testing.specs import ResolvedExperiment, derive_seed_plan, resolve_network
from thesis_testing.task_catalog import get_task_spec


def experiment_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Convert scientific configuration to stable primitive values only."""

    validate_experiment_config(config)
    controller = config.controller
    if type(controller) is StaticControllerConfig:
        controller_data = {"kind": "static"}
    else:
        controller_data = {
            "kind": "nchl",
            "initial_weight_low": controller.initial_weight_low,
            "initial_weight_high": controller.initial_weight_high,
            "eta": controller.eta,
        }

    evolution = config.evolution
    evolution_data = {
        "kind": "es1",
        "population_size": evolution.population_size,
        "generations": evolution.generations,
        "sigma": evolution.sigma,
    }

    return {
        "schema_version": config.schema_version,
        "name": config.name,
        "seed": config.seed,
        "task": config.task,
        "morphology": config.morphology,
        "network": {
            "hidden_sizes": list(config.network.hidden_sizes),
            "activation": config.network.activation,
            "bias": config.network.bias,
        },
        "controller": controller_data,
        "evolution": evolution_data,
        "evaluation": {
            "protocol": config.evaluation.protocol,
            "episodes_per_candidate": config.evaluation.episodes_per_candidate,
            "fitness_aggregation": config.evaluation.fitness_aggregation,
            "record_trace": config.evaluation.record_trace,
        },
        "runtime": {
            "device": config.runtime.device,
            "workers": config.runtime.workers,
            "torch_threads_per_worker": config.runtime.torch_threads_per_worker,
        },
    }


def _validate_canonical(value: object, path: str = "root") -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_canonical(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{path} contains a non-string key")
            _validate_canonical(item, f"{path}.{key}")
        return
    raise ValueError(f"{path} contains unsupported type {type(value).__name__}")


def canonical_json(config_or_data: ExperimentConfig | Mapping[str, Any]) -> str:
    """Serialize scientific identity with sorted keys and no non-finite values."""

    data: Mapping[str, Any]
    if isinstance(config_or_data, ExperimentConfig):
        data = experiment_to_dict(config_or_data)
    else:
        data = config_or_data
    _validate_canonical(data)
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def config_digest(config: ExperimentConfig) -> str:
    """Return the SHA-256 identity of all explicit scientific configuration."""

    return hashlib.sha256(canonical_json(config).encode("ascii")).hexdigest()


def resolve_experiment(config: ExperimentConfig) -> ResolvedExperiment:
    """Inspect a temporary EvoGym instance and return a closed, resolved spec."""

    validate_experiment_config(config)
    try:
        task = get_task_spec(config.task)
    except KeyError as error:
        raise ConfigError(f"task {error.args[0]}") from error
    try:
        morphology = get_reference_morphology(config.morphology)
    except KeyError as error:
        raise ConfigError(f"morphology {error.args[0]}") from error

    env = None
    try:
        env = make_env(task.env_id, morphology)
        env_contract = inspect_env(env)
    finally:
        if env is not None:
            env.close()
    if env_contract.max_episode_steps != task.expected_horizon:
        raise ValueError(
            f"task {task.env_id} horizon {env_contract.max_episode_steps} does not "
            f"match catalog {task.expected_horizon}"
        )

    return ResolvedExperiment(
        config=config,
        task=task,
        morphology=morphology,
        env_contract=env_contract,
        network=resolve_network(config.network, env_contract),
        seeds=derive_seed_plan(config.seed),
        config_digest=config_digest(config),
    )
