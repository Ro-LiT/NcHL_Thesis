"""Strict YAML-to-dataclass experiment configuration loader."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from thesis_testing.config.schema import (
    ES1Config,
    EvaluationConfig,
    ExperimentConfig,
    NcHLControllerConfig,
    NetworkConfig,
    RuntimeConfig,
    StaticControllerConfig,
    validate_experiment_config,
)


class ConfigError(ValueError):
    """A malformed or unsupported scientific configuration value."""


def _field_path(parent: str, field: str) -> str:
    return f"{parent}.{field}" if parent else field


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        label = path or "root"
        raise ConfigError(f"{label} must be a mapping")
    if not all(type(key) is str for key in value):
        label = path or "root"
        raise ConfigError(f"{label} keys must be strings")
    return value


def _check_keys(
    data: Mapping[str, Any], required: Iterable[str], path: str = ""
) -> None:
    required_set = set(required)
    for key in data:
        if key not in required_set:
            raise ConfigError(f"{_field_path(path, key)} is not allowed")
    for key in required:
        if key not in data:
            raise ConfigError(f"{_field_path(path, key)} is required")


def _string(value: object, path: str) -> str:
    if type(value) is not str or not value.strip():
        raise ConfigError(f"{path} must be a non-empty string")
    return value


def _integer(value: object, path: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ConfigError(f"{path} must be a {qualifier} integer")
    return value


def _float(value: object, path: str, *, positive: bool = True) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ConfigError(f"{path} must be a finite float")
    if positive and value <= 0.0:
        raise ConfigError(f"{path} must be a positive finite float")
    return value


def _boolean(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise ConfigError(f"{path} must be a boolean")
    return value


def _literal(value: object, path: str, choices: tuple[str, ...]) -> str:
    if type(value) is not str or value not in choices:
        formatted = ", ".join(repr(choice) for choice in choices)
        raise ConfigError(f"{path} must be one of: {formatted}")
    return value


def _parse_network(value: object) -> NetworkConfig:
    data = _mapping(value, "network")
    _check_keys(data, ("hidden_sizes", "activation", "bias"), "network")
    hidden_sizes_value = data["hidden_sizes"]
    if type(hidden_sizes_value) is not list:
        raise ConfigError("network.hidden_sizes must be a list")
    hidden_sizes: list[int | str] = []
    for index, width in enumerate(hidden_sizes_value):
        path = f"network.hidden_sizes[{index}]"
        if type(width) is str:
            if width != "input":
                raise ConfigError(f"{path} must be a positive integer or 'input'")
            hidden_sizes.append(width)
        elif type(width) is int and width > 0:
            hidden_sizes.append(width)
        else:
            raise ConfigError(f"{path} must be a positive integer or 'input'")
    activation = _literal(data["activation"], "network.activation", ("tanh",))
    bias = _boolean(data["bias"], "network.bias")
    if bias:
        raise ConfigError("network.bias must be false")
    return NetworkConfig(tuple(hidden_sizes), activation, bias)  # type: ignore[arg-type]


def _parse_controller(value: object):
    data = _mapping(value, "controller")
    if "kind" not in data:
        raise ConfigError("controller.kind is required")
    kind = _literal(data["kind"], "controller.kind", ("static", "nchl"))
    if kind == "static":
        _check_keys(data, ("kind",), "controller")
        return StaticControllerConfig()
    fields = ("kind", "initial_weight_low", "initial_weight_high", "eta")
    _check_keys(data, fields, "controller")
    low = _float(
        data["initial_weight_low"],
        "controller.initial_weight_low",
        positive=False,
    )
    high = _float(
        data["initial_weight_high"],
        "controller.initial_weight_high",
        positive=False,
    )
    eta = _float(data["eta"], "controller.eta")
    if low >= high:
        raise ConfigError(
            "controller.initial_weight_low must be less than "
            "controller.initial_weight_high"
        )
    return NcHLControllerConfig(low, high, eta)


def _parse_evolution(value: object):
    data = _mapping(value, "evolution")
    if "kind" not in data:
        raise ConfigError("evolution.kind is required")
    _literal(data["kind"], "evolution.kind", ("es1",))
    fields = ("kind", "population_size", "generations", "sigma")
    _check_keys(data, fields, "evolution")
    return ES1Config(
        kind="es1",
        population_size=_integer(
            data["population_size"], "evolution.population_size", minimum=1
        ),
        generations=_integer(
            data["generations"], "evolution.generations", minimum=1
        ),
        sigma=_float(data["sigma"], "evolution.sigma"),
    )


def _parse_evaluation(value: object) -> EvaluationConfig:
    data = _mapping(value, "evaluation")
    fields = (
        "protocol",
        "episodes_per_candidate",
        "fitness_aggregation",
        "record_trace",
    )
    _check_keys(data, fields, "evaluation")
    return EvaluationConfig(
        protocol=_literal(data["protocol"], "evaluation.protocol", ("nominal",)),
        episodes_per_candidate=_integer(
            data["episodes_per_candidate"],
            "evaluation.episodes_per_candidate",
            minimum=1,
        ),
        fitness_aggregation=_literal(
            data["fitness_aggregation"],
            "evaluation.fitness_aggregation",
            ("mean",),
        ),
        record_trace=_boolean(data["record_trace"], "evaluation.record_trace"),
    )


def _parse_runtime(value: object) -> RuntimeConfig:
    data = _mapping(value, "runtime")
    fields = ("device", "workers", "torch_threads_per_worker")
    _check_keys(data, fields, "runtime")
    workers = _integer(data["workers"], "runtime.workers", minimum=1)
    return RuntimeConfig(
        device=_literal(data["device"], "runtime.device", ("cpu",)),
        workers=workers,
        torch_threads_per_worker=_integer(
            data["torch_threads_per_worker"],
            "runtime.torch_threads_per_worker",
            minimum=1,
        ),
    )


def _parse_config(value: object) -> ExperimentConfig:
    data = _mapping(value, "")
    fields = (
        "schema_version",
        "name",
        "seed",
        "task",
        "morphology",
        "network",
        "controller",
        "evolution",
        "evaluation",
        "runtime",
    )
    _check_keys(data, fields)
    schema_version = _integer(data["schema_version"], "schema_version", minimum=0)
    if schema_version != 2:
        raise ConfigError("schema_version must be integer 2")
    config = ExperimentConfig(
        schema_version=schema_version,
        name=_string(data["name"], "name"),
        seed=_integer(data["seed"], "seed"),
        task=_string(data["task"], "task"),
        morphology=_string(data["morphology"], "morphology"),
        network=_parse_network(data["network"]),
        controller=_parse_controller(data["controller"]),
        evolution=_parse_evolution(data["evolution"]),
        evaluation=_parse_evaluation(data["evaluation"]),
        runtime=_parse_runtime(data["runtime"]),
    )
    validate_experiment_config(config)
    return config


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Safely load a strict schema-v2 YAML experiment configuration."""

    path = Path(path)
    try:
        text = path.read_text()
    except OSError as error:
        raise ConfigError(f"{path}: could not read configuration: {error}") from error
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigError(f"{path}: malformed YAML: {error}") from error
    return _parse_config(value)
