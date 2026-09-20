"""Immutable, versioned scientific configuration schema."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, TypeAlias


HiddenSize: TypeAlias = int | Literal["input"]


@dataclass(frozen=True)
class NetworkConfig:
    hidden_sizes: tuple[HiddenSize, ...]
    activation: Literal["tanh"]
    bias: bool


@dataclass(frozen=True)
class StaticControllerConfig:
    kind: Literal["static"] = "static"


@dataclass(frozen=True)
class NcHLControllerConfig:
    initial_weight_low: float
    initial_weight_high: float
    eta: float
    kind: Literal["nchl"] = "nchl"


ControllerConfig: TypeAlias = StaticControllerConfig | NcHLControllerConfig


@dataclass(frozen=True)
class ES1Config:
    kind: Literal["es1"]
    population_size: int
    generations: int
    sigma: float


EvolutionConfig: TypeAlias = ES1Config


@dataclass(frozen=True)
class EvaluationConfig:
    protocol: Literal["nominal"]
    episodes_per_candidate: int
    fitness_aggregation: Literal["mean"]
    record_trace: bool


@dataclass(frozen=True)
class RuntimeConfig:
    device: Literal["cpu"]
    workers: int
    torch_threads_per_worker: int


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    name: str
    seed: int
    task: str
    morphology: str
    network: NetworkConfig
    controller: ControllerConfig
    evolution: EvolutionConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig


def _exact_int(value: object) -> bool:
    return type(value) is int


def _exact_float(value: object) -> bool:
    return type(value) is float and math.isfinite(value)


def validate_experiment_config(config: ExperimentConfig) -> None:
    """Validate direct dataclass construction with full scientific field paths."""

    if type(config) is not ExperimentConfig:
        raise ValueError("config must be an ExperimentConfig")
    if not _exact_int(config.schema_version) or config.schema_version != 2:
        raise ValueError("schema_version must be integer 2")
    if type(config.name) is not str or not config.name.strip():
        raise ValueError("name must be a non-empty string")
    if not _exact_int(config.seed) or config.seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(config.task) is not str or not config.task.strip():
        raise ValueError("task must be a non-empty string")
    if type(config.morphology) is not str or not config.morphology.strip():
        raise ValueError("morphology must be a non-empty string")

    network = config.network
    if type(network) is not NetworkConfig:
        raise ValueError("network must be a NetworkConfig")
    if type(network.hidden_sizes) is not tuple:
        raise ValueError("network.hidden_sizes must be a tuple")
    for index, width in enumerate(network.hidden_sizes):
        if width == "input" and type(width) is str:
            continue
        if not _exact_int(width) or width <= 0:
            raise ValueError(
                f"network.hidden_sizes[{index}] must be a positive integer or 'input'"
            )
    if network.activation != "tanh" or type(network.activation) is not str:
        raise ValueError("network.activation must be 'tanh'")
    if type(network.bias) is not bool or network.bias:
        raise ValueError("network.bias must be false")

    controller = config.controller
    if type(controller) not in (StaticControllerConfig, NcHLControllerConfig):
        raise ValueError("controller must be a supported controller config")
    expected_controller_kind = (
        "static" if type(controller) is StaticControllerConfig else "nchl"
    )
    if controller.kind != expected_controller_kind:
        raise ValueError(f"controller.kind must be '{expected_controller_kind}'")
    if type(controller) is NcHLControllerConfig:
        if not _exact_float(controller.initial_weight_low):
            raise ValueError(
                "controller.initial_weight_low must be a finite float"
            )
        if not _exact_float(controller.initial_weight_high):
            raise ValueError(
                "controller.initial_weight_high must be a finite float"
            )
        if controller.initial_weight_low >= controller.initial_weight_high:
            raise ValueError(
                "controller.initial_weight_low must be less than "
                "controller.initial_weight_high"
            )
        if not _exact_float(controller.eta) or controller.eta <= 0.0:
            raise ValueError("controller.eta must be a positive finite float")

    evolution = config.evolution
    if type(evolution) is not ES1Config:
        raise ValueError("evolution must be an ES1Config")
    if evolution.kind != "es1":
        raise ValueError("evolution.kind must be 'es1'")
    if not _exact_int(evolution.population_size) or evolution.population_size <= 0:
        raise ValueError("evolution.population_size must be a positive integer")
    if not _exact_int(evolution.generations) or evolution.generations <= 0:
        raise ValueError("evolution.generations must be a positive integer")
    if not _exact_float(evolution.sigma) or evolution.sigma <= 0.0:
        raise ValueError("evolution.sigma must be a positive finite float")
    evaluation = config.evaluation
    if type(evaluation) is not EvaluationConfig:
        raise ValueError("evaluation must be an EvaluationConfig")
    if evaluation.protocol != "nominal":
        raise ValueError("evaluation.protocol must be 'nominal'")
    if (
        not _exact_int(evaluation.episodes_per_candidate)
        or evaluation.episodes_per_candidate < 1
    ):
        raise ValueError(
            "evaluation.episodes_per_candidate must be a positive integer"
        )
    if evaluation.fitness_aggregation != "mean":
        raise ValueError("evaluation.fitness_aggregation must be 'mean'")
    if type(evaluation.record_trace) is not bool:
        raise ValueError("evaluation.record_trace must be a boolean")

    runtime = config.runtime
    if type(runtime) is not RuntimeConfig:
        raise ValueError("runtime must be a RuntimeConfig")
    if runtime.device != "cpu":
        raise ValueError("runtime.device must be 'cpu'")
    if not _exact_int(runtime.workers) or runtime.workers < 1:
        raise ValueError("runtime.workers must be a positive integer")
    if (
        not _exact_int(runtime.torch_threads_per_worker)
        or runtime.torch_threads_per_worker < 1
    ):
        raise ValueError(
            "runtime.torch_threads_per_worker must be a positive integer"
        )
