"""Strict machine-readable scientific protocol and identity."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import yaml

REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_PATH = REPOSITORY / "configs" / "scientific_protocol.yaml"
ENVIRONMENT_PATH = REPOSITORY / "environment.yml"

ALLOWED_STUDY_OUTCOMES = {
    ("nominal", "heldout_fitness"),
}
ALLOWED_PRIORITIES = {"primary", "secondary", "descriptive"}
ALLOWED_COMPARISONS = {"nchl_vs_static"}
ALLOWED_TESTS = {"mann_whitney_u"}


@dataclass(frozen=True)
class ScientificClaim:
    id: str
    priority: str
    study: str
    outcome: str
    stratum: str
    strata: tuple[str, ...]
    comparison: str
    test: str
    correction_family: str
    direction: str
    inclusion: str
    planned_runs_per_controller: int


@dataclass(frozen=True)
class ScientificCondition:
    task: str
    morphology: str

    @property
    def label(self) -> str:
        return f"{self.task}:{self.morphology}"


@dataclass(frozen=True)
class ControllerEpisodeBudget:
    static: int
    nchl: int

    def for_controller(self, controller: str) -> int:
        if controller == "static":
            return self.static
        if controller == "nchl":
            return self.nchl
        raise ValueError(f"unknown controller: {controller}")


@dataclass(frozen=True)
class ScientificProtocol:
    schema_version: int
    protocol_id: str
    status: str
    sample_size_status: str
    conditions: tuple[ScientificCondition, ...]
    controllers: tuple[str, ...]
    training_seeds: tuple[int, ...]
    calibration_seeds: tuple[int, ...]
    population_size: int
    generations: int
    sigma: float
    nchl_parameters: tuple[str, ...]
    nchl_eta: float
    episodes_per_candidate: ControllerEpisodeBudget
    heldout_episodes_per_run: int
    claims: tuple[ScientificClaim, ...]

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(condition.task for condition in self.conditions))

def file_sha256(path: Path) -> str:
    """Return the SHA-256 of one required project identity file."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exact_keys(data: dict[str, Any], expected: tuple[str, ...], path: str) -> None:
    extras = set(data) - set(expected)
    missing = set(expected) - set(data)
    if extras:
        raise ValueError(f"{path} has unsupported fields: {sorted(extras)}")
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")


def _text(value: object, path: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _positive_int(value: object, path: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _positive_float(value: object, path: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{path} must be a positive finite float")
    return value


def _string_tuple(value: object, path: str) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise ValueError(f"{path} must be a non-empty list")
    result = tuple(_text(item, f"{path}[]") for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{path} contains duplicates")
    return result


def _seed_tuple(value: object, path: str) -> tuple[int, ...]:
    if type(value) is not list or not value:
        raise ValueError(f"{path} must be a non-empty list")
    result = tuple(value)
    if any(type(seed) is not int or seed < 0 for seed in result):
        raise ValueError(f"{path} must contain non-negative integers")
    if len(result) != len(set(result)):
        raise ValueError(f"{path} contains duplicates")
    return result


def load_scientific_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> ScientificProtocol:
    """Load and strictly validate the versioned thesis claim registry."""

    data = yaml.safe_load(Path(path).read_text())
    if type(data) is not dict:
        raise ValueError("protocol root must be a mapping")
    root_fields = (
        "schema_version", "protocol_id", "status", "conditions",
        "sample_size_status", "population_size", "generations", "sigma",
        "nchl_parameters", "nchl_eta",
        "episodes_per_candidate",
        "controllers", "training_seeds", "calibration_seeds",
        "heldout_episodes_per_run", "claims",
    )
    _exact_keys(data, root_fields, "protocol")
    if data["schema_version"] != 2 or type(data["schema_version"]) is not int:
        raise ValueError("protocol.schema_version must be integer 2")
    if data["status"] not in ("calibration", "frozen"):
        raise ValueError("protocol.status must be calibration or frozen")
    if data["sample_size_status"] not in (
        "pending_precision_and_power_pilot", "frozen_from_pilot_report",
        "frozen_by_resource_constraint",
    ):
        raise ValueError("protocol.sample_size_status is unsupported")
    if data["status"] == "frozen" and not data["sample_size_status"].startswith("frozen_"):
        raise ValueError("a frozen protocol requires a frozen sample size")
    raw_conditions = data["conditions"]
    if type(raw_conditions) is not list or not raw_conditions:
        raise ValueError("protocol.conditions must be a non-empty list")
    conditions = []
    for index, raw in enumerate(raw_conditions):
        if type(raw) is not dict:
            raise ValueError(f"protocol.conditions[{index}] must be a mapping")
        _exact_keys(raw, ("task", "morphology"), f"protocol.conditions[{index}]")
        conditions.append(
            ScientificCondition(
                task=_text(raw["task"], f"protocol.conditions[{index}].task"),
                morphology=_text(
                    raw["morphology"], f"protocol.conditions[{index}].morphology"
                ),
            )
        )
    labels = [condition.label for condition in conditions]
    if len(labels) != len(set(labels)):
        raise ValueError("protocol.conditions contains duplicates")
    controllers = _string_tuple(data["controllers"], "protocol.controllers")
    if controllers != ("static", "nchl"):
        raise ValueError("protocol.controllers must be [static, nchl]")

    nchl_parameters = _string_tuple(
        data["nchl_parameters"], "protocol.nchl_parameters"
    )
    if nchl_parameters != ("A", "B", "C", "D"):
        raise ValueError("protocol.nchl_parameters must be [A, B, C, D]")
    nchl_eta = _positive_float(data["nchl_eta"], "protocol.nchl_eta")
    raw_episodes = data["episodes_per_candidate"]
    if type(raw_episodes) is not dict:
        raise ValueError("protocol.episodes_per_candidate must be a mapping")
    _exact_keys(
        raw_episodes,
        ("static", "nchl"),
        "protocol.episodes_per_candidate",
    )
    episode_budget = ControllerEpisodeBudget(
        static=_positive_int(
            raw_episodes["static"], "protocol.episodes_per_candidate.static"
        ),
        nchl=_positive_int(
            raw_episodes["nchl"], "protocol.episodes_per_candidate.nchl"
        ),
    )
    training = _seed_tuple(data["training_seeds"], "protocol.training_seeds")
    calibration = _seed_tuple(data["calibration_seeds"], "protocol.calibration_seeds")
    if set(training) & set(calibration):
        raise ValueError("training and calibration seeds must be disjoint")
    raw_claims = data["claims"]
    if type(raw_claims) is not list or not raw_claims:
        raise ValueError("protocol.claims must be a non-empty list")
    claim_fields = (
        "id", "priority", "study", "outcome", "stratum", "comparison",
        "strata",
        "test", "correction_family", "direction", "inclusion",
        "planned_runs_per_controller",
    )
    claims = []
    for index, raw in enumerate(raw_claims):
        if type(raw) is not dict:
            raise ValueError(f"protocol.claims[{index}] must be a mapping")
        _exact_keys(raw, claim_fields, f"protocol.claims[{index}]")
        claim = ScientificClaim(
            id=_text(raw["id"], f"protocol.claims[{index}].id"),
            priority=_text(raw["priority"], f"protocol.claims[{index}].priority"),
            study=_text(raw["study"], f"protocol.claims[{index}].study"),
            outcome=_text(raw["outcome"], f"protocol.claims[{index}].outcome"),
            stratum=_text(raw["stratum"], f"protocol.claims[{index}].stratum"),
            strata=_string_tuple(raw["strata"], f"protocol.claims[{index}].strata"),
            comparison=_text(raw["comparison"], f"protocol.claims[{index}].comparison"),
            test=_text(raw["test"], f"protocol.claims[{index}].test"),
            correction_family=_text(raw["correction_family"], f"protocol.claims[{index}].correction_family"),
            direction=_text(raw["direction"], f"protocol.claims[{index}].direction"),
            inclusion=_text(raw["inclusion"], f"protocol.claims[{index}].inclusion"),
            planned_runs_per_controller=_positive_int(
                raw["planned_runs_per_controller"],
                f"protocol.claims[{index}].planned_runs_per_controller",
            ),
        )
        if claim.priority not in ALLOWED_PRIORITIES:
            raise ValueError(f"unsupported claim priority: {claim.priority}")
        if (claim.study, claim.outcome) not in ALLOWED_STUDY_OUTCOMES:
            raise ValueError(f"unsupported study/outcome: {claim.study}/{claim.outcome}")
        if claim.comparison not in ALLOWED_COMPARISONS:
            raise ValueError(f"unsupported comparison: {claim.comparison}")
        if claim.test not in ALLOWED_TESTS:
            raise ValueError(f"unsupported test: {claim.test}")
        claims.append(claim)
    ids = [claim.id for claim in claims]
    if len(ids) != len(set(ids)):
        raise ValueError("protocol claim IDs must be unique")
    condition_labels = {condition.label for condition in conditions}
    for claim in claims:
        if claim.study == "nominal" and set(claim.strata) != condition_labels:
            raise ValueError("nominal claim strata must equal the official conditions")
        if claim.planned_runs_per_controller != len(training):
            raise ValueError(
                "claim planned_runs_per_controller must equal the training seed count"
            )
    return ScientificProtocol(
        schema_version=2,
        protocol_id=_text(data["protocol_id"], "protocol.protocol_id"),
        status=data["status"],
        sample_size_status=data["sample_size_status"],
        conditions=tuple(conditions),
        controllers=controllers,
        training_seeds=training,
        calibration_seeds=calibration,
        population_size=_positive_int(
            data["population_size"], "protocol.population_size"
        ),
        generations=_positive_int(data["generations"], "protocol.generations"),
        sigma=_positive_float(data["sigma"], "protocol.sigma"),
        nchl_parameters=nchl_parameters,
        nchl_eta=nchl_eta,
        episodes_per_candidate=episode_budget,
        heldout_episodes_per_run=_positive_int(
            data["heldout_episodes_per_run"], "protocol.heldout_episodes_per_run"
        ),
        claims=tuple(claims),
    )


def scientific_identity() -> dict[str, str]:
    """Return stable identities embedded in every persisted result family."""

    protocol = load_scientific_protocol()
    return {
        "scientific_protocol_id": protocol.protocol_id,
        "scientific_protocol_status": protocol.status,
        "scientific_protocol_sha256": file_sha256(DEFAULT_PROTOCOL_PATH),
        "conda_environment_sha256": file_sha256(ENVIRONMENT_PATH),
    }


def claim_for(
    study: str, outcome: str, correction_family: str
) -> ScientificClaim:
    """Resolve one implemented hypothesis to exactly one registered claim."""

    matches = [
        claim
        for claim in load_scientific_protocol().claims
        if claim.study == study
        and claim.outcome == outcome
        and claim.correction_family == correction_family
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one claim for {study}/{outcome}/{correction_family}, got {len(matches)}"
        )
    return matches[0]
