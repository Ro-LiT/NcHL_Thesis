#!/usr/bin/env python3
"""Benchmark the real population path and estimate the frozen experiment workload."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np

from thesis_testing.config.loader import load_experiment_config
from thesis_testing.config.resolve import resolve_experiment
from thesis_testing.evaluation import evaluate_genome, evaluation_episode_seeds, make_controller
from thesis_testing.evogym_contract import make_env
from thesis_testing.parallel import PopulationEvaluator
from thesis_testing.protocol import load_scientific_protocol
from thesis_testing.reproducibility import collect_runtime_manifest
from thesis_testing.runtime_estimation import (
    TimingObservation,
    WorkloadStratum,
    estimate_workload,
    speed_rows,
)


REPOSITORY = Path(__file__).resolve().parents[1]
PROFILE_FILES = {
    ("Walker-v0", 40): "walker",
    ("Carrier-v0", 44): "carrier",
    ("Jumper-v0", 45): "jumper",
    ("Thrower-v0", 44): "thrower_biped",
    ("Thrower-v0", 92): "thrower_arm",
}


def _memory_bytes() -> int | None:
    try:
        if platform.system() == "Darwin":
            direct = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], check=True,
                capture_output=True, text=True,
            )
            return int(direct.stdout.strip())
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * pages)
    except (OSError, ValueError, subprocess.CalledProcessError):
        if platform.system() == "Darwin":
            try:
                output = subprocess.run(
                    ["hostinfo"], check=True, capture_output=True, text=True,
                ).stdout
                line = next(
                    line for line in output.splitlines()
                    if "Primary memory available" in line
                )
                gib = float(line.split(":", 1)[1].strip().split()[0])
                return int(gib * 1024**3)
            except (OSError, ValueError, StopIteration, subprocess.CalledProcessError):
                pass
        return None


def _profile(task: str, controller: str, width: int, workers: int):
    stem = PROFILE_FILES[(task, width)]
    path = REPOSITORY / "configs" / "nominal" / f"{stem}_{controller}.yaml"
    base = load_experiment_config(path)
    return resolve_experiment(
        replace(
            base,
            name=f"runtime_{stem}_{controller}_w{width}",
            network=replace(base.network, hidden_sizes=(width,)),
            runtime=replace(base.runtime, workers=workers, torch_threads_per_worker=1),
        )
    )


def _profiles(full: bool) -> list[tuple[str, str, int]]:
    profiles = [
        (task, controller, width)
        for task, width in PROFILE_FILES
        for controller in ("static", "nchl")
    ]
    if full:
        return profiles
    return [item for item in profiles if item[0] == "Walker-v0" and item[2] == 40]


def _full_workload() -> list[WorkloadStratum]:
    protocol = load_scientific_protocol()
    runs = len(protocol.training_seeds)
    widths = {
        ("Walker-v0", "biped"): 40,
        ("Carrier-v0", "biped"): 44,
        ("Jumper-v0", "biped"): 45,
        ("Thrower-v0", "biped"): 44,
        ("Thrower-v0", "thrower_arm"): 92,
    }
    work: list[WorkloadStratum] = []
    for condition in protocol.conditions:
        for controller in protocol.controllers:
            width = widths[(condition.task, condition.morphology)]
            work.append(WorkloadStratum(
                "nominal_training", condition.task, controller, width,
                runs, protocol.generations, protocol.population_size,
                protocol.episodes_per_candidate.for_controller(controller),
            ))
            work.append(WorkloadStratum(
                "nominal_heldout_and_weight_traces", condition.task, controller, width,
                runs, 1, protocol.heldout_episodes_per_run + 3, 1,
            ))
    return work


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    estimate = report.get("estimate")
    proxy = report.get("preliminary_proxy_estimate")
    lines = [
        "# Runtime calibration estimate", "",
        f"Generated: `{report['created_at_utc']}`", "",
        f"Status: **{report['status']}**", "",
        f"Workers requested: `{report['target_workers']}` spawned processes with one PyTorch thread each.", "",
        f"Logical CPUs: `{report['machine']['logical_cpu_count']}`; oversubscription: `{report['machine']['oversubscribed']}`.", "",
    ]
    if isinstance(estimate, dict):
        lines.extend([
            "## Full-workload estimate", "",
            f"Candidate episodes: `{estimate['total_candidate_episodes']:,}`", "",
            f"Central wall time: **{estimate['central_wall_hours']:.2f} h ({estimate['central_wall_days']:.2f} days)**", "",
            f"Conservative wall time: **{estimate['conservative_wall_hours']:.2f} h ({estimate['conservative_wall_days']:.2f} days)**", "",
            f"Central / conservative CPU-hours: `{estimate['central_cpu_hours']:.1f}` / `{estimate['conservative_cpu_hours']:.1f}`", "",
            "The conservative estimate uses the slowest measured repetition and then adds the declared contingency. Training runs are sequential; only candidates within a generation are parallel.", "",
        ])
    else:
        lines.extend([
            "This bounded smoke calibration does not cover every task/width stratum, so it intentionally does not claim a validated full-experiment estimate. Run with `--full-calibration`.", "",
        ])
        if isinstance(proxy, dict):
            lines.extend([
                "## Preliminary Walker-proxy range", "",
                f"Central proxy: **{proxy['central_wall_hours']:.2f} h ({proxy['central_wall_days']:.2f} days)**", "",
                f"Conservative proxy: **{proxy['conservative_wall_hours']:.2f} h ({proxy['conservative_wall_days']:.2f} days)**", "",
                "This range converts the frozen workload to 500-step-equivalent episodes, assumes unmeasured tasks run at Walker throughput, and assumes full horizons. It is for initial planning only, not the Gate-6 scheduling certificate.", "",
            ])
    lines.extend([
        "## Scope and assumptions", "",
        "- The workload is the official nominal Static-versus-NcHL matrix only.",
        "- Thrower/biped and Thrower/thrower_arm are measured as separate width strata.",
        "- Plotting and verification time is reported separately when measured and is not hidden inside simulator throughput.",
        "- The estimate is machine-specific and must be rerun after the protocol is frozen.", "",
    ])
    path.write_text("\n".join(lines))


def _preliminary_proxy(
    observations: list[TimingObservation], *, target_workers: int, contingency: float,
) -> dict[str, object] | None:
    samples = [
        item for item in observations
        if item.workers == target_workers and item.task == "Walker-v0"
        and item.network_width == 40
    ]
    if not {item.controller for item in samples} >= {"static", "nchl"}:
        return None
    horizons = {
        "Walker-v0": 500, "Carrier-v0": 500,
        "Jumper-v0": 500, "Thrower-v0": 300,
    }
    workload = _full_workload()
    equivalents = float(sum(
        item.candidate_episodes * horizons[item.task] / 500.0 for item in workload
    ))
    rates = np.asarray([item.episodes_per_second for item in samples], dtype=np.float64)
    starts = np.asarray([item.startup_seconds for item in samples], dtype=np.float64)
    training_runs = sum(
        item.new_runs for item in workload if item.stage.endswith("training")
    )
    central_seconds = (
        equivalents / float(np.median(rates))
        + training_runs * float(np.median(starts))
    )
    conservative_seconds = (
        equivalents / float(np.min(rates)) + training_runs * float(np.max(starts))
    ) * (1.0 + contingency)
    return {
        "status": "preliminary_unvalidated_walker_proxy",
        "full_horizon_500_step_equivalent_episodes": equivalents,
        "new_training_runs": training_runs,
        "measured_target_worker_samples": len(samples),
        "central_wall_hours": central_seconds / 3600.0,
        "conservative_wall_hours": conservative_seconds / 3600.0,
        "central_wall_days": central_seconds / 86400.0,
        "conservative_wall_days": conservative_seconds / 86400.0,
        "assumptions": [
            "unmeasured task/controller/width throughput equals measured Walker width-40 throughput",
            "Walker, Carrier, and Jumper use their full 500-step horizon",
            "Thrower uses its full 300-step horizon",
            "analysis/plotting/verifier overhead excluded",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/runtime_calibration"))
    parser.add_argument(
        "--workers", type=int, nargs="+", default=[1, 4, 8, 25],
        help="measured process counts; include 1 and --target-workers for a full calibration",
    )
    parser.add_argument(
        "--target-workers", type=int, default=25,
        help="measured process count used to project the complete workload",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--worker-waves", type=int, default=2)
    parser.add_argument("--minimum-candidates", type=int, default=8)
    parser.add_argument("--contingency", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--full-calibration", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 1 or args.worker_waves < 2 or args.minimum_candidates < 1:
        parser.error("repetitions/candidates must be positive and worker-waves must be at least two")
    if any(workers < 1 for workers in args.workers):
        parser.error("--workers values must be positive")
    if len(set(args.workers)) != len(args.workers):
        parser.error("--workers values must be unique")
    if args.target_workers not in args.workers:
        parser.error("--target-workers must be present in --workers")
    if args.full_calibration and (
        1 not in args.workers
        or args.repetitions < 3
    ):
        parser.error(
            "full calibration requires worker 1, the target worker count, "
            "and at least three repetitions"
        )

    cases = [
        (repetition, profile, workers)
        for repetition in range(args.repetitions)
        for profile in _profiles(args.full_calibration)
        for workers in args.workers
    ]
    random.Random(args.seed).shuffle(cases)
    observations: list[TimingObservation] = []
    for case_index, (repetition, (task, controller, width), workers) in enumerate(cases, start=1):
        resolved = _profile(task, controller, width, workers)
        rng = np.random.default_rng(args.seed + case_index)
        candidate_count = max(args.minimum_candidates, workers * args.worker_waves)
        population = rng.normal(0.0, 0.05, (candidate_count, make_controller(resolved).genome_size)).astype(np.float64)
        warmup = population[:max(1, workers)]
        started = time.perf_counter()
        evaluator = PopulationEvaluator(resolved)
        evaluator.evaluate(warmup)
        startup_seconds = time.perf_counter() - started
        measured = time.perf_counter()
        evaluator.evaluate(population)
        steady_seconds = time.perf_counter() - measured
        evaluator.close()

        env = make_env(resolved.task.env_id, resolved.morphology)
        diagnostic_controller = make_controller(resolved)
        try:
            _, diagnostic = evaluate_genome(
                env, diagnostic_controller, population[0],
                evaluation_episode_seeds(resolved), record_rewards=False,
            )
        finally:
            env.close()
        lengths = np.asarray([item.episode_length for item in diagnostic], dtype=np.float64)
        early = sum(item.terminated and not item.truncated for item in diagnostic)
        observation = TimingObservation(
            task, controller, width, workers, repetition,
            candidate_count * len(diagnostic), candidate_count / workers,
            startup_seconds, steady_seconds, float(np.mean(lengths)), early,
        )
        observations.append(observation)
        print(json.dumps({"case": case_index, "of": len(cases), **asdict(observation),
                          "episodes_per_second": observation.episodes_per_second}), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    raw_rows = [{**asdict(item), "episodes_per_second": item.episodes_per_second} for item in observations]
    _write_csv(args.output / "timing_observations.csv", raw_rows)
    speeds = speed_rows(observations)
    _write_csv(args.output / "throughput_summary.csv", speeds)
    estimate = None
    proxy = None
    status = "bounded_smoke_only"
    if args.full_calibration:
        estimate = estimate_workload(
            observations, _full_workload(), target_workers=args.target_workers,
            contingency=args.contingency,
        )
        status = "full_calibration_complete"
    else:
        proxy = _preliminary_proxy(
            observations, target_workers=args.target_workers,
            contingency=args.contingency,
        )
    manifest = collect_runtime_manifest(args.seed)
    machine = {
        "logical_cpu_count": os.cpu_count(), "physical_memory_bytes": _memory_bytes(),
        "target_worker_processes": args.target_workers,
        "torch_threads_per_worker": 1,
        "oversubscribed": args.target_workers > (os.cpu_count() or 0),
        "platform": platform.platform(),
    }
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status, "target_workers": args.target_workers,
        "calibration": {
            "repetitions": args.repetitions, "worker_waves": args.worker_waves,
            "workers": args.workers, "randomized_case_order_seed": args.seed,
            "full_task_and_width_coverage": args.full_calibration,
        },
        "machine": machine, "runtime_manifest": manifest,
        "observations": raw_rows, "throughput_summary": speeds,
        "estimate": estimate, "preliminary_proxy_estimate": proxy,
        "unmeasured_overhead": "analysis/plotting/verifier wall time; benchmark separately after pilot outputs exist",
    }
    (args.output / "runtime_estimate.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(args.output / "runtime_estimate.md", report)
    print(json.dumps({"status": status, "output": str(args.output),
                      "estimate": estimate}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
