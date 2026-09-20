#!/usr/bin/env python3
"""Run post-training experiments: held-out scores, weight dynamics and freezing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from thesis_testing.experiment_plots import (
    plot_training_curves, plot_heldout_fitness, plot_weight_dynamics, plot_freeze_trajectories,
)
from thesis_testing.freeze_analysis import (
    FREEZE_FITNESS_COLUMNS, FREEZE_RUN_COLUMNS, FREEZE_CHANGE_COLUMNS, FREEZE_SUMMARY_COLUMNS,
    evaluate_freeze_records, summarize_freeze_results,
)
from thesis_testing.config.loader import load_experiment_config
from thesis_testing.config.resolve import resolve_experiment
from thesis_testing.nominal_analysis import (
    WEIGHT_STAT_NAMES,
    aggregate_run_scores,
    cliffs_delta,
    completed_training_runs,
    convergence_metrics,
    evaluate_heldout_run,
    genome_sha256,
    heldout_episode_seeds,
    holm_adjust,
    layer_snapshot_rows,
    load_weight_trace,
    plasticity_summary,
    read_metrics,
    save_weight_trace,
    trace_weights,
)


def write_csv_rows(
    path: Path, rows: list[dict[str, object]], columns: tuple[str, ...] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and columns is None:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_heldout_comparison(
    heldout_rows: list[dict[str, object]], output: Path,
) -> dict[tuple[str, str, str, int], float]:
    per_run: dict[tuple[str, str, str, int], list[float]] = {}
    for row in heldout_rows:
        key = (
            str(row["task"]), str(row["morphology"]),
            str(row["controller"]), int(row["training_seed"]),
        )
        per_run.setdefault(key, []).append(float(row["fitness"]))
    scores = {key: float(np.mean(values)) for key, values in per_run.items()}
    summary_rows: list[dict[str, object]] = []
    for task, morphology, controller in sorted(
        {(key[0], key[1], key[2]) for key in scores}
    ):
        values = np.asarray(
            [
                value for key, value in scores.items()
                if key[:3] == (task, morphology, controller)
            ]
        )
        summary_rows.append(
            {
                "task": task,
                "morphology": morphology,
                "controller": controller,
                "independent_runs": values.size,
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "median": float(np.median(values)),
                "q1": float(np.quantile(values, 0.25)),
                "q3": float(np.quantile(values, 0.75)),
            }
        )
    write_csv_rows(output / "fitness_summary.csv", summary_rows)

    tests: list[dict[str, object]] = []
    conditions = sorted({key[:2] for key in scores})
    for task, morphology in conditions:
        static = [
            value for key, value in scores.items()
            if key[:3] == (task, morphology, "static")
        ]
        nchl = [
            value for key, value in scores.items()
            if key[:3] == (task, morphology, "nchl")
        ]
        if not static or not nchl:
            continue
        test = mannwhitneyu(static, nchl, alternative="two-sided")
        tests.append(
            {
                "task": task,
                "morphology": morphology,
                "static_n": len(static),
                "nchl_n": len(nchl),
                "mean_difference_static_minus_nchl": float(np.mean(static) - np.mean(nchl)),
                "median_difference_static_minus_nchl": float(np.median(static) - np.median(nchl)),
                "mann_whitney_u": float(test.statistic),
                "p_raw": float(test.pvalue),
                "p_holm": 0.0,
                "cliffs_delta_static_vs_nchl": cliffs_delta(static, nchl),
            }
        )
    adjusted = holm_adjust([float(row["p_raw"]) for row in tests]) if tests else []
    for row, value in zip(tests, adjusted):
        row["p_holm"] = float(value)
    write_csv_rows(output / "statistical_tests.csv", tests)

    return scores


def prepare_weight_traces(
    run_dirs: tuple[Path, ...],
    output: Path,
    count: int,
    *,
    overwrite: bool,
) -> list[tuple[dict[str, object], Path]]:
    traces_dir = output / "traces"
    trace_records: list[tuple[dict[str, object], Path]] = []
    summary_rows: list[dict[str, object]] = []
    plasticity_rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        resolved = resolve_experiment(load_experiment_config(run_dir / "config.yaml"))
        genome = np.load(run_dir / "best_genome.npy").astype(np.float64, copy=False)
        genome_digest = genome_sha256(run_dir / "best_genome.npy")
        run_id = (
            f"{resolved.task.env_id.split('-')[0].lower()}__{resolved.morphology.name}__"
            f"{resolved.config.controller.kind}__seed_{resolved.config.seed:03d}"
        )
        seeds = heldout_episode_seeds(resolved.config.seed, count, namespace="weight_analysis")
        for analysis_index, (environment_seed, controller_seed) in enumerate(seeds):
            metadata = {
                "run_id": run_id,
                "source_run": str(run_dir),
                "config_digest": resolved.config_digest,
                "genome_sha256": genome_digest,
                "training_seed": resolved.config.seed,
                "analysis_index": analysis_index,
                "environment_seed": environment_seed,
                "controller_seed": controller_seed,
                "task": resolved.task.env_id,
                "morphology": resolved.morphology.name,
                "controller": resolved.config.controller.kind,
                "hidden_sizes": resolved.network.hidden_dims,
            }
            filename = (
                f"{resolved.task.env_id}_{resolved.morphology.name}_"
                f"{resolved.config.controller.kind}_train{resolved.config.seed:03d}_"
                f"analysis{analysis_index:02d}.npz"
            )
            path = traces_dir / filename
            trace = None
            if path.is_file() and not overwrite:
                cached, cached_metadata = load_weight_trace(path)
                cache_keys = (
                    "run_id",
                    "config_digest",
                    "genome_sha256",
                    "environment_seed",
                    "controller_seed",
                )
                if all(cached_metadata.get(key) == metadata[key] for key in cache_keys):
                    trace = cached
            if trace is None:
                trace = trace_weights(
                    resolved,
                    genome,
                    environment_seed=environment_seed,
                    controller_seed=controller_seed,
                )
                metadata["episode_length"] = int(trace.rewards.size)
                save_weight_trace(path, trace, metadata)
            metadata["episode_length"] = int(trace.rewards.size)
            if resolved.config.controller.kind == "static":
                if not np.allclose(trace.update_rms, 0.0, rtol=0.0, atol=0.0):
                    raise RuntimeError(f"Static controller changed weights: {run_dir}")
            trace_records.append((metadata, path))
            index_rows.append(
                {
                    **metadata,
                    "trace_path": str(path),
                    "episode_length": int(trace.rewards.size),
                }
            )
            plasticity_rows.append({**metadata, **plasticity_summary(trace)})
            layer_rows.extend(
                {**metadata, **row} for row in layer_snapshot_rows(trace)
            )
            labels = ("W0", "W25", "W50", "W75", "Wfinal")
            for label, state in zip(labels, trace.snapshot_states):
                stats = trace.weight_summary[int(state), 0]
                summary_rows.append(
                    {
                        **metadata,
                        "snapshot": label,
                        "state": int(state),
                        **dict(zip(WEIGHT_STAT_NAMES, stats.tolist())),
                        "mean_update_rms": float(np.mean(trace.update_rms[:, 0])),
                        "mean_update_mean_abs": float(
                            np.mean(trace.update_mean_abs[:, 0])
                        ),
                    }
                )
    write_csv_rows(output / "trace_index.csv", index_rows)
    write_csv_rows(output / "weight_summary.csv", summary_rows)
    write_csv_rows(output / "plasticity_summary.csv", plasticity_rows)
    write_csv_rows(output / "layer_summary.csv", layer_rows)
    return trace_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/nominal_analysis"))
    parser.add_argument("--heldout-episodes", type=int, default=10)
    parser.add_argument("--weight-seeds", type=int, default=3)
    parser.add_argument("--skip-heldout", action="store_true")
    parser.add_argument("--skip-weight-traces", action="store_true", help="skip weight tracing and freeze analysis")
    parser.add_argument("--skip-freeze", action="store_true", help="keep weight analysis but omit frozen rollouts")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.heldout_episodes < 1 or args.weight_seeds < 1:
        parser.error("episode and weight-seed counts must be positive")
    run_dirs = completed_training_runs(args.results)
    if not run_dirs:
        parser.error("no completed training artifacts found")
    output_path = args.output.resolve()
    for run_dir in run_dirs:
        source_path = run_dir.resolve()
        if output_path == source_path or source_path in output_path.parents or output_path in source_path.parents:
            parser.error("analysis output must be separate from training artifact directories")
    args.output.mkdir(parents=True, exist_ok=True)

    convergence_rows = []
    for run_dir in run_dirs:
        config = load_experiment_config(run_dir / "config.yaml")
        metrics = convergence_metrics(read_metrics(run_dir / "metrics.csv"))
        convergence_rows.append(
            {
                "source_run": str(run_dir),
                "task": config.task,
                "morphology": config.morphology,
                "controller": config.controller.kind,
                "training_seed": config.seed,
                **metrics.__dict__,
            }
        )
    write_csv_rows(args.output / "convergence_summary.csv", convergence_rows)

    heldout_path = args.output / "heldout_fitness.csv"
    if not args.skip_heldout:
        expected_runs = {}
        for run_dir in run_dirs:
            resolved = resolve_experiment(load_experiment_config(run_dir / "config.yaml"))
            run_id = (
                f"{resolved.task.env_id.split('-')[0].lower()}__{resolved.morphology.name}__"
                f"{resolved.config.controller.kind}__seed_{resolved.config.seed:03d}"
            )
            expected_runs[run_id] = resolved.config_digest
        heldout_rows = []
        if heldout_path.is_file() and not args.overwrite:
            with heldout_path.open(newline="") as stream:
                cached_rows = list(csv.DictReader(stream))
            cached_counts: dict[str, int] = {}
            for row in cached_rows:
                cached_counts[str(row.get("run_id", ""))] = cached_counts.get(
                    str(row.get("run_id", "")), 0
                ) + 1
            cached_digests = {
                str(row.get("run_id", "")): str(row.get("config_digest", ""))
                for row in cached_rows
            }
            if set(cached_counts) == set(expected_runs) and cached_digests == expected_runs and all(
                count == args.heldout_episodes for count in cached_counts.values()
            ):
                heldout_rows = cached_rows
        if not heldout_rows:
            heldout_rows = [
                row
                for run_dir in run_dirs
                for row in evaluate_heldout_run(run_dir, args.heldout_episodes)
            ]
        write_csv_rows(heldout_path, heldout_rows)
    else:
        with heldout_path.open(newline="") as stream:
            heldout_rows = list(csv.DictReader(stream))
    write_csv_rows(args.output / "run_scores.csv", aggregate_run_scores(heldout_rows))
    heldout_scores = write_heldout_comparison(heldout_rows, args.output)

    trace_records = []
    freeze_run_scores = []
    if not args.skip_weight_traces:
        trace_records = prepare_weight_traces(
            run_dirs,
            args.output,
            args.weight_seeds,
            overwrite=args.overwrite,
        )
        if not args.skip_freeze:
            freeze_rows = evaluate_freeze_records(trace_records)
            freeze_run_scores, freeze_changes, freeze_summary = summarize_freeze_results(freeze_rows)
            write_csv_rows(args.output / "freeze_fitness.csv", freeze_rows, FREEZE_FITNESS_COLUMNS)
            write_csv_rows(args.output / "freeze_run_scores.csv", freeze_run_scores, FREEZE_RUN_COLUMNS)
            write_csv_rows(args.output / "freeze_change_by_run.csv", freeze_changes, FREEZE_CHANGE_COLUMNS)
            write_csv_rows(args.output / "freeze_summary.csv", freeze_summary, FREEZE_SUMMARY_COLUMNS)
    if not args.skip_plots:
        plot_training_curves(run_dirs, args.output / "figures")
        plot_heldout_fitness(heldout_scores, args.output)
        plot_weight_dynamics(trace_records, args.output)
        plot_freeze_trajectories(freeze_run_scores, args.output / "figures")
    print(json.dumps({"training_runs": len(run_dirs), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
