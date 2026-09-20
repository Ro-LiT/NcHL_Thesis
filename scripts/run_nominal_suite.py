#!/usr/bin/env python3
"""Launch, resume, or enumerate the frozen matched nominal suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from thesis_testing.nominal_suite import (
    completed_run_summary,
    enumerate_nominal_runs,
    run_index_row,
    write_run_index,
    write_suite_manifest,
)
from thesis_testing.protocol import load_scientific_protocol
from thesis_testing.training import train


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true", help="calibration seeds × 10 generations")
    mode.add_argument("--final", action="store_true", help="5 seeds × 500 generations")
    parser.add_argument("--config-dir", type=Path, default=Path("configs/nominal"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    phase = "pilot" if args.pilot else "final"
    protocol = load_scientific_protocol()
    if args.final and not args.dry_run and protocol.status != "frozen":
        parser.error("final runs are blocked until scientific_protocol.yaml status is frozen")
    seeds = args.seeds if args.seeds is not None else (
        list(protocol.calibration_seeds) if args.pilot else list(protocol.training_seeds)
    )
    specs = enumerate_nominal_runs(
        args.config_dir,
        args.output,
        phase=phase,
        seeds=seeds,
        workers=args.workers,
    )
    rows = []
    for spec in specs:
        complete, generations, best = completed_run_summary(spec)
        rows.append(
            run_index_row(
                spec,
                status="complete" if complete else "planned",
                completed_generations=generations,
                best_fitness=best,
            )
        )
        print(json.dumps(rows[-1], sort_keys=True))
    write_suite_manifest(
        args.output / "suite_manifest.json",
        specs,
        command=" ".join(sys.argv),
    )
    write_run_index(args.output / "run_index.csv", rows)
    if args.dry_run:
        return

    for index, spec in enumerate(specs):
        complete, generations, best = completed_run_summary(spec)
        if complete:
            print(f"skip complete {spec.output_dir}")
            continue
        use_resume = args.resume and (spec.output_dir / "checkpoint.pkl").is_file()
        rows[index] = run_index_row(
            spec,
            status="running",
            completed_generations=generations,
            best_fitness=best,
            started_from_checkpoint=use_resume,
        )
        write_run_index(args.output / "run_index.csv", rows)
        try:
            summary = train(
                spec.resolved,
                spec.output_dir,
                checkpoint_every=args.checkpoint_every,
                resume=use_resume,
            )
        except Exception:
            rows[index] = run_index_row(
                spec,
                status="failed",
                completed_generations=generations,
                best_fitness=best,
                started_from_checkpoint=use_resume,
            )
            write_run_index(args.output / "run_index.csv", rows)
            raise
        rows[index] = run_index_row(
            spec,
            status="complete",
            completed_generations=summary.completed_generations,
            best_fitness=summary.best_fitness,
            started_from_checkpoint=use_resume,
        )
        write_run_index(args.output / "run_index.csv", rows)


if __name__ == "__main__":
    main()
