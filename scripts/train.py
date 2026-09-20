#!/usr/bin/env python3
"""Train one resolved experiment into an explicit output directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from thesis_testing.config.loader import load_experiment_config
from thesis_testing.config.resolve import resolve_experiment
from thesis_testing.training import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    resolved = resolve_experiment(load_experiment_config(args.config))
    summary = train(
        resolved,
        args.output,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )
    print(f"completed_generations={summary.completed_generations}")
    print(f"best_fitness={summary.best_fitness:.12g}")
    print(f"best_genome={summary.best_genome_path}")


if __name__ == "__main__":
    main()
