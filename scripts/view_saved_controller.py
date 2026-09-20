#!/usr/bin/env python3
"""Render a trained Static or NcHL controller in its configured EvoGym task."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from thesis_testing.config import load_experiment_config, resolve_experiment
from thesis_testing.evaluation import make_controller, rollout
from thesis_testing.evogym_contract import make_env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open an EvoGym window and play a saved best controller.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python scripts/view_saved_controller.py \\
    hpc_rq1/nominal/walker/biped/static/seed_000

  python scripts/view_saved_controller.py \\
    hpc_rq1/nominal/walker/biped/nchl/seed_000 \\
    --environment-seed 12 --controller-seed 34 --episodes 3
""",
    )
    parser.add_argument(
        "run_directory",
        type=Path,
        help="training directory containing config.yaml and best_genome.npy",
    )
    parser.add_argument("--environment-seed", type=int, default=0)
    parser.add_argument(
        "--controller-seed",
        type=int,
        default=0,
        help="NcHL runtime-weight seed; accepted but has no effect for Static",
    )
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    if args.environment_seed < 0 or args.controller_seed < 0:
        parser.error("seeds must be non-negative")
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    config_path = args.run_directory / "config.yaml"
    genome_path = args.run_directory / "best_genome.npy"
    if not config_path.is_file() or not genome_path.is_file():
        parser.error(
            "run directory must contain config.yaml and best_genome.npy"
        )

    experiment = resolve_experiment(load_experiment_config(config_path))
    controller = make_controller(experiment)
    genome = np.load(genome_path, allow_pickle=False)
    if genome.dtype != np.float64:
        genome = genome.astype(np.float64)
    controller.set_genome(genome)

    print(
        f"task={experiment.task.env_id} "
        f"morphology={experiment.morphology.name} "
        f"controller={experiment.config.controller.kind} "
        f"genome_size={genome.size}",
        flush=True,
    )

    environment = make_env(
        experiment.task.env_id,
        experiment.morphology,
        render_mode="screen",
    )
    try:
        for episode_index in range(args.episodes):
            result = rollout(
                environment,
                controller,
                environment_seed=args.environment_seed + episode_index,
                controller_seed=args.controller_seed + episode_index,
            )
            print(
                f"episode={episode_index + 1} "
                f"fitness={result.fitness:.12g} "
                f"steps={result.episode_length} "
                f"terminated={result.terminated} "
                f"truncated={result.truncated}",
                flush=True,
            )
    finally:
        environment.close()


if __name__ == "__main__":
    main()
