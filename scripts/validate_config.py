#!/usr/bin/env python3
"""Load and resolve one experiment config without executing an episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from thesis_testing.config import (
    ConfigError,
    experiment_to_dict,
    load_experiment_config,
    resolve_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and resolve a thesis experiment configuration."
    )
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    try:
        config = load_experiment_config(args.config)
        resolved = resolve_experiment(config)
    except (ConfigError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print("Resolved configuration:")
    print(json.dumps(experiment_to_dict(config), indent=2, sort_keys=True))
    print()
    print(f"Task: {resolved.task.env_id} ({resolved.task.difficulty})")
    print(f"Morphology: {resolved.morphology.name}")
    print(
        "Environment: "
        f"O={resolved.env_contract.observation_dim}, "
        f"A={resolved.env_contract.action_dim}, "
        f"horizon={resolved.env_contract.max_episode_steps}"
    )
    print(f"Layer sizes: {resolved.network.layer_sizes}")
    print(f"Weight shapes (out, in): {resolved.network.weight_shapes}")
    print(
        f"Architecture counts: N={resolved.network.num_neurons}, "
        f"W={resolved.network.num_weights}"
    )
    print(
        "Seed plan: "
        f"run={resolved.seeds.run_seed}, "
        f"optimizer={resolved.seeds.optimizer_seed}, "
        f"initial_weight={resolved.seeds.initial_weight_seed}, "
        f"evaluation={resolved.seeds.evaluation_seed}"
    )
    print(f"Config digest: {resolved.config_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

