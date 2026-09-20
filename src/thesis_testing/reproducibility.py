"""CPU reproducibility and runtime provenance helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata
import os
from pathlib import Path
import platform
import random
import subprocess
from typing import Any

import numpy as np
import scipy
import torch

from thesis_testing.protocol import scientific_identity


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch CPU from one non-negative integer."""

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    # NumPy's legacy global RandomState accepts only uint32 seeds, while the
    # stable namespace derivation deliberately produces non-negative 63-bit seeds.
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _git_output(*args: str) -> str | None:
    root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(f"required distribution {name!r} is not installed") from error


def collect_runtime_manifest(seed: int | None = None) -> dict[str, Any]:
    """Collect versions and immutable source identifiers for one run."""

    dirty_output = _git_output("status", "--porcelain")
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "os": platform.system(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "gymnasium_version": _distribution_version("gymnasium"),
        "evogym_version": _distribution_version("evogym"),
        "project_version": _distribution_version("thesis-testing"),
        "project_git_sha": _git_output("rev-parse", "HEAD"),
        "project_git_dirty": None if dirty_output is None else bool(dirty_output),
        "seed": seed,
        **scientific_identity(),
    }
