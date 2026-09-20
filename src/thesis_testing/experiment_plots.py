"""PNG/PDF figures for training, held-out fitness, weight dynamics and freezing.

Plotting reads analysis data; it never trains or runs a simulator.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thesis_testing.config.loader import load_experiment_config
from thesis_testing.nominal_analysis import read_metrics
from thesis_testing.freeze_analysis import SNAPSHOT_FRACTIONS


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=220)
    plt.close(fig)


def plot_training_curves(run_dirs: tuple[Path, ...], output: Path) -> None:
    # Read each config/metrics file once rather than once per panel/controller.
    run_data = [
        (load_experiment_config(path / "config.yaml"), read_metrics(path / "metrics.csv"))
        for path in run_dirs
    ]
    conditions = sorted({(config.task, config.morphology) for config, _ in run_data})
    if not conditions:
        return
    panel_rows = math.ceil(len(conditions) / 2)
    fig, axes = plt.subplots(panel_rows, 2, figsize=(11, 4 * panel_rows), squeeze=False)
    rng = np.random.default_rng(2024)
    for axis, (task, morphology) in zip(axes.flat, conditions):
        for controller, color in (("static", "#2864dc"), ("nchl", "#d95732")):
            curves = []
            medians = []
            for config, metrics in run_data:
                if (
                    config.task == task
                    and config.morphology == morphology
                    and config.controller.kind == controller
                ):
                    curves.append([row["best_so_far_fitness"] for row in metrics])
                    medians.append([row["median_fitness"] for row in metrics])
            if not curves:
                continue
            length = min(map(len, curves))
            values = np.asarray([curve[:length] for curve in curves])
            mean = values.mean(axis=0)
            draws = np.asarray(
                [values[rng.integers(0, len(values), len(values))].mean(axis=0) for _ in range(1000)]
            )
            low, high = np.quantile(draws, [0.025, 0.975], axis=0)
            x = np.arange(length)
            axis.plot(x, mean, color=color, label=f"{controller} best-so-far")
            axis.fill_between(x, low, high, color=color, alpha=0.2)
            axis.plot(x, np.mean(np.asarray(medians)[:, :length], axis=0), color=color, alpha=0.45, linestyle="--")
        axis.set(
            title=f"{task} / {morphology}",
            xlabel="Generation",
            ylabel="Official fitness",
        )
        axis.legend(fontsize=8)
    for axis in axes.flat[len(conditions):]:
        axis.set_visible(False)
    save_figure(fig, output, "evolutionary_trends")


def plot_heldout_fitness(
    scores: dict[tuple[str, str, str, int], float], output: Path,
) -> None:
    conditions = sorted({key[:2] for key in scores})
    fig, axes = plt.subplots(
        1, max(1, len(conditions)),
        figsize=(3.2 * max(1, len(conditions)), 4), squeeze=False,
    )
    for axis, (task, morphology) in zip(axes.flat, conditions):
        static = [
            value for key, value in scores.items()
            if key[:3] == (task, morphology, "static")
        ]
        nchl = [
            value for key, value in scores.items()
            if key[:3] == (task, morphology, "nchl")
        ]
        axis.boxplot([static, nchl], tick_labels=["Static", "NcHL"], showmeans=True)
        axis.set(title=f"{task}\n{morphology}", ylabel="Mean held-out fitness")
    save_figure(fig, output / "figures", "heldout_nominal_fitness")


def plot_weight_dynamics(
    trace_records: list[tuple[dict[str, object], Path]], output: Path,
) -> None:
    if not trace_records:
        return
    conditions = sorted(
        {
            (str(metadata["task"]), str(metadata["morphology"]))
            for metadata, _ in trace_records
        }
    )
    rows = math.ceil(len(conditions) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(11, 4 * rows), squeeze=False)
    for axis, (task, morphology) in zip(axes.flat, conditions):
        for metadata, path in trace_records:
            if metadata["task"] != task or metadata["morphology"] != morphology:
                continue
            with np.load(path, allow_pickle=False) as archive:
                data = {name: archive[name] for name in ("snapshots", "weight_summary", "update_rms")}
            controller = str(metadata["controller"])
            indices = [-1] if controller == "static" else list(range(data["snapshots"].shape[0]))
            for index in indices:
                values = data["snapshots"][index]
                label = "Static" if controller == "static" else ("W0", "W25", "W50", "W75", "Wfinal")[index]
                axis.hist(values, bins=40, density=True, histtype="step", alpha=0.35, label=label)
        axis.set(title=f"{task} / {morphology}", xlabel="Weight", ylabel="Density")
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), fontsize=7)
    for axis in axes.flat[len(conditions):]:
        axis.set_visible(False)
    save_figure(fig, output / "figures", "weight_distributions")

    for field, name, ylabel in (("weight_summary", "weight_rms", "RMS(W)"), ("update_rms", "update_rms", "RMS(ΔW)")):
        fig, axes = plt.subplots(rows, 2, figsize=(11, 4 * rows), squeeze=False)
        for axis, (task, morphology) in zip(axes.flat, conditions):
            groups: dict[str, list[np.ndarray]] = {"static": [], "nchl": []}
            for metadata, path in trace_records:
                if metadata["task"] != task or metadata["morphology"] != morphology:
                    continue
                with np.load(path, allow_pickle=False) as archive:
                    data = {name: archive[name] for name in ("weight_summary", "update_rms")}
                values = data[field][:, 0, 3] if field == "weight_summary" else data[field][:, 0]
                x_old = np.linspace(0.0, 1.0, len(values))
                groups[str(metadata["controller"])].append(np.interp(np.linspace(0.0, 1.0, 101), x_old, values))
            for controller, color in (("static", "#2864dc"), ("nchl", "#d95732")):
                if groups[controller]:
                    axis.plot(np.linspace(0.0, 1.0, 101), np.mean(groups[controller], axis=0), color=color, label=controller)
            axis.set(
                title=f"{task} / {morphology}",
                xlabel="Normalized episode time",
                ylabel=ylabel,
            )
            axis.legend()
        for axis in axes.flat[len(conditions):]:
            axis.set_visible(False)
        save_figure(fig, output / "figures", name)



def plot_freeze_trajectories(run_scores: list[dict[str, object]], output: Path) -> None:
    """Show one trajectory per evolutionary run, after averaging its source traces."""

    if not run_scores:
        return
    conditions = sorted({(str(row["task"]), str(row["morphology"])) for row in run_scores})
    for show_delta, filename, ylabel in (
        (False, "freeze_fitness_over_lifetime", "Mean frozen-controller fitness"),
        (True, "freeze_delta_from_initial", "Frozen fitness change from W0"),
    ):
        panel_rows = math.ceil(len(conditions) / 2)
        figure, axes = plt.subplots(panel_rows, 2, figsize=(11, 4 * panel_rows), squeeze=False)
        for axis, (task, morphology) in zip(axes.flat, conditions):
            runs: dict[str, list[dict[str, object]]] = {}
            for row in run_scores:
                if (row["task"], row["morphology"]) == (task, morphology):
                    runs.setdefault(str(row["run_id"]), []).append(row)
            trajectories = []
            for run_id, rows in sorted(runs.items()):
                rows = sorted(rows, key=lambda row: float(row["snapshot_fraction"]))
                fractions = [float(row["snapshot_fraction"]) for row in rows]
                if tuple(fractions) != SNAPSHOT_FRACTIONS:
                    raise ValueError(f"incomplete freeze trajectory: {run_id}")
                values = np.asarray([float(row["mean_frozen_fitness"]) for row in rows])
                if show_delta:
                    values = values - values[0]
                trajectories.append(values)
                axis.plot(fractions, values, linewidth=1, marker="o", markersize=3, alpha=0.6,
                          label=f"Run seed {rows[0]['training_seed']}")
            axis.plot(SNAPSHOT_FRACTIONS, np.mean(trajectories, axis=0), color="black",
                      linewidth=2.5, label="Mean across runs")
            if show_delta:
                axis.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            axis.set(title=f"{task} / {morphology}", xlabel="Source lifetime stage", ylabel=ylabel)
            axis.set_xticks(SNAPSHOT_FRACTIONS, ["0%", "25%", "50%", "75%", "100%"])
            axis.legend(fontsize=7)
        for axis in axes.flat[len(conditions):]:
            axis.set_visible(False)
        save_figure(figure, output, filename)
