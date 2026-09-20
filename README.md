# NcHL Thesis Experiments

This repository compares a static neural-network controller with an NcHL
(neuron-centric Hebbian learning) controller on fixed EvoGym tasks.
It contains the experiment configurations, training code, result analysis, and
figure generation used for the thesis experiments.

## Local setup

Create the Conda environment, activate it, and expose the source directory to
Python:

```bash
conda env create --file environment.yml
conda activate thesis-testing
export PYTHONPATH="$PWD/src"
```

You can confirm the setup without running training:

```bash
python scripts/validate_config.py configs/nominal/walker_static.yaml
python scripts/run_smoke_rollout.py --manifest /tmp/nchl-smoke.json
```

## Run experiments locally

Train one configured experiment into a new directory:

```bash
python scripts/train.py configs/nominal/walker_static.yaml \
  --output results/local/walker_static
```

A completed run contains `config.yaml`, `metrics.csv`, `best_genome.npy`, and
`summary.json`. To run the matched local pilot across all configured conditions
with one seed and one worker:

```bash
python scripts/run_nominal_suite.py --pilot --seeds 0 --workers 1 \
  --output results/local/pilot
```

Use `--resume` with the same command if an incomplete run needs to continue.
The full experiment (`--final`) is substantially more expensive.

## Analyse and visualise results

`scripts/analyze_nominal.py` is the analysis and visualisation entry point. It
reads completed training directories, runs held-out and weight analyses, writes
CSV summaries, and creates PNG/PDF figures.

```bash
python scripts/analyze_nominal.py results/local/pilot \
  --output results/local/analysis
```

Figures are written to `results/local/analysis/figures/`, including training
curves, held-out fitness comparisons, weight dynamics, and freeze-analysis
plots. Use `--skip-plots` when you only want the CSV analysis data.
