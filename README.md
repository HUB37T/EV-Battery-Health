# EV Battery Health

This Jaringan Saraf Tiruan project estimates battery state of health (SOH) at observation time with multilayer perceptrons trained by backpropagation. The synthetic Kaggle dataset contains 50,000 observations from 2,600 batteries. A fixed split by `battery_id` keeps batteries disjoint across train, validation and test. See the [dataset card](docs/DATASET_CARD.md) and [project guide](CONTEXT.md).

## Two result sets

| Result set | Model | Test MAE (SOH points), mean ± sample SD |
|---|---|---:|
| Original PyTorch evaluation, 6 November 2026 | Baseline / candidate | 1.02684 ± 0.00311 / 0.95005 ± 0.00619 |
| NumPy follow-up snapshot, PR #1 | Baseline / candidate | 0.96018 ± 0.00641 / 0.97268 ± 0.00325 |

The [original PyTorch report and lock](results/final_evaluation/FINAL_REPORT.md) are the project's initial final evaluation. The [NumPy archive](results/numpy/README.md) documents the manual implementation and its follow-up on the **same 7,572 observations from 390 test batteries**. The NumPy scores are descriptive on a previously evaluated split. Both result sets report seeds 42, 123 and 2026. The original notebooks, reading HTMLs and experiment histories are retained at their published paths.

The NumPy work adds course-style manual gradients, min-max scaling, optimizer and learning-rate comparisons, SOH-band weighting, `max_error`, and a capacity demonstration. Selection and checkpointing use validation. The NumPy baseline has 2,209 parameters and the candidate has 553. See the [NumPy methodology](docs/METHODOLOGY.md) and [experiment report](results/numpy/experiments/EXPERIMENT_REPORT.md).

## Run the NumPy pipeline from a new clone

Use Python 3.12 or 3.13. From the repository root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

On macOS/Linux use `.venv/bin/python` in the commands below. Extract the three Kaggle files listed in [data/README.md](data/README.md) into `data/raw/`, then run:

```powershell
.\.venv\Scripts\python.exe -m ev_battery all
.\.venv\Scripts\python.exe -m pytest -q
```

`all` runs audit, EDA, preprocessing, training, feature-engineering ablation and the NumPy follow-up. The training stage rebuilds the two recorded model configurations in [the follow-up plan](configs/numpy/followup.json) for all three seeds. The evaluator reads their best validation checkpoints. Each local evaluation gets a directory under `data/processed/numpy_workspace/results/numpy_followup/<fingerprint>/`. The CLI prints the report path. The workspace also contains the fitted preprocessing, weights and run histories. Git ignores the workspace. The fingerprint captures local artifact identities, so duration or binary differences on another machine produce a separate evaluation directory.

With the virtual environment active, use `python -m ev_battery --list` to see individual stages and `python scripts/run_notebook.py --all` to execute the six [NumPy notebooks](notebooks/numpy/README.md), or `make all` and `make notebooks` with GNU Make. A first run trains all configurations on CPU. Small numerical differences can occur across environments.

For a runnable copy of the PyTorch pipeline, check out [the original release commit](https://github.com/HUB37T/EV-Battery-Health/tree/479ed842d141017591a3fe755647040c237dfd6c) in a separate clone or worktree and follow its README. The archived reports and executed notebooks are readable from the current branch.
