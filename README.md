# EV Battery Health: SOH Estimation with an MLP

Estimating the **state of health (SOH)** of electric-vehicle batteries that were
never seen during training, using a **multilayer perceptron trained with
backpropagation, implemented from scratch in NumPy** in the style of the course
modules (`bp_fit` and `bp_predict`). Final project for the Artificial Neural Networks course
(Jaringan Saraf Tiruan).

- **Data:** [EV Battery Health & Degradation Dataset](https://www.kaggle.com/datasets/vinay24baghira/ev-battery-health-and-degradation-dataset), 50,000 observations of 2,600 batteries, synthetic, CC0.
- **Task:** regression of `state_of_health_pct` at observation time.
- **Evaluation:** batteries are split 70/15/15 by `battery_id`, so no battery appears in more than one split.

## Results

Test split: 7,572 observations from 390 held-out batteries. Mean ± sample SD over
three training seeds (42, 123, 2026), errors in SOH percentage points. Both models
are trained with plain gradient descent (`learn_rate` 0.1, MSE loss) using the
from-scratch NumPy implementation.

| Model | Architecture | Parameters | Test MAE | Test RMSE | Test R² |
|---|---|---:|---:|---:|---:|
| Baseline | 67 → [32] sigmoid → 1 linear | 2,209 | **0.96018 ± 0.00641** | **1.40619 ± 0.00799** | **0.95785 ± 0.00048** |
| Candidate | 67 → [8] sigmoid → 1 linear | 553 | 0.97268 ± 0.00325 | 1.43615 ± 0.00465 | 0.95603 ± 0.00028 |
| Constant train median | none | 0 | 4.93 | 7.22 | -0.11 |

The candidate was selected on validation and locked before the test split was
opened. The three-seed validation check already showed both networks within seed
noise (0.99997 vs 0.99988), and on test the candidate is 0.0125 points behind the
baseline with a quarter of the parameters. Both models are weakest below 70% SOH
(88 observations from 23 batteries), where MAE is about three times the average.
See the [final report](results/final_evaluation/FINAL_REPORT.md).

## Project scope

| Stage | What was done | Report |
|---|---|---|
| EDA | Target and feature distributions, correlation and redundancy, missingness, outliers, chemistry and age patterns, temporal quality (train only) | [EDA](results/eda/EDA_REPORT.md) |
| Preprocessing | Group split, median imputation with missing indicators, one-hot encoding with an explicit missing category, standardisation vs min-max normalisation, outlier capping and log transforms, leakage-based and redundancy-based feature selection | [Preprocessing](results/preprocessing/PREPROCESSING_REPORT.md) |
| Experiments | Input neurons (67 vs 63), hidden layers ([2] to [128, 64, 32]), activation (sigmoid, tanh, ReLU), loss (MSE, MAE, Huber), learning rate, gradient descent vs Adam, SOH-band balancing, `max_error`, epochs (50, 100, 200), underfitting vs overfitting, three seeds | [Experiments](results/experiments/EXPERIMENT_REPORT.md) |
| Feature engineering | Four domain feature blocks tested as an ablation | [Feature engineering](results/feature_engineering/FEATURE_ENGINEERING_REPORT.md) |
| Evaluation | Six checkpoints hashed and locked before testing, errors per SOH band and per battery | [Final evaluation](results/final_evaluation/FINAL_REPORT.md) |

Inputs are tabular features of a single observation. Timesteps are not used
because `observation_date` is missing for 2,216 rows and cumulative counters are
not always monotonic. The reasoning behind every decision is in
[docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Repository structure

```text
EV-Battery-Health/
├── configs/              # Feature contract, preprocessing recipes, study and evaluation plans
├── data/
│   ├── raw/              # Kaggle archive (not committed)
│   ├── processed/        # Preprocessed arrays (not committed)
│   └── splits/           # Battery-level split manifest (committed)
├── docs/                 # Methodology and dataset card
├── models/               # Fitted pipelines and checkpoints (not committed)
├── notebooks/            # 00 audit to 05 final evaluation, narrative in Indonesian
├── results/              # Generated reports, figures and tables
├── scripts/              # Notebook execution
├── src/ev_battery/
│   ├── audit.py          # Schema, quality, leakage checks, split
│   ├── eda.py            # Exploratory analysis
│   ├── preprocessing.py  # Train-fitted transformer and array loader
│   ├── features.py       # Feature-engineering ablation
│   ├── model.py          # From-scratch NumPy MLP: bp_fit, bp_predict, backprop
│   ├── training.py       # Training runs, checkpoints and staged study
│   ├── evaluation.py     # Locked test evaluation
│   ├── reporting/        # Figures and Markdown reports per stage
│   └── __main__.py       # Pipeline CLI
├── tests/
└── pyproject.toml
```

## Getting started

Requires Python 3.12 or 3.13. The network itself only needs NumPy. Dependencies
are pinned exactly in `pyproject.toml` because run IDs and the evaluation lock
depend on bit-identical training.

```bash
python -m venv .venv
.venv/Scripts/activate          # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"         # or: pip install -r requirements.txt
```

Download the Kaggle archive into `data/raw/` (see [data/README.md](data/README.md)), then:

```bash
python -m ev_battery all              # audit → eda → preprocess → train → features → evaluate
python -m ev_battery --list           # list stages, each can also run on its own
python scripts/run_notebook.py --all  # re-execute notebooks 00 to 05
pytest                                # test suite
ruff check . && ruff format --check .
```

The full pipeline runs on CPU in about ten minutes. Runs whose code, data and
configuration fingerprints match are reused, so repeating `all` is fast.

## Methodology at a glance

- Split by battery, not by row, so repeated observations of a battery never leak across splits.
- Every preprocessing statistic is learned from train only.
- Validation selects configurations, the test split is opened once after models and seeds are locked.
- Every run is recorded, including the ones that lost, and every final seed is reported.
- `capacity_fade_pct` equals `100 - SOH` and is excluded from the inputs.

Team conventions are in [CONTRIBUTING.md](CONTRIBUTING.md), and dataset facts are
in the [dataset card](docs/DATASET_CARD.md).

## License

Code: [MIT](LICENSE). Dataset: CC0, according to the source documentation.
