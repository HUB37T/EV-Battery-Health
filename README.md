# EV Battery Health: SOH Estimation with a From-Scratch MLP

Estimating the **state of health (SOH)** of electric-vehicle batteries that were
never seen during training, with a **multilayer perceptron trained by
backpropagation and written from scratch in NumPy**. The implementation follows
the `bp_fit` and `bp_predict` functions of the course modules. Final project for
the Artificial Neural Networks course (Jaringan Saraf Tiruan).

- **Data:** [EV Battery Health & Degradation Dataset](https://www.kaggle.com/datasets/vinay24baghira/ev-battery-health-and-degradation-dataset) on Kaggle, 50,000 observations of 2,600 batteries, synthetic, CC0.
- **Task:** regression of `state_of_health_pct` at observation time.
- **Evaluation:** batteries are split 70/15/15 by `battery_id`, so no battery appears in more than one split.
- **Model:** NumPy MLP with hand-written forward pass, backpropagation, gradient descent and Adam. No deep-learning framework.

## Results

Test split: 7,572 observations from 390 held-out batteries. Mean ± sample SD over
three training seeds (42, 123, 2026), errors in SOH percentage points.

| Model | Architecture | Parameters | Test MAE | Test RMSE | Test R² |
|---|---|---:|---:|---:|---:|
| Baseline | 67 → [32] sigmoid → 1 linear | 2,209 | **0.96018 ± 0.00641** | **1.40619 ± 0.00799** | **0.95785 ± 0.00048** |
| Candidate | 67 → [8] sigmoid → 1 linear | 553 | 0.97268 ± 0.00325 | 1.43615 ± 0.00465 | 0.95603 ± 0.00028 |
| Constant train median | none | 0 | 4.93 | 7.22 | -0.11 |

Both networks use MSE loss and plain gradient descent (`learn_rate` 0.1). The
candidate was selected on validation and locked before testing. It matches the
baseline within seed noise with a quarter of the parameters, but does not beat it
on test. Both models are weakest below 70% SOH (88 observations from 23
batteries). The [final report](results/final_evaluation/FINAL_REPORT.md) has the
full analysis, and the [history of the test split](#history-of-the-test-split)
explains how these numbers relate to the earlier PyTorch version.

## Getting started

### Quick start

If you already have Python 3.12 or 3.13 and GNU Make:

```bash
git clone https://github.com/HUB37T/EV-Battery-Health.git
cd EV-Battery-Health
make venv && make install     # create .venv and install the package
make data                     # download the Kaggle files into data/raw/
make all                      # run the whole pipeline, about 10 minutes on CPU
make notebooks                # execute notebooks 00 to 05
```

### Step by step after cloning

**1. Check the prerequisites**

| Tool | Version | Needed for |
|---|---|---|
| Python | 3.12 or 3.13 | Everything |
| Git | any recent version | Cloning |
| GNU Make | optional | `make` shortcuts. Every command also has a plain `python` form |

On Windows, GNU Make can be installed with `scoop install make` or `choco install make`.

**2. Clone the repository**

```bash
git clone https://github.com/HUB37T/EV-Battery-Health.git
cd EV-Battery-Health
```

Run every command below from this repository root.

**3. Create a virtual environment and install the project**

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

With GNU Make the same is `make venv` followed by `make install`. The Makefile uses
`.venv` automatically, so it works without activating the environment. If you
created the environment with uv, install with `make install PIP="uv pip"`.

Dependencies are pinned exactly in `pyproject.toml`, because run IDs and the
evaluation lock depend on bit-identical training.

**4. Download the dataset**

The data is not stored in this repository. The pipeline expects three files, with
their original names, in `data/raw/`:

```text
data/raw/
├── ev_battery_health.csv
├── README.md
└── data_dictionary .csv      # the space before .csv is part of the original name
```

The dataset is public, so no Kaggle account is needed:

```bash
make data            # or: python scripts/download_data.py
```

The script downloads the archive, extracts only the three files and verifies the
CSV hash. To do it by hand instead, download the archive from the
[dataset page](https://www.kaggle.com/datasets/vinay24baghira/ev-battery-health-and-degradation-dataset)
and extract the three files into `data/raw/`.

The audit checks the CSV against a recorded SHA-256 hash and stops if the file
differs. See [data/README.md](data/README.md).

**5. Check the installation**

```bash
make test            # or: python -m pytest
```

The tests use small synthetic arrays, so they pass before the pipeline has run.

**6. Run the pipeline**

```bash
make all             # or: python -m ev_battery all
```

This runs six stages in order. Each stage can also run on its own, for example
`make train` or `python -m ev_battery train`.

| Stage | Command | What it does | Output |
|---|---|---|---|
| Audit | `make audit` | Schema, quality and leakage checks, battery split | `results/audit/` |
| EDA | `make eda` | Train-only exploratory analysis | `results/eda/` |
| Preprocessing | `make preprocess` | Fit four recipes on train, write model inputs | `results/preprocessing/`, `data/processed/` |
| Training | `make train` | Staged MLP study on train and validation | `results/experiments/`, `models/mlp/` |
| Feature engineering | `make features` | Domain feature ablation on validation | `results/feature_engineering/` |
| Evaluation | `make evaluate` | Score the six locked checkpoints on test | `results/final_evaluation/` |

The first run trains 34 networks and takes about ten minutes on a laptop CPU. Runs whose code, data and configuration fingerprints match are reused, so
running `make all` again takes seconds.

**7. Run the notebooks**

```bash
make notebooks       # or: python scripts/run_notebook.py --all
```

This executes notebooks 00 to 05 in place and saves their outputs. To work through
them interactively, open the `notebooks/` folder in VS Code (or install
`jupyterlab`) and select the `.venv` kernel. Run them in order, because each one
reads what the previous stage wrote. The narrative in the notebooks is in
Indonesian.

**8. Read the results**

Each stage writes one Markdown report. A good reading order:

1. [Audit](results/audit/AUDIT_REPORT.md) and [EDA](results/eda/EDA_REPORT.md)
2. [Preprocessing](results/preprocessing/PREPROCESSING_REPORT.md)
3. [MLP experiments](results/experiments/EXPERIMENT_REPORT.md)
4. [Feature engineering](results/feature_engineering/FEATURE_ENGINEERING_REPORT.md)
5. [Final evaluation](results/final_evaluation/FINAL_REPORT.md)

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Download ev_battery_health.csv into data/raw` | The data is missing. Run `make data` (step 4). |
| `CSV version differs from the audited one` | The CSV is not the audited version. Run `python scripts/download_data.py --force`. |
| `UnicodeEncodeError` on Windows | Set `PYTHONIOENCODING=utf-8` before running Python. The Makefile sets it automatically. |
| `The local checkpoints differ from the recorded lock` | Training on another CPU or package build can change the last bits of the weights. The evaluator then writes a separate reproduction to `data/processed/evaluation_reproduction/` and leaves the recorded results untouched. This is expected. |
| `No module named pip` | The environment was created with uv. Use `make install PIP="uv pip"` or `uv pip install -e ".[dev]"`. |
| `make: command not found` | Install GNU Make, or use the `python` commands shown next to each step. |

## Project scope

The assignment asks for EDA, preprocessing and experiments with an MLP trained by
backpropagation. Each requirement maps to a stage:

| Requirement | Where it is covered |
|---|---|
| EDA | Target and feature distributions, correlation and redundancy, missingness, outliers, chemistry and age patterns, temporal quality (train only) |
| Normalisation | Standardisation vs min-max scaling (`baseline_standard` vs `baseline_minmax`) |
| Missing values | Train-median imputation with missing indicators, explicit missing category for categoricals |
| Feature selection | Leakage-based exclusion, redundancy-based compact recipe, domain feature ablation |
| Balancing | SOH-band loss weighting (`inverse`, `inverse_sqrt`) as the regression counterpart of class balancing |
| Input neurons | 67 vs 63 inputs, plus 71 to 91 with engineered features |
| Hidden layers | [2], [8], [32], [64], [64, 32], [128, 64, 32] |
| Activation function | sigmoid, tanh, ReLU |
| Loss function | MSE, MAE, Huber |
| Epoch | Endpoints at 50, 100 and 200 epochs, early stopping, `max_error`, underfitting vs overfitting demo |
| Learning and optimizer | Learning rates 0.01 to 0.3, gradient descent vs Adam |

Inputs are tabular features of a single observation. Timesteps are not used
because `observation_date` is missing for 2,216 rows and cumulative counters are
not always monotonic. The reasoning behind every decision is in
[docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Repository structure

```text
EV-Battery-Health/
├── archive/              # Read-only records of earlier versions (see archive/README.md)
│   ├── pytorch/          # Original PyTorch release: notebooks, results, configs, guide
│   └── numpy_first_evaluation/  # First test scoring of the NumPy models
├── configs/              # Feature contract, preprocessing recipes, study and evaluation plans
├── data/
│   ├── raw/              # Kaggle files (not committed)
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
│   ├── model.py          # From-scratch NumPy MLP: bp_fit, bp_predict, backprop
│   ├── training.py       # Training runs, checkpoints and staged study
│   ├── final_models.py   # The two configurations frozen for the test evaluation
│   ├── features.py       # Feature-engineering ablation
│   ├── evaluation.py     # Locked test evaluation
│   ├── reporting/        # Figures and Markdown reports per stage
│   └── __main__.py       # Pipeline CLI
├── tests/
├── Makefile
└── pyproject.toml
```

Run `make` to list every shortcut.

## Methodology at a glance

- Split by battery, not by row, so repeated observations of a battery never leak across splits.
- Every preprocessing statistic is learned from train only.
- Validation selects configurations. The final configurations are fixed in `configs/final_evaluation.json` and hashed into a lock before any test prediction.
- Every run is recorded, including the ones that lost, and every final seed is reported.
- `capacity_fade_pct` equals `100 - SOH` and is excluded from the inputs.
- The backpropagation code reproduces the course `bp_fit` exactly and passes finite-difference gradient checks.

## History of the test split

The project was first built with PyTorch, which scored the test split on
6 November 2026 (candidate MAE 0.950, baseline 1.027). The project was then
rewritten to the from-scratch NumPy implementation to follow the course modules.
The NumPy configurations were chosen on validation only and scored on the same
test split on 8 November 2026 (WIB). No NumPy setting was changed because of a
test result. Because the same test batteries were used twice, a comparison
between the two versions is descriptive rather than an independent test. The
PyTorch release is kept read-only in [archive/pytorch/](archive/pytorch/README.md).

## Contributing and license

Team conventions are in [CONTRIBUTING.md](CONTRIBUTING.md) and dataset facts are
in the [dataset card](docs/DATASET_CARD.md). Code: [MIT](LICENSE). Dataset: CC0,
according to the source documentation.
