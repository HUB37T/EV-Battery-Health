# Notebooks

The notebooks are the presentation layer: questions, methods, tables, figures and
interpretation, with the narrative written in Indonesian. The logic lives in
`src/ev_battery/`, so the notebooks and the CLI produce the same artifacts. Run
them in order, because each stage reads the outputs of the previous one.

| Notebook | Stage | Module | Outputs |
|---|---|---|---|
| `00_data_audit.ipynb` | Schema, quality, leakage, split | `audit` | `results/audit/` |
| `01_eda.ipynb` | Train-only EDA | `eda` | `results/eda/` |
| `02_preprocessing.ipynb` | Missing values, encoding, scaling, outliers, feature selection | `preprocessing` | `results/preprocessing/`, `data/processed/` |
| `03_mlp_experiments.ipynb` | MLP and backpropagation: inputs, hidden layers, activation, loss, epochs, seeds | `model`, `training` | `results/experiments/`, `models/mlp/` |
| `04_feature_engineering.ipynb` | Domain feature ablation | `features` | `results/feature_engineering/` |
| `05_final_evaluation.ipynb` | Locked test evaluation and error analysis | `evaluation` | `results/final_evaluation/` |

From the repository root, with the environment installed:

```bash
python scripts/run_notebook.py --all                          # all notebooks, 00 to 05
python scripts/run_notebook.py notebooks/03_mlp_experiments.ipynb
```

Notebooks can also be opened directly in Jupyter or VS Code. The first cell finds
the project root through `pyproject.toml`. The test split is not loaded before
notebook 05.
