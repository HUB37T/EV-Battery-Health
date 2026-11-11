# Contributing

Working agreement for the project team. Read the [methodology](docs/METHODOLOGY.md)
first so that every change respects the same problem definition and protocol.

## Setup

Follow the step-by-step setup in the [README](README.md#getting-started). In short:

```bash
make venv && make install    # virtual environment and dependencies
make data                    # Kaggle files into data/raw/
make all                     # run the pipeline
```

## Ground rules

1. **One target, one split.** Every experiment uses `state_of_health_pct` and `data/splits/battery_split.csv`. Never re-split or change the seed to chase a better score.
2. **Fit on train only.** Imputers, encoders, scalers, caps and any data-driven selection learn from train. Validation chooses configurations.
3. **Test is locked.** Only `python -m ev_battery evaluate` touches the test split, and only for the configurations in `configs/final_evaluation.json`. Never tune on test results. The test split has already been scored by the PyTorch version and by the NumPy version, so any new idea needs validation evidence, not another look at test.
4. **Record every run.** Failed and losing runs stay in `results/experiments/` with their configuration. Never edit generated CSVs or reports by hand.
5. **MLP only.** The only predictive model is the feedforward MLP trained with backpropagation. Constant train mean and median references are sanity checks, not competing models.
6. **No claims without a run.** Every number in the report or slides must come from a file in `results/`.

## Making changes

- `archive/` holds read-only records of earlier versions. Never edit or regenerate it.
- Pipeline logic lives in `src/ev_battery/`. Notebooks call these functions and add the narrative, so do not duplicate logic inside notebooks.
- Configuration changes go into `configs/*.json`. Run IDs derive from a hash of code, data and configuration, so any change produces new runs instead of silently overwriting old ones.
- After changing code, regenerate the affected stages (`python -m ev_battery <stage>`) and re-execute the matching notebook with `python scripts/run_notebook.py`.
- Keep commits atomic, with a one-line conventional message (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `build:`, `chore:`).

Before opening a pull request:

```bash
ruff format . && ruff check .
pytest
```

With GNU Make, `make format` followed by `make check` does the same.

## Writing conventions

- Code, comments, error messages and documentation are in English.
- Notebook explanations are in Indonesian. Technical terms that are clearer in English stay in English (training, testing, validation, epoch, loss, batch, checkpoint, and so on).
- Avoid em dashes and unnecessary semicolons in prose.
- Always state units: SOH errors are in percentage points.

## Never commit

- Raw or processed data (`data/raw/`, `data/processed/`).
- Model weights or fitted pipelines (`models/`).
- Credentials such as `kaggle.json`, `.env` files or tokens.
- Virtual environments and editor folders.

`.gitignore` covers these paths, but check `git status` before every commit.

## Team roles

| Role | Responsibility | Deliverable |
|---|---|---|
| Data and EDA | Audit, data dictionary, quality checks, figures | Notebooks 00 and 01 |
| Preprocessing | Split, leakage audit, transformations, feature selection | Notebook 02 and fitted pipelines |
| Modeling | MLP, training loop, experiment matrix, run log | Notebooks 03 and 04 |
| Evaluation and reporting | Locked test evaluation, error analysis, report, slides | Notebook 05, final report and slides |

Roles can be combined. Each deliverable is reviewed by a member of another role.

## Presentation readiness

Every member should be able to answer these questions in their own words:

1. Why is this a regression problem, and what exactly is the target?
2. Why is the split made by battery instead of by row?
3. Why are `capacity_fade_pct` and the sensor proxies excluded?
4. How is the number of input neurons determined?
5. What do hidden layers and nonlinear activations add?
6. What is the difference between the MLP, backpropagation and the optimizer?
7. How does the gradient reach the first hidden layer?
8. What is the difference between an epoch, a batch and the learning rate?
9. Why are imputers and scalers fitted on train only?
10. How are SOH-band weights computed, and why is each sample's gradient divided by the batch size?
11. How was the final model chosen without using the test split?
12. What does the epoch experiment show?
13. How stable are the results across seeds?
14. Where does the model fail, and what are the limits of the conclusions?
15. How does someone else reproduce the results?

## Submission checklist

- [ ] Numbers in the report and slides match `results/final_evaluation/`.
- [ ] Every figure has labels, units and an interpretation.
- [ ] Limitations are stated: synthetic data, low-SOH weakness, unclipped linear output.
- [ ] The notebooks run top to bottom on a clean clone after `python -m ev_battery all`.
