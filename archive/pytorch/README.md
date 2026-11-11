# PyTorch release (archived)

The first complete version of the project, built with PyTorch autograd and
optimizers. It is kept read-only for comparison in the report. The current
pipeline does not read or write anything in this folder.

| Folder | Contents |
|---|---|
| `notebooks/` | The six executed notebooks of the release (`00` to `04`, including `03b`) |
| `results/` | Reports, figures, tables, run records and the evaluation lock of the release |
| `configs/` | Feature contract, preprocessing recipes, study and evaluation plans of the release |
| `docs/`, `models/` | Dataset card and model notes of the release, in Indonesian |
| `CONTEXT.md` | The original team guide (Indonesian) |

Links inside these files point to paths of the original tree, so a few of them do
not resolve here. Open the [release commit](https://github.com/HUB37T/EV-Battery-Health/tree/479ed842d141017591a3fe755647040c237dfd6c)
to browse the release exactly as it was.

## Final test results of the release

7,572 observations from 390 test batteries, mean ± sample SD over seeds 42, 123
and 2026, errors in SOH percentage points.

| Model | Configuration | Parameters | Test MAE | Test RMSE | Test R² |
|---|---|---:|---:|---:|---:|
| Baseline | `baseline_standard`, 67 → [64, 32] ReLU → 1, MSE, Adam lr 0.001 | 6,465 | 1.02684 ± 0.00311 | 1.49732 ± 0.00794 | 0.95221 ± 0.00051 |
| Candidate | `log_sensor_caps`, 67 → [32] sigmoid → 1, MAE, Adam lr 0.001, weight decay 1e-4 | 2,209 | 0.95005 ± 0.00619 | 1.42114 ± 0.01228 | 0.95694 ± 0.00074 |

Both used batch size 256 and early stopping on validation MAE. Below 70% SOH the
candidate scored 3.09049 against 2.80057 for the baseline.

## Running the release

The release code is not in the current tree. To run it, check out commit
`479ed842d141017591a3fe755647040c237dfd6c` in a separate clone or worktree and
follow the README of that commit:

```bash
git worktree add ../ev-battery-pytorch 479ed842d141017591a3fe755647040c237dfd6c
```

## Comparing with the NumPy version

| | PyTorch release | Current NumPy version |
|---|---|---|
| Gradients | autograd | hand-written backpropagation |
| Optimizer | Adam | plain gradient descent (Adam as comparison) |
| Candidate | log-cap recipe, [32] sigmoid, MAE loss | standard recipe, [8] sigmoid, MSE loss |
| Candidate test MAE | 0.950 | 0.973 |
| Baseline test MAE | 1.027 | 0.960 |

The test split was shared, so this comparison is descriptive. See
[archive/README.md](../README.md) for the timeline.
