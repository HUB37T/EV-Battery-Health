# Methodology

This document explains what the project does and why. It covers the problem
definition, data handling, the MLP and backpropagation, the experiment design and
the evaluation protocol. Numbers quoted here come from the generated reports in
[`results/`](../results/README.md), which remain the source of truth.

## 1. Problem definition

| Decision | Choice |
|---|---|
| Task | Regression |
| Target | `state_of_health_pct`, SOH in percent at observation time |
| Unit of observation | One measurement of one battery (`observation_id` is unique) |
| Prediction time | At the observation, using summaries known at that moment |
| Generalisation goal | Batteries that never appear in training |
| Primary metric | MAE in SOH percentage points (lower is better), with RMSE and R² as supporting metrics |

Out of scope: forecasting future SOH, remaining useful life (RUL), failure
classification, real fleet deployment, and any model family other than the MLP.

### Terminology

- **SOH** (state of health) describes battery health relative to its reference capacity. **SOC** (state of charge) is the current charge level. They are different quantities.
- **MLP** is the network architecture. **Backpropagation** computes the gradient of the loss with respect to every weight. **The optimizer** (Adam here) uses those gradients to update the weights. These three terms are not synonyms.
- **Train** fits parameters, **validation** chooses configurations, and **test** is evaluated once at the end with everything locked.
- **Leakage** is any path through which target, future or evaluation information influences training.

### Relation to the course material

McCulloch-Pitts neurons, Hebb nets, the perceptron and Adaline/Madaline motivate
weighted sums, biases and error-driven learning. SOM, LVQ and the Extreme Learning
Machine are covered in the course but are not used as models here. Every trained
model in this project is a feedforward MLP trained with backpropagation.

The implementation follows the course modules (BAB 6, BAB 7, LC 6 and LC 7): the
network is written from scratch in NumPy with `bp_fit` and `bp_predict`, sigmoid and
its derivative are written by hand, the bias is the last row of every weight matrix,
and the experiments use the same hyperparameter names (hidden neurons, `max_epoch`,
`max_error`, `learn_rate`). No deep-learning framework is used.

## 2. Data and split

The [Kaggle dataset](https://www.kaggle.com/datasets/vinay24baghira/ev-battery-health-and-degradation-dataset)
has 50,000 observations, 35 columns and 2,600 batteries. Its documentation states
that the data is fully synthetic and released under CC0. See the
[dataset card](DATASET_CARD.md) for verified facts.

Each battery is observed 7 to 37 times, so rows of the same battery are correlated.
A random row split would let the model see other observations of a test battery
during training. The split is therefore made **by `battery_id`**:

| Split | Batteries | Rows |
|---|---:|---:|
| Train | 1,820 | 34,929 |
| Validation | 390 | 7,499 |
| Test | 390 | 7,572 |

Battery IDs are sorted, permuted with NumPy `default_rng(42)` and assigned 70/15/15.
The manifest lives in `data/splits/battery_split.csv`, and the audit refuses to
overwrite it.

## 3. Leakage controls

| Column(s) | Reason for exclusion |
|---|---|
| `capacity_fade_pct` | Exactly `100 - SOH` on every non-missing row |
| `remaining_useful_life_cycles`, `failure_risk_label`, `degradation_category` | Alternative targets or labels derived from health |
| `battery_id`, `observation_id` | Identifiers used for splitting and tracking |
| `manufacture_date`, `observation_date`, `last_service_date` | Raw dates, `battery_age_days` already carries age |
| `internal_resistance_mohm`, `voltage_avg_v`, `current_avg_a`, `charge_time_avg_min` | The source states these are generated from SOH and chemistry, so they may act as target proxies |

The remaining 21 raw features (15 numeric, 5 categorical, 1 binary) form the
feature contract in `configs/feature_contract.json`. Every fitted statistic
(medians, modes, quantiles, scaler parameters, categories) is learned from train
only and then applied unchanged to validation and test.

## 4. Preprocessing decisions

| Topic | Decision | Rationale |
|---|---|---|
| Missing numeric values | Train median plus a missing indicator per feature | Keeps the information that a value was absent. Indicators exist even for features complete in train |
| Missing categories | Explicit `__MISSING__` category | Missingness is not a real category value and is not silently imputed |
| Invalid values | Out-of-domain values (for example a ratio above 1) become missing | Repairs nothing by guessing |
| Encoding | One-hot, unseen categories map to an all-zero block | No artificial order, stable input width |
| Normalisation | `StandardScaler` on numeric features, with a `MinMaxScaler` recipe (as `minmax_scale` in the course) as comparison. Target standardised separately | Features span very different ranges, which slows gradient descent |
| Outliers | Kept in the baseline. An alternative recipe caps three sensors at the 0.5% and 99.5% train quantiles and applies signed `log1p` to five skewed features | The source injects outliers without a mask, and extreme values can be genuine degraded batteries |
| Feature selection | Leakage removal (section 3) plus a compact recipe that drops two redundant features (throughput, rho 0.984 with cycles, and peak temperature, rho 0.920 with ambient) | Low marginal correlation is not a reason to drop a feature for a nonlinear model |
| Balancing | No resampling. SOH-band sample weights in the loss are tested as an experiment | The target is continuous, so class balancing does not apply. The regression counterpart is to weight rare SOH bands more |
| Timesteps | Not used, inputs are single observations | 2,216 rows lack `observation_date` and cumulative counters sometimes decrease |

This yields four recipes defined in `configs/preprocessing.json`:
`baseline_standard` (67 inputs), `baseline_minmax` (67 inputs), `log_sensor_caps`
(67 inputs) and `compact_log_sensor_caps` (63 inputs). The number of input neurons is counted
after encoding and indicators.

## 5. The multilayer perceptron

For one sample with `p` inputs and layers `l = 1 ... L`:

```text
a(0) = x
z(l) = W(l) a(l-1) + b(l)
a(l) = f_l(z(l))
y_hat = a(L)
```

`W(l)` has shape `n_l x n_(l-1)`, and `b(l)`, `z(l)` and `a(l)` have shape
`n_l x 1`. Hidden layers use a nonlinear activation. Without it, stacked linear
layers collapse into a single affine map. The output is one linear neuron
because SOH is an unbounded regression target on the standardised scale.

In code, the bias is stored as the last row of each weight matrix, so `W(l)` has
shape `(n_(l-1) + 1, n_l)` and the input of every layer gets an extra neuron fixed
at 1, exactly as in the course `bp_fit`.

Parameter count for the baseline network `67 -> [32] -> 1`:

```text
(67 + 1) * 32 + (32 + 1) * 1 = 2,176 + 33 = 2,209 parameters
```

and for the candidate `67 -> [8] -> 1`:

```text
(67 + 1) * 8 + (8 + 1) * 1 = 544 + 9 = 553 parameters
```

| Activation | Formula | Notes |
|---|---|---|
| ReLU | `max(0, z)` | Gradient 1 for `z > 0` and 0 otherwise, units can die |
| Sigmoid | `1 / (1 + exp(-z))` | Output in (0, 1), gradient shrinks when saturated |
| Tanh | `tanh(z)` | Output in (-1, 1), also saturates |
| Linear | `z` | Used for the regression output |

Weights start from a uniform distribution like the course (`uniform(-0.5, 0.5)` in
LC 7), with the range scaled to `sqrt(6 / (n_in + n_out))` so that sigmoid units
with 67 inputs do not start saturated. Biases start at zero. The code is in
[`src/ev_battery/model.py`](../src/ev_battery/model.py).

## 6. Training with backpropagation

### 6.1 One mini-batch

1. `forward`: compute the net input and activation of every layer.
2. `loss_gradient`: the derivative of the loss with respect to the output.
3. `backward`: propagate the deltas from the output to the first hidden layer and compute every weight gradient.
4. `update_weights`: plain gradient descent `W = W - learn_rate * gradient` (the course rule), or Adam.
5. After all batches (one epoch), evaluate train and validation without updating.

The course updates the weights after every sample. Here the gradient is averaged
over a mini-batch of 32 samples, because per-sample updates over 34,929 rows would
make every experiment take hours. With `batch_size=1`, no shuffling and a sigmoid
output, the implementation produces exactly the same weights as the course
`bp_fit`, which a unit test and notebook 03 verify.

Like the course, `bp_fit` stops when `max_epoch` is reached or when the epoch MSE
falls to `max_error`. The study adds early stopping on validation MAE. The training
loop around `bp_fit` is in [`src/ev_battery/training.py`](../src/ev_battery/training.py).

### 6.2 Losses and metrics

With error `e_i = y_hat_i - y_i` over `N` samples:

```text
MSE  = (1/N) * sum(e_i^2)
MAE  = (1/N) * sum(|e_i|)
RMSE = sqrt(MSE)
R^2  = 1 - sum(e_i^2) / sum((y_i - mean(y))^2)
```

MSE penalises large errors more strongly. MAE grows linearly and has a kink at
zero. Huber is quadratic near zero and linear beyond `delta`. A `delta` of 1 on
the standardised target corresponds to about 7.19 SOH points. Losses are
optimised on the standardised target, while every reported metric is converted
back to SOH percentage points. An MAE of 1 means an average absolute error of one
percentage point, not a 1% relative error.

### 6.3 The chain rule

For one sample with loss `0.5 * (y_hat - y)^2`:

```text
delta(L) = (y_hat - y) * f_L'(z(L))
delta(l) = (W(l+1)^T delta(l+1)) * f_l'(z(l))
dLoss/dW(l) = delta(l) a(l-1)^T
dLoss/db(l) = delta(l)
```

Plain gradient descent then updates `W <- W - lr * dLoss/dW`. This matches the
course code, which writes the error as `e = target - output` and adds
`learn_rate * delta * a` to the weights. Adam rescales each step using running
estimates of the gradient moments (`m` and `v`), but the gradients still come from
the same backpropagation.

### 6.4 Worked example

A single input, one ReLU hidden unit and one linear output (not battery data):

```text
x = 2, y = 1, w1 = 0.5, b1 = 0, w2 = 0.4, b2 = 0, lr = 0.1
```

Forward pass:

```text
z1 = 0.5 * 2 + 0 = 1
a1 = ReLU(1) = 1
y_hat = 0.4 * 1 + 0 = 0.4
loss = 0.5 * (0.4 - 1)^2 = 0.18
```

Backward pass, using the weights from the forward pass:

```text
delta_out = 0.4 - 1 = -0.6
grad_w2 = -0.6 * a1 = -0.6        grad_b2 = -0.6
delta_hidden = w2 * delta_out * ReLU'(z1) = 0.4 * -0.6 * 1 = -0.24
grad_w1 = -0.24 * x = -0.48       grad_b1 = -0.24
```

Update:

```text
w1 = 0.548, b1 = 0.024, w2 = 0.46, b2 = 0.06
```

The next forward pass gives `a1 = 1.12`, `y_hat = 0.5752` and a loss of about
`0.0902`, lower than before. A unit test and notebook 03 reproduce these numbers
with the project code. A common mistake is to compute `delta_hidden` with the already updated
`w2`.

## 7. Experiment design

All runs share the battery split, batch size 32, at most 200 epochs, early stopping
on validation MAE with patience 15 and min_delta 0.001 points, and seed 42 during the
search. The baseline mirrors the course defaults: hidden layer of 32 sigmoid neurons,
MSE loss, plain gradient descent with `learn_rate` 0.1.

The search changes one factor per stage and carries the validation winner forward:

| Stage | Values tested | Requirement covered | Selected (validation MAE) |
|---|---|---|---|
| Preprocessing | four recipes (67, 67, 67 and 63 inputs) | Input neurons, normalisation, feature selection | `baseline_standard` (1.007) |
| Hidden layers | [2], [8], [32], [64], [64, 32], [128, 64, 32] | Hidden layers and neurons | [8] (0.994) |
| Activation | sigmoid, tanh, ReLU | Activation function | sigmoid |
| Loss | MSE, MAE, Huber | Loss function | MSE |
| Learning rate | 0.01, 0.05, 0.1, 0.3 | Learning rate | 0.1 |
| Optimizer | gradient descent vs Adam (lr 0.001) | Optimizer | gradient descent |
| Balancing | none, `inverse_sqrt`, `inverse` SOH-band weights | Balancing | none |
| `max_error` | 0, 0.05, 0.1 | Stopping criterion | 0 (disabled) |
| Epochs | endpoints at 50, 100 and 200 of one run without early stopping | Epoch budget | diagnostic |
| Capacity demo | hidden [2] with `max_error` 0.5, hidden [128] on 100 batteries for 1,000 epochs | Underfitting vs overfitting (LC 7) | diagnostic |
| Seeds | 42, 123, 2026 for baseline and candidate | Stability | |
| Feature engineering | four domain feature blocks, then three-seed confirmation | Extra input features | control kept |

Every stage is compared on validation MAE in original units, never on raw loss
values, because MSE, MAE and Huber live on different scales. `best.npz` stores the
exact minimum validation MAE, and identical runs are reused through a fingerprint
of code, data and configuration. This sequential search is not a full factorial
design and does not prove a global optimum.

Most differences between configurations are around 0.01 MAE points. The three-seed
confirmation shows that this is the size of the seed noise: the baseline scored
0.99997 ± 0.0064 and the candidate 0.99988 ± 0.0083. The candidate therefore
matches the baseline with a quarter of the parameters rather than beating it.

The feature-engineering variant had to beat the control by at least 0.01 MAE
points on the three-seed mean and win at least two seeds. The best variant improved
the mean by only 0.00286 points, so the control stayed the main candidate.

## 8. Evaluation protocol

1. Fix every feature, preprocessing, architecture, checkpoint and seed decision on train and validation.
2. Write the plan to `configs/final_evaluation.json`.
3. Hash the six checkpoints, both pipelines, the test arrays and the configuration into `results/final_evaluation/evaluation_lock.json` before any test prediction exists.
4. Score the baseline and candidate on test for all three seeds in one step.
5. Report every seed, the mean and the sample SD. Never select a seed, ensemble, clip or retune after seeing test.
6. Report errors per SOH band and per battery, plus constant train mean and median references as a sanity check.

The sample SD across seeds measures training variability on a fixed split. It is
not a confidence interval over batteries.

## 9. Limitations

- The data is synthetic. Results describe the simulator, not real vehicles.
- Error is highest for SOH below 70%, which is rare in train. Both models overestimate SOH in that band, and SOH-band weighting trades better low-SOH error for worse overall error.
- The linear output can predict values outside 0 to 100. They are reported, not clipped.
- Validation informed many decisions, so validation scores are optimistic. The single locked test evaluation is the unbiased estimate.
- The candidate selected on seed 42 did not beat the baseline on test (MAE 0.973 vs 0.960), consistent with the seed confirmation. Selection gains of about 0.01 points are within seed noise.
- The test split was also scored by an earlier PyTorch version of this project. The switch to the from-scratch NumPy implementation was made to follow the course modules, not in response to test results, and no setting was tuned on test.
