# Dataset Card

## Source

[EV Battery Health & Degradation Dataset (Kaggle)](https://www.kaggle.com/datasets/vinay24baghira/ev-battery-health-and-degradation-dataset).

The version used in this project was downloaded on 6 November 2026 and is
identified by its CSV SHA-256:

```text
cacdf6afcf50db86583be3754207a89acbfd60b04dfff8f69b71e29cfd78218d
```

The archive contains `ev_battery_health.csv`, `README.md` and
`data_dictionary .csv` (with a space before the extension). The sample notebook
mentioned in the source README is not included. The audit refuses any CSV with a
different hash.

The source documentation describes the data as **fully synthetic**, produced by a
degradation-model generator, and licensed **CC0**. "Physics-informed" is the
author's description, not something this project validated against laboratory
data. Neither the generator code nor the masks of injected missing values and
outliers are available.

## Verified facts

- 50,000 observations, 35 columns, 2,600 unique batteries.
- One row is one observation of one battery, and `observation_id` is unique.
- 7 to 37 observations per battery (the source README says roughly 10 to 40).
- The target `state_of_health_pct` has no missing values and is in percent.
- Metadata, usage, environment and sensor columns contain missing values.
- Repeated observations of the same battery are correlated, so a random row split is not used.

Full numbers and the role of all 35 columns are in the
[audit report](../results/audit/AUDIT_REPORT.md) and
[`schema_audit.csv`](../results/audit/schema_audit.csv).

## Problem definition

**Regression of SOH at observation time, generalising to batteries not seen in
training.** This is not remaining-useful-life prediction, failure prediction or
forecasting of future SOH.

Assumption: the usage and environment summaries and metadata used as features are
known at observation time. The source does not document the aggregation windows,
so prediction before a session or into the future is not supported.

The split is made by `battery_id` without looking at the target: 1,820 train, 390
validation and 390 test batteries. Checks on the full dataset are limited to
integrity and basic quality. Any analysis that informs feature choices uses train
only.

## Feature policy

The machine-readable contract is
[`configs/feature_contract.json`](../configs/feature_contract.json). It is an
initial decision based on source definitions, not the result of statistical
selection.

| Group | Columns | Use |
|---|---|---|
| Target | `state_of_health_pct` | Prediction target |
| Baseline inputs | 21 columns: 15 numeric, 5 categorical, binary `recall_flag` | Model inputs |
| Identifiers | `battery_id`, `observation_id` | Splitting and tracking only |
| Direct leakage | `capacity_fade_pct` | Excluded, equals `100 - SOH` on every non-missing row |
| Alternative targets | `remaining_useful_life_cycles`, `failure_risk_label`, `degradation_category` | Excluded |
| Raw dates | `manufacture_date`, `observation_date`, `last_service_date` | Excluded, `battery_age_days` is kept |
| Sensor proxies | `internal_resistance_mohm`, `voltage_avg_v`, `current_avg_a`, `charge_time_avg_min` | Held out for a separate ablation |

The source README states that electrical measurements are derived from SOH and
chemistry and names internal resistance explicitly. The exact mechanism for each
sensor is unknown, so they are not all assumed to leak in the same way. Real
sensors can be valid inputs for current-state estimation, but results based on
synthetic proxies say nothing about physical relationships in real batteries.

## Temporal and quality findings

- 2,216 rows have no `observation_date`.
- No duplicate battery and date pairs exist among rows with a date.
- Battery age matches the date difference on all 43,199 comparable rows.
- `cycle_count` and `total_energy_throughput_kwh` often decrease at later dates even though they are described as cumulative.
- Part of the missing `last_service_date` values are structural, because `service_count = 0`.

Decision: a tabular baseline without windowing or timesteps. The data is not
repaired by sorting counters, forcing monotonicity, filling future dates or
removing extreme values.

## Limits of conclusions

The group split tests new batteries from the same simulator, not real-world
distributions or future calendar periods. Final test results (baseline MAE
0.96018 ± 0.00641 SOH points, candidate 0.97268 ± 0.00325) and their limitations,
including weaker performance below 70% SOH, are documented in the
[final report](../results/final_evaluation/FINAL_REPORT.md). The same test split
was scored earlier by the PyTorch version of the project, see
[archive/README.md](../archive/README.md).

## Reproduction

```bash
make data            # download and verify the archive
make audit           # or: python -m ev_battery audit
python scripts/run_notebook.py notebooks/00_data_audit.ipynb
```

The source CSV is never modified.
