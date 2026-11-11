"""Train-only fitted preprocessing, reusable at inference and in fold-local CV.

Run from repository root: python -m ev_battery.preprocessing
"""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from ev_battery.audit import TARGET, load_dataset, markdown_table
from ev_battery.paths import ROOT

MISSING = "__MISSING__"
# "minmax" maps train values to [0, 1], as minmax_scale does in the course modules.
SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler}
NONNEGATIVE = {
    "battery_age_days",
    "cycle_count",
    "avg_daily_distance_km",
    "avg_charging_power_kw",
    "charge_cycles_per_month",
    "total_energy_throughput_kwh",
    "service_count",
}
BOUNDS = {
    "fast_charge_ratio": (0, 1),
    "avg_depth_of_discharge_pct": (0, 100),
    "avg_state_of_charge_pct": (0, 100),
    "humidity_pct": (0, 100),
}
FORBIDDEN = {
    "battery_id",
    "observation_id",
    TARGET,
    "capacity_fade_pct",
    "remaining_useful_life_cycles",
    "failure_risk_label",
    "degradation_category",
    "manufacture_date",
    "observation_date",
    "last_service_date",
    "internal_resistance_mohm",
    "voltage_avg_v",
    "current_avg_a",
    "charge_time_avg_min",
}


class BatteryPreprocessor(TransformerMixin, BaseEstimator):
    """Strict named-column input -> dense float32 MLP input, target never used.

    Numeric, categorical, and binary rules have fixed output order. Unknown
    categories never refit the encoder. Missing indicators exist even if a
    feature happened to be complete during fit. No cross-row fill is performed.
    """

    def __init__(
        self,
        numeric_features,
        categorical_features,
        binary_features,
        log_features=(),
        clip_features=(),
        clip_quantiles=(0.005, 0.995),
        scaler="standard",
    ):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.binary_features = binary_features
        self.log_features = log_features
        self.clip_features = clip_features
        self.clip_quantiles = clip_quantiles
        self.scaler = scaler

    @property
    def required_columns(self):
        return list(self.numeric_features) + list(self.binary_features) + list(self.categorical_features)

    def _clean(self, X):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise TypeError("Input must be a DataFrame with unique column names.")
        required = self.required_columns
        missing, extra = set(required) - set(X.columns), set(X.columns) - set(required)
        if missing or extra:
            raise ValueError(f"Feature schema differs: missing={sorted(missing)}, extra={sorted(extra)}")
        if len(required) != len(set(required)) or set(required) & FORBIDDEN:
            raise ValueError(
                "Duplicate columns, IDs, the target, dates or sensor proxies are not allowed in the main pipeline."
            )
        cleaned = X.loc[:, required].copy()
        invalid = {}
        for c in list(self.numeric_features) + list(self.binary_features):
            raw = cleaned[c]
            value = pd.to_numeric(raw, errors="coerce").astype(float)
            value = value.where(np.isfinite(value))
            if c in NONNEGATIVE:
                value = value.where(value.ge(0))
            if c == "pack_capacity_kwh":
                value = value.where(value.gt(0))
            if c in BOUNDS:
                value = value.where(value.between(*BOUNDS[c]))
            if c == "service_count":
                value = value.where(np.isclose(value, np.round(value), equal_nan=False))
            if c in self.binary_features:
                value = value.where(value.isin([0, 1]))
            invalid[c] = int((raw.notna() & value.isna()).sum())
            cleaned[c] = value
        for c in self.categorical_features:
            value = cleaned[c].astype("string").str.strip()
            if value.eq(MISSING).any():
                raise ValueError(
                    f"Token {MISSING} is reserved for missing values and cannot be a real category of {c}."
                )
            invalid[c] = int((value.notna() & value.eq("")).sum())
            cleaned[c] = value.mask(value.eq("")).fillna(MISSING).astype(str)
        return cleaned, invalid

    def _numeric_values(self, clean):
        numeric = clean[list(self.numeric_features)].fillna(self.medians_).copy()
        for c in self.clip_features:
            numeric[c] = numeric[c].clip(self.clip_bounds_[c][0], self.clip_bounds_[c][1])
        for c in self.log_features:
            numeric[c] = np.sign(numeric[c]) * np.log1p(np.abs(numeric[c]))
        return numeric

    def fit(self, X, y=None):
        # Accept y for sklearn Pipeline compatibility, but never inspect/use it.
        del y
        clean, invalid = self._clean(X)
        if not len(clean):
            raise ValueError("Training frame is empty.")
        if not set(self.log_features).issubset(self.numeric_features) or not set(self.clip_features).issubset(
            self.numeric_features
        ):
            raise ValueError("Log and cap transforms apply only to selected numeric features.")
        qlo, qhi = self.clip_quantiles
        if not 0 <= qlo < qhi <= 1:
            raise ValueError("Invalid clipping quantiles.")
        self.medians_ = clean[list(self.numeric_features)].median()
        if self.medians_.isna().any():
            raise ValueError(
                "A numeric feature is entirely missing at fit time, an explicit decision is required."
            )
        self.binary_modes_ = {}
        for c in self.binary_features:
            mode = clean[c].mode()
            if mode.empty:
                raise ValueError(f"Binary feature {c} is entirely missing at fit time.")
            self.binary_modes_[c] = float(mode.iloc[0])
        self.clip_bounds_ = {
            c: tuple(float(v) for v in clean[c].quantile([qlo, qhi])) for c in self.clip_features
        }
        if self.scaler not in SCALERS:
            raise ValueError(f"Unknown scaler {self.scaler!r}.")
        self.scaler_ = SCALERS[self.scaler]().fit(self._numeric_values(clean))
        # Always allocate a distinct missing category, including future missing.
        categories = [sorted(set(clean[c]) | {MISSING}) for c in self.categorical_features]
        self.encoder_ = OneHotEncoder(
            categories=categories, handle_unknown="ignore", sparse_output=False, dtype=np.float32
        )
        self.encoder_.fit(clean[list(self.categorical_features)])
        numeric_names = [f"num__{c}" for c in self.numeric_features]
        numeric_missing = [f"missing__{c}" for c in self.numeric_features]
        binary_names = [f"binary__{c}" for c in self.binary_features]
        binary_missing = [f"missing__{c}" for c in self.binary_features]
        category_names = [f"cat__{c}" for c in self.encoder_.get_feature_names_out(self.categorical_features)]
        self.output_features_ = np.array(
            numeric_names + numeric_missing + binary_names + binary_missing + category_names, dtype=str
        )
        if len(set(self.output_features_)) != len(self.output_features_):
            raise ValueError("Output feature names are not unique.")
        self.fit_rows_ = len(clean)
        self.invalid_counts_train_ = invalid
        self.feature_names_in_ = np.array(self.required_columns, dtype=object)
        self.n_features_in_ = len(self.required_columns)
        self.n_features_out_ = len(self.output_features_)
        return self

    def transform(self, X):
        check_is_fitted(self, "output_features_")
        clean, _ = self._clean(X)
        numeric = self.scaler_.transform(self._numeric_values(clean))
        num_missing = clean[list(self.numeric_features)].isna().to_numpy(dtype=np.float32)
        binary = clean[list(self.binary_features)].fillna(self.binary_modes_).to_numpy(dtype=np.float32)
        binary_missing = clean[list(self.binary_features)].isna().to_numpy(dtype=np.float32)
        categorical = self.encoder_.transform(clean[list(self.categorical_features)])
        result = np.hstack([numeric, num_missing, binary, binary_missing, categorical]).astype(np.float32)
        if result.shape != (len(X), self.n_features_out_) or not np.isfinite(result).all():
            raise ValueError("Output shape changed or contains non-finite values.")
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "output_features_")
        return self.output_features_.copy()

    def diagnostic(self, X):
        check_is_fitted(self, "output_features_")
        clean, invalid = self._clean(X)
        unknown = {
            c: int((~clean[c].isin(self.encoder_.categories_[i])).sum())
            for i, c in enumerate(self.categorical_features)
        }
        capped = {
            c: int((clean[c].notna() & ~clean[c].between(*self.clip_bounds_[c])).sum())
            for c in self.clip_features
        }
        return {
            "invalid_to_missing": invalid,
            "unknown_categories": unknown,
            "clipped_rows_by_feature": capped,
        }


def scaler_center(scaler):
    """Train mean (standard) or train minimum (min-max) of every numeric feature."""
    return scaler.mean_ if isinstance(scaler, StandardScaler) else scaler.data_min_


def scaler_spread(scaler):
    """Train standard deviation (standard) or train range (min-max) of every numeric feature."""
    return scaler.scale_ if isinstance(scaler, StandardScaler) else scaler.data_range_


def read_settings(root=ROOT):
    contract = json.loads((root / "configs/feature_contract.json").read_text(encoding="utf-8"))
    settings = json.loads((root / "configs/preprocessing.json").read_text(encoding="utf-8"))
    if settings["categorical_missing_token"] != MISSING:
        raise ValueError("Configured missing token does not match the pipeline version.")
    return contract, settings


def make_preprocessor(recipe, contract, settings):
    spec = settings["recipes"][recipe]
    dropped = set(spec["drop_features"])
    if not dropped.issubset(contract["primary_features"]):
        raise ValueError("Unknown features in drop list.")
    return BatteryPreprocessor(
        numeric_features=[c for c in contract["numeric_features"] if c not in dropped],
        categorical_features=[c for c in contract["categorical_features"] if c not in dropped],
        binary_features=[c for c in contract["binary_features"] if c not in dropped],
        log_features=spec["log_features"],
        clip_features=spec["clip_features"],
        clip_quantiles=spec["clip_quantiles"],
        scaler=spec.get("scaler", "standard"),
    )


def load_splits(root=ROOT):
    frame, _, sha = load_dataset(root)
    manifest_path = root / "data/splits/battery_split.csv"
    audit = json.loads((root / "results/audit/provenance.json").read_text(encoding="utf-8"))
    if (
        sha != audit["dataset_sha256"]
        or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != audit["split_manifest_sha256"]
    ):
        raise ValueError("Dataset or split manifest differs from the locked audit.")
    manifest = pd.read_csv(manifest_path)
    if not manifest.battery_id.is_unique:
        raise ValueError("Battery IDs in the split manifest are not unique.")
    labels = frame.battery_id.map(manifest.set_index("battery_id").split)
    if labels.isna().any() or set(labels) != {"train", "validation", "test"}:
        raise ValueError("Some observations are not covered by the locked split.")
    splits = {s: frame.loc[labels.eq(s)].copy() for s in ["train", "validation", "test"]}
    groups = {s: set(f.battery_id) for s, f in splits.items()}
    if (
        groups["train"] & groups["validation"]
        or groups["train"] & groups["test"]
        or groups["validation"] & groups["test"]
    ):
        raise ValueError("A battery appears in more than one split.")
    return splits, audit


def parameter_tables(prep):
    numeric = []
    for i, c in enumerate(prep.numeric_features):
        lo, hi = prep.clip_bounds_.get(c, (np.nan, np.nan))
        numeric.append(
            {
                "feature": c,
                "median_train_original": prep.medians_[c],
                "cap_lower_train": lo,
                "cap_upper_train": hi,
                "signed_log1p": c in prep.log_features,
                "scaler": prep.scaler,
                "scaler_center_train": scaler_center(prep.scaler_)[i],
                "scaler_spread_train": scaler_spread(prep.scaler_)[i],
            }
        )
    categories = [
        {"feature": c, "category": value, "block_index": j}
        for i, c in enumerate(prep.categorical_features)
        for j, value in enumerate(prep.encoder_.categories_[i])
    ]
    output = pd.DataFrame(
        {"output_index": range(prep.n_features_out_), "output_feature": prep.get_feature_names_out()}
    )
    return pd.DataFrame(numeric), pd.DataFrame(categories), output


def load_prepared_split(recipe, split, root=ROOT):
    """Load trusted local artefacts, reject accidental mixing of recipes/runs."""
    _, settings = read_settings(root)
    if recipe not in settings["recipes"] or split not in {"train", "validation", "test"}:
        raise ValueError("Unknown recipe or split.")
    bundle = joblib.load(root / "models/preprocessing" / f"{recipe}.joblib")
    with np.load(root / "data/processed" / recipe / f"{split}.npz", allow_pickle=False) as stored:
        data = {key: stored[key] for key in stored.files}
    actual_state = joblib.hash((bundle["preprocessor"], bundle["target_scaler"]))
    if data["recipe"].item() != recipe or data["split"].item() != split or bundle["recipe"] != recipe:
        raise ValueError("Recipe or split identity differs.")
    if actual_state != bundle["state_hash"] or data["pipeline_state_hash"].item() != actual_state:
        raise ValueError("Arrays and fitted pipeline come from a different run or state.")
    if not np.array_equal(data["feature_names"], bundle["feature_names"]):
        raise ValueError("Feature order differs between arrays and pipeline.")
    if (
        data["X"].shape != (len(data["observation_id"]), len(bundle["feature_names"]))
        or not np.isfinite(data["X"]).all()
    ):
        raise ValueError("Invalid feature array.")
    n = len(data["observation_id"])
    if data["X"].dtype != np.float32 or len(data["battery_id"]) != n:
        raise ValueError("Array dtype or identifier length is invalid.")
    for key in ["y_scaled", "y_original"]:
        if data[key].shape != (n,) or not np.isfinite(data[key]).all():
            raise ValueError("Invalid target array.")
    return data, bundle


def make_figures(train, variants, output):
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    figures = []

    def save(fig, name):
        fig.savefig(output / name, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figures.append(name)

    selected = ["cycle_count", "avg_ambient_temp_c", "total_energy_throughput_kwh"]
    labels = {
        "cycle_count": "Equivalent cycles",
        "avg_ambient_temp_c": "Ambient temperature (°C)",
        "total_energy_throughput_kwh": "Cumulative energy throughput (kWh)",
    }
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for row, c in enumerate(selected):
        axes[row, 0].hist(train[c].dropna(), bins=40, color="#247e88", edgecolor="white")
        axes[row, 0].set_title("Observed values before preprocessing", fontsize=9)
        axes[row, 0].set_xlabel(labels[c])
        for col, recipe in enumerate(["baseline_standard", "log_sensor_caps"], 1):
            prep = variants[recipe]["preprocessor"]
            idx = list(prep.get_feature_names_out()).index(f"num__{c}")
            axes[row, col].hist(
                variants[recipe]["train_X"][:, idx],
                bins=40,
                color="#3674b5" if col == 1 else "#c47a21",
                edgecolor="white",
            )
            axes[row, col].set_title(recipe, fontsize=9)
            axes[row, col].set_xlabel("Standardised value")
        for ax in axes[row]:
            ax.set_ylabel("Train observations")
    fig.suptitle("Effect of transformations on train (no observations removed)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "01_before_after_train.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for recipe in ["baseline_standard", "log_sensor_caps"]:
        prep = variants[recipe]["preprocessor"]
        values = variants[recipe]["train_X"][:, : len(prep.numeric_features)]
        axes[0].plot(range(len(prep.numeric_features)), values.mean(axis=0), "o-", label=recipe)
        axes[1].plot(range(len(prep.numeric_features)), values.std(axis=0), "o-", label=recipe)
    axes[0].axhline(0, color="gray", ls="--")
    axes[1].axhline(1, color="gray", ls="--")
    axes[0].set(title="Train numeric means after scaling", ylabel="Mean")
    axes[1].set(title="Train numeric standard deviations after scaling (ddof=0)", ylabel="Standard deviation")
    for ax in axes:
        ax.set_xlabel("Numeric feature index (see parameter table)")
        ax.legend(fontsize=8)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0].set_ylim(-0.01, 0.01)
    axes[1].set_ylim(0.99, 1.01)
    fig.tight_layout()
    save(fig, "02_scaler_train_checks.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(variants)
    counts = [variants[n]["preprocessor"].n_features_out_ for n in names]
    bars = ax.barh(names, counts, color=["#247e88", "#7a5aa6", "#3674b5", "#c47a21"][: len(names)])
    for bar, count in zip(bars, counts):
        ax.text(count + 0.5, bar.get_y() + bar.get_height() / 2, str(count), va="center")
    ax.set(
        xlabel="Input neurons after preprocessing",
        title="MLP input width per recipe",
        xlim=(0, max(counts) + 8),
    )
    fig.tight_layout()
    save(fig, "03_input_dimensions.png")
    return figures


def write_report(summary, checks, target_scaler, settings, variants, output):
    dimensions = summary.loc[
        summary.split.eq("train"),
        ["recipe", "rows", "raw_features", "input_neurons", "numeric_missing_after", "finite"],
    ]
    recipes = []
    for name, entry in variants.items():
        prep = entry["preprocessor"]
        diag = prep.diagnostic(entry["train_raw"])
        recipes.append(
            {
                "recipe": name,
                "raw_features": len(prep.required_columns),
                "input_neurons": prep.n_features_out_,
                "capped_train_feature_cells": sum(diag["clipped_rows_by_feature"].values()),
                "log_features": ", ".join(prep.log_features) or "none",
            }
        )
    report = f"""# Preprocessing for the Backpropagation MLP

## Scope

Every pipeline is fitted **only on the 34,929 train observations (1,820 batteries)**. Validation (7,499 rows) and test (7,572 rows) are transformed with the frozen train parameters. This stage trains no model, performs no tuning, and makes no claim about overfitting or underfitting.

Preprocessing puts inputs on a scale that suits gradient-based optimisation, handles absent information explicitly, and provides a redundancy ablation. Generalisation is judged later from train and validation learning curves, MLP capacity, regularisation, early stopping and seed variation. The initial feature contract and the split manifest are unchanged.

## Recipes

{markdown_table(pd.DataFrame(recipes))}

The default recipe is **{settings["default_recipe"]}** because it keeps all 21 permitted raw features as a reference. The other recipes were fixed before training. They are alternatives, not experiment winners. `baseline_standard` vs `baseline_minmax` compares two normalisation methods. `baseline_standard` vs `log_sensor_caps` tests log transforms and caps together, so their individual effects are not isolated. `compact_log_sensor_caps` vs `log_sensor_caps` tests dropping two redundant features.

## 1. Processing order

1. Validate the named schema. IDs, the target, derived health labels, dates and sensor proxies may not enter the main pipeline. Column order may differ. Missing or extra columns are rejected.
2. Convert numerics and require finite values. Values that violate a definitional bound (ratios, percentages, binary flags, selected non-negative counters) become missing. Extreme but valid values are kept, and negative altitude is valid. There is no monotonic repair, forward/backward fill, or cross-row imputation.
3. Numeric features: train-median imputation. A missing indicator is created for every numeric feature, including features that were complete during fit, so new missingness at inference is still encoded.
4. Alternative recipes only: cap the selected sensors at the 0.5% and 99.5% quantiles of **non-missing train values before imputation**. No rows, old batteries or low-SOH observations are removed. Caps are a robustness hypothesis for heavy tails, not a verified error label.
5. Signed `log1p` for selected right-skewed features: `log1p(x)` for non-negative values and `sign(x) * log1p(abs(x))` for altitude. The target is not log-transformed.
6. Numeric features are scaled on train **after** imputation, capping and log transforms. Most recipes use `StandardScaler` (train mean 0, standard deviation 1). `baseline_minmax` uses `MinMaxScaler` (train range mapped to [0, 1]), the normalisation used in the course modules. Validation and test are never rescaled to their own statistics.
7. `recall_flag` is imputed with the train mode, stays 0/1, and gets a missing indicator. The rare flag is excluded from IQR outlier rules.
8. Categorical features: trim whitespace, encode empty values as the explicit `{MISSING}` category, and one-hot encode without an artificial order. Unseen categories map to an all-zero block, do not become the missing category, and do not change the input width. The missing token is allocated at fit time even when a feature is complete in train.
9. Concatenate numeric values, numeric indicators, binary values, binary indicators and one-hot blocks into a dense **float32** array. Output names and order are stored. The target is stored separately.

No value is filled from future observations or battery-level groupings. Inference needs only a single row of features that are available at prediction time.

## 2. Feature selection

- `capacity_fade_pct` (100 minus SOH), RUL, risk and degradation labels, and IDs are excluded to prevent leakage and memorisation. The four sensor proxies stay outside the main scope.
- The compact recipe drops `total_energy_throughput_kwh` (train Spearman correlation 0.984 with `cycle_count`) and `peak_temp_during_fast_charge_c` (0.920 with ambient temperature).
- Age and cycle count are both kept despite their correlation because they represent calendar ageing and usage respectively. Features with a weak marginal correlation to the target are not removed automatically, since an MLP can use interactions.
- Removing redundancy reduces input width and parameter count but can discard information. The compact recipe is therefore an ablation, **not the final feature selection**. Recipes are compared on validation.

## 3. Target and balancing

The original SOH is kept: no target clipping, no binning into classes, no oversampling, no SMOTE and no removal of low-SOH rows. Class balancing does not apply to a continuous target, and evaluation follows the original split distribution. The regression counterpart, weighting the loss by SOH band, is tested as an experiment in notebook 03.

The target scaler is fitted on `y_train` only: mean **{target_scaler.mean_[0]:.6f}**, scale **{target_scaler.scale_[0]:.6f}** percentage points. A standardised target lets the linear output neuron start near zero. It does not fix tail behaviour and does not guarantee better scores.

Arrays store `y_scaled` for optimisation and `y_original` for evaluation. Convert predictions back with `bundle['target_scaler'].inverse_transform(...)`. MAE and RMSE are always reported in original SOH percentage points. The Huber delta is defined on the scaled target, because changing the scale also changes loss and gradient magnitudes.

## 4. Input readiness

{markdown_table(dimensions)}

All splits keep aligned ID, X and y arrays. Holdout summaries are limited to shape and validity. No target distribution or model score is computed for them here. All arrays are finite, output names are unique and consistent, and the target round trip is checked. Per-split details are in `split_readiness.csv`.

![Before and after transformation](01_before_after_train.png)

Train only. The spike at the median is a consequence of imputation. The missing indicator keeps the information that the value was absent instead of pretending the median was measured. Log transforms and caps compress tails but also remove detail, which is why the recipes are compared on validation.

![Scaling checks](02_scaler_train_checks.png)

The mean and standard deviation checks cover the numeric block only. One-hot, binary and indicator columns are not forced to unit variance. Per-feature standardisation does not remove correlation between features.

![Input dimensions](03_input_dimensions.png)

The number of input neurons is counted **after preprocessing**, including indicators and one-hot columns. For a `[64, 32]` network with one output, the parameter count is `(p*64 + 64) + (64*32 + 32) + (32*1 + 1)`.

## 5. Integrity checks

{markdown_table(checks)}

`tests/test_preprocessing.py` additionally covers train-only parameters, new categories and new missingness, invalid values, column order, leaking schemas, serialisation, and that transforming holdout data does not change the fitted state. Small synthetic frames are used for behavioural checks only.

## 6. Artifacts and usage

- `data/processed/<recipe>/{{train,validation,test}}.npz`: float32 `X`, `y_scaled`, `y_original`, `observation_id`, `battery_id` and `feature_names`. Local only, ignored by Git.
- `models/preprocessing/<recipe>.joblib`: preprocessor, target scaler, recipe, source and split identity, feature names and package versions. Local only. Load joblib files only from trusted sources. `load_prepared_split` verifies recipe, split, feature order and state hash so arrays are never mixed with another pipeline.
- `results/preprocessing/`: this report, integrity checks, three train figures, train parameters, categories, output names and provenance.
- `configs/preprocessing.json`: reviewable recipe definitions. The initial feature contract is not overwritten.

```python
from ev_battery.preprocessing import load_prepared_split

data, bundle = load_prepared_split("baseline_standard", "train")
X_train, y_train = data["X"], data["y_scaled"]
# New rows: X_new = bundle["preprocessor"].transform(new_rows[bundle["required_columns"]])
# Back to SOH: bundle["target_scaler"].inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
```

The precomputed arrays are for this fixed split only. For cross-validation inside train, **fit a fresh preprocessor and target scaler on every training fold**. scikit-learn is used for transformations only. The only predictive model in this project is the MLP.

Reproduce with `python -m ev_battery preprocess`, then `python scripts/run_notebook.py notebooks/02_preprocessing.ipynb`. The raw CSV, split manifest and feature contract are never modified.

## 7. Questions for the experiments

1. Start from `baseline_standard`, then compare alternatives with the same architecture, seeds and training policy.
2. High train and validation error suggests underfitting or an optimisation problem: check capacity, learning rate, epochs and features.
3. Improving train error with worsening validation error suggests overfitting: check hidden size, the epoch budget, early stopping and seed stability.
4. Never add or remove features, or choose caps, based on the test split.

## References

- [scikit-learn: common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
- [OneHotEncoder and `handle_unknown`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)

The executed version is scikit-learn 1.7.2, recorded in `provenance.json`.
"""
    (output / "PREPROCESSING_REPORT.md").write_text(report, encoding="utf-8", newline="\n")


def run_preprocessing(root: Path = ROOT):
    splits, audit = load_splits(root)
    train = splits["train"]
    contract, settings = read_settings(root)
    target_scaler = StandardScaler().fit(train[[TARGET]])
    target_state_before = joblib.hash(target_scaler)
    output = root / "results/preprocessing"
    output.mkdir(parents=True, exist_ok=True)
    model_dir = root / "models/preprocessing"
    model_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    checks = []
    variants = {}
    source_info = {
        "dataset_sha256": audit["dataset_sha256"],
        "split_manifest_sha256": audit["split_manifest_sha256"],
        "fit_split": "train",
        "fit_rows": len(train),
        "fit_batteries": train.battery_id.nunique(),
        "fit_observation_ids_sha256": hashlib.sha256("\n".join(train.observation_id).encode()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "settings_sha256": hashlib.sha256((root / "configs/preprocessing.json").read_bytes()).hexdigest(),
        "packages": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "joblib", "matplotlib"]},
        "target": {
            "column": TARGET,
            "mean_train": float(target_scaler.mean_[0]),
            "scale_train": float(target_scaler.scale_[0]),
            "unit": "percentage points for inverse-transformed MAE/RMSE",
        },
        "default_recipe": settings["default_recipe"],
        "selection": "not ranked by model scores, no predictive model fitted",
    }
    for name in settings["recipes"]:
        prep = make_preprocessor(name, contract, settings)
        X_raw = train.loc[:, prep.required_columns]
        prep.fit(X_raw)
        params, categories, names = parameter_tables(prep)
        params.to_csv(output / f"{name}_numeric_parameters.csv", index=False, lineterminator="\n")
        categories.to_csv(output / f"{name}_categories.csv", index=False, lineterminator="\n")
        names.to_csv(output / f"{name}_output_features.csv", index=False, lineterminator="\n")
        state_before = joblib.hash(prep)
        pipeline_state = joblib.hash((prep, target_scaler))
        data_dir = root / "data/processed" / name
        data_dir.mkdir(parents=True, exist_ok=True)
        arrays = {}
        max_roundtrip = 0.0
        for split, frame in splits.items():
            X = prep.transform(frame.loc[:, prep.required_columns])
            ys = target_scaler.transform(frame[[TARGET]]).ravel().astype(np.float32)
            restored = target_scaler.inverse_transform(ys.astype(float).reshape(-1, 1)).ravel()
            max_roundtrip = max(max_roundtrip, float(np.max(np.abs(restored - frame[TARGET].to_numpy()))))
            if not np.isfinite(ys).all() or not np.isfinite(X).all():
                raise ValueError("Non-finite values after transform.")
            np.savez_compressed(
                data_dir / f"{split}.npz",
                X=X,
                y_scaled=ys,
                y_original=frame[TARGET].to_numpy(),
                battery_id=frame.battery_id.to_numpy(dtype=str),
                observation_id=frame.observation_id.to_numpy(dtype=str),
                feature_names=prep.get_feature_names_out(),
                recipe=np.array(name),
                split=np.array(split),
                pipeline_state_hash=np.array(pipeline_state),
            )
            summary.append(
                {
                    "recipe": name,
                    "split": split,
                    "rows": len(frame),
                    "batteries": frame.battery_id.nunique(),
                    "raw_features": len(prep.required_columns),
                    "input_neurons": X.shape[1],
                    "numeric_missing_after": int(np.isnan(X).sum()),
                    "finite": bool(np.isfinite(X).all()),
                    "dtype": str(X.dtype),
                }
            )
            arrays[split] = X
        unchanged = state_before == joblib.hash(prep)
        if not unchanged:
            raise AssertionError("Transform changed the fitted state.")
        if max_roundtrip > 1e-5:
            raise AssertionError("Target inverse-transform error is too large.")
        bundle = {
            "preprocessor": prep,
            "target_scaler": target_scaler,
            "recipe": name,
            "state_hash": pipeline_state,
            "required_columns": prep.required_columns,
            "feature_names": prep.get_feature_names_out(),
            "provenance": source_info,
        }
        bundle_path = model_dir / f"{name}.joblib"
        joblib.dump(bundle, bundle_path)
        loaded = joblib.load(bundle_path)
        restored_X = loaded["preprocessor"].transform(X_raw.iloc[:16])
        if not np.array_equal(restored_X, arrays["train"][:16]):
            raise AssertionError("Serialization changed the output.")
        checks.extend(
            [
                {"recipe": name, "check": "transform_state_unchanged", "result": "PASS"},
                {"recipe": name, "check": "all_split_float32_finite_and_same_width", "result": "PASS"},
                {"recipe": name, "check": "target_inverse_max_error_pp", "result": f"{max_roundtrip:.9f}"},
                {"recipe": name, "check": "serialization_exact_first_16_rows", "result": "PASS"},
            ]
        )
        variants[name] = {
            "preprocessor": prep,
            "train_raw": X_raw,
            "train_X": arrays["train"],
            "bundle_path": bundle_path,
        }
    summary = pd.DataFrame(summary)
    checks = pd.DataFrame(checks)
    if target_state_before != joblib.hash(target_scaler):
        raise AssertionError("Transform changed the target scaler state.")
    summary.to_csv(output / "split_readiness.csv", index=False, lineterminator="\n")
    checks.to_csv(output / "integrity_checks.csv", index=False, lineterminator="\n")
    figures = make_figures(train, variants, output)
    write_report(summary, checks, target_scaler, settings, variants, output)
    source_info["recipes"] = {
        name: {
            "raw_features": len(v["preprocessor"].required_columns),
            "input_neurons": v["preprocessor"].n_features_out_,
            "train_diagnostic": v["preprocessor"].diagnostic(v["train_raw"]),
        }
        for name, v in variants.items()
    }
    (output / "provenance.json").write_text(
        json.dumps(source_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return {
        "splits": splits,
        "variants": variants,
        "summary": summary,
        "checks": checks,
        "target_scaler": target_scaler,
        "settings": settings,
        "contract": contract,
        "provenance": source_info,
        "figures": figures,
        "output": output,
    }


if __name__ == "__main__":
    # Persist objects under their importable module, not __main__.
    from ev_battery.preprocessing import run_preprocessing as run_canonical

    result = run_canonical()
    print(result["summary"].to_string(index=False))
    print(result["output"] / "PREPROCESSING_REPORT.md")
