"""Reproducible, non-destructive audit, no preprocessing fit or model training.

Run from the repository root: python -m ev_battery.audit
"""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ev_battery.paths import ROOT

TARGET = "state_of_health_pct"
SEED = 42
EXPECTED_SHA256 = "cacdf6afcf50db86583be3754207a89acbfd60b04dfff8f69b71e29cfd78218d"
SOURCE = "https://www.kaggle.com/datasets/vinay24baghira/ev-battery-health-and-degradation-dataset"
DATE_COLUMNS = ["manufacture_date", "observation_date", "last_service_date"]
CATEGORICAL = ["chemistry", "manufacturer", "vehicle_model", "climate_zone", "bms_firmware_version"]
PROXY_COLUMNS = ["internal_resistance_mohm", "voltage_avg_v", "current_avg_a", "charge_time_avg_min"]
OTHER_TARGETS = ["remaining_useful_life_cycles", "failure_risk_label", "degradation_category"]


def load_dataset(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    path = root / "data/raw/ev_battery_health.csv"
    if not path.is_file():
        raise FileNotFoundError("Download ev_battery_health.csv into data/raw, see data/README.md.")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA256:
        raise ValueError(
            "CSV version differs from the audited one. Re-audit the version and split, never overwrite earlier results silently."
        )
    frame = pd.read_csv(path)
    dictionary = pd.read_csv(root / "data/raw/data_dictionary .csv")
    if set(frame.columns) != set(dictionary["column"]) or len(frame.columns) != len(dictionary):
        raise ValueError("CSV schema does not match the data dictionary.")
    if frame["battery_id"].isna().any() or frame["observation_id"].duplicated().any():
        raise ValueError("Missing IDs or non-unique observation_id, review the split.")
    if frame[TARGET].isna().any():
        raise ValueError("Missing targets, define a policy before splitting.")
    return frame, dictionary, sha


def feature_contract(frame: pd.DataFrame, sha: str) -> dict:
    excluded = {
        "battery_id": "Group ID used for splitting, not a feature.",
        "observation_id": "Observation ID used for tracking, not a feature.",
        "capacity_fade_pct": "Direct identity 100 - SOH, target leakage.",
        **{c: "Alternative target or health label, not an input for SOH regression." for c in OTHER_TARGETS},
        **{c: "Raw date excluded from the baseline, battery_age_days is used instead." for c in DATE_COLUMNS},
        **{
            c: "Candidate synthetic sensor proxy, only for a separate ablation after reviewing origin and availability."
            for c in PROXY_COLUMNS
        },
    }
    primary = [c for c in frame if c not in excluded and c != TARGET]
    return {
        "task": "Regression of SOH at observation time for batteries not seen in train (not forecasting or RUL)",
        "target": TARGET,
        "target_unit": "percent, MAE/RMSE errors in percentage points",
        "dataset_sha256": sha,
        "seed": SEED,
        "split_unit": "battery_id",
        "split_ratios_batteries": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "primary_features": primary,
        "categorical_features": [c for c in primary if c in CATEGORICAL],
        "numeric_features": [c for c in primary if c not in CATEGORICAL and c != "recall_flag"],
        "binary_features": ["recall_flag"],
        "excluded_features": excluded,
        "proxy_ablation_candidates": PROXY_COLUMNS,
        "availability_assumption": "Usage, environment summaries and metadata are known at observation time. The source does not document aggregation windows, no claim of prediction before a session or into the future.",
        "timesteps": "Not used: dates are missing and cumulative counters are not always monotonic, requires further temporal audit.",
        "preprocessing_status": "Not fitted here. Imputation, encoding, scaling and data-driven selection may learn from train only.",
        "balancing": "No class balancing because the target is continuous.",
    }


def build_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    ids = np.array(sorted(frame["battery_id"].unique()))
    shuffled = np.random.default_rng(SEED).permutation(ids)
    # Integer arithmetic avoids int(2600 * 0.70) rounding down to 1819.
    n_train, n_validation = len(ids) * 70 // 100, len(ids) * 15 // 100
    labels = np.repeat(
        ["train", "validation", "test"], [n_train, n_validation, len(ids) - n_train - n_validation]
    )
    manifest = (
        pd.DataFrame({"battery_id": shuffled, "split": labels})
        .sort_values("battery_id")
        .reset_index(drop=True)
    )
    row_split = frame["battery_id"].map(manifest.set_index("battery_id")["split"])
    groups = {
        s: set(manifest.loc[manifest["split"].eq(s), "battery_id"]) for s in ["train", "validation", "test"]
    }
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]
    assert len(set.union(*groups.values())) == len(ids) and row_split.notna().all()
    counts = pd.DataFrame(
        {
            "split": ["train", "validation", "test"],
            "batteries": [len(groups[s]) for s in ["train", "validation", "test"]],
            "rows": [int(row_split.eq(s).sum()) for s in ["train", "validation", "test"]],
        }
    )
    counts["row_pct"] = counts["rows"] / len(frame) * 100
    return manifest, row_split, counts


def audit_quality(frame: pd.DataFrame, dictionary: pd.DataFrame, contract: dict) -> tuple[pd.DataFrame, dict]:
    roles = {c: "excluded" for c in contract["excluded_features"]}
    roles.update({c: "primary_feature" for c in contract["primary_features"]})
    roles[TARGET] = "target"
    schema = dictionary.copy()
    schema["actual_dtype"] = schema["column"].map(frame.dtypes.astype(str))
    schema["missing_count"] = schema["column"].map(frame.isna().sum())
    schema["missing_pct"] = schema["missing_count"] / len(frame) * 100
    schema["unique_non_null"] = schema["column"].map(frame.nunique())
    schema["role"] = schema["column"].map(roles)
    schema["decision"] = (
        schema["column"]
        .map(contract["excluded_features"])
        .fillna("Target")
        .where(schema["role"].ne("primary_feature"), "Baseline candidate, preprocessing not fitted yet.")
    )
    dates = frame[DATE_COLUMNS].apply(pd.to_datetime, errors="coerce")
    valid_date = frame["observation_date"].notna()
    delta = (dates["observation_date"] - dates["manufacture_date"]).dt.days - frame["battery_age_days"]
    fade = frame["capacity_fade_pct"].notna()
    summary = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "unique_batteries": int(frame["battery_id"].nunique()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_observation_ids": int(frame["observation_id"].duplicated().sum()),
        "missing_targets": int(frame[TARGET].isna().sum()),
        "missing_observation_dates": int((~valid_date).sum()),
        "duplicate_battery_date_non_null": int(
            frame.loc[valid_date].duplicated(["battery_id", "observation_date"]).sum()
        ),
        "date_parse_errors": {c: int((frame[c].notna() & dates[c].isna()).sum()) for c in DATE_COLUMNS},
        "observation_before_manufacture": int((dates["observation_date"] < dates["manufacture_date"]).sum()),
        "service_after_observation": int((dates["last_service_date"] > dates["observation_date"]).sum()),
        "service_before_manufacture": int((dates["last_service_date"] < dates["manufacture_date"]).sum()),
        "age_check_eligible": int(delta.notna().sum()),
        "age_date_mismatches": int(delta.dropna().ne(0).sum()),
        "fade_identity_eligible": int(fade.sum()),
        "fade_identity_max_abs_error": float(
            (frame.loc[fade, "capacity_fade_pct"] + frame.loc[fade, TARGET] - 100).abs().max()
        ),
        "observations_per_battery": frame.groupby("battery_id").size().describe().to_dict(),
        "date_ranges": {
            c: {"min": str(dates[c].min().date()), "max": str(dates[c].max().date())} for c in DATE_COLUMNS
        },
        "service_missing_count_zero": int(
            (dates["last_service_date"].isna() & frame["service_count"].eq(0)).sum()
        ),
        "service_missing_count_positive": int(
            (dates["last_service_date"].isna() & frame["service_count"].gt(0)).sum()
        ),
        "service_missing_count_missing": int(
            (dates["last_service_date"].isna() & frame["service_count"].isna()).sum()
        ),
        "infinite_numeric_values": int(np.isinf(frame.select_dtypes("number")).sum().sum()),
    }
    violations = {}
    for c in ["cycle_count", "battery_age_days", "total_energy_throughput_kwh"]:
        ordered = (
            frame.assign(parsed_date=dates["observation_date"])
            .dropna(subset=["parsed_date", c])
            .sort_values(["battery_id", "parsed_date", "observation_id"])
        )
        lag = ordered.groupby("battery_id")[["parsed_date", c]].shift()
        eligible = ordered["parsed_date"].gt(lag["parsed_date"])
        violations[c] = {
            "eligible_pairs": int(eligible.sum()),
            "decreasing_pairs": int((eligible & ordered[c].lt(lag[c])).sum()),
        }
    summary["temporal_counter_checks"] = violations
    summary["within_battery_static_conflicts"] = {
        c: int(frame.groupby("battery_id")[c].nunique().gt(1).sum())
        for c in [
            "chemistry",
            "manufacturer",
            "vehicle_model",
            "pack_capacity_kwh",
            "manufacture_date",
            "climate_zone",
        ]
    }
    ranges = {
        "state_of_health_pct": (0, 100),
        "capacity_fade_pct": (0, 100),
        "fast_charge_ratio": (0, 1),
        "avg_depth_of_discharge_pct": (0, 100),
        "avg_state_of_charge_pct": (0, 100),
        "humidity_pct": (0, 100),
        "recall_flag": (0, 1),
    }
    summary["outside_basic_ranges"] = {
        c: int((frame[c].notna() & ~frame[c].between(lo, hi)).sum()) for c, (lo, hi) in ranges.items()
    }
    summary["unexpected_recall_values"] = int(
        (frame["recall_flag"].notna() & ~frame["recall_flag"].isin([0, 1])).sum()
    )
    return schema, summary


def profile_train(train: pd.DataFrame, contract: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric = contract["numeric_features"] + [TARGET] + PROXY_COLUMNS
    statistics = (
        train[numeric]
        .describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
        .T.rename_axis("column")
        .reset_index()
    )
    flags = []
    for c in numeric:
        q1, q3 = train[c].quantile([0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        valid = train[c].notna()
        n = int((valid & ~train[c].between(lo, hi)).sum())
        flags.append(
            {
                "column": c,
                "iqr_lower_train": lo,
                "iqr_upper_train": hi,
                "flagged_rows": n,
                "non_missing_rows": int(valid.sum()),
                "flagged_pct_of_non_missing": n / valid.sum() * 100,
            }
        )
    bins = [-np.inf, 70, 80, 90, 95, np.inf]
    bands = pd.cut(
        train[TARGET], bins=bins, right=False, labels=["<70", "70–<80", "80–<90", "90–<95", ">=95"]
    )
    distribution = bands.value_counts(sort=False).rename_axis("soh_band_pct").reset_index(name="train_rows")
    distribution["train_pct"] = distribution["train_rows"] / len(train) * 100
    return statistics, pd.DataFrame(flags), distribution


def make_figures(train: pd.DataFrame, contract: dict, output: Path) -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.facecolor": "white",
        }
    )
    paths = []

    def save(fig, name):
        path = output / name
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    missing = train.isna().mean().mul(100).sort_values()
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(missing.index, missing.values, color="#207c83")
    ax.set(
        xlabel="Missing in train (%)",
        title="Missing values per column (train only)",
        xlim=(0, max(42, missing.max() + 3)),
    )
    ax.grid(axis="x", alpha=0.18)
    save(fig, "01_missing_train.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(train[TARGET], bins=np.arange(55, 102, 2), color="#207c83", edgecolor="white")
    axes[0].set(xlabel="SOH (%)", ylabel="Observations", title="Train target distribution")
    chemistry = train["chemistry"].fillna("Missing").value_counts()
    axes[1].bar(chemistry.index, chemistry.values, color="#3665a0")
    axes[1].set(xlabel="Battery chemistry", ylabel="Observations", title="Train chemistry mix (per row)")
    save(fig, "02_target_chemistry_train.png")

    selected = [
        "battery_age_days",
        "cycle_count",
        "fast_charge_ratio",
        "avg_ambient_temp_c",
        "avg_depth_of_discharge_pct",
        "internal_resistance_mohm",
    ]
    plot = train.sample(min(5000, len(train)), random_state=SEED)
    labels = {
        "battery_age_days": "Battery age (days)",
        "cycle_count": "Equivalent cycles",
        "fast_charge_ratio": "Fast-charge ratio (0 to 1)",
        "avg_ambient_temp_c": "Ambient temperature (°C)",
        "avg_depth_of_discharge_pct": "Depth of discharge (%)",
        "internal_resistance_mohm": "Internal resistance (mΩ)",
    }
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, c in zip(axes.flat, selected):
        ax.scatter(plot[c], plot[TARGET], s=5, alpha=0.2, color="#207c83", rasterized=True)
        ax.set(xlabel=labels[c], ylabel="SOH (%)")
        if c == "internal_resistance_mohm":
            ax.set_title("Sensor proxy: not a baseline feature", fontsize=9)
    fig.suptitle("Feature and target relationships (fixed sample of 5,000 train rows)", y=1.01)
    fig.tight_layout()
    save(fig, "03_relationships_train.png")

    sensor = ["avg_ambient_temp_c", "internal_resistance_mohm", "current_avg_a", "charge_time_avg_min"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    titles = ["Ambient temperature", "Internal resistance", "Average current", "Charge duration"]
    units = ["°C", "mΩ", "A", "min"]
    for ax, c, title, unit in zip(axes, sensor, titles, units):
        ax.boxplot(train[c].dropna(), widths=0.4, flierprops={"markersize": 2, "alpha": 0.3})
        ax.set(title=title, ylabel=unit, xticks=[])
    fig.suptitle("Sensor ranges and outlier candidates (train, nothing removed)")
    fig.tight_layout()
    save(fig, "04_sensor_ranges_train.png")
    return paths


def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value):
        if pd.isna(value):
            return "n/a"
        return str(value).replace("|", "/").replace("\n", " ")

    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    lines += [
        "| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join(lines)


def write_report(summary, schema, contract, counts, train, bands, outliers, output):
    temporal = pd.DataFrame(summary["temporal_counter_checks"]).T.rename_axis("counter").reset_index()
    missing = schema.loc[
        schema["missing_count"].gt(0), ["column", "missing_count", "missing_pct"]
    ].sort_values("missing_count", ascending=False)
    report = f"""# Dataset Audit and Initial Train Profile

## Key decisions

- Task: **regression of SOH at observation time for unseen batteries**, target `{TARGET}`.
- The source README describes the data as synthetic, not real fleet telemetry. No model is trained at this stage.
- Full dataset: **{summary["rows"]:,} rows, {summary["columns"]} columns, {summary["unique_batteries"]:,} batteries**.
- 70/15/15 split by **battery_id** with seed {SEED}. No battery appears in more than one subset. The ratios apply to batteries, so row counts are not exactly 70/15/15.
- Distribution and relationship profiles and IQR bounds use train only. Global checks are limited to schema, IDs, dates, missingness, basic ranges and the target identity. The test split is not used for any selection or scoring.
- `{len(contract["primary_features"])}` raw candidate features **before encoding**. The number of input neurons is only known after preprocessing is fitted on train.

## Source and version

[Kaggle dataset]({SOURCE}). Column definitions come from `data/raw/README.md` and `data/raw/data_dictionary .csv` in the downloaded archive. The source README states a **CC0** licence and fully synthetic data. The dictionary file name contains a space before `.csv` and is kept as is. The archive does not contain the sample notebook mentioned in its README.

CSV SHA-256: `{contract["dataset_sha256"]}`.
Environment details and documentation hashes are stored in `provenance.json`. The split manifest `../../data/splits/battery_split.csv` is locked to this CSV version. All outputs can be regenerated without modifying the CSV.

## Split manifest

{markdown_table(counts.round(3))}

The manifest has one row per battery. Join on `battery_id` to select rows. Never split randomly by observation. The algorithm sorts battery IDs, permutes them with NumPy `default_rng(42)`, and assigns train, validation and test in that order. There is no stratification on the target and the seed was not chosen from model scores.

## Global integrity

- Fully duplicated rows: {summary["duplicate_rows"]}. Duplicated `observation_id`: {summary["duplicate_observation_ids"]}.
- Missing targets: {summary["missing_targets"]}. Infinite numeric values: {summary["infinite_numeric_values"]}.
- Missing `observation_date`: {summary["missing_observation_dates"]}. Duplicated battery and date among rows with a date: {summary["duplicate_battery_date_non_null"]}. Missing dates are not treated as same-day measurements.
- Non-empty dates that failed to parse: {sum(summary["date_parse_errors"].values())}.
- Observation before manufacture: {summary["observation_before_manufacture"]}. Service after observation: {summary["service_after_observation"]}. Service before manufacture: {summary["service_before_manufacture"]}.
- Age inconsistent with the date difference: {summary["age_date_mismatches"]} of {summary["age_check_eligible"]:,} checkable rows.
- Observations per battery: {int(summary["observations_per_battery"]["min"])} to {int(summary["observations_per_battery"]["max"])}, mean {summary["observations_per_battery"]["mean"]:.2f}. The source README says roughly 10 to 40. The audited numbers are used.
- No conflicting non-missing values were found in the static attributes checked (chemistry, manufacturer, model, nominal capacity, manufacture date, climate zone). This does not prove that every column is constant per battery.

## Missing values

This table is a quality audit of the whole CSV, not imputer statistics. Imputation statistics are computed from train only.

{markdown_table(missing.round(3))}

`last_service_date` is not always missing because of data corruption: {summary["service_missing_count_zero"]:,} rows with a missing service date have `service_count = 0`, {summary["service_missing_count_missing"]:,} also have a missing `service_count`, and {summary["service_missing_count_positive"]:,} have `service_count > 0`. These dates must not all be imputed as real service dates, so the column is excluded from the baseline.

## Target leakage and sensor proxies

Across the {summary["fade_identity_eligible"]:,} rows that have `capacity_fade_pct`, **capacity_fade_pct + SOH = 100** with a maximum absolute deviation of {summary["fade_identity_max_abs_error"]}. This is a direct identity, so `capacity_fade_pct` is never used as an input.

The three alternative targets (`remaining_useful_life_cycles`, `failure_risk_label`, `degradation_category`) are also excluded. The source README states that the electrical measurements are derived from SOH and chemistry, naming internal resistance explicitly. Internal resistance, voltage, current and charge time are therefore conservatively excluded from the baseline and kept only as candidates for a separate sensor ablation. A real measured sensor can be valid for current-state estimation, but in this synthetic dataset such columns may act as target proxies.

Feature roles and reasons are stored in `../../configs/feature_contract.json` and `schema_audit.csv`. The main features cover chemistry and metadata, age and usage, and environment. Availability of every summary at observation time is an assumption, because the source does not document aggregation windows.

## Temporal feasibility

{markdown_table(temporal)}

The check compares consecutive valid measurements of the same battery after dropping missing dates or counter values, and only where the date increases. Pairs may skip missing observations. `cycle_count` and throughput are defined as cumulative yet frequently decrease. This is treated as an inconsistency of the synthetic data rather than repaired by sorting, cumulative maxima or mass deletion.

**Decision:** a tabular baseline without timesteps. Forecasting and windowing are deferred. The group split tests generalisation to unseen batteries, not to future time periods.

## Initial train profile

Train rows: {len(train):,}. Train SOH: min {train[TARGET].min():.2f}, median {train[TARGET].median():.2f}, mean {train[TARGET].mean():.2f}, max {train[TARGET].max():.2f} percent.

{markdown_table(bands.round(3))}

The target is unevenly distributed. Because the target is continuous, SMOTE or class balancing is not applied. Final evaluation must report errors per SOH band with sample counts instead of relying on aggregate MAE alone.

![Missing values in train](01_missing_train.png)

Missingness affects both numeric and categorical features. Median imputation and an explicit missing category are reasonable baselines. No imputer or scaler is fitted in this audit.

![Target and chemistry](02_target_chemistry_train.png)

Chemistry shares are counted per observation, not per battery, and observations of the same battery are correlated. Row shares must not be read as real fleet prevalence.

![Feature and target relationships](03_relationships_train.png)

The plots use a deterministic sample of at most 5,000 train rows and skip missing pairs. They show observational relationships inside a simulator, not causal effects. Internal resistance is shown to discuss proxy risk. It is not part of the baseline.

![Sensor ranges](04_sensor_ranges_train.png)

Points beyond the whiskers are candidates for inspection, not confirmed errors. `outlier_flags_train.csv` applies the 1.5 x IQR rule on train without removing or clipping values. The source mentions injected sensor outliers but provides no mask of affected rows.

## Next stage

The deeper train-only EDA is in `../eda/EDA_REPORT.md`. Preprocessing, the MLP study and the final evaluation follow in later stages. This report contains the audit and an initial profile only, **not the full EDA, final preprocessing or any prediction results**.
"""
    (output / "AUDIT_REPORT.md").write_text(report, encoding="utf-8", newline="\n")


def run_audit(root: Path = ROOT) -> dict:
    frame, dictionary, sha = load_dataset(root)
    contract = feature_contract(frame, sha)
    manifest, row_split, counts = build_split(frame)
    schema, summary = audit_quality(frame, dictionary, contract)
    train = frame.loc[row_split.eq("train")].copy()
    stats, outliers, bands = profile_train(train, contract)
    output = root / "results/audit"
    output.mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(exist_ok=True)
    (root / "data/splits").mkdir(exist_ok=True)
    manifest_path = root / "data/splits/battery_split.csv"
    if manifest_path.exists():
        old = pd.read_csv(manifest_path)
        if not old.equals(manifest):
            raise ValueError(
                "A different split manifest already exists, refusing to overwrite the locked split."
            )
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    for name, table in {
        "schema_audit": schema,
        "split_summary": counts,
        "numeric_train": stats,
        "outlier_flags_train": outliers,
        "target_bands_train": bands,
    }.items():
        table.to_csv(output / f"{name}.csv", index=False, lineterminator="\n")
    for path, value in [
        (root / "configs/feature_contract.json", contract),
        (output / "quality_summary.json", summary),
    ]:
        path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    provenance = {
        "source_url": SOURCE,
        "dataset_sha256": sha,
        "source_documentation_sha256": {
            name: hashlib.sha256((root / "data/raw" / name).read_bytes()).hexdigest()
            for name in ["README.md", "data_dictionary .csv"]
        },
        "split_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "packages": {
            p: version(p) for p in ["numpy", "pandas", "matplotlib", "nbformat", "nbclient", "ipykernel"]
        },
        "algorithm": "numpy.random.default_rng(42).permutation(sorted(unique battery_id)), first 70%, next 15%, remainder test",
        "scope": "Global integrity checks, train-only exploratory profiles/plots. No preprocessing fit or modeling.",
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    make_figures(train, contract, output)
    write_report(summary, schema, contract, counts, train, bands, outliers, output)
    return {
        "frame": frame,
        "dictionary": dictionary,
        "contract": contract,
        "manifest": manifest,
        "row_split": row_split,
        "split_summary": counts,
        "schema": schema,
        "quality": summary,
        "train": train,
        "statistics_train": stats,
        "outliers_train": outliers,
        "bands_train": bands,
        "output": output,
    }


if __name__ == "__main__":
    result = run_audit()
    print(result["split_summary"].to_string(index=False))
    print("Primary raw features:", len(result["contract"]["primary_features"]))
    print("Report:", result["output"] / "AUDIT_REPORT.md")
