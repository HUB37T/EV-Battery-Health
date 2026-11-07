"""Train-only EDA. No predictive models, feature fitting, or holdout summaries.

Run: python -m ev_battery.eda
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
from matplotlib.patches import Patch

from ev_battery.audit import PROXY_COLUMNS, TARGET, load_dataset, markdown_table
from ev_battery.paths import ROOT

SEED = 42
BOOTSTRAPS = 1000
MIN_BATTERIES = 30
CHEMISTRIES = ["LFP", "NMC", "NCA", "LMO"]
COLORS = dict(zip(CHEMISTRIES, ["#007f78", "#3674b5", "#c47a21", "#a4548e"]))
AGE_EDGES = [0, 365, 730, 1095, 1460, 1825, np.inf]
AGE_LABELS = ["≤1 year", "1-2 years", "2-3 years", "3-4 years", "4-5 years", ">5 years"]
LABELS = {
    "pack_capacity_kwh": "Nominal capacity (kWh)",
    "battery_age_days": "Battery age (days)",
    "cycle_count": "Equivalent cycles",
    "avg_daily_distance_km": "Daily distance (km)",
    "avg_charging_power_kw": "Average charging power (kW)",
    "fast_charge_ratio": "Fast-charge ratio (0 to 1)",
    "avg_depth_of_discharge_pct": "Depth of discharge (%)",
    "avg_state_of_charge_pct": "Average state of charge (%)",
    "charge_cycles_per_month": "Charge cycles per month",
    "total_energy_throughput_kwh": "Cumulative energy throughput (kWh)",
    "avg_ambient_temp_c": "Ambient temperature (°C)",
    "humidity_pct": "Humidity (%)",
    "altitude_m": "Altitude (m)",
    "peak_temp_during_fast_charge_c": "Peak fast-charge temperature (°C)",
    "service_count": "Service count",
    "internal_resistance_mohm": "Internal resistance (mΩ)",
    "voltage_avg_v": "Average voltage (V)",
    "current_avg_a": "Average current (A)",
    "charge_time_avg_min": "Charge duration (min)",
    "recall_flag": "Recall flag (0/1)",
    TARGET: "State of Health (%)",
    "capacity_fade_pct": "Capacity fade (%), excluded",
    "remaining_useful_life_cycles": "Remaining cycles, alternative target",
}


def load_train(root: Path = ROOT):
    frame, dictionary, sha = load_dataset(root)
    manifest_path = root / "data/splits/battery_split.csv"
    provenance = json.loads((root / "results/audit/provenance.json").read_text(encoding="utf-8"))
    contract = json.loads((root / "configs/feature_contract.json").read_text(encoding="utf-8"))
    assert sha == contract["dataset_sha256"] == provenance["dataset_sha256"]
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == provenance["split_manifest_sha256"]
    manifest = pd.read_csv(manifest_path)
    assert manifest.battery_id.is_unique and set(manifest.split) == {"train", "validation", "test"}
    train_ids = manifest.loc[manifest.split.eq("train"), "battery_id"]
    train = frame.loc[frame.battery_id.isin(train_ids)].copy()
    assert len(train) == 34929 and train.battery_id.nunique() == 1820
    # Discard the full CSV here, downstream analysis receives train only.
    return train, dictionary, contract, provenance


def weighted_quantile(values, weights, q):
    order = np.argsort(np.asarray(values), kind="stable")
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    return float(values[np.searchsorted(np.cumsum(weights), q * weights.sum())])


def safe_corr(x, y, method="pearson"):
    """Undefined correlations stay missing instead of becoming a misleading zero."""
    paired = pd.concat([x, y], axis=1).dropna()
    if len(paired) < 2 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
        return np.nan
    return paired.iloc[:, 0].corr(paired.iloc[:, 1], method=method)


def safe_median(series):
    values = series.dropna()
    return float(values.median()) if len(values) else np.nan


def prepare_metadata(train):
    columns = ["chemistry", "manufacturer", "vehicle_model", "climate_zone", "pack_capacity_kwh"]
    assert train.groupby("battery_id")[columns].nunique().le(1).all().all()
    metadata = train.groupby("battery_id")[columns].first()
    out = train.copy()
    # EDA-only grouping label, never overwrite the raw feature or export imputed X.
    out["chemistry_group"] = out.battery_id.map(metadata.chemistry)
    out["age_band"] = pd.cut(out.battery_age_days, AGE_EDGES, labels=AGE_LABELS, right=True)
    return out, metadata


def correlations(train, columns):
    rows = []
    for c in columns:
        sub = train[["battery_id", c, TARGET]].dropna()
        between = sub.groupby("battery_id")[[c, TARGET]].mean()
        centered = sub[[c, TARGET]] - sub.groupby("battery_id")[[c, TARGET]].transform("mean")
        within_r = centered[c].corr(centered[TARGET]) if centered[c].std() > 1e-10 else np.nan
        rows.append(
            {
                "feature": c,
                "pairwise_rows": len(sub),
                "batteries": sub.battery_id.nunique(),
                "pearson_pooled": safe_corr(sub[c], sub[TARGET]),
                "spearman_pooled": safe_corr(sub[c], sub[TARGET], method="spearman"),
                "pearson_between_battery_means": safe_corr(between[c], between[TARGET]),
                "pearson_within_battery_centered": within_r,
            }
        )
    return pd.DataFrame(rows)


def make_tables(train, dictionary, contract):
    d, metadata = prepare_metadata(train)
    numeric = contract["numeric_features"] + ["recall_flag"] + PROXY_COLUMNS
    tables = {}
    tables["numeric_distributions"] = (
        d[numeric + [TARGET]]
        .describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        .T.rename_axis("feature")
        .reset_index()
    )
    rows = []
    for c in numeric + [TARGET]:
        x = d[c].dropna()
        q1, q3 = x.quantile([0.25, 0.75])
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        rows.append(
            {
                "feature": c,
                "n": len(x),
                "skewness": x.skew(),
                "iqr_low": low,
                "iqr_high": high,
                "iqr_flag_n": int((~x.between(low, high)).sum()),
                "iqr_flag_pct": (~x.between(low, high)).mean() * 100,
                "median": x.median(),
                "p99": x.quantile(0.99),
                "max": x.max(),
            }
        )
    tables["tails_and_iqr"] = pd.DataFrame(rows)
    categories = []
    for c in contract["categorical_features"] + ["recall_flag", "failure_risk_label", "degradation_category"]:
        tmp = d.assign(group=d[c].astype("string").fillna("Missing"))
        tab = tmp.groupby("group").agg(
            rows=(TARGET, "size"),
            batteries=("battery_id", "nunique"),
            median_soh=(TARGET, "median"),
            mean_soh=(TARGET, "mean"),
            median_age_days=("battery_age_days", "median"),
            median_cycles=("cycle_count", "median"),
        )
        tab["row_pct"] = tab.rows / len(d) * 100
        tab = tab.reset_index().rename(columns={"group": "category"})
        tab.insert(0, "feature", c)
        categories.append(tab)
    tables["categorical_profiles"] = pd.concat(categories, ignore_index=True)

    pair_columns = numeric + [TARGET]
    tables["spearman_matrix"] = d[pair_columns].corr(method="spearman")
    mask = d[pair_columns].notna().astype("int64")
    tables["correlation_pair_counts"] = mask.T.dot(mask)
    assoc = correlations(d, numeric)
    assoc["role"] = np.where(assoc.feature.isin(PROXY_COLUMNS), "proxy_not_baseline", "primary")
    tables["target_associations"] = assoc
    base_columns = contract["numeric_features"] + ["recall_flag"]
    matrix = d[base_columns].corr(method="spearman")
    pairs = []
    for i, c in enumerate(base_columns):
        for other in base_columns[i + 1 :]:
            rho = matrix.loc[c, other]
            if abs(rho) >= 0.70:
                pairs.append(
                    {
                        "feature_a": c,
                        "feature_b": other,
                        "spearman": rho,
                        "pairwise_rows": int(d[[c, other]].notna().all(axis=1).sum()),
                    }
                )
    tables["redundant_pairs"] = pd.DataFrame(pairs).sort_values("spearman", key=abs, ascending=False)

    # One contribution per battery in each age band. No observation-level CI.
    battery_age = (
        d.dropna(subset=["age_band", "chemistry_group"])
        .groupby(["battery_id", "chemistry_group", "age_band"], observed=True)[TARGET]
        .mean()
        .reset_index(name="battery_band_mean_soh")
    )
    rng = np.random.default_rng(SEED)
    curves = []
    for (chem, age), group in battery_age.groupby(["chemistry_group", "age_band"], observed=True):
        values = group.battery_band_mean_soh.to_numpy()
        boot = values[rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))].mean(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        rows_n = int((d.chemistry_group.eq(chem) & d.age_band.eq(age)).sum())
        curves.append(
            {
                "chemistry": chem,
                "age_band": str(age),
                "rows": rows_n,
                "batteries": len(values),
                "battery_balanced_mean_soh": values.mean(),
                "ci95_low": lo,
                "ci95_high": hi,
                "support_at_least_30_batteries": len(values) >= MIN_BATTERIES,
            }
        )
    tables["chemistry_age_profiles"] = pd.DataFrame(curves)
    tables["battery_age_means"] = battery_age
    battery = d.groupby("battery_id").agg(
        observations=(TARGET, "size"),
        mean_soh=(TARGET, "mean"),
        median_soh=(TARGET, "median"),
        min_soh=(TARGET, "min"),
        max_age_days=("battery_age_days", "max"),
    )
    tables["battery_profiles"] = battery.join(metadata)

    # Descriptive associations within fixed chemistry/age strata, no p-values.
    stratified = []
    selected = [
        "fast_charge_ratio",
        "avg_depth_of_discharge_pct",
        "avg_ambient_temp_c",
        "charge_cycles_per_month",
        "service_count",
    ]
    for (chem, age), group in d.groupby(["chemistry_group", "age_band"], observed=True):
        for c in selected:
            valid = group[["battery_id", c, TARGET]].dropna()
            if len(valid) >= 100 and valid.battery_id.nunique() >= MIN_BATTERIES:
                stratified.append(
                    {
                        "chemistry": chem,
                        "age_band": str(age),
                        "feature": c,
                        "rows": len(valid),
                        "batteries": valid.battery_id.nunique(),
                        "spearman": safe_corr(valid[c], valid[TARGET], method="spearman"),
                    }
                )
    tables["stratified_associations"] = pd.DataFrame(stratified)

    missing_rows = []
    for c in train.columns:
        mask = train[c].isna()
        if not mask.any():
            continue
        missing_rows.append(
            {
                "feature": c,
                "missing_rows": int(mask.sum()),
                "missing_pct": mask.mean() * 100,
                "observed_rows": int((~mask).sum()),
                "median_soh_missing": train.loc[mask, TARGET].median(),
                "median_soh_observed": train.loc[~mask, TARGET].median(),
                "median_gap_missing_minus_observed": train.loc[mask, TARGET].median()
                - train.loc[~mask, TARGET].median(),
                "median_age_missing": safe_median(train.loc[mask, "battery_age_days"]),
                "median_age_observed": safe_median(train.loc[~mask, "battery_age_days"]),
            }
        )
    tables["missingness_target_profiles"] = pd.DataFrame(missing_rows)
    missing_columns = [c for c in train if train[c].isna().any()]
    phi = train[missing_columns].isna().astype(float).corr()
    tables["missingness_phi"] = phi
    missing_pairs = [
        {"feature_a": a, "feature_b": b, "phi": phi.loc[a, b]}
        for i, a in enumerate(missing_columns)
        for b in missing_columns[i + 1 :]
    ]
    tables["missingness_pairs"] = pd.DataFrame(missing_pairs).sort_values("phi", key=abs, ascending=False)

    ordered = (
        d.assign(date=pd.to_datetime(d.observation_date, errors="coerce"))
        .dropna(subset=["date"])
        .sort_values(["battery_id", "date", "observation_id"])
    )
    lag = ordered.groupby("battery_id")[
        ["date", TARGET, "cycle_count", "total_energy_throughput_kwh"]
    ].shift()
    temporal = ordered[["battery_id", "observation_id", "date", "chemistry_group"]].copy()
    temporal["gap_days"] = (ordered.date - lag.date).dt.days
    # Match recorded precision at boundaries: exactly 0.01 must not become >0.01.
    assert np.allclose(d[TARGET] * 100, (d[TARGET] * 100).round(), atol=1e-8, rtol=0)
    temporal["delta_soh_pp"] = (ordered[TARGET] - lag[TARGET]).round(2)
    temporal["delta_cycles"] = ordered.cycle_count - lag.cycle_count
    temporal["delta_energy_kwh"] = ordered.total_energy_throughput_kwh - lag.total_energy_throughput_kwh
    temporal = temporal.loc[temporal.gap_days.gt(0)]
    tables["temporal_pairs"] = temporal
    levels = [
        {"label": "down >0.01 pp", "rows": int(temporal.delta_soh_pp.lt(-0.01).sum())},
        {"label": "stable ±0.01 pp", "rows": int(temporal.delta_soh_pp.abs().le(0.01).sum())},
        {"label": "up >0.01 pp", "rows": int(temporal.delta_soh_pp.gt(0.01).sum())},
    ]
    tables["temporal_soh_changes"] = pd.DataFrame(levels)
    tables["temporal_soh_changes"]["pct_pairs"] = tables["temporal_soh_changes"].rows / len(temporal) * 100

    grid = d.dropna(subset=["age_band", "fast_charge_ratio"]).copy()
    grid["fast_band"] = pd.cut(
        grid.fast_charge_ratio, [-0.001, 0.2, 0.4, 0.6, 1], labels=["0–0.2", ">0.2–0.4", ">0.4–0.6", ">0.6–1"]
    )
    tables["age_fastcharge_grid"] = (
        grid.groupby(["age_band", "fast_band"], observed=True)
        .agg(median_soh=(TARGET, "median"), rows=(TARGET, "size"), batteries=("battery_id", "nunique"))
        .reset_index()
    )

    sensitivity = []
    for c in numeric:
        sub = d[[c, TARGET]].dropna()
        q1, q3 = sub[c].quantile([0.25, 0.75])
        keep = sub[c].between(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
        sensitivity.append(
            {
                "feature": c,
                "rows_all": len(sub),
                "rows_inside_iqr_fences": int(keep.sum()),
                "spearman_all": safe_corr(sub[c], sub[TARGET], method="spearman"),
                "spearman_inside_iqr_fences": safe_corr(
                    sub.loc[keep, c], sub.loc[keep, TARGET], method="spearman"
                ),
                "inside_feature_unique_values": sub.loc[keep, c].nunique(),
            }
        )
    tables["outlier_association_sensitivity"] = pd.DataFrame(sensitivity)

    # Cover every raw column, including IDs/dates/alternate targets without making them features.
    inventory = dictionary.copy()
    inventory["missing_train"] = inventory.column.map(train.isna().sum())
    inventory["unique_train_non_null"] = inventory.column.map(train.nunique())
    inventory["analysis_role"] = inventory.column.map(
        {
            **{c: "primary" for c in contract["primary_features"]},
            **{c: "excluded" for c in contract["excluded_features"]},
            TARGET: "target",
        }
    )
    tables["column_coverage"] = inventory
    return d, tables


def make_figures(d, tables, contract, output):
    figures = []
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )

    def save(fig, name):
        fig.savefig(output / name, dpi=140, bbox_inches="tight")
        plt.close(fig)
        figures.append(name)

    weights = 1 / d.groupby("battery_id")[TARGET].transform("size")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.arange(54, 102, 2)
    axes[0].hist(
        d[TARGET],
        bins=bins,
        weights=np.ones(len(d)) / len(d),
        alpha=0.65,
        color="#247e88",
        label="Equal weight per observation",
    )
    axes[0].hist(
        d[TARGET],
        bins=bins,
        weights=weights / weights.sum(),
        histtype="step",
        linewidth=2,
        color="#c47a21",
        label="Equal weight per battery",
    )
    axes[0].set(xlabel="SOH (%)", ylabel="Share per bin", title="Target: check for observation-count bias")
    axes[0].legend(fontsize=8)
    x = np.sort(d[TARGET])
    axes[1].plot(x, np.arange(1, len(x) + 1) / len(x), color="#247e88")
    axes[1].axvline(80, color="#bd623b", ls="--", label="80%: source end-of-life reference")
    axes[1].set(xlabel="SOH (%)", ylabel="Cumulative share", title="Train ECDF (per observation)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "01_target_and_weighting.png")

    numeric = contract["numeric_features"] + ["recall_flag"] + PROXY_COLUMNS
    for page, start in enumerate(range(0, len(numeric), 10), 1):
        columns = numeric[start : start + 10]
        fig, axes = plt.subplots(5, 2, figsize=(12, 15))
        for ax, c in zip(axes.flat, columns):
            bins = [-0.5, 0.5, 1.5] if c == "recall_flag" else 35
            ax.hist(d[c].dropna(), bins=bins, color="#247e88", edgecolor="white", linewidth=0.3)
            if c == "recall_flag":
                ax.set_xticks([0, 1])
            ax.set(xlabel=LABELS[c], ylabel="Observations")
            role = "proxy, not baseline" if c in PROXY_COLUMNS else "input candidate"
            ax.set_title(f"{role} · n={d[c].notna().sum():,}, missing={d[c].isna().mean():.1%}", fontsize=9)
        for ax in list(axes.flat)[len(columns) :]:
            ax.set_visible(False)
        fig.suptitle(f"All numeric candidates in train, part {page}", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        save(fig, f"02_numeric_distributions_{page}.png")

    cats = contract["categorical_features"] + ["recall_flag"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    profiles = tables["categorical_profiles"]
    for ax, c in zip(axes.flat, cats):
        tab = profiles.loc[profiles.feature.eq(c)].sort_values("rows")
        bars = ax.barh(
            tab.category,
            tab.row_pct,
            color=["#a9b4b9" if s == "Missing" else "#247e88" for s in tab.category],
        )
        for bar, n in zip(bars, tab.rows):
            inside = bar.get_width() > 75
            ax.text(
                bar.get_width() - 1 if inside else bar.get_width() + 0.6,
                bar.get_y() + bar.get_height() / 2,
                f"n={n:,}",
                va="center",
                ha="right" if inside else "left",
                color="white" if inside else "black",
                fontsize=8,
            )
        ax.set(title=c, xlabel="% of train observations", xlim=(0, min(100, max(tab.row_pct) * 1.32)))
    fig.suptitle("Category support and imbalance (Missing shown explicitly)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "03_categorical_support.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bychem = (
        d.groupby("chemistry_group")
        .agg(
            median=(TARGET, "median"),
            q25=(TARGET, lambda s: s.quantile(0.25)),
            q75=(TARGET, lambda s: s.quantile(0.75)),
            n=(TARGET, "size"),
        )
        .reindex(CHEMISTRIES)
    )
    for i, chem in enumerate(CHEMISTRIES):
        row = bychem.loc[chem]
        axes[0].errorbar(
            row["median"],
            i,
            xerr=[[row["median"] - row.q25], [row.q75 - row["median"]]],
            fmt="o",
            capsize=4,
            color=COLORS[chem],
        )
    axes[0].set(
        yticks=range(4),
        yticklabels=[f"{c} (n={int(bychem.loc[c, 'n']):,})" for c in CHEMISTRIES],
        xlabel="SOH (%)",
        title="Median and IQR per observation",
        xlim=(65, 100.5),
    )
    curves = tables["chemistry_age_profiles"]
    for chem in CHEMISTRIES:
        sub = curves[curves.chemistry.eq(chem)].set_index("age_band").reindex(AGE_LABELS)
        axes[1].plot(range(6), sub.battery_balanced_mean_soh, "o-", color=COLORS[chem], label=chem)
        axes[1].fill_between(range(6), sub.ci95_low, sub.ci95_high, color=COLORS[chem], alpha=0.12)
        small = ~sub.support_at_least_30_batteries.fillna(False).astype(bool)
        axes[1].scatter(
            np.arange(6)[small],
            sub.battery_balanced_mean_soh[small],
            marker="x",
            s=80,
            color="black",
            zorder=5,
        )
    axes[1].set(
        xticks=range(6),
        xticklabels=["≤1", "1-2", "2-3", "3-4", "4-5", ">5"],
        xlabel="Age band (years)",
        ylabel="Mean SOH, equal weight per battery (%)",
        title="Same age band, 95% battery bootstrap",
    )
    axes[1].legend(fontsize=8, ncol=4)
    axes[1].text(
        0.01, 0.02, "x = fewer than 30 batteries, weak evidence", transform=axes[1].transAxes, fontsize=8
    )
    fig.tight_layout()
    save(fig, "04_chemistry_and_age.png")

    support = curves.pivot(index="chemistry", columns="age_band", values="batteries").reindex(
        index=CHEMISTRIES, columns=AGE_LABELS
    )
    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(support, cmap="Blues", aspect="auto")
    for (r, c), value in np.ndenumerate(support.to_numpy()):
        ax.text(
            c,
            r,
            str(int(value)),
            ha="center",
            va="center",
            color="white" if value > support.max().max() * 0.65 else "#20343c",
        )
    ax.set(
        xticks=range(6),
        xticklabels=AGE_LABELS,
        yticks=range(4),
        yticklabels=CHEMISTRIES,
        title="Supporting batteries per chemistry and age band",
    )
    fig.colorbar(im, ax=ax, label="Unique batteries in cell")
    fig.tight_layout()
    save(fig, "05_chemistry_age_support.png")

    columns = contract["numeric_features"] + ["recall_flag", TARGET]
    corr = tables["spearman_matrix"].loc[columns, columns]
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    labels = [LABELS[c] for c in columns]
    ax.set(
        xticks=range(len(columns)),
        xticklabels=labels,
        yticks=range(len(columns)),
        yticklabels=labels,
        title="Spearman correlation of baseline features and SOH (pairwise complete, train)",
    )
    plt.setp(ax.get_xticklabels(), rotation=65, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    for (r, c), value in np.ndenumerate(corr.to_numpy()):
        if r != c and abs(value) >= 0.70:
            ax.text(
                c,
                r,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(value) > 0.75 else "black",
            )
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman ρ (annotated where |ρ| ≥ 0.70)")
    fig.tight_layout()
    save(fig, "06_baseline_correlations.png")

    assoc = tables["target_associations"].sort_values("spearman_pooled")
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    axes[0].barh(
        [LABELS[c] for c in assoc.feature],
        assoc.spearman_pooled,
        color=["#c47a21" if c in PROXY_COLUMNS else "#247e88" for c in assoc.feature],
    )
    axes[0].axvline(0, color="gray", lw=0.8)
    axes[0].set(
        xlim=(-1, 1),
        xlabel="Spearman ρ (non-missing pairs)",
        title="Target association is not a selection decision",
    )
    axes[0].legend(
        handles=[
            Patch(color="#247e88", label="Baseline candidate"),
            Patch(color="#c47a21", label="Proxy, not baseline"),
        ],
        loc="upper left",
        bbox_to_anchor=(0, -0.10),
        fontsize=8,
        ncol=2,
    )
    comp = assoc.loc[
        assoc.feature.isin(
            [
                "battery_age_days",
                "cycle_count",
                "total_energy_throughput_kwh",
                "service_count",
                "fast_charge_ratio",
                "avg_ambient_temp_c",
            ]
        )
    ].copy()
    y = np.arange(len(comp))
    for offset, field, label, color in [
        (-0.22, "pearson_pooled", "All rows", "#247e88"),
        (0, "pearson_between_battery_means", "Battery means", "#c47a21"),
        (0.22, "pearson_within_battery_centered", "Centered within battery", "#926098"),
    ]:
        axes[1].barh(y + offset, comp[field], height=0.2, label=label, color=color)
    axes[1].set(
        yticks=y,
        yticklabels=[LABELS[c] for c in comp.feature],
        xlim=(-1, 1),
        xlabel="Pearson r (not the Spearman values on the left)",
        title="Between-battery vs within-battery variation",
    )
    axes[1].axvline(0, color="gray", lw=0.8)
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.10), ncol=2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "07_target_associations_and_dependence.png")

    sample = d.sample(min(6000, len(d)), random_state=SEED)
    xcols = [
        "battery_age_days",
        "cycle_count",
        "total_energy_throughput_kwh",
        "fast_charge_ratio",
        "avg_depth_of_discharge_pct",
        "avg_ambient_temp_c",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, c in zip(axes.flat, xcols):
        for chem in CHEMISTRIES:
            sub = sample[sample.chemistry_group.eq(chem)]
            x = sub[c] / 1000 if c == "total_energy_throughput_kwh" else sub[c]
            ax.scatter(x, sub[TARGET], s=4, alpha=0.16, color=COLORS[chem], rasterized=True)
        ax.set(
            xlabel="Cumulative energy (thousand kWh)" if c == "total_energy_throughput_kwh" else LABELS[c],
            ylabel="SOH (%)",
        )
    fig.legend(
        handles=[Patch(color=COLORS[c], label=c) for c in CHEMISTRIES],
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1),
    )
    fig.suptitle("Numeric relationships by chemistry (fixed sample of 6,000 train rows)", y=1.03)
    fig.tight_layout()
    save(fig, "08_relationships_by_chemistry.png")

    grid = tables["age_fastcharge_grid"]
    order = ["0–0.2", ">0.2–0.4", ">0.4–0.6", ">0.6–1"]
    med = grid.pivot(index="age_band", columns="fast_band", values="median_soh").reindex(
        index=AGE_LABELS, columns=order
    )
    count = grid.pivot(index="age_band", columns="fast_band", values="rows").reindex(
        index=AGE_LABELS, columns=order
    )
    bats = grid.pivot(index="age_band", columns="fast_band", values="batteries").reindex(
        index=AGE_LABELS, columns=order
    )
    supported = count.ge(100) & bats.ge(30)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(med.where(supported), cmap="YlGnBu", vmin=65, vmax=100, aspect="auto")
    for (r, c), value in np.ndenumerate(med.to_numpy()):
        ok = bool(supported.iloc[r, c])
        label = f"{value:.1f}%\nn={int(count.iloc[r, c]):,}" if ok else "low support"
        ax.text(
            c, r, label, ha="center", va="center", fontsize=9, color="white" if ok and value > 90 else "black"
        )
    ax.set(
        xticks=range(4),
        xticklabels=order,
        yticks=range(6),
        yticklabels=AGE_LABELS,
        xlabel="Fast-charge ratio",
        title="Median SOH by age and fast-charge ratio\nCells shown with ≥100 rows and ≥30 batteries, chemistry not balanced",
    )
    fig.colorbar(im, ax=ax, label="Median SOH (%)")
    fig.tight_layout()
    save(fig, "09_age_fastcharge_conditioning.png")

    cats = ["manufacturer", "vehicle_model", "climate_zone", "bms_firmware_version"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, c in zip(axes.flat, cats):
        tab = profiles.loc[profiles.feature.eq(c) & profiles.category.ne("Missing")].sort_values("median_soh")
        ax.barh(tab.category, tab.median_soh, color="#247e88")
        for i, (_, row) in enumerate(tab.iterrows()):
            ax.text(
                96.2,
                i,
                f"n={int(row.rows):,}, median age {row.median_age_days:.0f} d",
                fontsize=8,
                va="center",
            )
        ax.set(xlim=(90, 106), xlabel="Median SOH (%), axis starts at 90", title=c)
    fig.suptitle("Raw category comparison (check support and age differences)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "10_category_comparisons.png")

    missing = tables["missingness_target_profiles"]
    primary_missing = missing[missing.feature.isin(contract["primary_features"])].sort_values(
        "median_gap_missing_minus_observed"
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    axes[0].barh(primary_missing.feature, primary_missing.median_gap_missing_minus_observed, color="#247e88")
    axes[0].axvline(0, color="gray", lw=0.8)
    axes[0].set(
        xlabel="Median SOH, missing minus observed (pp)",
        title="Main-feature missingness vs SOH (descriptive)",
    )
    missing_count = d[contract["primary_features"]].isna().sum(axis=1)
    freq = missing_count.value_counts().sort_index()
    axes[1].bar(freq.index, freq.values, color="#3674b5")
    axes[1].set(
        xlabel="Missing main features in one observation",
        ylabel="Observations",
        title="Missing values per row (21 main features)",
    )
    fig.tight_layout()
    save(fig, "11_missingness_diagnostics.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    service = d.last_service_date.isna()
    for ax, c, label in [(axes[0], TARGET, "SOH (%)"), (axes[1], "battery_age_days", "Age (days)")]:
        groups = [d.loc[service, c].dropna(), d.loc[~service, c].dropna()]
        ax.boxplot(groups, tick_labels=["Service date missing", "Service date observed"], showfliers=False)
        ax.set(ylabel=label, title="IQR and whiskers (outliers hidden in plot only)")
    fig.suptitle("Missing service dates track usage stage, not a causal service effect")
    fig.tight_layout()
    save(fig, "12_service_missingness_context.png")

    tails = tables["tails_and_iqr"].sort_values("iqr_flag_pct")
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    axes[0].barh([LABELS[c] for c in tails.feature], tails.iqr_flag_pct, color="#c47a21")
    axes[0].set(
        xlabel="% of observed values outside 1.5×IQR fences", title="Outlier candidates, not error labels"
    )
    x = d.charge_time_avg_min.dropna()
    axes[1].hist(x, bins=np.geomspace(x.min(), x.max(), 40), color="#247e88", edgecolor="white")
    axes[1].set_xscale("log")
    axes[1].set(
        xlabel="Charge duration (min, log axis)",
        ylabel="Observations",
        title="Long duration tail (data unchanged)",
    )
    fig.tight_layout()
    save(fig, "13_outliers_and_tail.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    dated = d.assign(date=pd.to_datetime(d.observation_date)).dropna(subset=["date"])
    for ax, chem in zip(axes.flat, CHEMISTRIES):
        sub = dated[dated.chemistry_group.eq(chem)]
        ids = sub.groupby("battery_id").size().loc[lambda x: x.ge(10)].sort_index().index[:3]
        for bid in ids:
            trajectory = sub[sub.battery_id.eq(bid)].sort_values("date")
            ax.plot(trajectory.date, trajectory[TARGET], "o-", ms=3, lw=1, label=bid)
        ax.set(title=f"{chem}: first three IDs with ≥10 valid dates", ylabel="SOH (%)", ylim=(50, 102))
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Example train trajectories (IDs chosen without using the target)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "14_battery_trajectories.png")

    pairs = tables["temporal_pairs"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].hist(pairs.gap_days, bins=40, color="#247e88", edgecolor="white")
    axes[0].set(xlabel="Gap between valid observations (days)", ylabel="Pairs", title="Irregular sampling")
    axes[1].hist(pairs.delta_soh_pp, bins=65, color="#3674b5", edgecolor="white")
    axes[1].axvline(0, color="#a4548e", ls="--")
    axes[1].set(xlabel="SOH change (pp)", title="SOH can increase between observations")
    counts = dated.date.dt.to_period("M").value_counts().sort_index()
    axes[2].plot(counts.index.to_timestamp(), counts.values, color="#247e88")
    axes[2].set(xlabel="Observation month", ylabel="Dated observations", title="Train calendar coverage")
    axes[2].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    save(fig, "15_temporal_quality.png")

    battery = tables["battery_profiles"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(battery.observations, bins=np.arange(6.5, 38.5), color="#247e88", edgecolor="white")
    axes[0].set(
        xlabel="Observations per battery", ylabel="Batteries", title="Independent units: 1,820 batteries"
    )
    axes[1].scatter(battery.observations, battery.mean_soh, s=9, alpha=0.3, color="#247e88")
    axes[1].set(
        xlabel="Observations per battery",
        ylabel="Battery mean SOH (%)",
        title="Do batteries with more rows differ?",
    )
    fig.tight_layout()
    save(fig, "16_battery_observation_structure.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, c in zip(axes[:2], ["failure_risk_label", "degradation_category"]):
        tab = profiles[profiles.feature.eq(c)].sort_values("rows", ascending=False)
        ax.bar(tab.category, tab.row_pct, color="#a4548e")
        ax.set(title=c, ylabel="% of train observations")
        ax.tick_params(axis="x", rotation=20)
        for i, row in enumerate(tab.itertuples()):
            ax.text(i, row.row_pct + 1, f"{row.row_pct:.1f}%", ha="center", fontsize=8)
        ax.set_ylim(0, tab.row_pct.max() + 12)
    axes[2].scatter(
        sample[TARGET], sample.remaining_useful_life_cycles, s=4, alpha=0.15, color="#a4548e", rasterized=True
    )
    axes[2].set(
        xlabel="SOH (%)",
        ylabel="Remaining useful life (cycles)",
        title="Alternative target, not an SOH input",
    )
    fig.suptitle("Alternative targets explored for context, the regression task is unchanged")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "17_alternative_targets.png")
    return figures


def write_report(d, tables, contract, figures, output):
    primary = contract["primary_features"]
    weights = 1 / d.groupby("battery_id")[TARGET].transform("size")
    assoc = tables["target_associations"].set_index("feature")
    missing = tables["missingness_target_profiles"]
    service = missing.set_index("feature").loc["last_service_date"]
    curve = tables["chemistry_age_profiles"]
    one_to_two = curve[curve.age_band.eq(AGE_LABELS[1])][
        ["chemistry", "rows", "batteries", "battery_balanced_mean_soh", "ci95_low", "ci95_high"]
    ]
    temporal = tables["temporal_pairs"]
    increments = temporal.delta_soh_pp.gt(0.01)
    rho_n = tables["battery_profiles"].observations.corr(
        tables["battery_profiles"].mean_soh, method="spearman"
    )
    strongest = (
        tables["target_associations"]
        .sort_values("spearman_pooled", key=abs, ascending=False)[
            ["feature", "role", "pairwise_rows", "batteries", "pearson_pooled", "spearman_pooled"]
        ]
        .head(10)
    )
    redundancy = tables["redundant_pairs"]
    missing_small = missing[missing.feature.isin(primary)]
    sparse = curve[~curve.support_at_least_30_batteries][["chemistry", "age_band", "rows", "batteries"]]
    count_matrix = tables["correlation_pair_counts"]
    pair_values = count_matrix.to_numpy()[~np.eye(len(count_matrix), dtype=bool)]
    report = f"""# Exploratory Data Analysis (Train Only)

## Summary

1. The analysis uses **34,929 observations from 1,820 train batteries**. Validation and test are not summarised, plotted or used to choose patterns.
2. SOH is skewed towards healthy batteries: median **{d[TARGET].median():.2f}%**, mean **{d[TARGET].mean():.2f}%**, **{d[TARGET].ge(95).mean() * 100:.2f}%** of observations at or above 95% and **{d[TARGET].lt(80).mean() * 100:.2f}%** below 80%. Errors on low SOH must be reported separately.
3. Cycle count has the strongest association with SOH among the main features (Spearman **{assoc.loc["cycle_count", "spearman_pooled"]:.3f}**). Age and cumulative energy are also strong. The three are strongly related to each other, so they are not three independent pieces of evidence.
4. Chemistries differ in their SOH distribution. Within the same age band, with equal weight per battery, the differences remain, but usage conditions are not fully controlled. This is a simulator pattern, not a causal effect of chemistry in a real fleet.
5. For the main features, the median SOH gap between missing and observed rows ranges from **{missing_small.median_gap_missing_minus_observed.min():.2f} to {missing_small.median_gap_missing_minus_observed.max():.2f} percentage points**. This does not prove MCAR. Missing service dates differ strongly and are linked to the usage stage.
6. Across **{len(temporal):,}** consecutive pairs with valid dates, SOH increases by more than 0.01 points in **{increments.sum():,} ({increments.mean() * 100:.2f}%)** pairs. Intervals are irregular and cumulative counters are inconsistent, so timesteps are not built and SOH increases are not interpreted as physical recovery.
7. No preprocessing is fitted, no final feature selection is made and no model is trained here. The raw data and the split manifest are unchanged.

## 1. Questions, source and units

Goal: describe distributions, relationships, data quality and observation structure before designing preprocessing and the MLP experiments.

Source: [Kaggle EV Battery Health & Degradation](https://www.kaggle.com/datasets/vinay24baghira/ev-battery-health-and-degradation-dataset). The README and data dictionary in the archive describe the data as **fully synthetic** (CC0), with injected missing values and outliers, and electrical measurements related to SOH and chemistry. The generator and the injection masks are not available.

The CSV and manifest hashes are recorded in `eda_provenance.json`. The CSV is read only to select train rows from the manifest. Every EDA function receives train only. Data is never combined across subsets to recover missing values.

Two summary units are used:
- **Observation:** one measurement row. Batteries with more rows carry more weight.
- **Battery:** the grouping unit. For the chemistry and age curves, each battery contributes one mean SOH per age band, which is then averaged across batteries.

`chemistry_group` is an EDA-only label: the single non-missing chemistry value that is consistent across the battery's **train** observations. The raw chemistry column stays unchanged and its missingness is still counted. It is not an imputer for model inputs. Raw category comparisons use the original row values, including Missing.

No p-values are computed, because the 34,929 rows are not independent. Correlations and group differences are descriptive. No causal claims or battery safety recommendations are drawn from a simulator.

## 2. Target distribution and units of analysis

- Train SOH: {d[TARGET].min():.2f}% to {d[TARGET].max():.2f}%, median {d[TARGET].median():.2f}%, mean {d[TARGET].mean():.2f}%.
- Exactly 100%: {d[TARGET].eq(100).sum():,} rows. Exactly 55%: {d[TARGET].eq(55).sum():,} rows. Pile-up at the observed bounds is consistent with clipping or rounding in the simulator, but the generator is not available to confirm it.
- Median with equal weight per battery: {weighted_quantile(d[TARGET], weights, 0.5):.2f}%. Mean with equal weight per battery: {np.average(d[TARGET], weights=weights):.2f}%.
- Spearman correlation between observations per battery and battery mean SOH: {rho_n:.3f}. This checks for bias from the number of observations. It does not prove independent sampling.

![Target and weighting](01_target_and_weighting.png)

Both histograms use the same bins. The orange line weights each row by 1 / (rows of its battery). The ECDF shows the share of observations at or below each SOH value. The 80% line is the end-of-life reference from the dataset documentation, not a class threshold introduced here.

![Observation structure](16_battery_observation_structure.png)

**Implication:** use MAE and RMSE on the original target scale, not a regression "accuracy". Final evaluation should include SOH bands and per-battery summaries. The nominal sample size is not the number of independent batteries.

## 3. All numeric and categorical features

![Numeric features, part 1](02_numeric_distributions_1.png)

![Numeric features, part 2](02_numeric_distributions_2.png)

All 15 numeric candidates, `recall_flag` and the four sensor proxy candidates are covered. Every panel shows observed and missing counts. Histograms keep outliers. Minimum, p1, p5, quartiles, p95, p99, maximum and skewness are in `numeric_distributions.csv` and `tails_and_iqr.csv`.

![Category support](03_categorical_support.png)

The Missing category is not a new battery type. Firmware and recall status can change between observations and should not be treated as static attributes. Batteries in changing categories can overlap, so their counts must not be summed as if categories were mutually exclusive per battery. `recall_flag = 1` is rare. Row counts are shown so that small groups are not over-interpreted.

**Implication:** numeric imputation, encoding of unseen categories, numeric scaling and binary handling are designed in preprocessing. Weakly correlated features and rare categories are not removed on the basis of EDA alone.

## 4. Chemistry and age: a fairer comparison

![Chemistry and age](04_chemistry_and_age.png)

The left panel shows observation-level median and IQR, not confidence intervals. The right panel groups age into fixed yearly bands, averages SOH per battery within each band, and gives each battery equal weight.

Example for the **{AGE_LABELS[1]}** band:

{markdown_table(one_to_two.round(3))}

Bands in the right panel are 95% percentile bootstrap intervals from **{BOOTSTRAPS:,} battery resamples within each chemistry and age cell** (seed 42). They are pointwise intervals for the cell mean, not for the whole curve or for differences between chemistries, and they do not address confounding. They are not used as significance tests. The same battery can appear in several age bands, so points along a curve are correlated.

![Chemistry and age support](05_chemistry_age_support.png)

Cells with fewer than {MIN_BATTERIES} batteries are marked with x on the curve. Low-support cells:

{markdown_table(sparse)}

**Conclusion:** chemistry and age should be considered together because the patterns differ between chemistries. These curves summarise observations and batteries across age bands. They are not controlled degradation curves of single batteries and do not show that any chemistry is always better.

## 5. Numeric relationships, nonlinearity and redundancy

![Baseline correlations](06_baseline_correlations.png)

Spearman measures rank-based monotonic association and does not assume a straight line. Pearson complements it for linear association. Correlations use pairwise-complete rows without imputation. Pair counts range from **{int(pair_values.min()):,} to {int(pair_values.max()):,}** in the stored matrix. See `correlation_pair_counts.csv`.

Strongest target associations among baseline and proxy candidates:

{markdown_table(strongest.round(3))}

![Associations and dependence](07_target_associations_and_dependence.png)

The right panel compares **Pearson** coefficients, not the Spearman values on the left: pooled over all observed pairs, between batteries using per-battery means, and within batteries after subtracting each battery's mean. Centering is not a causal model and does not control for age, chemistry or other variables. Features that are constant within a battery have no defined centered correlation. Those values are left empty.

Main-feature pairs with |Spearman| at or above 0.70 (an exploratory threshold, not a removal rule):

{markdown_table(redundancy.round(3))}

![Relationships by chemistry](08_relationships_by_chemistry.png)

The scatter plots use a deterministic sample of 6,000 train rows with no fitted lines. More cycles and greater age generally go with lower SOH, but the spread and the chemistry differences show that a single linear relationship does not describe all observations. This motivates testing an MLP. It does not prove that an MLP will score best.

**Implication:** overlapping usage features are compared in controlled experiments later. Weak pooled correlations for fast charging, depth of discharge and temperature do not prove that these features are useless or that the factors do not matter for real batteries. In particular, the train Spearman correlation between fast-charge ratio and SOH is only about -0.003. The source mentions a fast-charge penalty in the simulator, but without the generator it is impossible to check how observed features relate to latent parameters.

## 6. Conditional patterns: do not rely on pooled correlations

![Age and fast charging](09_age_fastcharge_conditioning.png)

The grid uses fixed age and ratio bands and shows median SOH with the number of observations. Only cells with at least 100 rows and 30 batteries are shown. Chemistry and other usage factors are not balanced. The grid helps separate age variation from fast-charge variation. It does not identify a causal effect of charging.

`stratified_associations.csv` computes Spearman correlations for fast-charge ratio, depth of discharge, temperature, charging frequency and `service_count` within chemistry by age cells with the same support rules. Age bands are still wide, so confounding within cells is possible. Cells are not pooled into a single "effect" and are not cherry-picked.

![Category comparisons](10_category_comparisons.png)

Manufacturer, model, climate and firmware categories are compared together with their support and median age. The SOH axis starts at 90%, marked explicitly, so the variation is visible. Small differences are not automatically important, and age differences can explain part of the raw gap.

## 7. Missingness

![Missingness diagnostics](11_missingness_diagnostics.png)

The missing versus observed median gap is descriptive only: missingness can be related to conditions, to the injection process or to group composition. Small gaps do not prove Missing Completely At Random. Correlations between missing indicators (`missingness_phi.csv`) and all ordered pairs (`missingness_pairs.csv`) are stored without changing the data. The maximum |phi| is about 0.017 in train, so no large linear co-missingness appears among the indicator pairs checked. Nonlinear or conditional dependence is not ruled out.

![Service date context](12_service_missingness_context.png)

Median SOH is **{service.median_soh_missing:.2f}%** when the service date is missing and **{service.median_soh_observed:.2f}%** when it is observed. Median ages are **{service.median_age_missing:.0f}** and **{service.median_age_observed:.0f}** days. This is consistent with different usage stages and is **not** evidence that servicing reduces health. Service dates are not imputed here.

## 8. Outliers and sensitivity

![Outliers and tails](13_outliers_and_tail.png)

The 1.5 x IQR fences only mark values that are extreme relative to the train distribution. Low target values and rare binary flags can also be flagged without being errors, so this rule is not used to remove outliers.

The charge-duration histogram uses a log axis to show the tail. The raw values are not log-transformed for the model at this stage. `outlier_association_sensitivity.csv` compares Spearman on all pairs with Spearman inside each feature's IQR fences. This is a sensitivity analysis, not a decision to drop rows, and the denominators are recorded. For `recall_flag` the fences keep only zeros, so the filtered correlation is undefined and left empty. For cycles, age and energy the strong negative relationship remains inside the fences, which does not prove that removing outliers would improve predictions. There is no official mask of injected outliers.

## 9. Temporal structure and simulator consistency

![Battery trajectories](14_battery_trajectories.png)

Examples are chosen deterministically: the first three IDs in alphabetical order per chemistry with at least 10 dated observations. SOH values are not used to choose them. These trajectories do not summarise the population, and missing dates are not invented or connected as known intervals.

![Temporal quality](15_temporal_quality.png)

{markdown_table(tables["temporal_soh_changes"].round(3))}

Differences are computed between dated observations of the same battery with increasing dates. Pairs may skip observations whose date is missing. SOH changes are rounded to the recorded precision of 0.01 points before classifying them as down, stable or up, so floating-point artefacts do not move values at the boundary. Because intervals vary, the delta-SOH histogram is **not a degradation rate per day**. SOH increases in some pairs. The source describes the data as synthetic and noisy, but the mechanism behind each increase cannot be confirmed without the generator. Decreasing cycle and energy counters were already identified in the audit.

**Implication:** keeping a tabular baseline and deferring timesteps is justified. Trajectories are not forced to be monotonic or sorted by the target. The group split tests new batteries, not new calendar periods.

## 10. Alternative targets: understood, not used as inputs

![Alternative targets](17_alternative_targets.png)

Risk and degradation labels help to understand the simulator mix and class imbalance, but the task remains SOH regression. In train, Low, Medium and High risk account for about 85.26%, 8.84% and 5.90%, consistent with the composition described by the source. Normal degradation reaches 92.13%. RUL, risk and degradation are neither features nor auxiliary tasks. `capacity_fade_pct` equals 100 minus SOH and stays excluded. A perfect correlation with it is not predictive skill.

## 11. Decisions supported by the EDA

| Finding | Implication for later stages | Limit of the conclusion |
| --- | --- | --- |
| Target skewed towards high SOH | Report errors per SOH band and per battery | Do not automatically rebalance a continuous target |
| Age, cycles and energy strongly related | Important candidate features, test redundant subsets | Not three independent causes |
| Chemistries differ within the same age band | Keep chemistry, allow nonlinear interactions | Usage conditions not fully controlled |
| Numeric and categorical missingness, structural service missingness | Train-only imputation and encoding of unseen categories | Not all missingness is equivalent or MCAR |
| Very different feature ranges | Standardise numeric features in the baseline | Scaler statistics are fitted in preprocessing, not here |
| Sensor tails and rare values | Evaluate targeted outlier handling and sensitivity | IQR is not ground truth for errors |
| Repeated observations and temporal quality | Keep the group split, defer timesteps | No forecasting claim |
| Proxies and derived targets | Keep leaking columns excluded, sensors only in a separate ablation | Not a universal ban on real measured sensors |

The initial feature list is **unchanged**. No score-based feature selection, model, imputer, encoder, scaler, balancing or tuning is applied in this stage.

## 12. Reproduction

- Notebook: `../../notebooks/01_eda.ipynb` with questions, methods, tables, figures, interpretation and checks.
- Commands from the repository root: `python -m ev_battery eda` and `python scripts/run_notebook.py notebooks/01_eda.ipynb`.
- All 35 columns are listed in `column_coverage.csv`, including IDs, dates and alternative targets with their roles.
- Every table is computed from train. Figures are saved as 140 dpi PNG.
- No naive row-level p-values, no causal conclusions, and no use of holdout data to refine the narrative.

Main limitations: synthetic data, unavailable generator, undocumented aggregation windows, missing values and outliers without original masks, repeated observations, small-support groups, and confounding that is not fully controlled.
"""
    (output / "EDA_REPORT.md").write_text(report, encoding="utf-8", newline="\n")


def run_eda(root: Path = ROOT):
    train, dictionary, contract, provenance = load_train(root)
    d, tables = make_tables(train, dictionary, contract)
    output = root / "results/eda"
    output.mkdir(parents=True, exist_ok=True)
    matrices = {"spearman_matrix", "correlation_pair_counts", "missingness_phi", "battery_profiles"}
    for name, table in tables.items():
        # Keep the detailed temporal pairs local, aggregate change tables suffice in Git.
        table.to_csv(output / f"{name}.csv", index=name in matrices, lineterminator="\n")
    figures = make_figures(d, tables, contract, output)
    write_report(d, tables, contract, figures, output)
    meta = {
        "dataset_sha256": provenance["dataset_sha256"],
        "split_manifest_sha256": provenance["split_manifest_sha256"],
        "scope": "train only",
        "train_rows": len(train),
        "train_batteries": train.battery_id.nunique(),
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_unit": "battery-band means within each chemistry/age cell, pointwise percentile interval",
        "minimum_batteries_for_supported_cell": MIN_BATTERIES,
        "python": platform.python_version(),
        "figures": figures,
        "columns_covered": dictionary.column.tolist(),
        "no_fitted_preprocessing_or_models": True,
    }
    meta["packages"] = {
        p: version(p) for p in ["numpy", "pandas", "matplotlib", "scipy", "nbformat", "nbclient"]
    }
    meta["feature_contract_sha256"] = hashlib.sha256(
        (root / "configs/feature_contract.json").read_bytes()
    ).hexdigest()
    meta["eda_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / "eda_provenance.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return {
        "train": train,
        "analysis": d,
        "tables": tables,
        "contract": contract,
        "output": output,
        "figures": figures,
        "provenance": meta,
    }


if __name__ == "__main__":
    result = run_eda()
    print(
        f"EDA train: {len(result['train']):,} observations, {result['train'].battery_id.nunique():,} batteries"
    )
    print(f"Saved {len(result['figures'])} figures and {len(result['tables'])} tables.")
    print(result["output"] / "EDA_REPORT.md")
