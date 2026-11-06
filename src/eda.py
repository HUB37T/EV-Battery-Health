"""Train-only EDA. No predictive models, feature fitting, or holdout summaries.

Run: python -m src.eda
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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.data_audit import ROOT, TARGET, PROXY_COLUMNS, load_dataset, markdown_table

SEED = 42
BOOTSTRAPS = 1000
MIN_BATTERIES = 30
CHEMISTRIES = ["LFP", "NMC", "NCA", "LMO"]
COLORS = dict(zip(CHEMISTRIES, ["#007f78", "#3674b5", "#c47a21", "#a4548e"]))
AGE_EDGES = [0, 365, 730, 1095, 1460, 1825, np.inf]
AGE_LABELS = ["≤1 tahun", "1–2 tahun", "2–3 tahun", "3–4 tahun", "4–5 tahun", ">5 tahun"]
LABELS = {
    "pack_capacity_kwh": "Kapasitas nominal (kWh)",
    "battery_age_days": "Umur baterai (hari)",
    "cycle_count": "Siklus ekuivalen (siklus)",
    "avg_daily_distance_km": "Jarak harian (km)",
    "avg_charging_power_kw": "Daya pengisian rata-rata (kW)",
    "fast_charge_ratio": "Rasio pengisian cepat (0–1)",
    "avg_depth_of_discharge_pct": "Depth of discharge (%)",
    "avg_state_of_charge_pct": "State of charge rata-rata (%)",
    "charge_cycles_per_month": "Frekuensi pengisian (/bulan)",
    "total_energy_throughput_kwh": "Energi kumulatif (kWh)",
    "avg_ambient_temp_c": "Suhu lingkungan (°C)",
    "humidity_pct": "Kelembapan (%)",
    "altitude_m": "Ketinggian (m)",
    "peak_temp_during_fast_charge_c": "Suhu puncak fast charge (°C)",
    "service_count": "Jumlah servis",
    "internal_resistance_mohm": "Resistansi internal (mΩ)",
    "voltage_avg_v": "Tegangan rata-rata (V)",
    "current_avg_a": "Arus rata-rata (A)",
    "charge_time_avg_min": "Durasi pengisian (menit)",
    "recall_flag": "Recall flag (0/1)",
    TARGET: "State of Health (%)",
    "capacity_fade_pct": "Capacity fade (%) — dilarang",
    "remaining_useful_life_cycles": "Sisa siklus — target alternatif",
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
    # Discard the full CSV here; downstream analysis receives train only.
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
    # EDA-only grouping label; never overwrite the raw feature or export imputed X.
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
        rows.append({"feature": c, "pairwise_rows": len(sub), "batteries": sub.battery_id.nunique(),
                     "pearson_pooled": safe_corr(sub[c], sub[TARGET]),
                     "spearman_pooled": safe_corr(sub[c], sub[TARGET], method="spearman"),
                     "pearson_between_battery_means": safe_corr(between[c], between[TARGET]),
                     "pearson_within_battery_centered": within_r})
    return pd.DataFrame(rows)


def make_tables(train, dictionary, contract):
    d, metadata = prepare_metadata(train)
    numeric = contract["numeric_features"] + ["recall_flag"] + PROXY_COLUMNS
    tables = {}
    tables["numeric_distributions"] = d[numeric + [TARGET]].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T.rename_axis("feature").reset_index()
    rows = []
    for c in numeric + [TARGET]:
        x = d[c].dropna()
        q1, q3 = x.quantile([.25, .75])
        low, high = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)
        rows.append({"feature": c, "n": len(x), "skewness": x.skew(), "iqr_low": low, "iqr_high": high,
                     "iqr_flag_n": int((~x.between(low, high)).sum()),
                     "iqr_flag_pct": (~x.between(low, high)).mean()*100,
                     "median": x.median(), "p99": x.quantile(.99), "max": x.max()})
    tables["tails_and_iqr"] = pd.DataFrame(rows)
    categories = []
    for c in contract["categorical_features"] + ["recall_flag", "failure_risk_label", "degradation_category"]:
        tmp = d.assign(group=d[c].astype("string").fillna("Missing"))
        tab = tmp.groupby("group").agg(rows=(TARGET, "size"), batteries=("battery_id", "nunique"), median_soh=(TARGET, "median"), mean_soh=(TARGET, "mean"), median_age_days=("battery_age_days", "median"), median_cycles=("cycle_count", "median"))
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
        for other in base_columns[i+1:]:
            rho = matrix.loc[c, other]
            if abs(rho) >= .70:
                pairs.append({"feature_a": c, "feature_b": other, "spearman": rho, "pairwise_rows": int(d[[c, other]].notna().all(axis=1).sum())})
    tables["redundant_pairs"] = pd.DataFrame(pairs).sort_values("spearman", key=abs, ascending=False)

    # One contribution per battery in each age band. No observation-level CI.
    battery_age = d.dropna(subset=["age_band", "chemistry_group"]).groupby(["battery_id", "chemistry_group", "age_band"], observed=True)[TARGET].mean().reset_index(name="battery_band_mean_soh")
    rng = np.random.default_rng(SEED)
    curves = []
    for (chem, age), group in battery_age.groupby(["chemistry_group", "age_band"], observed=True):
        values = group.battery_band_mean_soh.to_numpy()
        boot = values[rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))].mean(axis=1)
        lo, hi = np.quantile(boot, [.025, .975])
        rows_n = int((d.chemistry_group.eq(chem) & d.age_band.eq(age)).sum())
        curves.append({"chemistry": chem, "age_band": str(age), "rows": rows_n, "batteries": len(values), "battery_balanced_mean_soh": values.mean(), "ci95_low": lo, "ci95_high": hi, "support_at_least_30_batteries": len(values) >= MIN_BATTERIES})
    tables["chemistry_age_profiles"] = pd.DataFrame(curves)
    tables["battery_age_means"] = battery_age
    battery = d.groupby("battery_id").agg(observations=(TARGET, "size"), mean_soh=(TARGET, "mean"), median_soh=(TARGET, "median"), min_soh=(TARGET, "min"), max_age_days=("battery_age_days", "max"))
    tables["battery_profiles"] = battery.join(metadata)

    # Descriptive associations within fixed chemistry/age strata; no p-values.
    stratified = []
    selected = ["fast_charge_ratio", "avg_depth_of_discharge_pct", "avg_ambient_temp_c", "charge_cycles_per_month", "service_count"]
    for (chem, age), group in d.groupby(["chemistry_group", "age_band"], observed=True):
        for c in selected:
            valid = group[["battery_id", c, TARGET]].dropna()
            if len(valid) >= 100 and valid.battery_id.nunique() >= MIN_BATTERIES:
                stratified.append({"chemistry": chem, "age_band": str(age), "feature": c, "rows": len(valid), "batteries": valid.battery_id.nunique(), "spearman": safe_corr(valid[c], valid[TARGET], method="spearman")})
    tables["stratified_associations"] = pd.DataFrame(stratified)

    missing_rows = []
    for c in train.columns:
        mask = train[c].isna()
        if not mask.any():
            continue
        missing_rows.append({"feature": c, "missing_rows": int(mask.sum()), "missing_pct": mask.mean()*100,
                             "observed_rows": int((~mask).sum()), "median_soh_missing": train.loc[mask, TARGET].median(),
                             "median_soh_observed": train.loc[~mask, TARGET].median(),
                             "median_gap_missing_minus_observed": train.loc[mask, TARGET].median()-train.loc[~mask, TARGET].median(),
                             "median_age_missing": safe_median(train.loc[mask, "battery_age_days"]), "median_age_observed": safe_median(train.loc[~mask, "battery_age_days"])})
    tables["missingness_target_profiles"] = pd.DataFrame(missing_rows)
    missing_columns = [c for c in train if train[c].isna().any()]
    phi = train[missing_columns].isna().astype(float).corr()
    tables["missingness_phi"] = phi
    missing_pairs = [{"feature_a": a, "feature_b": b, "phi": phi.loc[a, b]} for i, a in enumerate(missing_columns) for b in missing_columns[i+1:]]
    tables["missingness_pairs"] = pd.DataFrame(missing_pairs).sort_values("phi", key=abs, ascending=False)

    ordered = d.assign(date=pd.to_datetime(d.observation_date, errors="coerce")).dropna(subset=["date"]).sort_values(["battery_id", "date", "observation_id"])
    lag = ordered.groupby("battery_id")[["date", TARGET, "cycle_count", "total_energy_throughput_kwh"]].shift()
    temporal = ordered[["battery_id", "observation_id", "date", "chemistry_group"]].copy()
    temporal["gap_days"] = (ordered.date-lag.date).dt.days
    # Match recorded precision at boundaries: exactly 0.01 must not become >0.01.
    assert np.allclose(d[TARGET]*100, (d[TARGET]*100).round(), atol=1e-8, rtol=0)
    temporal["delta_soh_pp"] = (ordered[TARGET]-lag[TARGET]).round(2)
    temporal["delta_cycles"] = ordered.cycle_count-lag.cycle_count
    temporal["delta_energy_kwh"] = ordered.total_energy_throughput_kwh-lag.total_energy_throughput_kwh
    temporal = temporal.loc[temporal.gap_days.gt(0)]
    tables["temporal_pairs"] = temporal
    levels = [{"label": "turun >0.01 pp", "rows": int(temporal.delta_soh_pp.lt(-.01).sum())}, {"label": "stabil ±0.01 pp", "rows": int(temporal.delta_soh_pp.abs().le(.01).sum())}, {"label": "naik >0.01 pp", "rows": int(temporal.delta_soh_pp.gt(.01).sum())}]
    tables["temporal_soh_changes"] = pd.DataFrame(levels)
    tables["temporal_soh_changes"]["pct_pairs"] = tables["temporal_soh_changes"].rows / len(temporal) * 100

    grid = d.dropna(subset=["age_band", "fast_charge_ratio"]).copy()
    grid["fast_band"] = pd.cut(grid.fast_charge_ratio, [-.001, .2, .4, .6, 1], labels=["0–0.2", ">0.2–0.4", ">0.4–0.6", ">0.6–1"])
    tables["age_fastcharge_grid"] = grid.groupby(["age_band", "fast_band"], observed=True).agg(median_soh=(TARGET, "median"), rows=(TARGET, "size"), batteries=("battery_id", "nunique")).reset_index()

    sensitivity = []
    for c in numeric:
        sub = d[[c, TARGET]].dropna()
        q1, q3 = sub[c].quantile([.25, .75])
        keep = sub[c].between(q1-1.5*(q3-q1), q3+1.5*(q3-q1))
        sensitivity.append({"feature": c, "rows_all": len(sub), "rows_inside_iqr_fences": int(keep.sum()), "spearman_all": safe_corr(sub[c], sub[TARGET], method="spearman"), "spearman_inside_iqr_fences": safe_corr(sub.loc[keep, c], sub.loc[keep, TARGET], method="spearman"), "inside_feature_unique_values": sub.loc[keep, c].nunique()})
    tables["outlier_association_sensitivity"] = pd.DataFrame(sensitivity)

    # Cover every raw column, including IDs/dates/alternate targets without making them features.
    inventory = dictionary.copy()
    inventory["missing_train"] = inventory.column.map(train.isna().sum())
    inventory["unique_train_non_null"] = inventory.column.map(train.nunique())
    inventory["analysis_role"] = inventory.column.map({**{c: "primary" for c in contract["primary_features"]}, **{c: "excluded" for c in contract["excluded_features"]}, TARGET: "target"})
    tables["column_coverage"] = inventory
    return d, tables


def make_figures(d, tables, contract, output):
    figures = []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})
    def save(fig, name):
        fig.savefig(output / name, dpi=140, bbox_inches="tight")
        svg = output / name.replace(".png", ".svg")
        fig.savefig(svg, bbox_inches="tight")
        # Matplotlib pads path lines with spaces; normalize generated XML for Git.
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
        plt.close(fig)
        figures.append(name)

    weights = 1 / d.groupby("battery_id")[TARGET].transform("size")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.arange(54, 102, 2)
    axes[0].hist(d[TARGET], bins=bins, weights=np.ones(len(d))/len(d), alpha=.65, color="#247e88", label="Bobot sama per observasi")
    axes[0].hist(d[TARGET], bins=bins, weights=weights/weights.sum(), histtype="step", linewidth=2, color="#c47a21", label="Bobot sama per baterai")
    axes[0].set(xlabel="SOH (%)", ylabel="Proporsi per bin", title="Target: cek bias jumlah pengamatan")
    axes[0].legend(fontsize=8)
    x = np.sort(d[TARGET])
    axes[1].plot(x, np.arange(1, len(x)+1)/len(x), color="#247e88")
    axes[1].axvline(80, color="#bd623b", ls="--", label="80%: batas referensi sumber")
    axes[1].set(xlabel="SOH (%)", ylabel="Proporsi kumulatif", title="ECDF train (per observasi)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "01_target_and_weighting.png")

    numeric = contract["numeric_features"] + ["recall_flag"] + PROXY_COLUMNS
    for page, start in enumerate(range(0, len(numeric), 10), 1):
        columns = numeric[start:start+10]
        fig, axes = plt.subplots(5, 2, figsize=(12, 15))
        for ax, c in zip(axes.flat, columns):
            bins = [-.5, .5, 1.5] if c == "recall_flag" else 35
            ax.hist(d[c].dropna(), bins=bins, color="#247e88", edgecolor="white", linewidth=.3)
            if c == "recall_flag":
                ax.set_xticks([0, 1])
            ax.set(xlabel=LABELS[c], ylabel="Observasi")
            role = "proxy; bukan baseline" if c in PROXY_COLUMNS else "kandidat input"
            ax.set_title(f"{role} · n={d[c].notna().sum():,}, kosong={d[c].isna().mean():.1%}", fontsize=9)
        for ax in list(axes.flat)[len(columns):]:
            ax.set_visible(False)
        fig.suptitle(f"Distribusi seluruh kandidat numerik — train, bagian {page}", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, .97))
        save(fig, f"02_numeric_distributions_{page}.png")

    cats = contract["categorical_features"] + ["recall_flag"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    profiles = tables["categorical_profiles"]
    for ax, c in zip(axes.flat, cats):
        tab = profiles.loc[profiles.feature.eq(c)].sort_values("rows")
        bars = ax.barh(tab.category, tab.row_pct, color=["#a9b4b9" if s == "Missing" else "#247e88" for s in tab.category])
        for bar, n in zip(bars, tab.rows):
            inside = bar.get_width() > 75
            ax.text(bar.get_width()-1 if inside else bar.get_width()+.6, bar.get_y()+bar.get_height()/2, f"n={n:,}", va="center", ha="right" if inside else "left", color="white" if inside else "black", fontsize=8)
        ax.set(title=c, xlabel="% observasi train", xlim=(0, min(100, max(tab.row_pct)*1.32)))
    fig.suptitle("Kategori dan ketimpangan dukungan: Missing tetap ditampilkan", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .97))
    save(fig, "03_categorical_support.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bychem = d.groupby("chemistry_group").agg(median=(TARGET, "median"), q25=(TARGET, lambda s:s.quantile(.25)), q75=(TARGET, lambda s:s.quantile(.75)), n=(TARGET,"size")).reindex(CHEMISTRIES)
    for i, chem in enumerate(CHEMISTRIES):
        row = bychem.loc[chem]
        axes[0].errorbar(row["median"], i, xerr=[[row["median"]-row.q25], [row.q75-row["median"]]], fmt="o", capsize=4, color=COLORS[chem])
    axes[0].set(yticks=range(4), yticklabels=[f"{c} (n={int(bychem.loc[c,'n']):,})" for c in CHEMISTRIES], xlabel="SOH (%)", title="Median dan IQR per observasi", xlim=(65, 100.5))
    curves = tables["chemistry_age_profiles"]
    for chem in CHEMISTRIES:
        sub = curves[curves.chemistry.eq(chem)].set_index("age_band").reindex(AGE_LABELS)
        axes[1].plot(range(6), sub.battery_balanced_mean_soh, "o-", color=COLORS[chem], label=chem)
        axes[1].fill_between(range(6), sub.ci95_low, sub.ci95_high, color=COLORS[chem], alpha=.12)
        small = ~sub.support_at_least_30_batteries.fillna(False).astype(bool)
        axes[1].scatter(np.arange(6)[small], sub.battery_balanced_mean_soh[small], marker="x", s=80, color="black", zorder=5)
    axes[1].set(xticks=range(6), xticklabels=["≤1", "1–2", "2–3", "3–4", "4–5", ">5"], xlabel="Rentang umur (tahun)", ylabel="Rerata SOH, bobot sama per baterai (%)", title="Bandingkan umur; 95% bootstrap antar-baterai")
    axes[1].legend(fontsize=8, ncol=4)
    axes[1].text(.01, .02, "× = dukungan <30 baterai; bukan hasil kuat", transform=axes[1].transAxes, fontsize=8)
    fig.tight_layout()
    save(fig, "04_chemistry_and_age.png")

    support = curves.pivot(index="chemistry", columns="age_band", values="batteries").reindex(index=CHEMISTRIES, columns=AGE_LABELS)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    im=ax.imshow(support, cmap="Blues", aspect="auto")
    for (r,c), value in np.ndenumerate(support.to_numpy()):
        ax.text(c,r,str(int(value)),ha="center",va="center",color="white" if value>support.max().max()*.65 else "#20343c")
    ax.set(xticks=range(6),xticklabels=AGE_LABELS,yticks=range(4),yticklabels=CHEMISTRIES,title="Jumlah baterai pendukung per kimia × umur")
    fig.colorbar(im,ax=ax,label="Baterai unik dalam sel")
    fig.tight_layout()
    save(fig,"05_chemistry_age_support.png")

    columns=contract["numeric_features"]+["recall_flag",TARGET]
    corr=tables["spearman_matrix"].loc[columns,columns]
    fig,ax=plt.subplots(figsize=(12,10))
    im=ax.imshow(corr,vmin=-1,vmax=1,cmap="RdBu_r")
    labels=[LABELS[c] for c in columns]
    ax.set(xticks=range(len(columns)),xticklabels=labels,yticks=range(len(columns)),yticklabels=labels,title="Spearman: fitur baseline dan SOH — pairwise complete train")
    plt.setp(ax.get_xticklabels(),rotation=65,ha="right",fontsize=8)
    plt.setp(ax.get_yticklabels(),fontsize=8)
    for (r,c),value in np.ndenumerate(corr.to_numpy()):
        if r!=c and abs(value)>=.70:
            ax.text(c,r,f"{value:.2f}",ha="center",va="center",fontsize=7,color="white" if abs(value)>.75 else "black")
    fig.colorbar(im,ax=ax,shrink=.7,label="ρ Spearman; anotasi |ρ| ≥0.70")
    fig.tight_layout()
    save(fig,"06_baseline_correlations.png")

    assoc=tables["target_associations"].sort_values("spearman_pooled")
    fig,axes=plt.subplots(1,2,figsize=(13,8))
    axes[0].barh([LABELS[c] for c in assoc.feature],assoc.spearman_pooled,color=["#c47a21" if c in PROXY_COLUMNS else "#247e88" for c in assoc.feature])
    axes[0].axvline(0,color="gray",lw=.8)
    axes[0].set(xlim=(-1,1),xlabel="ρ Spearman (pasangan nonkosong)",title="Asosiasi target bukan keputusan seleksi")
    axes[0].legend(handles=[Patch(color="#247e88",label="Kandidat baseline"),Patch(color="#c47a21",label="Proxy: bukan baseline")],loc="upper left",bbox_to_anchor=(0,-.10),fontsize=8,ncol=2)
    comp=assoc.loc[assoc.feature.isin(["battery_age_days","cycle_count","total_energy_throughput_kwh","service_count","fast_charge_ratio","avg_ambient_temp_c"])].copy()
    y=np.arange(len(comp))
    for offset,field,label,color in [(-.22,"pearson_pooled","Semua baris", "#247e88"),(0,"pearson_between_battery_means","Rerata per baterai","#c47a21"),(.22,"pearson_within_battery_centered","Centered dalam baterai","#926098")]:
        axes[1].barh(y+offset,comp[field],height=.2,label=label,color=color)
    axes[1].set(yticks=y,yticklabels=[LABELS[c] for c in comp.feature],xlim=(-1,1),xlabel="r Pearson (berbeda dari Spearman di kiri)",title="Pisahkan variasi antar/dalam baterai")
    axes[1].axvline(0,color="gray",lw=.8)
    axes[1].legend(fontsize=8,loc="upper left",bbox_to_anchor=(0,-.10),ncol=2)
    fig.tight_layout(rect=(0,.05,1,1))
    save(fig,"07_target_associations_and_dependence.png")

    sample=d.sample(min(6000,len(d)),random_state=SEED)
    xcols=["battery_age_days","cycle_count","total_energy_throughput_kwh","fast_charge_ratio","avg_depth_of_discharge_pct","avg_ambient_temp_c"]
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for ax,c in zip(axes.flat,xcols):
        for chem in CHEMISTRIES:
            sub=sample[sample.chemistry_group.eq(chem)]
            x = sub[c]/1000 if c == "total_energy_throughput_kwh" else sub[c]
            ax.scatter(x,sub[TARGET],s=4,alpha=.16,color=COLORS[chem],rasterized=True)
        ax.set(xlabel="Energi kumulatif (ribu kWh)" if c == "total_energy_throughput_kwh" else LABELS[c],ylabel="SOH (%)")
    fig.legend(handles=[Patch(color=COLORS[c],label=c) for c in CHEMISTRIES],loc="upper center",ncol=4,bbox_to_anchor=(.5,1))
    fig.suptitle("Relasi numerik dengan konteks kimia: sampel tetap 6.000 baris train",y=1.03)
    fig.tight_layout()
    save(fig,"08_relationships_by_chemistry.png")

    grid=tables["age_fastcharge_grid"]
    order=["0–0.2",">0.2–0.4",">0.4–0.6",">0.6–1"]
    med=grid.pivot(index="age_band",columns="fast_band",values="median_soh").reindex(index=AGE_LABELS,columns=order)
    count=grid.pivot(index="age_band",columns="fast_band",values="rows").reindex(index=AGE_LABELS,columns=order)
    bats=grid.pivot(index="age_band",columns="fast_band",values="batteries").reindex(index=AGE_LABELS,columns=order)
    supported=count.ge(100)&bats.ge(30)
    fig,ax=plt.subplots(figsize=(9,6))
    im=ax.imshow(med.where(supported),cmap="YlGnBu",vmin=65,vmax=100,aspect="auto")
    for (r,c),value in np.ndenumerate(med.to_numpy()):
        ok=bool(supported.iloc[r,c])
        label=f"{value:.1f}%\nn={int(count.iloc[r,c]):,}" if ok else "dukungan rendah"
        ax.text(c,r,label,ha="center",va="center",fontsize=9,color="white" if ok and value>90 else "black")
    ax.set(xticks=range(4),xticklabels=order,yticks=range(6),yticklabels=AGE_LABELS,xlabel="Rasio fast charge",title="Median SOH menurut umur × fast charge\nSel tampil jika ≥100 baris dan ≥30 baterai; kimia belum diseimbangkan")
    fig.colorbar(im,ax=ax,label="Median SOH (%)")
    fig.tight_layout()
    save(fig,"09_age_fastcharge_conditioning.png")

    cats=["manufacturer","vehicle_model","climate_zone","bms_firmware_version"]
    fig,axes=plt.subplots(2,2,figsize=(12,10))
    for ax,c in zip(axes.flat,cats):
        tab=profiles.loc[profiles.feature.eq(c)&profiles.category.ne("Missing")].sort_values("median_soh")
        ax.barh(tab.category,tab.median_soh,color="#247e88")
        for i,(_,row) in enumerate(tab.iterrows()):
            ax.text(96.2,i,f"n={int(row.rows):,}; umur med={row.median_age_days:.0f} hari",fontsize=8,va="center")
        ax.set(xlim=(90,106),xlabel="Median SOH (%); sumbu mulai 90",title=c)
    fig.suptitle("Perbandingan kategori mentah: periksa dukungan dan perbedaan umur",fontsize=14)
    fig.tight_layout(rect=(0,0,1,.97))
    save(fig,"10_category_comparisons.png")

    missing=tables["missingness_target_profiles"]
    primary_missing=missing[missing.feature.isin(contract["primary_features"])].sort_values("median_gap_missing_minus_observed")
    fig,axes=plt.subplots(1,2,figsize=(13,8))
    axes[0].barh(primary_missing.feature,primary_missing.median_gap_missing_minus_observed,color="#247e88")
    axes[0].axvline(0,color="gray",lw=.8)
    axes[0].set(xlabel="Median SOH missing − terisi (poin persentase)",title="Missing fitur utama vs SOH (deskriptif)")
    missing_count=d[contract["primary_features"]].isna().sum(axis=1)
    freq=missing_count.value_counts().sort_index()
    axes[1].bar(freq.index,freq.values,color="#3674b5")
    axes[1].set(xlabel="Jumlah fitur utama kosong dalam satu observasi",ylabel="Jumlah observasi",title="Beban missing per baris (21 fitur utama)")
    fig.tight_layout()
    save(fig,"11_missingness_diagnostics.png")

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    service=d.last_service_date.isna()
    for ax,c,label in [(axes[0],TARGET,"SOH (%)"),(axes[1],"battery_age_days","Umur (hari)")]:
        groups=[d.loc[service,c].dropna(),d.loc[~service,c].dropna()]
        ax.boxplot(groups,tick_labels=["Tanggal servis kosong","Tanggal servis terisi"],showfliers=False)
        ax.set(ylabel=label,title="IQR/whisker; outlier disembunyikan hanya di grafik")
    fig.suptitle("Missing tanggal servis berkaitan dengan tahap pemakaian; bukan efek kausal servis")
    fig.tight_layout()
    save(fig,"12_service_missingness_context.png")

    tails=tables["tails_and_iqr"].sort_values("iqr_flag_pct")
    fig,axes=plt.subplots(1,2,figsize=(13,8))
    axes[0].barh([LABELS[c] for c in tails.feature],tails.iqr_flag_pct,color="#c47a21")
    axes[0].set(xlabel="% nilai terisi di luar pagar 1.5×IQR",title="Kandidat outlier, bukan label kesalahan")
    x=d.charge_time_avg_min.dropna()
    axes[1].hist(x,bins=np.geomspace(x.min(),x.max(),40),color="#247e88",edgecolor="white")
    axes[1].set_xscale("log")
    axes[1].set(xlabel="Durasi pengisian (menit; sumbu log)",ylabel="Observasi",title="Ekor durasi panjang: data tidak diubah")
    fig.tight_layout()
    save(fig,"13_outliers_and_tail.png")

    fig,axes=plt.subplots(2,2,figsize=(12,8))
    dated=d.assign(date=pd.to_datetime(d.observation_date)).dropna(subset=["date"])
    for ax,chem in zip(axes.flat,CHEMISTRIES):
        sub=dated[dated.chemistry_group.eq(chem)]
        ids=sub.groupby("battery_id").size().loc[lambda x:x.ge(10)].sort_index().index[:3]
        for bid in ids:
            trajectory=sub[sub.battery_id.eq(bid)].sort_values("date")
            ax.plot(trajectory.date,trajectory[TARGET],"o-",ms=3,lw=1,label=bid)
        ax.set(title=f"{chem}: tiga ID pertama dengan ≥10 tanggal valid",ylabel="SOH (%)",ylim=(50,102))
        ax.legend(fontsize=7)
        ax.tick_params(axis="x",rotation=25)
    fig.suptitle("Contoh lintasan nyata dalam train; pemilihan ID tidak memakai nilai target")
    fig.tight_layout(rect=(0,0,1,.96))
    save(fig,"14_battery_trajectories.png")

    pairs=tables["temporal_pairs"]
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    axes[0].hist(pairs.gap_days,bins=40,color="#247e88",edgecolor="white")
    axes[0].set(xlabel="Selang antarobservasi valid (hari)",ylabel="Pasangan",title="Sampling tidak seragam")
    axes[1].hist(pairs.delta_soh_pp,bins=65,color="#3674b5",edgecolor="white")
    axes[1].axvline(0,color="#a4548e",ls="--")
    axes[1].set(xlabel="Perubahan SOH (poin persentase)",title="SOH bisa naik antarobservasi")
    counts=dated.date.dt.to_period("M").value_counts().sort_index()
    axes[2].plot(counts.index.to_timestamp(),counts.values,color="#247e88")
    axes[2].set(xlabel="Bulan observasi",ylabel="Observasi bertanggal",title="Cakupan kalender train")
    axes[2].tick_params(axis="x",rotation=35)
    fig.tight_layout()
    save(fig,"15_temporal_quality.png")

    battery=tables["battery_profiles"]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].hist(battery.observations,bins=np.arange(6.5,38.5),color="#247e88",edgecolor="white")
    axes[0].set(xlabel="Pengamatan per baterai",ylabel="Baterai",title="Unit independen: 1.820 baterai")
    axes[1].scatter(battery.observations,battery.mean_soh,s=9,alpha=.3,color="#247e88")
    axes[1].set(xlabel="Pengamatan per baterai",ylabel="Rerata SOH baterai (%)",title="Apakah baterai dengan banyak baris berbeda?")
    fig.tight_layout()
    save(fig,"16_battery_observation_structure.png")

    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,c in zip(axes[:2],["failure_risk_label","degradation_category"]):
        tab=profiles[profiles.feature.eq(c)].sort_values("rows",ascending=False)
        ax.bar(tab.category,tab.row_pct,color="#a4548e")
        ax.set(title=c,ylabel="% observasi train")
        ax.tick_params(axis="x",rotation=20)
        for i,row in enumerate(tab.itertuples()):
            ax.text(i,row.row_pct+1,f"{row.row_pct:.1f}%",ha="center",fontsize=8)
        ax.set_ylim(0,tab.row_pct.max()+12)
    axes[2].scatter(sample[TARGET],sample.remaining_useful_life_cycles,s=4,alpha=.15,color="#a4548e",rasterized=True)
    axes[2].set(xlabel="SOH (%)",ylabel="Remaining useful life (siklus)",title="Target alternatif, bukan input SOH")
    fig.suptitle("Eksplorasi target alternatif untuk memahami dataset, tanpa mengubah tugas regresi")
    fig.tight_layout(rect=(0,0,1,.94))
    save(fig,"17_alternative_targets.png")
    return figures


def write_report(d, tables, contract, figures, output):
    primary=contract["primary_features"]
    weights=1/d.groupby("battery_id")[TARGET].transform("size")
    assoc=tables["target_associations"].set_index("feature")
    missing=tables["missingness_target_profiles"]
    service=missing.set_index("feature").loc["last_service_date"]
    curve=tables["chemistry_age_profiles"]
    one_to_two=curve[curve.age_band.eq("1–2 tahun")][["chemistry","rows","batteries","battery_balanced_mean_soh","ci95_low","ci95_high"]]
    temporal=tables["temporal_pairs"]
    increments=temporal.delta_soh_pp.gt(.01)
    rho_n=tables["battery_profiles"].observations.corr(tables["battery_profiles"].mean_soh,method="spearman")
    strongest=tables["target_associations"].sort_values("spearman_pooled",key=abs,ascending=False)[["feature","role","pairwise_rows","batteries","pearson_pooled","spearman_pooled"]].head(10)
    redundancy=tables["redundant_pairs"]
    missing_small=missing[missing.feature.isin(primary)]
    sparse=curve[~curve.support_at_least_30_batteries][["chemistry","age_band","rows","batteries"]]
    count_matrix=tables["correlation_pair_counts"]
    pair_values=count_matrix.to_numpy()[~np.eye(len(count_matrix),dtype=bool)]
    report=f"""# EDA Mendalam EV Battery Health

## Ringkasan yang perlu dipahami kelompok

1. Analisis ini memakai **34.929 observasi dari 1.820 baterai train**. Validation dan test tidak diringkas, divisualisasikan, atau digunakan untuk memilih pola.
2. Target SOH condong ke kondisi tinggi: median **{d[TARGET].median():.2f}%**, rerata **{d[TARGET].mean():.2f}%**, **{d[TARGET].ge(95).mean()*100:.2f}%** observasi mencapai ≥95%, sedangkan **{d[TARGET].lt(80).mean()*100:.2f}%** di bawah 80%. Error pada SOH rendah perlu dilaporkan terpisah kelak.
3. Jumlah siklus paling kuat berkaitan dengan SOH di antara fitur utama (Spearman **{assoc.loc['cycle_count','spearman_pooled']:.3f}**). Umur dan energi kumulatif juga kuat. Ketiganya saling berkaitan, sehingga bukan tiga bukti pengaruh yang independen.
4. Kimia baterai berbeda dalam distribusi SOH. Perbandingan dalam rentang umur yang sama dengan bobot sama per baterai masih menunjukkan perbedaan, tetapi belum mengontrol semua kondisi penggunaan. Ini pola simulator, bukan efek kausal kimia pada armada nyata.
5. Missing fitur utama menunjukkan perbedaan median SOH missing–terisi sebesar **{missing_small.median_gap_missing_minus_observed.min():.2f} hingga {missing_small.median_gap_missing_minus_observed.max():.2f} poin persentase**. Ini tidak membuktikan MCAR. Missing tanggal servis sangat berbeda dan berkaitan dengan tahap pemakaian.
6. Pada **{len(temporal):,}** pasangan berurutan bertanggal valid, SOH naik >0.01 poin pada **{increments.sum():,} ({increments.mean()*100:.2f}%)** pasangan. Interval tidak seragam dan counter kumulatif tidak konsisten; jangan langsung membangun timestep atau menyebut kenaikan SOH sebagai pemulihan fisik.
7. Belum ada preprocessing fit, feature selection final, training, atau penilaian performa. Seluruh data asli dan manifest split dipertahankan.

## 1. Pertanyaan, sumber, dan cara membaca hasil

Tujuan EDA: menjelaskan distribusi, hubungan, kualitas, dan struktur pengamatan sebelum menentukan preprocessing serta eksperimen MLP/backpropagation.

Sumber: [Kaggle EV Battery Health & Degradation](https://www.kaggle.com/datasets/vinay24baghira/ev-battery-health-and-degradation-dataset). README dan kamus di dalam arsip menyebut data **sepenuhnya sintetis** dengan lisensi CC0, missing dan outlier yang diinjeksi, serta pengukuran listrik terkait SOH/kimia. Generator dan mask injeksi tidak tersedia di proyek ini.

EDA menggunakan versi CSV dan manifest yang hash-nya dicatat pada `eda_provenance.json`. Kode membaca CSV untuk memuat train sesuai manifest, lalu seluruh fungsi EDA hanya menerima train. Tidak menggabungkan data antar-subset untuk memulihkan missing.

Ada dua unit ringkasan:
- **Observasi:** satu baris pengukuran; baterai dengan banyak baris mendapat bobot lebih besar.
- **Baterai:** unit kelompok. Untuk kurva kimia–umur, setiap baterai menyumbang satu rerata SOH per rentang umur, kemudian dirata-ratakan antar-baterai.

Label `chemistry_group` hanya dipakai untuk EDA: satu nilai kimia nonkosong yang konsisten dalam pengamatan **train** baterai tersebut. Kolom kimia asli tetap utuh dan missing-nya tetap dihitung. Ini bukan imputer untuk input model. Perbandingan kategori mentah memakai nilai baris asli, termasuk Missing.

Tidak ada uji p-value berbasis asumsi 34.929 baris independen. Korelasi dan perbedaan kelompok bersifat deskriptif. Tidak ada klaim sebab akibat atau rekomendasi keselamatan baterai dari simulator.

## 2. Distribusi target dan representasi unit

- SOH train: {d[TARGET].min():.2f}–{d[TARGET].max():.2f}%, median {d[TARGET].median():.2f}%, rerata {d[TARGET].mean():.2f}%.
- Nilai tepat 100%: {d[TARGET].eq(100).sum():,}; tepat 55%: {d[TARGET].eq(55).sum():,}. Penumpukan di batas yang teramati konsisten dengan kemungkinan pembatasan/rounding simulator, tetapi kode generator tidak tersedia untuk memastikan mekanismenya.
- Median dengan bobot sama per baterai: {weighted_quantile(d[TARGET],weights,.5):.2f}%. Rerata berbobot sama per baterai: {np.average(d[TARGET],weights=weights):.2f}%.
- Korelasi Spearman jumlah observasi dengan rerata SOH per baterai: {rho_n:.3f}; ini pemeriksaan potensi bias banyaknya pengamatan, bukan bukti independensi sampling.

![Target dan pembobotan](01_target_and_weighting.png)

Histogram memakai bin yang sama; garis oranye memakai bobot baris 1/jumlah baris baterai. ECDF menunjukkan proporsi target pada/bawah suatu nilai. Garis 80% mengikuti referensi end-of-life dalam dokumentasi dataset, bukan ambang kelas yang baru dibuat kelompok.

![Struktur pengamatan](16_battery_observation_structure.png)

**Implikasi:** gunakan MAE/RMSE pada skala target asli; jangan memakai accuracy regresi. Pada evaluasi kelak, sertakan rentang SOH dan ringkasan per baterai. Jumlah sampel nominal bukan jumlah baterai independen.

## 3. Seluruh fitur numerik dan kategori

![Numerik bagian 1](02_numeric_distributions_1.png)

![Numerik bagian 2](02_numeric_distributions_2.png)

Seluruh 15 kandidat numerik, recall_flag, dan empat kandidat proxy sensor dicakup. Setiap panel mencantumkan jumlah terisi dan missing. Histogram tidak membuang outlier. Bentuk miring, diskret, atau rentang lebar dapat dilihat langsung; statistik minimum, p1, p5, kuartil, p95, p99, maksimum, dan skewness tersedia di `numeric_distributions.csv` dan `tails_and_iqr.csv`.

![Dukungan kategori](03_categorical_support.png)

Kategori Missing bukan jenis baterai baru. Firmware dan recall dapat berubah antarobservasi; jangan dianggap atribut statis hanya karena bentuknya kategori. Jumlah baterai dalam kategori yang berubah dapat tumpang tindih dan tidak boleh dijumlahkan seolah-olah kategori saling eksklusif pada tingkat baterai. Recall=1 jarang; banyaknya baris per kategori tercantum agar pembaca tidak menyimpulkan dari kelompok kecil.

**Implikasi:** imputer numerik, encoding kategori tak dikenal, skala numerik, dan perlakuan biner perlu dirancang pada tahap preprocessing. Tidak otomatis menghapus fitur berkorelasi rendah atau kategori langka dari EDA saja.

## 4. Kimia dan umur: perbandingan yang lebih adil

![Kimia dan umur](04_chemistry_and_age.png)

Panel kiri merangkum median/IQR observasi, bukan interval kepercayaan. Panel kanan mengelompokkan umur menjadi rentang tahunan tetap, merata-ratakan SOH per baterai dalam rentang itu, lalu memberi bobot sama pada setiap baterai.

Contoh rentang **1–2 tahun**:

{markdown_table(one_to_two.round(3))}

Pita pada panel kanan adalah interval persentil bootstrap 95% dari **{BOOTSTRAPS:,} resampling baterai dalam setiap sel kimia–umur**, seed 42. Ini interval pointwise untuk rerata sel, bukan untuk keseluruhan kurva atau perbedaan kimia, dan tidak mengatasi confounding. Tidak digunakan untuk uji signifikansi. Baterai yang sama dapat muncul di beberapa rentang umur, sehingga titik-titik sepanjang kurva tetap berkaitan.

![Dukungan kimia dan umur](05_chemistry_age_support.png)

Sel dengan <{MIN_BATTERIES} baterai diberi tanda × pada kurva. Dukungan rendah:

{markdown_table(sparse)}

**Kesimpulan terbatas:** kimia dan umur layak dipertimbangkan bersama karena pola antarkimia tidak identik. Kurva ini adalah ringkasan lintas observasi/baterai pada rentang umur, bukan kurva degradasi terkontrol satu baterai atau bukti kimia tertentu selalu lebih baik.

## 5. Hubungan numerik, nonlinearitas, dan redundansi

![Korelasi baseline](06_baseline_correlations.png)

Spearman mengukur hubungan monoton berbasis ranking dan tidak mengharuskan garis lurus. Pearson melengkapi hubungan linear. Korelasi dihitung pairwise-complete, tanpa mengisi missing; jumlah pasangan antarvariabel berkisar **{int(pair_values.min()):,}–{int(pair_values.max()):,}** pada matriks yang disimpan. Matriks jumlah pasangan terpisah ada di `correlation_pair_counts.csv`.

Asosiasi target terbesar pada kandidat baseline/proxy:

{markdown_table(strongest.round(3))}

![Asosiasi dan dependensi](07_target_associations_and_dependence.png)

Panel kanan membandingkan **Pearson**, bukan nilai Spearman panel kiri: pooled pada seluruh pasangan terisi; antar-baterai pada rerata pasangan per baterai; dan dalam-baterai setelah mengurangi rerata masing-masing baterai. Centering bukan model kausal dan tidak mengontrol umur/kimia atau semua variabel lain. Fitur konstan dalam baterai tidak punya korelasi centered yang terdefinisi; nilai tersebut dibiarkan kosong dalam tabel.

Pasangan fitur utama dengan |Spearman| ≥0.70 (ambang eksplorasi, bukan aturan penghapusan):

{markdown_table(redundancy.round(3))}

![Hubungan menurut kimia](08_relationships_by_chemistry.png)

Scatter memakai sampel deterministik 6.000 baris train tanpa fitting garis/model. Warna menambah konteks kimia. Makin banyak siklus/umur umumnya berasosiasi dengan SOH lebih rendah, tetapi sebaran dan perbedaan kimia menunjukkan satu hubungan linear tunggal tidak cukup menggambarkan semua observasi. Ini alasan menguji MLP, bukan bukti MLP pasti memberi skor terbaik.

**Implikasi:** bandingkan fitur pemakaian yang saling tumpang tindih melalui eksperimen terkontrol nanti. Korelasi rendah fast charge/DOD/suhu secara pooled tidak membuktikan fitur tidak berguna atau faktor tersebut tidak memengaruhi baterai nyata. Secara khusus, Spearman fast charge–SOH pada train hanya sekitar −0.003; sumber menyebut penalti fast charge dalam simulator, tetapi generator tidak tersedia untuk memeriksa bagaimana fitur observasi dan parameter laten saling terkait.

## 6. Pola bersyarat: jangan tertipu korelasi pooled

![Umur dan fast charge](09_age_fastcharge_conditioning.png)

Grid memakai rentang umur/rasio tetap, menampilkan median SOH dan n observasi. Hanya sel dengan ≥100 baris dan ≥30 baterai ditampilkan. Kimia dan penggunaan lain belum diseimbangkan. Grid membantu membedakan variasi umur dari rasio fast charge; tidak mengidentifikasi efek kausal charging.

Tabel `stratified_associations.csv` menghitung Spearman fast charge, DOD, suhu, frekuensi pengisian, dan service_count di dalam sel kimia × umur dengan dukungan yang sama. Rentang umur masih lebar; confounding di dalam sel tetap mungkin. Tidak merangkum semua sel menjadi satu “efek” atau memilih hanya sel yang menarik.

![Perbandingan kategori](10_category_comparisons.png)

Kategori produsen/model/iklim/firmware dibandingkan bersama dukungan dan median umur. Sumbu SOH dimulai 90% agar variasi terlihat dan ditandai eksplisit. Selisih kecil tidak otomatis penting; perbedaan usia dapat menjelaskan sebagian perbedaan mentah. Jangan menyebut merek/model fiktif paling awet tanpa kontrol kondisi dan validasi lebih lanjut.

## 7. Missingness: apa yang hilang dan apakah terkait kondisi?

![Diagnostik missing](11_missingness_diagnostics.png)

Perbedaan median missing–terisi hanya ringkasan deskriptif: missing dapat terkait kondisi, sumber injeksi, atau komposisi kelompok. Selisih kecil tidak membuktikan data Missing Completely At Random. Korelasi indikator missing (`missingness_phi.csv`) dan semua pasangan terurut (`missingness_pairs.csv`) disimpan tanpa mengubah data. Nilai maksimum |phi| sekitar 0.017 pada train, sehingga tidak tampak ko-missing linear besar pada pasangan indikator yang diperiksa; ketergantungan nonlinear atau bersyarat belum dikesampingkan.

![Konteks tanggal servis](12_service_missingness_context.png)

Median SOH untuk tanggal servis kosong adalah **{service.median_soh_missing:.2f}%**, dibanding **{service.median_soh_observed:.2f}%** ketika terisi; median umur masing-masing **{service.median_age_missing:.0f}** dan **{service.median_age_observed:.0f}** hari. Ini konsisten dengan tahap penggunaan berbeda, **bukan** bukti servis menurunkan kesehatan. Tidak melakukan pengisian tanggal dan tidak menambahkan indikator servis sebagai fitur melalui notebook ini.

## 8. Outlier dan sensitivitas kesimpulan

![Outlier dan ekor](13_outliers_and_tail.png)

Pagar 1.5×IQR hanya menandai nilai ekstrem terhadap distribusi train. Nilai target rendah dan flag biner langka juga dapat ditandai IQR tanpa menjadi kesalahan. Jangan menerapkan satu aturan ini untuk menghapus seluruh outlier.

Histogram durasi menggunakan sumbu log agar ekor terlihat; nilai asli tidak dilog-transform untuk model. `outlier_association_sensitivity.csv` membandingkan Spearman semua pasangan dengan pasangan di dalam pagar IQR fitur tersebut. Ini analisis sensitivitas, bukan keputusan membuang baris; denominator perbandingan dicatat. Untuk recall_flag, pagar IQR menyisakan hanya 0 sehingga korelasi setelah penyaringan tidak terdefinisi dan dibiarkan kosong. Untuk siklus/umur/energi, hubungan negatif kuat tetap tampak pada subset di dalam pagar, tanpa membuktikan bahwa penghapusan outlier akan memperbaiki prediksi. Tidak ada mask resmi outlier injeksi untuk menilai apakah titik tertentu pasti salah.

## 9. Struktur temporal dan konsistensi simulator

![Contoh lintasan](14_battery_trajectories.png)

Contoh dipilih secara deterministik: tiga ID pertama secara alfabet dalam setiap kimia dengan ≥10 observasi bertanggal. Nilai SOH tidak dipakai untuk memilih contoh. Lintasan ini tidak merangkum seluruh populasi; tanggal yang kosong tidak dibuat-buat atau dihubungkan sebagai interval yang diketahui.

![Kualitas temporal](15_temporal_quality.png)

{markdown_table(tables['temporal_soh_changes'].round(3))}

Selisih dihitung antarobservasi bertanggal valid dari baterai sama, pada tanggal yang meningkat. Pasangan dapat melompati observasi yang tanggalnya kosong. Perubahan SOH dibulatkan ke presisi pencatatan 0.01 poin sebelum klasifikasi naik/stabil/turun, agar artefak floating-point tidak menggeser nilai tepat di batas. Karena selang waktu berbeda, histogram ΔSOH **bukan laju degradasi per hari**. SOH naik pada sebagian pasangan; sumber menyebut data sintetis/noisy, tetapi mekanisme tiap kenaikan tidak dapat dipastikan tanpa generator. Counter siklus/energi yang turun sudah teridentifikasi pada audit sebelumnya.

**Implikasi:** mempertahankan baseline tabular dan menunda timesteps tetap masuk akal. Jangan memaksa monoton atau menyortir berdasarkan target untuk merapikan lintasan. Group split menguji baterai baru, bukan periode kalender baru.

## 10. Target alternatif: dipahami, bukan dimasukkan ke input

![Target alternatif](17_alternative_targets.png)

Label risiko/degradasi membantu membaca komposisi simulator dan ketimpangan kelas, tetapi tugas tetap regresi SOH. Pada train, risiko Low/Medium/High masing-masing sekitar 85.26%/8.84%/5.90%, konsisten dengan perkiraan komposisi yang disebut sumber. Degradasi Normal mencapai 92.13%. RUL/risiko/degradasi tidak dijadikan fitur atau tugas tambahan. Kolom capacity_fade_pct memiliki identitas langsung 100−SOH dan tetap dilarang; korelasi sempurna bukan prestasi prediktif.

## 11. Keputusan yang didukung EDA dan yang belum diputuskan

| Temuan | Implikasi tahap berikutnya | Batas kesimpulan |
| --- | --- | --- |
| Target condong ke SOH tinggi | Analisis error per rentang SOH dan per baterai | Tidak otomatis menyeimbangkan target kontinu |
| Umur/siklus/energi berkaitan kuat | Kandidat fitur penting; uji subset redundan nanti | Tidak menyatakan ketiganya penyebab independen |
| Kimia berbeda pada rentang umur sama | Pertahankan kimia sebagai kandidat; pertimbangkan interaksi nonlinear | Belum mengontrol seluruh kondisi penggunaan |
| Missing numerik/kategori dan servis struktural | Rancang imputasi train-only dan encoding kategori baru | Jangan menganggap semua missing setara/MCAR |
| Rentang fitur sangat berbeda | Standardisasi numerik sebagai baseline preprocessing | Statistik scaler belum dihitung/disimpan untuk model |
| Ekor sensor/nilai langka | Evaluasi penanganan outlier terarah dan sensitivitas | IQR bukan ground truth kesalahan |
| Pengamatan berulang dan kualitas waktu | Pertahankan group split, tunda timestep | Tidak mengklaim forecasting |
| Proxy dan target turunan | Tetap keluarkan kebocoran; sensor hanya ablation terpisah | Bukan larangan universal terhadap sensor terukur nyata |

Daftar fitur awal **tidak diubah**. Belum ada feature selection berbasis skor, model, imputer, encoder, scaler, balancing, atau tuning. Fokus pekerjaan ini hanya pemahaman data.

## 12. Reproduksi dan bahan penilaian EDA

- Notebook: `../../notebooks/01_eda_train.ipynb`, berisi pertanyaan, metode, tabel, gambar, interpretasi, dan pemeriksaan.
- Perintah dari root: `python -m src.eda` dan `python scripts/run_notebook.py notebooks/01_eda_train.ipynb`.
- Semua 35 kolom tercantum pada `column_coverage.csv`, termasuk ID/tanggal/target alternatif dengan peran yang jelas.
- Seluruh tabel merupakan hasil hitung aktual train; grafik tersimpan sebagai PNG resolusi 140 dpi dan SVG untuk ekspor laporan.
- Tidak menggunakan p-value naif antarbaris, tidak menyimpulkan sebab akibat, dan tidak mengintip holdout untuk menyempurnakan narasi.

Keterbatasan utama: data sintetis, generator tidak tersedia, interval fitur agregat tidak lengkap, missing/outlier tanpa mask asli, pengamatan berulang, sel kelompok dengan dukungan kecil, dan confounding yang belum seluruhnya dikontrol.
"""
    (output/"EDA_REPORT.md").write_text(report,encoding="utf-8")


def run_eda(root: Path = ROOT):
    train,dictionary,contract,provenance=load_train(root)
    d,tables=make_tables(train,dictionary,contract)
    output=root/"results/eda"
    output.mkdir(parents=True,exist_ok=True)
    matrices={"spearman_matrix","correlation_pair_counts","missingness_phi","battery_profiles"}
    for name,table in tables.items():
        # Keep the detailed temporal pairs local; aggregate change tables suffice in Git.
        table.to_csv(output/f"{name}.csv",index=name in matrices,lineterminator="\n")
    figures=make_figures(d,tables,contract,output)
    write_report(d,tables,contract,figures,output)
    meta={"dataset_sha256":provenance["dataset_sha256"],"split_manifest_sha256":provenance["split_manifest_sha256"],"scope":"train only", "train_rows":len(train),"train_batteries":train.battery_id.nunique(),"seed":SEED,"bootstrap_replicates":BOOTSTRAPS,"bootstrap_unit":"battery-band means within each chemistry/age cell; pointwise percentile interval", "minimum_batteries_for_supported_cell":MIN_BATTERIES,"python":platform.python_version(),"figures":figures,"columns_covered":dictionary.column.tolist(),"no_fitted_preprocessing_or_models":True}
    meta["packages"]={p:version(p) for p in ["numpy","pandas","matplotlib","scipy","nbformat","nbclient"]}
    meta["feature_contract_sha256"]=hashlib.sha256((root/"configs/feature_contract.json").read_bytes()).hexdigest()
    meta["eda_source_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output/"eda_provenance.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return {"train":train,"analysis":d,"tables":tables,"contract":contract,"output":output,"figures":figures,"provenance":meta}


if __name__=="__main__":
    result=run_eda()
    print(f"EDA train: {len(result['train']):,} observations, {result['train'].battery_id.nunique():,} batteries")
    print(f"Saved {len(result['figures'])} figures and {len(result['tables'])} tables.")
    print(result["output"]/"EDA_REPORT.md")
