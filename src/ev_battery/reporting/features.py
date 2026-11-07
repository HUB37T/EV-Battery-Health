"""Report feature engineering ablations without rewriting the original study."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ev_battery import features as features_module
from ev_battery.audit import markdown_table
from ev_battery.features import load_engineered_data, load_raw_observations
from ev_battery.paths import ROOT
from ev_battery.reporting.experiments import config_sentence
from ev_battery.training import load_training_data

NAMES = {
    "control_no_domain_features": "Control, no domain features",
    "fe_usage_rates": "Usage ratios",
    "fe_context": "Temperature delta + fast charge × DOD",
    "fe_chemistry": "Chemistry × exposure",
    "fe_combined": "All engineered features",
}


def read_results(root=ROOT):
    output = root / "results/feature_engineering"
    study = json.loads((output / "study.json").read_text(encoding="utf-8"))
    if study["source_sha256"] != hashlib.sha256(Path(features_module.__file__).read_bytes()).hexdigest():
        raise ValueError("Feature-engineering results come from a different source version, rerun the study.")
    tables = {
        name: pd.read_csv(output / f"{name}.csv")
        for name in [
            "screening",
            "seed_confirmation",
            "seed_summary",
            "paired_seed_improvements",
            "validation_bands",
            "run_records",
        ]
    }
    return output, study, tables


def build_report(root=ROOT):
    output, study, tables = read_results(root)
    best = study["selected_engineered_variant"]
    summary = tables["seed_summary"].set_index("variant")
    control = summary.loc["control_no_domain_features"]
    fe = summary.loc[best]
    selected = tables["screening"].set_index("variant").loc[best]
    control_selected = tables["screening"].set_index("variant").loc["control_no_domain_features"]
    bands = (
        tables["validation_bands"]
        .groupby(["variant", "band"], sort=False)
        .agg(
            n_rows=("rows", "first"),
            batteries=("batteries", "first"),
            mae_mean_pp=("mae_pp", "mean"),
            mae_sd_pp=("mae_pp", "std"),
        )
        .reset_index()
    )
    bands.to_csv(output / "validation_band_summary.csv", index=False, lineterminator="\n")
    base = load_training_data(study["reference_config"]["recipe"], root)
    raw = load_raw_observations(base, root)
    full = load_engineered_data("fe_combined", root)
    prep = full["bundle"]["preprocessor"]
    extras = prep._arithmetic(raw["train"])
    diagnostics = prep.describe_train(raw["train"])
    diagnostics["raw_min_train"] = [extras[c].min() for c in diagnostics.feature]
    diagnostics["raw_max_train"] = [extras[c].max() for c in diagnostics.feature]
    y = pd.Series(base["train"]["y_original"], index=extras.index)
    diagnostics["spearman_target_train"] = [extras[c].corr(y, method="spearman") for c in diagnostics.feature]
    diagnostics.to_csv(output / "train_feature_diagnostics.csv", index=False, lineterminator="\n")
    figures = []
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )

    def save(fig, name):
        fig.savefig(output / name, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figures.append(name)

    screen = tables["screening"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(
        [NAMES[n] for n in screen.variant],
        screen.validation_mae_pp,
        color=["#aebcc3" if n != best else "#247e88" for n in screen.variant],
    )
    for i, r in enumerate(screen.itertuples()):
        ax.text(
            r.validation_mae_pp + 0.01,
            i,
            f"{r.validation_mae_pp:.4f} · p={r.input_neurons}",
            va="center",
            fontsize=9,
        )
    ax.set(
        xlabel="Validation MAE (SOH points), seed 42",
        title="Screening changes feature blocks only, model and procedure fixed",
        xlim=(0, screen.validation_mae_pp.max() * 1.24),
    )
    fig.tight_layout()
    save(fig, "01_feature_screening.png")

    confirmations = tables["seed_confirmation"]
    gains = tables["paired_seed_improvements"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for seed, group in confirmations.groupby("seed"):
        tab = group.set_index("variant").reindex(["control_no_domain_features", best])
        axes[0].plot([0, 1], tab.validation_mae_pp, "o-", label=f"seed {seed}")
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["Control", "Best FE"],
        ylabel="Validation MAE (SOH points)",
        title="Paired comparison over three seeds",
    )
    axes[0].legend(fontsize=8)
    axes[1].bar(gains.seed.astype(str), gains.mae_improvement_pp, color="#247e88")
    axes[1].axhline(
        study["mean_validation_mae_improvement_pp"], color="#a4548e", label="Mean gain over three seeds"
    )
    axes[1].axhline(
        study["decision_threshold_pp"], color="gray", ls="--", label="Promotion threshold for the mean"
    )
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set(
        xlabel="Seed",
        ylabel="Control MAE minus FE MAE (SOH points)",
        title="Positive gain = lower FE error (threshold is not a statistical test)",
    )
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "02_paired_seed_improvements.png")

    order = ["<70", "70–<80", "80–<90", "90–<95", ">=95"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for offset, variant, color, label in [
        (-0.18, "control_no_domain_features", "#aebcc3", "Control"),
        (0.18, best, "#247e88", "Best FE"),
    ]:
        tab = bands[bands.variant.eq(variant)].set_index("band").reindex(order)
        ax.bar(
            np.arange(5) + offset,
            tab.mae_mean_pp,
            width=0.35,
            yerr=tab.mae_sd_pp,
            capsize=4,
            color=color,
            label=label,
        )
    counts = bands[bands.variant.eq(best)].set_index("band").reindex(order).n_rows
    ax.set(
        xticks=range(5),
        xticklabels=[f"{b}\nn={int(n)}" for b, n in zip(order, counts)],
        ylabel="MAE (SOH points), mean ± SD over three seeds",
        xlabel="Actual validation SOH band (%)",
        title="Check the low-SOH tail, not only the overall score",
    )
    ax.legend()
    fig.tight_layout()
    save(fig, "03_validation_tail_errors.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for run_id, label, color in [
        (control_selected.run_id, "Control", "#aebcc3"),
        (selected.run_id, "Best FE", "#247e88"),
    ]:
        history = pd.read_csv(root / "results/experiments/runs" / run_id / "history.csv")
        history = history.loc[history.epoch.ge(5)]
        axes[0].plot(history.epoch, history.validation_mae_pp, label=label, color=color)
    axes[0].set(
        xlabel="Epoch (from 5)",
        ylabel="Validation MAE (SOH points)",
        title="Seed-42 curves with the same early-stopping rule",
    )
    axes[0].legend()
    axes[1].bar(["Control", "Best FE"], [control.parameters, fe.parameters], color=["#aebcc3", "#247e88"])
    axes[1].set(
        ylabel="MLP parameters",
        title=f"Hidden layers fixed at {list(study['reference_config']['hidden_layers'])}, input width grows",
    )
    fig.tight_layout()
    save(fig, "04_learning_and_complexity.png")
    formulas = pd.DataFrame(
        [
            {
                "block": "usage_rates",
                "feature": "fe_cycles_per_day",
                "formula": "cycle_count / battery_age_days",
                "unit": "cycles/day, snapshot ratio",
            },
            {
                "block": "usage_rates",
                "feature": "fe_energy_per_cycle_kwh",
                "formula": "total_energy_throughput_kwh / cycle_count",
                "unit": "kWh/cycle",
            },
            {
                "block": "context",
                "feature": "fe_charge_ambient_delta_c",
                "formula": "peak_temp_during_fast_charge_c - avg_ambient_temp_c",
                "unit": "°C",
            },
            {
                "block": "context",
                "feature": "fe_fastcharge_dod_interaction",
                "formula": "fast_charge_ratio * avg_depth_of_discharge_pct / 100",
                "unit": "ratio interaction",
            },
            {
                "block": "chemistry",
                "feature": "fe_chem_<chemistry>_log_cycles",
                "formula": "I(chemistry=c) * log1p(cycle_count)",
                "unit": "interaction, 4 columns from train chemistries",
            },
            {
                "block": "chemistry",
                "feature": "fe_chem_<chemistry>_log_age_years",
                "formula": "I(chemistry=c) * log1p(battery_age_days / 365.25)",
                "unit": "interaction, 4 columns from train chemistries",
            },
        ]
    )
    formulas.to_csv(output / "feature_definitions.csv", index=False, lineterminator="\n")
    recommendation = (
        "keep the control without domain features as the main candidate"
        if not study["promoted_by_prespecified_gate"]
        else "promote the best engineered variant as the provisional candidate based on validation"
    )
    percentage = study["mean_validation_mae_improvement_pp"] / control.mae_mean_pp * 100
    complexity = (fe.parameters / control.parameters - 1) * 100
    source = f"""# Feature Engineering Ablation for the MLP

## Result and recommendation

**Recommendation: {recommendation}.** The new features were genuinely tested, and a small improvement is not by itself a reason to add complexity.

- The control is the notebook 03 candidate, **not** the original baseline.
- Three-seed mean validation MAE: control **{control.mae_mean_pp:.5f} ± {control.mae_sd_pp:.5f}**, best engineered variant **{fe.mae_mean_pp:.5f} ± {fe.mae_sd_pp:.5f}** SOH points.
- Mean paired gain **{study["mean_validation_mae_improvement_pp"]:.5f} points** ({percentage:.2f}% relative to the control MAE), better on **{study["seed_wins"]}/3** seeds.
- Inputs: control {int(control_selected.input_neurons)}, best variant {int(selected.input_neurons)}. Parameters {int(control.parameters):,} to {int(fe.parameters):,} (**+{complexity:.1f}%**), hidden layers fixed at {list(study["reference_config"]["hidden_layers"])}.
- Promotion rule fixed before training: mean gain of at least **{study["decision_threshold_pp"]:.2f} points** and better on at least two seeds. Passed: **{study["promoted_by_prespecified_gate"]}**. This is a practical heuristic, not a significance test or a physical threshold.
- The test split is not transformed, loaded or evaluated. The synthetic data still limits generalisation.

## 1. Question and protocol

Do derived features reduce MLP error when the hidden layers, activation, output, loss, learning rate, batch size, seeds, checkpointing and early stopping are unchanged? The control is frozen from run `{study["reference_candidate_run_id"]}`: {config_sentence(study["reference_config"], int(control_selected.input_neurons))}, at most {study["reference_config"]["max_epochs"]} epochs, patience {study["reference_config"]["patience"]}.

Four variants are screened on seed 42, then only the variant with the lowest screening MAE is confirmed on seeds 42, 123 and 2026. The three existing control runs are reused with identical fingerprints. This final version adds **{study["new_training_runs"]} new training runs**, and the notebook 03 study history stays intact.

Validation is used for both screening and confirmation, so this is not yet an independent final evaluation. The search is not extended until results look good, and no best seed is selected.

## 2. Feature definitions

{markdown_table(formulas)}

Chemistry categories are taken from the non-missing categories of the frozen train encoder. For a known chemistry, the non-matching interaction columns are zero. For a missing or unseen chemistry, or a missing exposure, the interaction becomes missing. Unknown values are never mapped to one of the known chemistries.

Ratios are only computed when the denominator is positive and both operands are present after domain checks. No epsilon or floor changes the meaning of a division. Unavailable values stay missing until the extra imputation. The target, `capacity_fade_pct`, RUL, risk and degradation labels, IDs, future data and group fills are never used.

Cumulative ratios are snapshot summaries, not degradation rates or evidence of a physical mechanism. Non-monotonic cycle and energy counters remain a limitation of the source, and feature engineering does not repair them.

## 3. Extra preprocessing and fairness

All **67 base pipeline outputs remain bit-for-bit identical** as the input prefix. New features are appended together with their own missing indicators. Medians, 0.5% and 99.5% caps for the two ratios and the temperature delta, and the extra scaler are fitted on train only. The two ratios use signed `log1p` after capping. Chemistry interactions are built on log exposure and are not logged twice. The fast-charge by depth-of-discharge product is not capped because it is already a valid product of ratios.

The target scaler is reused without refitting. No observations or low-SOH rows are removed and no new test arrays are created. The source CSV is read in chunks with input and ID columns only, and test rows are dropped while loading, before any arithmetic or statistics. Target labels are not read in this step.

The MLP configuration is identical except for the input width and the resulting parameter count. Because parameters increase, this ablation measures the combined effect of the feature representation and input capacity, not a causal effect of the features at a fixed parameter count. All parameter counts are recorded.

## 4. Screening

{markdown_table(tables["screening"].round(5))}

![Feature screening](01_feature_screening.png)

The screening winner is `{best}`. A block can make results worse, so not every derived feature has to be used, and losing variants are kept in the record. Single-seed screening can be sensitive to optimisation noise, which is why it is followed by three-seed confirmation. Only the winner is confirmed, so a very small gap to the ratio variant does not prove that the ranking is stable on other seeds.

## 5. Paired confirmation

{markdown_table(tables["seed_summary"].round(5))}

{markdown_table(tables["paired_seed_improvements"].round(5))}

![Paired seed improvements](02_paired_seed_improvements.png)

SD is the sample SD over three seeds on a fixed split, not a population CI or evidence of significance. The 0.01-point threshold applies to the **mean gain**, not to each seed. The engineered variant is better on {study["seed_wins"]} of 3 seeds. With n = 3, both the direction and the size of the gain matter.

## 6. Low-SOH tail and complexity

{markdown_table(bands.round(5))}

![Errors by SOH band](03_validation_tail_errors.png)

Read each band's mean error together with its observation and battery counts and the seed variation. Battery counts can overlap across bands. A feature is not promoted only because one small validation group looks better.

![Learning curves and parameters](04_learning_and_complexity.png)

Learning curves show seed-42 validation from epoch 5. Differences in the number of epochs come from the same early-stopping rule. More features mean more parameters, and adding features does not guarantee less overfitting or underfitting.

## 7. Train feature diagnostics

`train_feature_diagnostics.csv` contains missingness, raw ranges, medians, caps, scaler parameters and the **train** target correlation of every engineered feature. Per-variant parameters and output feature names are stored as well. Correlations support interpretation only and are not used to choose further subsets after seeing validation results. In train, the two ratios are missing in about 10.38% and 11.07% of rows, because a missing operand makes the ratio missing. Derived features add no independent measurements and can increase redundancy and complexity, so a small improvement is plausible without proving its cause.

## 8. Artifacts and reproduction

- Notebook `04_feature_engineering.ipynb`, separate so that the notebook 03 study is not rewritten.
- Config: `configs/feature_engineering.json`.
- Code: `src/ev_battery/features.py`, reusing the MLP training loop unchanged.
- Engineered train and validation arrays: `data/processed/<variant>/`, local and ignored by Git.
- Inference bundle: `models/feature_engineering/<variant>.joblib`. `bundle['preprocessor'].transform(new_rows[bundle['required_columns']])` computes the base and extra features consistently. Load only trusted artifacts.
- MLP checkpoints: `models/mlp/<run_id>/`, local. Config, history, summary and status are in `results/experiments/runs/<run_id>/`.
- Study summary: `results/feature_engineering/study.json`. The control stays recorded in `results/experiments/study.json`.

Reproduce with `python -m ev_battery features`, then run notebook 04. Runs with identical source, data and config fingerprints are reused. For cross-validation, fit the base and extra transforms on every training fold.

## 9. Limits and next stage

The current result supports the decision to **{recommendation}** under the practical rule fixed in advance, while keeping the engineered variant as a valid ablation result. This is not a population-level statistical claim, a test score, or evidence of validity on real batteries.

The next stage locks the configuration and seeds for the test evaluation in notebook 05. The test split is never used to choose new features or adjust the decision threshold.
"""
    (output / "FEATURE_ENGINEERING_REPORT.md").write_text(source, encoding="utf-8", newline="\n")
    return {"figures": figures, "report": output / "FEATURE_ENGINEERING_REPORT.md"}


if __name__ == "__main__":
    print(build_report()["report"])
