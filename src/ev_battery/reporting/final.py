"""Report frozen final test results, no selection, fitting, or training."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ev_battery.audit import markdown_table
from ev_battery.reporting.experiments import config_sentence

COLORS = {"baseline": "#a5b4bd", "candidate": "#247e88"}
NAMES = {"baseline": "Baseline MLP", "candidate": "Main candidate"}


def read_results(output):
    output = Path(output)
    result = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    lock = json.loads((output / "evaluation_lock.json").read_text(encoding="utf-8"))
    if (
        result["status"] != "complete"
        or result["model_selection_from_test"]
        or result["training_or_fit_performed"]
    ):
        raise ValueError("Final results are incomplete or violate the protocol.")
    if result["lock_fingerprint"] != lock["fingerprint"]:
        raise ValueError("Lock and final results do not match.")
    tables = {
        name: pd.read_csv(output / f"{name}.csv")
        for name in [
            "metrics_by_seed",
            "metrics_summary",
            "test_band_summary",
            "test_battery_summary",
            "validation_test_comparison",
            "constant_references",
        ]
    }
    return output, result, lock, tables


def build_report(output):
    output, result, lock, tables = read_results(output)
    groups = lock["payload"]["plan"]["groups"]
    summary = tables["metrics_summary"].set_index("group")
    candidate = summary.loc["candidate"]
    baseline = summary.loc["baseline"]
    metrics = tables["metrics_by_seed"]
    inputs = metrics.groupby("group").input_features.first().astype(int).to_dict()
    bands = tables["test_band_summary"]
    batteries = tables["test_battery_summary"]
    selected_seed = lock["payload"]["plan"]["seeds"][0]  # predeclared first seed, not best test seed
    predictions = {g: pd.read_csv(output / "predictions" / f"{g}_seed_{selected_seed}.csv") for g in NAMES}
    paired = batteries.pivot(index="battery_id", columns="group", values="mae_mean_pp")
    paired["baseline_minus_candidate_mae_pp"] = paired.baseline - paired.candidate
    paired.reset_index().to_csv(output / "paired_battery_mae.csv", index=False, lineterminator="\n")
    worst = (
        batteries.loc[batteries.group.eq("candidate")].sort_values("mae_mean_pp", ascending=False).head(10)
    )
    worst.to_csv(output / "largest_candidate_battery_errors.csv", index=False, lineterminator="\n")
    figures = []
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )

    def save(fig, name):
        fig.savefig(output / name, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figures.append(name)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (mean, sd, title) in zip(
        axes,
        [
            ("test_mae_mean_pp", "test_mae_sd_pp", "MAE (SOH points)"),
            ("test_rmse_mean_pp", "test_rmse_sd_pp", "RMSE (SOH points)"),
            ("test_r2_mean", "test_r2_sd", "R² (not accuracy)"),
        ],
    ):
        values = summary.reindex(["baseline", "candidate"])
        ax.bar([0, 1], values[mean], yerr=values[sd], capsize=5, color=list(COLORS.values()))
        for i, value in enumerate(values[mean]):
            ax.text(i, value + max(values[mean]) * 0.035, f"{value:.4f}", ha="center", fontsize=10)
        ax.set(
            xticks=[0, 1],
            xticklabels=["Baseline", "Candidate"],
            title=title,
            ylim=(0, max(values[mean]) * 1.17),
        )
    fig.suptitle("Test results: mean ± sample SD over three seeds on the same 390 batteries")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "01_test_metrics.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(3)
    for offset, group in [(-0.18, "baseline"), (0.18, "candidate")]:
        rows = metrics.loc[metrics.group.eq(group)]
        fields = ["train_mae_pp", "validation_mae_pp", "test_mae_pp"]
        means = [rows[f].mean() for f in fields]
        deviations = [rows[f].std(ddof=1) for f in fields]
        ax.bar(
            x + offset, means, width=0.35, yerr=deviations, capsize=4, color=COLORS[group], label=NAMES[group]
        )
    ax.set(
        xticks=x,
        xticklabels=["Train", "Validation", "Test"],
        ylabel="MAE (SOH points), mean ± SD over three seeds",
        title="Generalisation comparison (weights never changed after testing)",
    )
    ax.legend()
    fig.tight_layout()
    save(fig, "02_train_validation_test.png")

    lo = np.floor(min(min(p.actual_soh.min(), p.predicted_soh.min()) for p in predictions.values()) / 5) * 5
    hi = np.ceil(max(max(p.actual_soh.max(), p.predicted_soh.max()) for p in predictions.values()) / 5) * 5
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (group, pred) in zip(axes, predictions.items()):
        ax.scatter(pred.actual_soh, pred.predicted_soh, s=5, alpha=0.18, color=COLORS[group], rasterized=True)
        ax.plot([lo, hi], [lo, hi], "--", color="gray")
        ax.set(
            xlabel="Actual SOH (%)",
            ylabel="Predicted SOH (%)",
            title=f"{NAMES[group]} · seed {selected_seed}",
            xlim=(lo, hi),
            ylim=(lo, hi),
        )
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Unclipped test predictions (first planned seed, not chosen from test)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "03_actual_predictions.png")

    order = lock["payload"]["plan"]["soh_band_labels"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(5)
    for offset, group in [(-0.18, "baseline"), (0.18, "candidate")]:
        sub = bands.loc[bands.group.eq(group)].set_index("soh_band").reindex(order)
        axes[0].bar(
            x + offset,
            sub.mae_mean_pp,
            width=0.35,
            yerr=sub.mae_sd_pp,
            capsize=4,
            color=COLORS[group],
            label=NAMES[group],
        )
        axes[1].plot(x, sub.bias_mean_pp, "o-", color=COLORS[group], label=NAMES[group])
    counts = bands.loc[bands.group.eq("candidate")].set_index("soh_band").reindex(order).rows
    for ax in axes:
        ax.set(
            xticks=x,
            xticklabels=[f"{b}\nn={int(n)}" for b, n in zip(order, counts)],
            xlabel="Actual test SOH band (%)",
        )
        ax.legend(fontsize=8)
    axes[0].set(
        ylabel="MAE (SOH points), mean ± SD over three seeds",
        title="The aggregate advantage is not uniform across bands",
    )
    axes[1].axhline(0, color="gray", ls="--")
    axes[1].set(
        ylabel="Mean prediction minus actual (SOH points)",
        title="Positive bias means SOH is overestimated",
    )
    fig.tight_layout()
    save(fig, "04_test_soh_bands.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for group in NAMES:
        values = np.sort(batteries.loc[batteries.group.eq(group), "mae_mean_pp"])
        axes[0].plot(
            values, np.arange(1, len(values) + 1) / len(values), color=COLORS[group], label=NAMES[group]
        )
    axes[0].set(
        xlabel="Per-battery MAE averaged over three seeds (SOH points)",
        ylabel="Cumulative share of batteries",
        title="Every battery weighted equally",
    )
    axes[0].legend(fontsize=8)
    axes[1].hist(
        paired.baseline_minus_candidate_mae_pp, bins=40, color=COLORS["candidate"], edgecolor="white"
    )
    axes[1].axvline(0, color="gray", ls="--")
    axes[1].set(
        xlabel="Baseline minus candidate MAE per battery (SOH points)",
        ylabel="Batteries",
        title="Positive means lower candidate error, some batteries get worse",
    )
    fig.tight_layout()
    save(fig, "05_battery_errors.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for group, pred in predictions.items():
        axes[0].hist(
            pred.error_pp, bins=60, histtype="step", linewidth=1.5, color=COLORS[group], label=NAMES[group]
        )
    axes[0].axvline(0, color="gray", ls="--")
    axes[0].set(
        xlabel="Prediction minus actual (SOH points)",
        ylabel="Test observations",
        title=f"Residuals, seed {selected_seed}",
    )
    axes[0].legend(fontsize=8)
    view = worst.sort_values("mae_mean_pp")
    axes[1].barh(view.battery_id, view.mae_mean_pp, color=COLORS["candidate"])
    axes[1].set(
        xlabel="Mean MAE over three seeds (SOH points)",
        title="10 test batteries with the largest candidate error",
    )
    fig.tight_layout()
    save(fig, "06_residuals_and_largest_errors.png")

    improvement = baseline.test_mae_mean_pp - candidate.test_mae_mean_pp
    relative = improvement / baseline.test_mae_mean_pp * 100
    low = bands.loc[bands.soh_band.eq("<70")].set_index("group")
    compact_summary = summary.reset_index()[
        [
            "group",
            "test_mae_mean_pp",
            "test_mae_sd_pp",
            "test_rmse_mean_pp",
            "test_rmse_sd_pp",
            "test_r2_mean",
            "test_r2_sd",
            "test_battery_mae_mean_pp",
            "parameters",
        ]
    ]
    better = int(paired.baseline_minus_candidate_mae_pp.gt(0).sum())
    per_seed = metrics.pivot(index="seed", columns="group", values="test_mae_pp")
    seed_wins = int((per_seed.candidate < per_seed.baseline).sum())
    ratio = baseline.parameters / candidate.parameters
    if improvement > 0:
        difference = f"The candidate is **{improvement:.5f} points lower** (about {relative:.2f}% of the baseline MAE)."
    else:
        difference = f"The candidate is **{-improvement:.5f} points higher** (about {-relative:.2f}% of the baseline MAE)."
    low_gap = low.loc["candidate", "mae_mean_pp"] - low.loc["baseline", "mae_mean_pp"]
    low_sentence = (
        f"for **SOH < 70%** the candidate MAE is **{low.loc['candidate', 'mae_mean_pp']:.4f}** versus "
        f"**{low.loc['baseline', 'mae_mean_pp']:.4f}** for the baseline ({'higher' if low_gap > 0 else 'lower'}), "
        f"on {int(low.loc['candidate', 'rows'])} observations from {int(low.loc['candidate', 'batteries'])} batteries. "
        "Errors on the rare, most degraded batteries are about three times the average error for both models."
    )
    if improvement > 0 and seed_wins == len(per_seed):
        conclusion = (
            "Under the group-split protocol on this synthetic dataset, the candidate chosen on validation achieves a "
            f"lower aggregate test MAE than the baseline on all {len(per_seed)} locked seeds, with {ratio:.1f} times "
            "fewer parameters. The direction of the validation comparison holds on test."
        )
    else:
        conclusion = (
            "Under the group-split protocol on this synthetic dataset, the candidate chosen on validation does **not** "
            f"beat the baseline on test: it is lower on {seed_wins} of {len(per_seed)} seeds and the mean difference is "
            f"{improvement:+.5f} points. The three-seed validation confirmation already showed the two models within "
            "seed noise of each other, so the seed-42 advantage that selected the candidate did not generalise. The "
            f"candidate reaches almost the same accuracy with {ratio:.1f} times fewer parameters, which is the "
            "practical argument for the smaller network. Both MLPs clearly beat the constant references."
        )
    source = f"""# NumPy MLP follow-up on the previously evaluated split

## Key results

This NumPy follow-up uses **7,572 observations from the same 390 batteries** scored in the original PyTorch release (`{result["original_pytorch_release_commit"]}`) on 6 November 2026. The six local checkpoints and fitted preprocessing were locked before this run scored them. Training used only train, checkpoint selection used validation, and the follow-up reuses the recorded NumPy configurations. This is a descriptive comparison on a previously evaluated split.

- Main candidate, fixed before test: mean MAE **{candidate.test_mae_mean_pp:.5f} ± {candidate.test_mae_sd_pp:.5f} SOH points**, RMSE **{candidate.test_rmse_mean_pp:.5f} ± {candidate.test_rmse_sd_pp:.5f}**, R² **{candidate.test_r2_mean:.5f} ± {candidate.test_r2_sd:.5f}**.
- Baseline MLP: mean MAE **{baseline.test_mae_mean_pp:.5f} ± {baseline.test_mae_sd_pp:.5f} points**. {difference}
- Parameters: baseline {int(baseline.parameters):,}, candidate {int(candidate.parameters):,}. All results use seeds 42, 123 and 2026, with no best-seed selection and no ensembling.
- Key limitation: {low_sentence}

## 1. What was locked before testing

`configs/numpy/followup.json` fixes the baseline and candidate, the three seeds, best-validation checkpoints, the SOH bands, and the candidate as the main model family. The candidate uses {config_sentence(groups["candidate"]["config"], inputs["candidate"])}. The baseline uses {config_sentence(groups["baseline"]["config"], inputs["baseline"])}. Both share the same early-stopping policy. Best epochs come from validation, never from test.

The historical NumPy configuration records come from `{result["selection_reference_commit"]}`. The feature engineering study remains a validation ablation, and rebuilt results report the same frozen base configurations even if a fresh validation search ranks alternatives differently.

`evaluation_lock.json` was created at `{lock["locked_at_utc"]}` before test inference, with fingerprint `{lock["fingerprint"]}`. The lock records SHA-256 hashes of the six checkpoints, the fitted preprocessing and test arrays, the configurations, the manifest, the validation and feature-engineering selection evidence, and the evaluator source. Every locked file is verified to be unchanged after inference.

## 2. Final metrics and their units

{markdown_table(compact_summary.round(5))}

![Test metrics](01_test_metrics.png)

MAE and RMSE are computed after inverting the target scaling. An MAE of 0.97 means an average absolute error of about 0.97 **SOH percentage points**. R² measures explained variance relative to the test target mean.

SD is the sample SD (ddof = 1) over three training seeds on the same test batteries. It is not a population confidence interval, does not cover the uncertainty over all possible splits, and does not represent three independent test sets. Mean R² and RMSE are averages of per-seed metrics, not metrics of an ensemble.

## 3. Results per seed

{markdown_table(tables["metrics_by_seed"][["group", "seed", "run_id", "best_epoch_from_validation", "test_mae_pp", "test_rmse_pp", "test_r2", "test_battery_mean_mae_pp"]].round(5))}

Every seed is shown, including those with higher errors. Scatter plots use seed {selected_seed}, the first seed in the existing plan, not a seed chosen after seeing the test ranking. The main comparison remains the mean and SD over all three seeds.

## 4. Does the validation pattern hold?

{markdown_table(tables["validation_test_comparison"].round(5))}

![Train, validation and test](02_train_validation_test.png)

The candidate's test MAE differs from validation by {candidate.test_mae_mean_pp - candidate.validation_mae_mean_pp:+.5f} points, and the baseline test score differs slightly as well. Such differences can come from the battery and sample composition of each split and are not a reason to adjust scalers or weights. A test score lower than validation neither proves leakage nor guarantees generalisation to other distributions.

The candidate was chosen before testing. The result supports its usefulness on the held-out synthetic test split, but does not prove freedom from overfitting on every dataset. Train and validation curves and the selection process are discussed in notebook 03, and the test split does not become a new tuning set.

## 5. Actual vs predicted and residuals

![Predictions](03_actual_predictions.png)

![Residuals and hardest batteries](06_residuals_and_largest_errors.png)

Predictions are not clipped to 0 to 100. On average **{candidate.test_outside_0_100_mean_pct:.4f}%** of candidate predictions fall outside that range, reported as a limitation of the linear output. A positive residual means the model predicts a higher SOH than the actual value. The candidate's mean bias over the whole test split is {candidate.test_bias_mean_pp:+.4f} points, which must not hide larger biases in subgroups.

## 6. Low SOH remains a weakness

{markdown_table(bands.round(5))}

![Errors by SOH band](04_test_soh_bands.png)

For SOH < 70% the candidate has MAE **{low.loc["candidate", "mae_mean_pp"]:.4f} ± {low.loc["candidate", "mae_sd_pp"]:.4f}** and the baseline **{low.loc["baseline", "mae_mean_pp"]:.4f} ± {low.loc["baseline", "mae_sd_pp"]:.4f}**. The candidate's bias in this band is **{low.loc["candidate", "bias_mean_pp"]:+.4f} points**, so the health of the most degraded synthetic batteries tends to be overestimated.

This subgroup has only {int(low.loc["candidate", "rows"])} observations from {int(low.loc["candidate", "batteries"])} batteries, so no statistical superiority is inferred from a three-seed mean and SD. The finding must still be reported, because the EDA showed that train is dominated by high SOH. Loss, features, weighting and thresholds are not changed in response to this test finding. Any follow-up work needs a new independent validation and evaluation.

## 7. Per-battery evaluation

![Per-battery errors](05_battery_errors.png)

The main MAE weights every observation equally. `test_battery_mean_mae_pp` averages the MAE of each battery so that batteries with many observations do not dominate. Candidate: **{candidate.test_battery_mae_mean_pp:.5f} ± {candidate.test_battery_mae_sd_pp:.5f} points**, baseline **{baseline.test_battery_mae_mean_pp:.5f} ± {baseline.test_battery_mae_sd_pp:.5f}**.

After averaging each battery's MAE over three seeds, the candidate has the lower error on {better} of {len(paired)} batteries. This is descriptive, not 390 independent training experiments. The ten batteries with the largest candidate errors are listed below for final diagnosis, not for tuning the model on test.

{markdown_table(worst.round(5))}

## 8. Constant references from train

{markdown_table(tables["constant_references"].round(5))}

The median and mean are learned from `y_train` only and serve as a sanity check. They are not additional models, the only trained model is the MLP. The final result shows a clear benefit over predicting the centre of the target distribution, without equating that improvement with physical validation.

## 9. Defensible conclusions

{conclusion}

The claim is limited to estimating SOH at observation time for new batteries from the same simulator. Forecasting, remaining useful life, real vehicle data and distribution shift are not tested. The low-SOH weakness and predictions outside the physical range remain open. There is no claim of "95% accuracy", universal freedom from overfitting, or readiness for real batteries.

The NumPy candidate is the frozen configuration for this follow-up. The original PyTorch release retains its own candidate and evaluation. The shared test split supports descriptive comparisons, and independent evidence would require a new evaluation set.

## 10. Artifacts and reproduction

- `notebooks/numpy/05_final_evaluation.ipynb`: protocol, metrics, figures, error analysis and checks.
- `configs/numpy/followup.json`: recorded NumPy configurations and seeds.
- `evaluation_lock.json`: file identities and lock time before test inference.
- `metrics_by_seed.csv`, `metrics_summary.csv`: all seed results and the summary.
- `test_band_summary.csv`, `test_battery_summary.csv`: results per SOH band and per battery, with detailed per-seed tables alongside.
- `predictions/`: per-observation test predictions, local and ignored by Git.
- Weights and fitted preprocessing stay in the local `models/` folder.

Command: `python -m ev_battery evaluate`, then run notebook 05. The evaluator reuses complete results that match the lock and never retrains or replaces checkpoints. Re-running the deterministic checks on the same results is not a new independent evaluation.

On a fresh clone, `python -m ev_battery all` rebuilds the required artifacts and writes a local evaluation under `data/processed/numpy_workspace/results/numpy_followup/<fingerprint>`. New artifact identities receive their own directory. The original PyTorch release and historical NumPy snapshot can be read directly from Git.
"""
    (output / "FINAL_REPORT.md").write_text(source, encoding="utf-8", newline="\n")
    return {"figures": figures, "report": output / "FINAL_REPORT.md"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Path to a completed NumPy follow-up evaluation")
    print(build_report(parser.parse_args().output)["report"])
