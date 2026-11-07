"""Read completed train/validation experiments, generate figures and discussion."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from ev_battery.audit import markdown_table
from ev_battery.paths import ROOT

STAGE_PARAMETER = {
    "preprocessing": "recipe",
    "hidden_layers": "hidden_layers",
    "activation": "activation",
    "loss": "loss",
    "learning_rate": "learning_rate",
    "optimizer": "optimizer",
    "balancing": "sample_weighting",
    "max_error": "max_error",
}


def read_results(root=ROOT):
    output = root / "results/experiments"
    study = json.loads((output / "study.json").read_text(encoding="utf-8"))
    if study["status"] != "complete" or study["test_loaded"]:
        raise ValueError("Study is incomplete or the test boundary was violated.")
    runs = pd.read_csv(output / "run_summary.csv")
    tables = {
        name: pd.read_csv(output / f"{name}.csv")
        for name in [
            "stage_comparisons",
            "epoch_budget_comparison",
            "capacity_demo",
            "seed_confirmation",
            "seed_summary",
            "constant_references",
            "candidate_validation_bands",
            "balancing_validation_bands",
        ]
    }
    summaries = {}
    histories = {}
    for run_id in runs.run_id:
        summaries[run_id] = json.loads(
            (output / "runs" / run_id / "summary.json").read_text(encoding="utf-8")
        )
        histories[run_id] = pd.read_csv(output / "runs" / run_id / "history.csv")
    return output, study, runs, tables, summaries, histories


def stage_label(row):
    stage = row.stage
    if stage == "optimizer":
        return f"{row.optimizer} (lr={row.learning_rate:g})"
    value = getattr(row, STAGE_PARAMETER[stage])
    return f"{value:g}" if isinstance(value, float) else str(value)


def build_figures(root=ROOT):
    output, study, runs, tables, summaries, histories = read_results(root)
    figures = []
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )

    def save(fig, name):
        fig.savefig(output / name, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figures.append(name)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, (label, key) in enumerate(
        [("Baseline", "baseline_run_id"), ("Candidate seed 42", "candidate_run_id")]
    ):
        run = summaries[study[key]]
        history = histories[study[key]]
        for split, color in [("train", "#247e88"), ("validation", "#c47a21")]:
            axes[row, 0].plot(history.epoch, history[f"{split}_objective_scaled"], color=color, label=split)
            zoom = history.loc[history.epoch.ge(5)]
            axes[row, 1].plot(zoom.epoch, zoom[f"{split}_mae_pp"], color=color, label=split)
        for ax in axes[row]:
            ax.axvline(
                run["best_epoch"], ls="--", color="#a4548e", lw=1, label=f"best epoch {run['best_epoch']}"
            )
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=8)
        axes[row, 0].set(title=f"{label}: {run['config']['loss']} loss on standardised y", ylabel="Loss")
        axes[row, 1].set(title=f"{label}: MAE on original SOH, epoch ≥5", ylabel="MAE (percentage points)")
    fig.suptitle("Learning curves at the end of every epoch (checkpoint chosen by validation MAE)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "01_learning_curves.png")

    stages = tables["stage_comparisons"]
    names = list(dict.fromkeys(stages.stage))
    cols = 2
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows))
    for ax, stage in zip(axes.flat, names):
        group = stages.loc[stages.stage.eq(stage)]
        labels = [stage_label(r) for r in group.itertuples()]
        ax.barh(
            labels, group.validation_mae_pp, color=["#247e88" if s else "#aebcc3" for s in group.selected]
        )
        for i, value in enumerate(group.validation_mae_pp):
            ax.text(value + 0.01, i, f"{value:.3f}", va="center", fontsize=9)
        ax.set(
            title=stage,
            xlabel="Validation MAE (percentage points)",
            xlim=(0, group.validation_mae_pp.max() * 1.25),
        )
    for ax in list(axes.flat)[len(names) :]:
        ax.axis("off")
    fig.suptitle("Staged experiments: one factor per stage, the green bar is carried forward")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "02_stage_comparisons.png")

    candidate = summaries[study["candidate_run_id"]]
    config = candidate["config"]
    blocks = (
        [f"Input\n{candidate['input_features']} neurons"]
        + [
            f"Hidden {i + 1}\n{units} · {config['activation']}"
            for i, units in enumerate(config["hidden_layers"])
        ]
        + ["Output\n1 · linear"]
    )
    fig, ax = plt.subplots(figsize=(12, 3.7))
    ax.set_xlim(0, len(blocks) * 2)
    ax.set_ylim(-1, 2)
    ax.axis("off")
    for i, label in enumerate(blocks):
        x = i * 2 + 0.15
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.6), 1.65, 0.9, boxstyle="round,pad=.06", facecolor="#e1f0ed", edgecolor="#247e88"
            )
        )
        ax.text(x + 0.825, 1.05, label, ha="center", va="center", fontsize=11)
        if i:
            ax.annotate(
                "",
                xy=(x - 0.08, 1.05),
                xytext=(x - 0.3, 1.05),
                arrowprops={"arrowstyle": "->", "color": "#247e88"},
            )
    ax.annotate(
        "",
        xy=(0.3, 0.1),
        xytext=(len(blocks) * 2 - 0.3, 0.1),
        arrowprops={"arrowstyle": "->", "color": "#a4548e", "lw": 2},
    )
    optimizer = "gradient descent" if config["optimizer"] == "gd" else "Adam"
    ax.text(
        len(blocks),
        -0.23,
        f"Backpropagation (NumPy, from scratch): deltas flow back through every layer, {optimizer} updates the weights",
        ha="center",
        fontsize=10,
    )
    ax.text(
        len(blocks),
        1.8,
        f"Candidate MLP: {candidate['parameters']:,} parameters · {config['loss']} loss · {optimizer} lr={config['learning_rate']:g}",
        ha="center",
        fontsize=13,
    )
    save(fig, "03_candidate_architecture.png")

    fixed = tables["epoch_budget_comparison"]
    history = histories[study["fixed_epoch_run_id"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for split, color in [("train", "#247e88"), ("validation", "#c47a21")]:
        axes[0].plot(history.epoch, history[f"{split}_mae_pp"], color=color, label=split)
        axes[0].scatter(fixed.epoch, fixed[f"{split}_mae"], color=color, zorder=3)
    axes[0].set(
        title="One trajectory without early stopping", xlabel="Epoch", ylabel="MAE (percentage points)"
    )
    axes[0].legend()
    x = np.arange(len(fixed))
    axes[1].bar(x - 0.18, fixed.train_mae, width=0.35, color="#247e88", label="train")
    axes[1].bar(x + 0.18, fixed.validation_mae, width=0.35, color="#c47a21", label="validation")
    axes[1].set(
        xticks=x,
        xticklabels=fixed.epoch.astype(str),
        xlabel="max_epoch",
        ylabel="MAE (percentage points)",
        title="Weights at the end of each epoch budget",
    )
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "04_epoch_budget.png")

    seeds = tables["seed_confirmation"]
    seed_summary = tables["seed_summary"].set_index("group")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for seed in sorted(seeds.seed.unique()):
        pair = seeds[seeds.seed.eq(seed)].set_index("group").reindex(["baseline", "candidate"])
        axes[0].plot([0, 1], pair.validation_mae_pp, "o-", label=f"seed {seed}")
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["Baseline", "Candidate"],
        ylabel="Validation MAE (percentage points)",
        title="All seeds reported, no best-seed selection",
    )
    axes[0].legend(fontsize=8)
    selected = seed_summary.reindex(["baseline", "candidate"])
    axes[1].bar(
        [0, 1],
        selected.validation_mae_mean,
        yerr=selected.validation_mae_std,
        capsize=6,
        color=["#aebcc3", "#247e88"],
    )
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["Baseline", "Candidate"],
        ylabel="Validation MAE (percentage points)",
        title="Mean ± sample SD over three seeds",
    )
    fig.tight_layout()
    save(fig, "05_seed_confirmation.png")

    predictions = pd.read_csv(output / "runs" / study["candidate_run_id"] / "validation_predictions.csv")
    error = predictions.predicted_soh - predictions.actual_soh
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].scatter(
        predictions.actual_soh, predictions.predicted_soh, s=5, alpha=0.18, color="#247e88", rasterized=True
    )
    axes[0].plot([50, 105], [50, 105], ls="--", color="gray")
    axes[0].set(xlabel="Actual SOH (%)", ylabel="Predicted SOH (%)", title="Validation, candidate seed 42")
    axes[1].hist(error, bins=50, color="#247e88", edgecolor="white")
    axes[1].axvline(0, ls="--", color="gray")
    axes[1].set(
        xlabel="Prediction minus actual (percentage points)", ylabel="Observations", title="Residuals"
    )
    bands = tables["candidate_validation_bands"]
    axes[2].bar(bands.soh_band, bands.mae_pp, color="#c47a21")
    for i, row in enumerate(bands.itertuples()):
        axes[2].text(i, row.mae_pp + 0.03, f"n={row.rows}", ha="center", fontsize=8)
    axes[2].set(xlabel="Actual SOH band (%)", ylabel="MAE (percentage points)", title="Error per SOH band")
    axes[2].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    save(fig, "06_validation_errors.png")

    capacity = tables["capacity_demo"]
    fig, axes = plt.subplots(1, len(capacity), figsize=(12, 4), sharey=True)
    for ax, row in zip(np.atleast_1d(axes), capacity.itertuples()):
        history = histories[row.run_id]
        for split, color in [("train", "#247e88"), ("validation", "#c47a21")]:
            ax.plot(history.epoch, history[f"{split}_mae_pp"], color=color, label=split)
        ax.set(
            title=f"{row.scenario}: hidden [{row.hidden_layers}], {row.train_rows:,} train rows",
            xlabel="Epoch",
            ylabel="MAE (percentage points)",
        )
        ax.legend(fontsize=8)
    fig.suptitle("Underfitting and overfitting, reproduced in the style of LC 7")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "07_underfitting_overfitting.png")

    balancing = tables["balancing_validation_bands"]
    order = list(dict.fromkeys(balancing.soh_band))
    schemes = list(dict.fromkeys(balancing.sample_weighting))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.8 / len(schemes)
    for i, scheme in enumerate(schemes):
        sub = balancing[balancing.sample_weighting.eq(scheme)].set_index("soh_band").reindex(order)
        ax.bar(
            np.arange(len(order)) + (i - (len(schemes) - 1) / 2) * width,
            sub.mae_pp,
            width=width,
            label=scheme,
        )
    counts = balancing[balancing.sample_weighting.eq(schemes[0])].set_index("soh_band").reindex(order).rows
    ax.set(
        xticks=range(len(order)),
        xticklabels=[f"{b}\nn={int(n)}" for b, n in zip(order, counts)],
        xlabel="Actual validation SOH band (%)",
        ylabel="MAE (percentage points)",
        title="SOH-band sample weighting (balancing) shifts error between bands",
    )
    ax.legend(title="sample_weighting")
    fig.tight_layout()
    save(fig, "08_balancing_by_band.png")
    return figures


def config_sentence(c, input_features):
    """Readable one-line description of a run configuration."""
    optimizer = "gradient descent" if c["optimizer"] == "gd" else "Adam"
    return (
        f"`{c['recipe']}`, {input_features} inputs, hidden {list(c['hidden_layers'])} {c['activation']},"
        f" linear output, {c['loss'].upper()} loss, {optimizer} with learn_rate {c['learning_rate']:g},"
        f" batch size {c['batch_size']}, sample weighting `{c['sample_weighting']}`, max_error {c['max_error']:g}"
    )


def describe_config(run):
    return config_sentence(run["config"], run["input_features"])


def write_report(root=ROOT):
    output, study, runs, tables, summaries, histories = read_results(root)
    baseline = summaries[study["baseline_run_id"]]
    candidate = summaries[study["candidate_run_id"]]
    bcfg, cfg = baseline["config"], candidate["config"]
    confirmation = tables["seed_summary"].set_index("group")
    stages = tables["stage_comparisons"]
    epochs = tables["epoch_budget_comparison"]
    capacity = tables["capacity_demo"].set_index("scenario")
    fixed_summary = summaries[study["fixed_epoch_run_id"]]
    balancing = tables["balancing_validation_bands"]
    low = balancing.loc[balancing.soh_band.eq("<70")].set_index("sample_weighting")

    winners = []
    for stage in dict.fromkeys(stages.stage):
        group = stages.loc[stages.stage.eq(stage)]
        best = group.loc[group.selected].iloc[0]
        winners.append(
            {
                "stage": stage,
                "values tested": ", ".join(stage_label(r) for r in group.itertuples()),
                "selected": stage_label(best),
                "validation MAE (pp)": round(best.validation_mae_pp, 4),
            }
        )
    winners = pd.DataFrame(winners)
    stage_table = stages.assign(value=[stage_label(r) for r in stages.itertuples()])[
        [
            "stage",
            "value",
            "validation_mae_pp",
            "train_mae_pp",
            "epochs_actual",
            "best_epoch",
            "stop_reason",
            "selected",
        ]
    ]
    comparison = pd.DataFrame(
        [
            {
                "model": label,
                "recipe": r["config"]["recipe"],
                "hidden": str(list(r["config"]["hidden_layers"])),
                "activation": r["config"]["activation"],
                "loss": r["config"]["loss"],
                "optimizer": r["config"]["optimizer"],
                "learn_rate": r["config"]["learning_rate"],
                "parameters": r["parameters"],
                "best_epoch": r["best_epoch"],
                "epochs_actual": r["epochs_actual"],
                "train_MAE_pp": r["train"]["mae"],
                "validation_MAE_pp": r["validation"]["mae"],
                "validation_R2": r["validation"]["r2"],
            }
            for label, r in [("Baseline seed 42", baseline), ("Candidate seed 42", candidate)]
        ]
    )
    under, over = capacity.loc["underfitting"], capacity.loc["overfitting"]
    source = f"""# MLP Experiments with Backpropagation

## Key results

- The MLP and backpropagation are implemented **from scratch in NumPy** (`src/ev_battery/model.py`), following `bp_fit` and `bp_predict` from the course modules. No deep-learning framework is used.
- **{study["unique_training_runs"]} unique training runs** completed, including the three-seed confirmation and the underfitting and overfitting demonstration.
- Baseline ({describe_config(baseline)}): validation MAE **{baseline["validation"]["mae"]:.4f}** SOH percentage points.
- Candidate ({describe_config(candidate)}): validation MAE **{candidate["validation"]["mae"]:.4f}**, RMSE **{candidate["validation"]["rmse"]:.4f}**, R² **{candidate["validation"]["r2"]:.4f}**.
- Seed confirmation: baseline **{confirmation.loc["baseline", "validation_mae_mean"]:.4f} ± {confirmation.loc["baseline", "validation_mae_std"]:.4f}**, candidate **{confirmation.loc["candidate", "validation_mae_mean"]:.4f} ± {confirmation.loc["candidate", "validation_mae_std"]:.4f}**, three seeds on the fixed split.
- Validation was reused throughout the search, so these are not test scores. The test split is evaluated once in notebook 05.

## 1. Implementation

The network follows the structure of the course `bp_fit`:

- `layer_conf` lists the neurons per layer, for example `({candidate["input_features"]}, {", ".join(map(str, cfg["hidden_layers"]))}, 1)`.
- Weights are a list of matrices of shape `(n_in + 1, n_out)`. The last row is the bias, which corresponds to an extra input neuron fixed at 1.
- Forward pass: `net = [a, 1] @ W`, then the activation (`sig`, `tanh` or `relu`) in hidden layers and a linear output for regression.
- Backward pass: the output delta is the loss gradient, `delta = (delta_next @ W_next[:-1].T) * f'(net)` for hidden layers, and the weight gradient is `[a, 1].T @ delta`.
- Update: plain gradient descent `W = W - learn_rate * gradient` as in the course, or Adam (written by hand) as a comparison.
- Stopping: `max_epoch`, `max_error` on the epoch MSE (as in the course), and early stopping on validation MAE with patience {cfg["patience"]}.

Updates use mini-batches of {cfg["batch_size"]} samples, because per-sample updates over {study["train_rows"]:,} rows would make each experiment take hours. The gradient formula is identical: with `batch_size=1`, no shuffling and a sigmoid output, a unit test shows that the implementation produces exactly the same weights as the course `bp_fit`. A second test checks every gradient against finite differences.

![Candidate architecture](03_candidate_architecture.png)

## 2. Reference configuration

The baseline mirrors the course defaults: sigmoid hidden units, MSE loss and plain gradient descent with learn_rate {bcfg["learning_rate"]:g}, batch size {bcfg["batch_size"]}, at most {bcfg["max_epochs"]} epochs and seed 42 during the search.

{markdown_table(comparison.round(4))}

![Learning curves](01_learning_curves.png)

Train and validation losses are computed with the same weights at the end of every epoch. Epoch 0 is the untrained network and serves only as a reference.

## 3. Staged experiment matrix

The search changes one factor per stage and carries the validation winner forward:

{markdown_table(winners)}

![Stage comparisons](02_stage_comparisons.png)

All runs:

{markdown_table(stage_table.round(4))}

This is a sequential search, **not a full factorial design**: it does not isolate every interaction and does not prove a global optimum. Raw loss values are never compared across MSE, MAE and Huber, every decision uses validation MAE in original SOH units. The Huber delta of 1 on the standardised target corresponds to about 7.19 SOH points.

## 4. Optimizer: gradient descent vs Adam

Plain gradient descent (the course rule) uses one fixed learn_rate for every weight. Adam keeps running averages of each gradient and its square, which gives every weight its own adaptive step size. The optimizer stage compares the two on the configuration selected so far, see the `optimizer` rows above.

## 5. Balancing with SOH-band weights

The target is continuous, so class balancing in the classification sense does not apply. The regression counterpart is to weight each training sample by the inverse frequency of its SOH band, so that rare low-SOH observations contribute more to the loss. Weights are computed on train only and normalised to a mean of 1.

{markdown_table(balancing.round(4))}

![Balancing by band](08_balancing_by_band.png)

For SOH below 70%, validation MAE is **{low.loc["none", "mae_pp"]:.3f}** without weighting, **{low.loc["inverse_sqrt", "mae_pp"]:.3f}** with `inverse_sqrt` and **{low.loc["inverse", "mae_pp"]:.3f}** with `inverse` weights. The selection rule uses overall validation MAE, so balancing is only selected if it also improves the aggregate error.

## 6. Epoch budget

![Epoch budget](04_epoch_budget.png)

{markdown_table(epochs[["epoch", "train_mae", "validation_mae", "validation_rmse", "validation_r2"]].round(4))}

The table uses the **endpoint weights at epochs 50, 100 and 200** of one trajectory without early stopping. Because the trajectories are nested, this is a controlled budget experiment, not three independent replicates. The best checkpoint of this diagnostic trajectory is at epoch {fixed_summary["best_epoch"]} (MAE {fixed_summary["validation"]["mae"]:.4f}). The run is not promoted to the candidate.

## 7. Underfitting and overfitting

Reproducing LC 7 on this dataset:

{markdown_table(tables["capacity_demo"].round(4))}

![Underfitting and overfitting](07_underfitting_overfitting.png)

- **Underfitting:** hidden [{under.hidden_layers}], max_epoch {int(under.max_epochs)} and max_error {under.max_error:g}. Training stopped after {int(under.epochs_actual)} epoch(s) ({under.stop_reason}). Train MAE {under.train_mae_pp:.3f} and validation MAE {under.validation_mae_pp:.3f} are both high: the model is too small and stops learning too early.
- **Overfitting:** hidden [{over.hidden_layers}] trained for {int(over.epochs_actual)} epochs without early stopping on only {int(over.train_rows):,} rows ({int(over.train_rows / study["train_rows"] * 100)}% of train). After the last epoch, train MAE is {over.final_train_mae_pp:.3f} while validation MAE is {over.final_validation_mae_pp:.3f}, a gap of {over.final_gap_pp:.3f} points: the network memorises the small training set. Its best validation epoch was {int(summaries[over.run_id]["best_epoch"])}, after which validation error rises while train error keeps falling.

The full study does not overfit strongly because it trains on {study["train_rows"]:,} rows with early stopping.

## 8. Seed confirmation

![Seed confirmation](05_seed_confirmation.png)

{markdown_table(tables["seed_confirmation"].round(4))}

SD is the sample standard deviation (ddof = 1) over three training seeds, which control the initial weights and the shuffling order. It is not a confidence interval over batteries.

## 9. Validation error analysis

![Validation errors](06_validation_errors.png)

{markdown_table(tables["candidate_validation_bands"].round(4))}

Predictions are not clipped to 0 to 100. The seed-42 candidate produces {candidate["validation"]["outside_0_100_pct"]:.3f}% predictions outside that range. The mean per-battery MAE is **{candidate["validation"]["battery_mean_mae"]:.4f}** points.

## 10. Sanity check against train constants

{markdown_table(tables["constant_references"].round(4))}

The train median and mean are not competing models. They show that the MLP does more than predict the dominant high-SOH region.

## 11. Artifacts and reproduction

- `configs/mlp_study.json`: baseline, stages and protocol.
- `results/experiments/run_summary.csv`: one row per unique training run.
- `runs/<run_id>/`: config, per-epoch history, summary and status.
- `models/mlp/<run_id>/`: best, last and fixed-epoch weights as `.npz`, local and ignored by Git.

Command: `python -m ev_battery train`, or run notebook 03. Identical runs are reused through a fingerprint of the code, data and configuration.
"""
    (output / "EXPERIMENT_REPORT.md").write_text(source, encoding="utf-8", newline="\n")
    return output / "EXPERIMENT_REPORT.md"


def build_report(root=ROOT):
    figures = build_figures(root)
    report = write_report(root)
    return {"figures": figures, "report": report}


if __name__ == "__main__":
    print(build_report()["report"])
