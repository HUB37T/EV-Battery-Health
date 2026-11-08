"""Training runs and the staged MLP study (train and validation only).

Every run trains the from-scratch NumPy MLP in ``model.py`` with ``bp_fit``. This
module adds what a reproducible study needs around it: per-epoch validation
monitoring, best-checkpoint selection, early stopping, caching of identical runs,
and the staged experiment matrix defined in ``configs/mlp_study.json``.

python -m ev_battery.training --baseline
python -m ev_battery.training
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from ev_battery.model import (
    MLP,
    backward,
    bp_fit,
    forward,
    init_weights,
    loss_gradient,
    loss_value,
    n_parameters,
)
from ev_battery.paths import ROOT
from ev_battery.preprocessing import load_prepared_split

SOH_BAND_EDGES = [-np.inf, 70, 80, 90, 95, np.inf]
SOH_BAND_LABELS = ["<70", "70–<80", "80–<90", "90–<95", ">=95"]
FIXED_EPOCHS = (50, 100, 200)


@dataclass(frozen=True)
class RunConfig:
    recipe: str = "baseline_standard"
    hidden_layers: tuple = (32,)
    activation: str = "sigmoid"
    loss: str = "mse"
    huber_delta: float = 1.0
    optimizer: str = "gd"
    learning_rate: float = 0.1
    batch_size: int = 32
    max_epochs: int = 200
    max_error: float = 0.0
    early_stopping: bool = True
    patience: int = 15
    min_delta_pp: float = 0.001
    sample_weighting: str = "none"
    train_batteries: int = 0
    seed: int = 42

    def __post_init__(self):
        object.__setattr__(self, "hidden_layers", tuple(int(h) for h in self.hidden_layers))


def regression_metrics(y, pred, battery_ids=None):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if (
        y.ndim != 1
        or pred.shape != y.shape
        or not len(y)
        or not np.isfinite(y).all()
        or not np.isfinite(pred).all()
    ):
        raise ValueError("Targets and predictions must be finite vectors of equal shape.")
    error = pred - y
    denominator = float(np.square(y - y.mean()).sum())
    metrics = {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": float(1 - np.square(error).sum() / denominator) if denominator else None,
        "bias_pp": float(error.mean()),
        "outside_0_100_pct": float(((pred < 0) | (pred > 100)).mean() * 100),
    }
    if battery_ids is not None:
        if len(battery_ids) != len(y):
            raise ValueError("battery_ids length does not match the targets.")
        metrics["battery_mean_mae"] = float(
            pd.Series(np.abs(error)).groupby(np.asarray(battery_ids)).mean().mean()
        )
    return metrics


def source_sha256():
    """Hash of every source file that defines a training run (model and loop)."""
    digest = hashlib.sha256()
    for path in [Path(__file__).with_name("model.py"), Path(__file__)]:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_training_data(recipe, root=ROOT):
    """Load the train and validation arrays of one recipe. The test split is never requested."""
    train, bundle = load_prepared_split(recipe, "train", root)
    validation, other = load_prepared_split(recipe, "validation", root)
    if bundle["state_hash"] != other["state_hash"] or set(train["battery_id"]) & set(
        validation["battery_id"]
    ):
        raise ValueError("Pipelines differ or train/validation batteries overlap.")
    digest = hashlib.sha256()
    for data in [train, validation]:
        for key in ["X", "y_scaled", "y_original", "observation_id", "battery_id", "feature_names"]:
            digest.update(np.ascontiguousarray(data[key]).tobytes())
    digest.update(bundle["state_hash"].encode())
    return {"train": train, "validation": validation, "bundle": bundle, "fingerprint": digest.hexdigest()}


def configure_runtime(seed):
    """Seed NumPy's legacy global generator. Training itself uses explicit generators."""
    np.random.seed(seed)


def predict_scaled(model, X):
    return model.predict(X)


def original_scale(scaled, bundle):
    return (
        bundle["target_scaler"].inverse_transform(np.asarray(scaled, dtype=np.float64).reshape(-1, 1)).ravel()
    )


def soh_band_weights(y_original, scheme):
    """Per-sample weights that balance SOH bands (regression form of class balancing).

    ``inverse`` gives equal total weight per band across the training set before
    mini-batch normalisation. Each update divides by its batch weight sum, so
    effective contributions depend on batch composition. ``inverse_sqrt`` is milder.
    """
    if scheme == "none":
        return None
    bands = pd.cut(y_original, SOH_BAND_EDGES, right=False, labels=False)
    counts = np.bincount(bands, minlength=len(SOH_BAND_LABELS)).astype(float)
    if scheme == "inverse":
        per_band = np.divide(1.0, counts, out=np.zeros_like(counts), where=counts > 0)
    elif scheme == "inverse_sqrt":
        per_band = np.divide(1.0, np.sqrt(counts), out=np.zeros_like(counts), where=counts > 0)
    else:
        raise ValueError(f"Unknown sample weighting {scheme!r}.")
    weights = per_band[bands]
    return weights / weights.mean()


def subset_train(train, n_batteries):
    """Deterministic subset of train batteries, used only for the overfitting demonstration."""
    if not n_batteries:
        return train
    ids = np.array(sorted(set(train["battery_id"])))
    if n_batteries > len(ids):
        raise ValueError("train_batteries exceeds the number of train batteries.")
    chosen = set(np.random.default_rng(0).permutation(ids)[:n_batteries])
    mask = np.isin(train["battery_id"], list(chosen))
    return {
        key: (
            value[mask] if key in {"X", "y_scaled", "y_original", "observation_id", "battery_id"} else value
        )
        for key, value in train.items()
    }


def evaluate(model, data, bundle, loss, huber_delta):
    scaled = model.predict(data["X"])
    objective = loss_value(loss, scaled, data["y_scaled"], huber_delta=huber_delta)
    metrics = regression_metrics(data["y_original"], original_scale(scaled, bundle), data["battery_id"])
    return objective, metrics


def save_checkpoint(path, model, meta):
    arrays = {f"w{i}": w for i, w in enumerate(model.weights)}
    np.savez(path, meta=np.array(json.dumps(meta, sort_keys=True)), **arrays)


def load_checkpoint(path, expected_data=None):
    with np.load(path, allow_pickle=False) as stored:
        meta = json.loads(stored["meta"].item())
        weights = [stored[f"w{i}"] for i in range(len(stored.files) - 1)]
    cfg = RunConfig(**meta["config"])
    model = MLP(weights, cfg.activation)
    if expected_data is not None and (
        meta["data_fingerprint"] != expected_data["fingerprint"]
        or meta["feature_names"] != expected_data["train"]["feature_names"].tolist()
    ):
        raise ValueError("Checkpoint does not match the preprocessed inputs.")
    return model, meta


def validate_config(config):
    if (
        config.batch_size < 1
        or config.max_epochs < 1
        or config.patience < 1
        or config.learning_rate <= 0
        or config.min_delta_pp < 0
        or config.max_error < 0
        or config.train_batteries < 0
        or not config.hidden_layers
    ):
        raise ValueError("Invalid training hyperparameters.")


def train_run(config: RunConfig, data=None, root=ROOT):
    validate_config(config)
    if data is None:
        data = load_training_data(config.recipe, root)
    identity = {
        "config": asdict(config),
        "data_fingerprint": data["fingerprint"],
        "training_source_sha256": source_sha256(),
        "numpy": np.__version__,
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    run_id = "run_" + fingerprint[:12]
    output = root / "results/experiments/runs" / run_id
    checkpoints = root / "models/mlp" / run_id
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.is_file():
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") != fingerprint:
            raise ValueError("Run ID collision or changed provenance.")
        if (
            cached.get("status") == "complete"
            and (checkpoints / "best.npz").is_file()
            and (checkpoints / "last.npz").is_file()
            and (output / "history.csv").is_file()
        ):
            return cached
    write_json(output / "config.json", identity)
    write_json(output / "status.json", {"status": "running", "run_id": run_id, "fingerprint": fingerprint})
    configure_runtime(config.seed)
    train = subset_train(data["train"], config.train_batteries)
    val = data["validation"]
    bundle = data["bundle"]
    layer_conf = (train["X"].shape[1], *config.hidden_layers, 1)
    sample_weight = soh_band_weights(train["y_original"], config.sample_weighting)
    base_meta = {
        "run_id": run_id,
        "config": asdict(config),
        "layer_conf": list(layer_conf),
        "feature_names": train["feature_names"].tolist(),
        "data_fingerprint": data["fingerprint"],
        "fingerprint": fingerprint,
    }
    history = []
    fixed_rows = []
    tracker = {"best_mae": float("inf"), "best_epoch": 0, "reference": float("inf"), "stale": 0}
    started = time.perf_counter()

    def monitor(epoch, weights, epoch_mse=None):
        model = MLP([w.copy() for w in weights], config.activation)
        tr_loss, tr = evaluate(model, train, bundle, config.loss, config.huber_delta)
        va_loss, va = evaluate(model, val, bundle, config.loss, config.huber_delta)
        history.append(
            {
                "epoch": epoch,
                "train_objective_scaled": tr_loss,
                "validation_objective_scaled": va_loss,
                "epoch_mse_scaled": epoch_mse,
                "train_mae_pp": tr["mae"],
                "validation_mae_pp": va["mae"],
                "train_rmse_pp": tr["rmse"],
                "validation_rmse_pp": va["rmse"],
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return model, tr, va

    def on_epoch_end(epoch, weights, epoch_mse):
        model, tr, va = monitor(epoch, weights, epoch_mse)
        if va["mae"] < tracker["best_mae"]:
            tracker.update(best_mae=va["mae"], best_epoch=epoch)
            save_checkpoint(checkpoints / "best.npz", model, {**base_meta, "epoch": epoch})
        if va["mae"] < tracker["reference"] - config.min_delta_pp:
            tracker.update(reference=va["mae"], stale=0)
        else:
            tracker["stale"] += 1
        if not config.early_stopping and epoch in FIXED_EPOCHS:
            save_checkpoint(checkpoints / f"epoch_{epoch:03d}.npz", model, {**base_meta, "epoch": epoch})
            fixed_rows.append(
                {
                    "run_id": run_id,
                    "epoch": epoch,
                    **{"train_" + k: v for k, v in tr.items()},
                    **{"validation_" + k: v for k, v in va.items()},
                }
            )
        if epoch % 10 == 0:
            pd.DataFrame(history).to_csv(output / "history.csv", index=False, lineterminator="\n")
        if epoch % 50 == 0:
            print(f"  epoch {epoch}: train MAE={tr['mae']:.4f}, validation MAE={va['mae']:.4f}", flush=True)
        return config.early_stopping and tracker["stale"] >= config.patience

    print(
        f"START {run_id} {config.recipe} conf={layer_conf} {config.activation}/{config.loss}"
        f" {config.optimizer} lr={config.learning_rate} weighting={config.sample_weighting} seed={config.seed}",
        flush=True,
    )
    try:
        initial = init_weights(layer_conf, config.seed)
        monitor(0, initial)
        # Diagnostics of the very first mini-batch: the gradient must be finite and non-zero.
        first = slice(0, config.batch_size)
        nets, activations = forward(train["X"][first], initial, config.activation)
        grads = backward(
            nets,
            activations,
            initial,
            loss_gradient(
                config.loss,
                activations[-1],
                train["y_scaled"][first].reshape(-1, 1),
                None,
                config.huber_delta,
            ),
            config.activation,
        )
        norm = float(np.sqrt(sum(np.sum(g**2) for g in grads)))
        if not np.isfinite(norm) or norm == 0:
            raise FloatingPointError("First gradient is invalid or zero.")
        weights, epoch, final_mse = bp_fit(
            train["X"],
            train["y_scaled"],
            layer_conf,
            max_epoch=config.max_epochs,
            max_error=config.max_error,
            learn_rate=config.learning_rate,
            activation=config.activation,
            loss=config.loss,
            optimizer=config.optimizer,
            batch_size=config.batch_size,
            seed=config.seed,
            sample_weight=sample_weight,
            huber_delta=config.huber_delta,
            w=initial,
            on_epoch_end=on_epoch_end,
        )
        save_checkpoint(
            checkpoints / "last.npz", MLP(weights, config.activation), {**base_meta, "epoch": epoch}
        )
        pd.DataFrame(history).to_csv(output / "history.csv", index=False, lineterminator="\n")
        if fixed_rows:
            pd.DataFrame(fixed_rows).to_csv(
                output / "fixed_epoch_metrics.csv", index=False, lineterminator="\n"
            )
        model, _ = load_checkpoint(checkpoints / "best.npz", data)
        tr_loss, tr = evaluate(model, train, bundle, config.loss, config.huber_delta)
        va_loss, va = evaluate(model, val, bundle, config.loss, config.huber_delta)
        if not np.isclose(va["mae"], tracker["best_mae"], atol=1e-9):
            raise AssertionError("Best checkpoint does not reproduce the best validation MAE.")
        pd.DataFrame(
            {
                "observation_id": val["observation_id"],
                "battery_id": val["battery_id"],
                "actual_soh": val["y_original"],
                "predicted_soh": original_scale(model.predict(val["X"]), bundle),
            }
        ).to_csv(output / "validation_predictions.csv", index=False, lineterminator="\n")
        stop_reason = (
            "max_error"
            if final_mse <= config.max_error
            else "early_stopping"
            if epoch < config.max_epochs
            else "max_epoch"
        )
        summary = {
            "status": "complete",
            "run_id": run_id,
            "fingerprint": fingerprint,
            "config": asdict(config),
            "layer_conf": list(layer_conf),
            "input_features": int(layer_conf[0]),
            "train_rows": len(train["X"]),
            "parameters": n_parameters(model.weights),
            "epochs_actual": epoch,
            "best_epoch": tracker["best_epoch"],
            "stop_reason": stop_reason,
            "final_epoch_mse_scaled": final_mse,
            "training_seconds": time.perf_counter() - started,
            "train": tr,
            "validation": va,
            "train_objective_scaled": tr_loss,
            "validation_objective_scaled": va_loss,
            "first_batch_gradient_l2_norm": norm,
            "checkpoint_best": str((checkpoints / "best.npz").relative_to(root)).replace("\\", "/"),
            "checkpoint_last": str((checkpoints / "last.npz").relative_to(root)).replace("\\", "/"),
            "identity": identity,
        }
        write_json(summary_path, summary)
        write_json(output / "status.json", {"status": "complete", "run_id": run_id})
        print(
            f"DONE {run_id}: best epoch={tracker['best_epoch']}/{epoch} ({stop_reason}),"
            f" val MAE={va['mae']:.4f}, train MAE={tr['mae']:.4f}, seconds={summary['training_seconds']:.1f}",
            flush=True,
        )
        return summary
    except Exception as exc:
        pd.DataFrame(history).to_csv(output / "history.csv", index=False, lineterminator="\n")
        write_json(output / "status.json", {"status": "failed", "run_id": run_id, "error": str(exc)})
        raise


def flatten_summary(run):
    config = run["config"]
    return {
        "run_id": run["run_id"],
        **config,
        "hidden_layers": "-".join(map(str, config["hidden_layers"])),
        "input_neurons": run["input_features"],
        "parameters": run["parameters"],
        "epochs_actual": run["epochs_actual"],
        "best_epoch": run["best_epoch"],
        "stop_reason": run["stop_reason"],
        "seconds": run["training_seconds"],
        **{"train_" + k: v for k, v in run["train"].items()},
        **{"validation_" + k: v for k, v in run["validation"].items()},
    }


def stage_variants(cfg, spec):
    """Expand one stage of the study config into run configurations."""
    if "param" in spec:
        values = [tuple(v) if spec["param"] == "hidden_layers" else v for v in spec["values"]]
        return [replace(cfg, **{spec["param"]: v}) for v in values]
    return [replace(cfg, **variant) for variant in spec["variants"]]


def describe(config):
    return {
        "recipe": config["recipe"],
        "hidden_layers": "-".join(map(str, config["hidden_layers"])),
        "activation": config["activation"],
        "loss": config["loss"],
        "optimizer": config["optimizer"],
        "learning_rate": config["learning_rate"],
        "sample_weighting": config["sample_weighting"],
        "max_error": config["max_error"],
    }


def band_errors(predictions):
    predictions = predictions.copy()
    predictions["soh_band"] = pd.cut(
        predictions.actual_soh, SOH_BAND_EDGES, right=False, labels=SOH_BAND_LABELS
    )
    predictions["absolute_error"] = (predictions.predicted_soh - predictions.actual_soh).abs()
    return (
        predictions.groupby("soh_band", observed=False)
        .agg(
            rows=("actual_soh", "size"),
            batteries=("battery_id", "nunique"),
            mae_pp=("absolute_error", "mean"),
        )
        .reset_index()
    )


def run_study(root=ROOT):
    settings = json.loads((root / "configs/mlp_study.json").read_text(encoding="utf-8"))
    if tuple(settings["fixed_epoch_budgets"]) != FIXED_EPOCHS:
        raise ValueError("Unsupported study configuration.")
    initial = RunConfig(**settings["baseline"])
    data_cache = {}
    runs = {}
    stages = []
    output = root / "results/experiments"
    output.mkdir(parents=True, exist_ok=True)

    def execute(cfg):
        if cfg.recipe not in data_cache:
            data_cache[cfg.recipe] = load_training_data(cfg.recipe, root)
        result = train_run(cfg, data_cache[cfg.recipe], root)
        runs[result["run_id"]] = result
        pd.DataFrame([flatten_summary(r) for r in runs.values()]).to_csv(
            output / "run_summary.csv", index=False, lineterminator="\n"
        )
        return result

    def stage(name, configs):
        results = [execute(cfg) for cfg in configs]
        winner = min(results, key=lambda r: (r["validation"]["mae"], r["parameters"], r["run_id"]))
        for result in results:
            stages.append(
                {
                    "stage": name,
                    "run_id": result["run_id"],
                    "selected": result["run_id"] == winner["run_id"],
                    "validation_mae_pp": result["validation"]["mae"],
                    "train_mae_pp": result["train"]["mae"],
                    "epochs_actual": result["epochs_actual"],
                    "best_epoch": result["best_epoch"],
                    "stop_reason": result["stop_reason"],
                    "seconds": result["training_seconds"],
                    **describe(result["config"]),
                }
            )
        pd.DataFrame(stages).to_csv(output / "stage_comparisons.csv", index=False, lineterminator="\n")
        print(
            f"SELECT {name}: {winner['run_id']} validation MAE={winner['validation']['mae']:.4f}", flush=True
        )
        return RunConfig(**winner["config"]), winner

    baseline = execute(initial)
    cfg, candidate = initial, baseline
    for spec in settings["stages"]:
        cfg, candidate = stage(spec["name"], stage_variants(cfg, spec))

    fixed = execute(replace(cfg, early_stopping=False, max_error=0.0, max_epochs=max(FIXED_EPOCHS)))
    pd.read_csv(output / "runs" / fixed["run_id"] / "fixed_epoch_metrics.csv").to_csv(
        output / "epoch_budget_comparison.csv", index=False, lineterminator="\n"
    )

    capacity_rows = []
    for name, overrides in settings["capacity_demo"].items():
        overrides = {k: tuple(v) if k == "hidden_layers" else v for k, v in overrides.items()}
        run = execute(replace(cfg, **overrides))
        last = pd.read_csv(output / "runs" / run["run_id"] / "history.csv").iloc[-1]
        capacity_rows.append(
            {
                "scenario": name,
                "run_id": run["run_id"],
                "hidden_layers": "-".join(map(str, run["config"]["hidden_layers"])),
                "train_rows": run["train_rows"],
                "max_epochs": run["config"]["max_epochs"],
                "max_error": run["config"]["max_error"],
                "epochs_actual": run["epochs_actual"],
                "stop_reason": run["stop_reason"],
                "train_mae_pp": run["train"]["mae"],
                "validation_mae_pp": run["validation"]["mae"],
                "gap_pp": run["validation"]["mae"] - run["train"]["mae"],
                "final_train_mae_pp": last.train_mae_pp,
                "final_validation_mae_pp": last.validation_mae_pp,
                "final_gap_pp": last.validation_mae_pp - last.train_mae_pp,
            }
        )
    pd.DataFrame(capacity_rows).to_csv(output / "capacity_demo.csv", index=False, lineterminator="\n")

    confirmations = []
    for group, selected_cfg in [("baseline", initial), ("candidate", cfg)]:
        for seed in settings["confirmation_seeds"]:
            run = execute(replace(selected_cfg, seed=seed))
            confirmations.append(
                {
                    "group": group,
                    "seed": seed,
                    "run_id": run["run_id"],
                    "validation_mae_pp": run["validation"]["mae"],
                    "validation_rmse_pp": run["validation"]["rmse"],
                    "validation_r2": run["validation"]["r2"],
                    "train_mae_pp": run["train"]["mae"],
                    "best_epoch": run["best_epoch"],
                }
            )
    confirmation = pd.DataFrame(confirmations)
    confirmation.to_csv(output / "seed_confirmation.csv", index=False, lineterminator="\n")
    confirmation.groupby("group").agg(
        n_seeds=("seed", "size"),
        validation_mae_mean=("validation_mae_pp", "mean"),
        validation_mae_std=("validation_mae_pp", "std"),
        validation_rmse_mean=("validation_rmse_pp", "mean"),
        validation_r2_mean=("validation_r2", "mean"),
    ).reset_index().to_csv(output / "seed_summary.csv", index=False, lineterminator="\n")

    val = data_cache[initial.recipe]["validation"]
    train = data_cache[initial.recipe]["train"]
    constants = []
    for name, value in [
        ("train_median", np.median(train["y_original"])),
        ("train_mean", np.mean(train["y_original"])),
    ]:
        constants.append(
            {
                "reference": name,
                "constant_soh": float(value),
                **regression_metrics(
                    val["y_original"], np.full(len(val["y_original"]), value), val["battery_id"]
                ),
            }
        )
    pd.DataFrame(constants).to_csv(output / "constant_references.csv", index=False, lineterminator="\n")
    band_errors(pd.read_csv(output / "runs" / candidate["run_id"] / "validation_predictions.csv")).to_csv(
        output / "candidate_validation_bands.csv", index=False, lineterminator="\n"
    )
    balancing = [s for s in stages if s["stage"] == "balancing"]
    band_tables = []
    for row in balancing:
        table = band_errors(pd.read_csv(output / "runs" / row["run_id"] / "validation_predictions.csv"))
        table.insert(0, "sample_weighting", row["sample_weighting"])
        band_tables.append(table)
    if band_tables:
        pd.concat(band_tables).to_csv(
            output / "balancing_validation_bands.csv", index=False, lineterminator="\n"
        )

    study = {
        "status": "complete",
        "baseline_run_id": baseline["run_id"],
        "candidate_run_id": candidate["run_id"],
        "fixed_epoch_run_id": fixed["run_id"],
        "capacity_demo_run_ids": {row["scenario"]: row["run_id"] for row in capacity_rows},
        "unique_training_runs": len(runs),
        "train_rows": len(train["X"]),
        "validation_rows": len(val["X"]),
        "test_loaded": False,
        "candidate_config": asdict(cfg),
        "baseline_config": asdict(initial),
        "confirmation_seeds": settings["confirmation_seeds"],
        "source_sha256": source_sha256(),
        "settings_sha256": hashlib.sha256((root / "configs/mlp_study.json").read_bytes()).hexdigest(),
        "packages": {p: version(p) for p in ["numpy", "pandas", "scikit-learn"]},
        "data_fingerprints": {recipe: data["fingerprint"] for recipe, data in data_cache.items()},
        "selection": "Sequential seed-42 validation MAE, three-seed confirmation, no global optimum or unbiased final test claim.",
        "total_training_seconds": sum(r["training_seconds"] for r in runs.values()),
    }
    write_json(output / "study.json", study)
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    if args.baseline:
        settings = json.loads((ROOT / "configs/mlp_study.json").read_text(encoding="utf-8"))
        train_run(RunConfig(**settings["baseline"]))
    else:
        run_study()
