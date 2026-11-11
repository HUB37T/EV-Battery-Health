"""Score the six frozen checkpoints once on the test split. No fitting or training.

The lock is written before any test prediction. It records content hashes of the
checkpoints, the fitted preprocessing, the test arrays and the configurations, so
rebuilding identical artifacts reproduces the same lock. If a recorded lock exists
and the local artifacts differ (for example after training on another machine),
the evaluation is written to ``data/processed/evaluation_reproduction/<id>``
instead and the recorded results stay untouched.

python -m ev_battery evaluate
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ev_battery.final_models import final_config, load_plan
from ev_battery.paths import ROOT
from ev_battery.preprocessing import load_prepared_split
from ev_battery.training import (
    RunConfig,
    configure_runtime,
    load_checkpoint,
    load_training_data,
    original_scale,
    predict_scaled,
    regression_metrics,
    write_json,
)

RECORDED = Path("results/final_evaluation")
REPRODUCTION = Path("data/processed/evaluation_reproduction")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(path):
    """Hash what a file contains, ignoring incidental bytes.

    NumPy writes the current time into every ``.npz`` archive, so identical weights
    saved twice give different file hashes. ``.npz`` files are therefore hashed by
    array name, dtype, shape and values, and ``.joblib`` bundles by their fitted
    preprocessing state. Every other file is hashed byte for byte.
    """
    path = Path(path)
    if path.suffix == ".npz":
        digest = hashlib.sha256()
        with np.load(path, allow_pickle=False) as stored:
            for name in sorted(stored.files):
                array = np.ascontiguousarray(stored[name])
                digest.update(f"{name}|{array.dtype.str}|{array.shape}|".encode())
                digest.update(array.tobytes())
        return digest.hexdigest()
    if path.suffix == ".joblib":
        bundle = joblib.load(path)
        return joblib.hash((bundle["preprocessor"], bundle["target_scaler"]))
    return file_sha256(path)


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def lock_payload(output, payload):
    """Create the audit lock before inference, refuse to replace a different lock."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / "evaluation_lock.json"
    fingerprint = canonical_sha256(payload)
    if path.exists():
        locked = json.loads(path.read_text(encoding="utf-8"))
        if locked["fingerprint"] != fingerprint or canonical_sha256(locked["payload"]) != fingerprint:
            raise ValueError(
                "Evaluation is already locked with different artifacts or protocol, "
                "refusing to overwrite test results."
            )
        return locked
    locked = {
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "fingerprint": fingerprint,
        "payload": payload,
    }
    write_json(path, locked)
    return locked


def test_split_size(root):
    """Rows and batteries of the test split as recorded by the audit."""
    summary = pd.read_csv(root / "results/audit/split_summary.csv").set_index("split")
    return int(summary.loc["test", "rows"]), int(summary.loc["test", "batteries"])


def selection_evidence(root, plan):
    """Record whether the pre-test decisions on validation agree with the frozen plan."""
    study = json.loads((root / "results/experiments/study.json").read_text(encoding="utf-8"))
    features = json.loads((root / "results/feature_engineering/study.json").read_text(encoding="utf-8"))
    candidate = final_config(plan, "candidate", 42)
    return {
        "validation_study_candidate_matches_plan": RunConfig(**study["candidate_config"]) == candidate,
        "feature_engineering_recommendation": features["recommendation"],
    }


def prepare_locked_inputs(root):
    plan = load_plan(root)
    runs_path = "results/experiments/final_runs.csv"
    runs = pd.read_csv(root / runs_path)
    expected_pairs = {(group, seed) for group in plan["groups"] for seed in plan["seeds"]}
    if len(runs) != len(expected_pairs) or set(zip(runs.group, runs.seed)) != expected_pairs:
        raise ValueError("Run the train stage first to build all six final checkpoints.")
    training = {}
    models = []
    files = {
        "configs/final_evaluation.json",
        "configs/feature_contract.json",
        "configs/preprocessing.json",
        "data/splits/battery_split.csv",
        runs_path,
    }
    for group in plan["groups"]:
        for seed in plan["seeds"]:
            expected = final_config(plan, group, seed)
            recipe = expected.recipe
            if recipe not in training:
                training[recipe] = load_training_data(recipe, root)
                files.update((f"models/preprocessing/{recipe}.joblib", f"data/processed/{recipe}/test.npz"))
            data = training[recipe]
            run_id = runs.loc[runs.group.eq(group) & runs.seed.eq(seed), "run_id"].item()
            summary_path = root / "results/experiments/runs" / run_id / "summary.json"
            record = json.loads(summary_path.read_text(encoding="utf-8"))
            if RunConfig(**record["config"]) != expected or record["status"] != "complete":
                raise ValueError("A checkpoint configuration differs from the final plan.")
            path = root / record["checkpoint_best"]
            model, checkpoint = load_checkpoint(path, data)
            if checkpoint["run_id"] != run_id or checkpoint["epoch"] != record["best_epoch"]:
                raise ValueError("Checkpoint does not match the best validation epoch.")
            files.add(path.relative_to(root).as_posix())
            models.append(
                {
                    "group": group,
                    "seed": seed,
                    "recipe": recipe,
                    "run_id": run_id,
                    "record": record,
                    "model": model,
                }
            )
    payload = {
        "plan": plan,
        "source_sha256": file_sha256(Path(__file__)),
        "checkpoint_selection": "best validation epoch of each frozen configuration",
        "test_predictions_computed_at_lock": False,
        "files_content_sha256": {name: content_sha256(root / name) for name in sorted(files)},
        "checkpoints": [
            {
                "group": item["group"],
                "seed": item["seed"],
                "run_id": item["run_id"],
                "best_epoch": item["record"]["best_epoch"],
                "validation_mae_pp": item["record"]["validation"]["mae"],
            }
            for item in models
        ],
        "data_fingerprints": {recipe: data["fingerprint"] for recipe, data in training.items()},
        **selection_evidence(root, plan),
    }
    return plan, training, models, payload


def output_directory(root, fingerprint):
    """The recorded results, or a separate reproduction when the recorded lock differs."""
    lock_path = root / RECORDED / "evaluation_lock.json"
    if lock_path.is_file():
        locked = json.loads(lock_path.read_text(encoding="utf-8"))
        if locked["fingerprint"] != fingerprint:
            return root / REPRODUCTION / fingerprint[:16]
    return root / RECORDED


def describe_subgroup(y, pred, battery_ids):
    metrics = regression_metrics(y, pred, battery_ids)
    return {
        "rows": len(y),
        "batteries": len(set(battery_ids)),
        "mae_pp": metrics["mae"],
        "rmse_pp": metrics["rmse"],
        "bias_pp": metrics["bias_pp"],
    }


def evaluate_locked_predictions(model, test, bundle):
    """Inference only. Targets never enter the model or a fitted transformer."""
    scaled = predict_scaled(model, test["X"])
    return original_scale(scaled, bundle)


def run_final_evaluation(root=ROOT):
    plan, training, models, payload = prepare_locked_inputs(root)
    output = output_directory(root, canonical_sha256(payload))
    if output != root / RECORDED:
        print(
            f"The local checkpoints differ from the recorded lock in {RECORDED.as_posix()}. "
            f"Writing a separate reproduction to {output.relative_to(root).as_posix()}.",
            flush=True,
        )
    locked = lock_payload(output, payload)
    configure_runtime(42)
    print(
        f"LOCKED {len(models)} checkpoints before test predictions: {locked['fingerprint'][:16]}", flush=True
    )
    status_path = output / "evaluation.json"
    required = [
        "metrics_by_seed.csv",
        "metrics_summary.csv",
        "test_bands_by_seed.csv",
        "test_band_summary.csv",
        "test_batteries_by_seed.csv",
        "test_battery_summary.csv",
        "validation_test_comparison.csv",
        "constant_references.csv",
    ]
    required.extend(f"predictions/{item['group']}_seed_{item['seed']}.csv" for item in models)
    if status_path.is_file():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous["lock_fingerprint"] != locked["fingerprint"]:
            raise ValueError("Cached final results do not match the lock.")
        if previous["status"] == "complete" and all((output / name).is_file() for name in required):
            return {**previous, "output": output}
    test_rows, test_batteries = test_split_size(root)
    test_data = {}
    manifest = pd.read_csv(root / "data/splits/battery_split.csv")
    expected_test_ids = set(manifest.loc[manifest.split.eq("test"), "battery_id"])
    for recipe, data in training.items():
        test, bundle = load_prepared_split(recipe, "test", root)
        if (
            bundle["state_hash"] != data["bundle"]["state_hash"]
            or set(test["battery_id"]) != expected_test_ids
        ):
            raise ValueError("Test split does not match the frozen preprocessing or manifest.")
        if set(test["battery_id"]) & (
            set(data["train"]["battery_id"]) | set(data["validation"]["battery_id"])
        ):
            raise ValueError("Test batteries leaked into train/validation.")
        if len(test["X"]) != test_rows or len(expected_test_ids) != test_batteries:
            raise ValueError("Test size does not match the audit.")
        test_data[recipe] = test
    first = test_data[plan["groups"]["baseline"]["config"]["recipe"]]
    for test in test_data.values():
        if not np.array_equal(first["observation_id"], test["observation_id"]) or not np.array_equal(
            first["y_original"], test["y_original"]
        ):
            raise ValueError("Test observations or targets are misaligned across recipes.")
    rows = []
    band_tables = []
    battery_tables = []
    predictions_dir = output / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    for item in models:
        group, seed, recipe = item["group"], item["seed"], item["recipe"]
        test = test_data[recipe]
        bundle = training[recipe]["bundle"]
        pred = evaluate_locked_predictions(item["model"], test, bundle)
        metrics = regression_metrics(test["y_original"], pred, test["battery_id"])
        record = item["record"]
        rows.append(
            {
                "group": group,
                "seed": seed,
                "run_id": item["run_id"],
                "recipe": recipe,
                "input_features": record["input_features"],
                "parameters": record["parameters"],
                "best_epoch_from_validation": record["best_epoch"],
                "train_mae_pp": record["train"]["mae"],
                "validation_mae_pp": record["validation"]["mae"],
                "test_mae_pp": metrics["mae"],
                "test_rmse_pp": metrics["rmse"],
                "test_r2": metrics["r2"],
                "test_bias_pp": metrics["bias_pp"],
                "test_battery_mean_mae_pp": metrics["battery_mean_mae"],
                "test_outside_0_100_pct": metrics["outside_0_100_pct"],
                "test_rows": len(pred),
                "test_batteries": len(expected_test_ids),
            }
        )
        frame = pd.DataFrame(
            {
                "observation_id": test["observation_id"],
                "battery_id": test["battery_id"],
                "actual_soh": test["y_original"],
                "predicted_soh": pred,
            }
        )
        frame["error_pp"] = frame.predicted_soh - frame.actual_soh
        frame["absolute_error_pp"] = frame.error_pp.abs()
        frame["soh_band"] = pd.cut(
            frame.actual_soh, plan["soh_band_edges"], right=False, labels=plan["soh_band_labels"]
        )
        if frame.soh_band.isna().any():
            raise ValueError("Test target outside the locked SOH bands.")
        frame.to_csv(predictions_dir / f"{group}_seed_{seed}.csv", index=False, lineterminator="\n")
        for band, sub in frame.groupby("soh_band", observed=False):
            if not len(sub):
                continue
            band_tables.append(
                {
                    "group": group,
                    "seed": seed,
                    "soh_band": str(band),
                    **describe_subgroup(sub.actual_soh, sub.predicted_soh, sub.battery_id),
                }
            )
        for battery, sub in frame.groupby("battery_id"):
            battery_tables.append(
                {
                    "group": group,
                    "seed": seed,
                    "battery_id": battery,
                    "mean_actual_soh": sub.actual_soh.mean(),
                    **describe_subgroup(sub.actual_soh, sub.predicted_soh, sub.battery_id),
                }
            )
        print(
            f"TEST {group} seed={seed}: MAE={metrics['mae']:.5f}, RMSE={metrics['rmse']:.5f}, R2={metrics['r2']:.5f}",
            flush=True,
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics_by_seed.csv", index=False, lineterminator="\n")
    summary = (
        metrics.groupby("group")
        .agg(
            n_seeds=("seed", "size"),
            test_rows=("test_rows", "first"),
            test_batteries=("test_batteries", "first"),
            parameters=("parameters", "first"),
            test_mae_mean_pp=("test_mae_pp", "mean"),
            test_mae_sd_pp=("test_mae_pp", "std"),
            test_rmse_mean_pp=("test_rmse_pp", "mean"),
            test_rmse_sd_pp=("test_rmse_pp", "std"),
            test_r2_mean=("test_r2", "mean"),
            test_r2_sd=("test_r2", "std"),
            test_battery_mae_mean_pp=("test_battery_mean_mae_pp", "mean"),
            test_battery_mae_sd_pp=("test_battery_mean_mae_pp", "std"),
            validation_mae_mean_pp=("validation_mae_pp", "mean"),
            train_mae_mean_pp=("train_mae_pp", "mean"),
            test_bias_mean_pp=("test_bias_pp", "mean"),
            test_outside_0_100_mean_pct=("test_outside_0_100_pct", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output / "metrics_summary.csv", index=False, lineterminator="\n")
    bands = pd.DataFrame(band_tables)
    bands.to_csv(output / "test_bands_by_seed.csv", index=False, lineterminator="\n")
    bands.groupby(["group", "soh_band"], sort=False).agg(
        rows=("rows", "first"),
        batteries=("batteries", "first"),
        mae_mean_pp=("mae_pp", "mean"),
        mae_sd_pp=("mae_pp", "std"),
        rmse_mean_pp=("rmse_pp", "mean"),
        bias_mean_pp=("bias_pp", "mean"),
    ).reset_index().to_csv(output / "test_band_summary.csv", index=False, lineterminator="\n")
    batteries = pd.DataFrame(battery_tables)
    batteries.to_csv(output / "test_batteries_by_seed.csv", index=False, lineterminator="\n")
    battery_summary = (
        batteries.groupby(["group", "battery_id"])
        .agg(
            rows=("rows", "first"),
            mean_actual_soh=("mean_actual_soh", "first"),
            mae_mean_pp=("mae_pp", "mean"),
            mae_sd_pp=("mae_pp", "std"),
            rmse_mean_pp=("rmse_pp", "mean"),
            bias_mean_pp=("bias_pp", "mean"),
        )
        .reset_index()
    )
    battery_summary.to_csv(output / "test_battery_summary.csv", index=False, lineterminator="\n")
    comparison = summary[["group", "train_mae_mean_pp", "validation_mae_mean_pp", "test_mae_mean_pp"]].copy()
    comparison["test_minus_validation_pp"] = comparison.test_mae_mean_pp - comparison.validation_mae_mean_pp
    comparison.to_csv(output / "validation_test_comparison.csv", index=False, lineterminator="\n")
    constants = []
    train_y = training[plan["groups"]["baseline"]["config"]["recipe"]]["train"]["y_original"]
    for name, value in [("train_median", np.median(train_y)), ("train_mean", train_y.mean())]:
        constants.append(
            {
                "reference": name,
                "constant_from_train": float(value),
                **regression_metrics(
                    first["y_original"], np.full(len(first["y_original"]), value), first["battery_id"]
                ),
            }
        )
    pd.DataFrame(constants).to_csv(output / "constant_references.csv", index=False, lineterminator="\n")
    # Verify that inference did not change any frozen source/model/data files.
    for relative, sha in locked["payload"]["files_content_sha256"].items():
        if content_sha256(root / relative) != sha:
            raise AssertionError("A locked artifact changed during evaluation.")
    result = {
        "status": "complete",
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "lock_fingerprint": locked["fingerprint"],
        "primary_group_fixed_before_test": plan["primary_group_fixed_before_test"],
        "model_selection_from_test": False,
        "training_or_fit_performed": False,
        "test_evaluated": True,
        "test_split_previously_evaluated": plan["test_history"]["test_split_previously_evaluated"],
        "n_checkpoints": len(models),
        "test_rows": test_rows,
        "test_batteries": test_batteries,
        "seeds": plan["seeds"],
        "prediction_policy": "Unclipped, original SOH scale, per-seed predictions, no ensemble.",
        "sd_interpretation": "Sample SD over three training seeds on the same test batteries, not population confidence interval.",
        "source_sha256": file_sha256(Path(__file__)),
        "all_locked_files_unchanged_after_inference": True,
    }
    write_json(status_path, result)
    return {**result, "output": output}


if __name__ == "__main__":
    print(run_final_evaluation()["output"])
