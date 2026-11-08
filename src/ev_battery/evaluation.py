"""Score rebuilt NumPy checkpoints in a local, fingerprinted follow-up directory."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ev_battery.followup import load_plan
from ev_battery.paths import ROOT, prepare_workspace
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


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def lock_payload(output, payload):
    """Create the audit lock before inference, refuse to replace changed models."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / "evaluation_lock.json"
    fingerprint = canonical_sha256(payload)
    if path.exists():
        locked = json.loads(path.read_text(encoding="utf-8"))
        if locked["fingerprint"] != fingerprint or canonical_sha256(locked["payload"]) != fingerprint:
            raise ValueError(
                "Evaluation is already locked with different artifacts or protocol. Do not overwrite test results, use a separate reproduction output."
            )
        return locked
    locked = {
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "fingerprint": fingerprint,
        "payload": payload,
    }
    write_json(path, locked)
    return locked


def prepare_locked_inputs(root):
    plan = load_plan(root)
    confirmation_path = "results/experiments/followup_runs.csv"
    confirmation = pd.read_csv(root / confirmation_path)
    expected_pairs = {(group, seed) for group in plan["groups"] for seed in plan["seeds"]}
    if (
        len(confirmation) != len(expected_pairs)
        or set(zip(confirmation.group, confirmation.seed)) != expected_pairs
    ):
        raise ValueError("Run the NumPy train stage to rebuild all six follow-up checkpoints.")
    training = {}
    models = []
    files = {
        "configs/followup.json",
        "configs/feature_contract.json",
        "configs/preprocessing.json",
        "data/splits/battery_split.csv",
        confirmation_path,
    }
    for group, spec in plan["groups"].items():
        recipe = spec["recipe"]
        if recipe not in training:
            training[recipe] = load_training_data(recipe, root)
        data = training[recipe]
        files.update((f"models/preprocessing/{recipe}.joblib", f"data/processed/{recipe}/test.npz"))
        for seed in plan["seeds"]:
            row = confirmation.loc[confirmation.group.eq(group) & confirmation.seed.eq(seed)].iloc[0]
            run_id = row.run_id
            record_path = root / "results/experiments/runs" / run_id / "summary.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            expected = replace(RunConfig(**spec["config"]), seed=seed)
            if RunConfig(**record["config"]) != expected or record["status"] != "complete":
                raise ValueError("Rebuilt checkpoint configuration differs from the NumPy follow-up plan.")
            path = root / record["checkpoint_best"]
            model, checkpoint = load_checkpoint(path, data)
            if checkpoint["run_id"] != run_id or checkpoint["epoch"] != record["best_epoch"]:
                raise ValueError("Checkpoint does not match the best validation epoch.")
            files.update((path.relative_to(root).as_posix(), record_path.relative_to(root).as_posix()))
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
        "selection_reference_commit": plan["selection_reference_commit"],
        "original_pytorch_release_commit": plan["original_pytorch_release_commit"],
        "checkpoint_selection": "best validation epoch of the recorded NumPy configuration",
        "test_split_previously_evaluated": True,
        "files_sha256": {name: file_sha256(root / name) for name in sorted(files)},
        "data_fingerprints": {recipe: data["fingerprint"] for recipe, data in training.items()},
    }
    return plan, training, models, payload


def evaluation_directory(root, fingerprint):
    return root / "results/numpy_followup" / fingerprint


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
    output = evaluation_directory(root, canonical_sha256(payload))
    locked = lock_payload(output, payload)
    configure_runtime(42)
    print(
        f"LOCKED {len(models)} NumPy checkpoints for this follow-up: {locked['fingerprint'][:16]}", flush=True
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
            return previous
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
        if len(test["X"]) != 7572 or len(expected_test_ids) != 390:
            raise ValueError("Test size does not match the audit.")
        test_data[recipe] = test
    first = test_data[plan["groups"]["baseline"]["recipe"]]
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
    train_y = training[plan["groups"]["baseline"]["recipe"]]["train"]["y_original"]
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
    for relative, sha in locked["payload"]["files_sha256"].items():
        if file_sha256(root / relative) != sha:
            raise AssertionError("A locked artifact changed during evaluation.")
    result = {
        "status": "complete",
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "lock_fingerprint": locked["fingerprint"],
        "primary_group_fixed_before_test": plan["primary_group_fixed_before_test"],
        "model_selection_from_test": False,
        "training_or_fit_performed": False,
        "test_evaluated": True,
        "evaluation_scope": plan["evaluation_scope"],
        "selection_reference_commit": plan["selection_reference_commit"],
        "original_pytorch_release_commit": plan["original_pytorch_release_commit"],
        "test_split_previously_evaluated": True,
        "n_checkpoints": len(models),
        "test_rows": 7572,
        "test_batteries": 390,
        "seeds": plan["seeds"],
        "prediction_policy": "Unclipped, original SOH scale, per-seed predictions, no ensemble.",
        "sd_interpretation": "Sample SD over three training seeds on the same test batteries, not population confidence interval.",
        "source_sha256": file_sha256(Path(__file__)),
        "all_locked_files_unchanged_after_inference": True,
    }
    write_json(status_path, result)
    return result


if __name__ == "__main__":
    prepare_workspace()
    run_final_evaluation()
