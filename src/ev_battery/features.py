"""Feature-only ablation on the frozen candidate-03 MLP, train/validation only."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from ev_battery.followup import load_plan
from ev_battery.paths import ROOT
from ev_battery.preprocessing import MISSING
from ev_battery.training import RunConfig, load_training_data, train_run, write_json


class EngineeredPreprocessor:
    """Append train-fitted numeric features to a frozen, already-fitted base.

    No target, ID, cross-row operation, future observation or health label is
    used. For fold-local CV construct a new train-fitted base for each fold.
    """

    def __init__(self, base_preprocessor, blocks, clip_quantiles=(0.005, 0.995)):
        self.base_preprocessor = copy.deepcopy(base_preprocessor)
        self.blocks = tuple(blocks)
        self.clip_quantiles = tuple(clip_quantiles)
        if (
            not self.blocks
            or len(set(self.blocks)) != len(self.blocks)
            or not set(self.blocks).issubset({"usage_rates", "context", "chemistry"})
        ):
            raise ValueError("Unknown or duplicate feature-engineering block.")

    @property
    def required_columns(self):
        return self.base_preprocessor.required_columns

    def _arithmetic(self, X):
        # Reuse the same project's cleaning rules, without mutating/refitting them.
        clean, _ = self.base_preprocessor._clean(X)
        values = pd.DataFrame(index=clean.index)
        if "usage_rates" in self.blocks:
            values["fe_cycles_per_day"] = clean.cycle_count.div(
                clean.battery_age_days.where(clean.battery_age_days.gt(0))
            )
            values["fe_energy_per_cycle_kwh"] = clean.total_energy_throughput_kwh.div(
                clean.cycle_count.where(clean.cycle_count.gt(0))
            )
        if "context" in self.blocks:
            values["fe_charge_ambient_delta_c"] = (
                clean.peak_temp_during_fast_charge_c - clean.avg_ambient_temp_c
            )
            values["fe_fastcharge_dod_interaction"] = clean.fast_charge_ratio * (
                clean.avg_depth_of_discharge_pct / 100
            )
        if "chemistry" in self.blocks:
            known = clean.chemistry.isin(self.chemistries_)
            for exposure, name in [
                (clean.cycle_count, "cycles"),
                (clean.battery_age_days / 365.25, "age_years"),
            ]:
                valid = known & exposure.notna()
                logged = np.log1p(exposure)
                for chemistry in self.chemistries_:
                    column = f"fe_chem_{chemistry}_log_{name}"
                    values[column] = np.where(
                        valid, np.where(clean.chemistry.eq(chemistry), logged, 0.0), np.nan
                    )
        return values.replace([np.inf, -np.inf], np.nan)

    def _numeric_values(self, raw):
        values = raw.fillna(self.medians_).copy()
        for c, (lo, hi) in self.clip_bounds_.items():
            values[c] = values[c].clip(lo, hi)
        for c in self.log_features_:
            values[c] = np.sign(values[c]) * np.log1p(np.abs(values[c]))
        return values

    def fit(self, X, y=None):
        del y
        check_is_fitted(self.base_preprocessor, "output_features_")
        self.base_state_hash_ = joblib.hash(self.base_preprocessor)
        idx = list(self.base_preprocessor.categorical_features).index("chemistry")
        self.chemistries_ = tuple(
            str(c) for c in self.base_preprocessor.encoder_.categories_[idx] if c != MISSING
        )
        raw = self._arithmetic(X)
        self.extra_columns_ = list(raw.columns)
        self.medians_ = raw.median()
        if self.medians_.isna().any():
            raise ValueError("Derived feature is entirely missing in train.")
        self.log_features_ = [
            c for c in self.extra_columns_ if c in {"fe_cycles_per_day", "fe_energy_per_cycle_kwh"}
        ]
        cap_columns = self.log_features_ + (["fe_charge_ambient_delta_c"] if "context" in self.blocks else [])
        lo, hi = self.clip_quantiles
        if not 0 <= lo < hi <= 1:
            raise ValueError("Invalid clipping quantiles.")
        self.clip_bounds_ = {c: tuple(float(x) for x in raw[c].quantile([lo, hi])) for c in cap_columns}
        self.scaler_ = StandardScaler().fit(self._numeric_values(raw))
        self.fit_rows_ = len(X)
        self.source_hash_ = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.output_features_ = np.array(
            list(self.base_preprocessor.get_feature_names_out())
            + ["engineered__" + c for c in self.extra_columns_]
            + ["missing__" + c for c in self.extra_columns_],
            dtype=str,
        )
        self.n_features_out_ = len(self.output_features_)
        if joblib.hash(self.base_preprocessor) != self.base_state_hash_:
            raise AssertionError("Base pipeline changed while fitting extra features.")
        return self

    def transform(self, X):
        if not hasattr(self, "output_features_"):
            raise NotFittedError("Fit feature engineering on train first.")
        base = self.base_preprocessor.transform(X)
        raw = self._arithmetic(X)
        extra = self.scaler_.transform(self._numeric_values(raw)).astype(np.float32)
        missing = raw.isna().to_numpy(dtype=np.float32)
        values = np.hstack([base, extra, missing]).astype(np.float32)
        if not np.isfinite(values).all() or values.shape != (len(X), self.n_features_out_):
            raise ValueError("Invalid feature-engineering output.")
        return values

    def get_feature_names_out(self):
        if not hasattr(self, "output_features_"):
            raise NotFittedError("Fit feature engineering on train first.")
        return self.output_features_.copy()

    def describe_train(self, X):
        raw = self._arithmetic(X)
        records = []
        for i, c in enumerate(self.extra_columns_):
            lo, hi = self.clip_bounds_.get(c, (np.nan, np.nan))
            records.append(
                {
                    "feature": c,
                    "missing_train": int(raw[c].isna().sum()),
                    "missing_train_pct": raw[c].isna().mean() * 100,
                    "median_train_before_log": self.medians_[c],
                    "cap_low_train": lo,
                    "cap_high_train": hi,
                    "signed_log1p_after_cap": c in self.log_features_,
                    "scaler_mean_train": self.scaler_.mean_[i],
                    "scaler_scale_train": self.scaler_.scale_[i],
                }
            )
        return pd.DataFrame(records)


def read_settings(root=ROOT):
    return json.loads((root / "configs/feature_engineering.json").read_text(encoding="utf-8"))


def load_raw_observations(base_data, root=ROOT):
    required = base_data["bundle"]["required_columns"]
    allowed = {s: set(base_data[s]["battery_id"]) for s in ["train", "validation"]}
    path = root / "data/raw/ev_battery_health.csv"
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != base_data["bundle"]["provenance"]["dataset_sha256"]:
        raise ValueError("Source CSV changed.")
    records = {"train": [], "validation": []}
    # Read allowed inputs only, never health target/labels. Test rows discarded
    # in each chunk before arithmetic, fit, summaries, or model evaluation.
    for chunk in pd.read_csv(path, usecols=["battery_id", "observation_id"] + required, chunksize=10000):
        for split in records:
            records[split].append(chunk.loc[chunk.battery_id.isin(allowed[split])])
    result = {}
    for split, chunks in records.items():
        frame = pd.concat(chunks, ignore_index=True)
        if not np.array_equal(frame.observation_id.to_numpy(dtype=str), base_data[split]["observation_id"]):
            raise ValueError("Raw and prepared row order differ.")
        result[split] = frame.loc[:, required].copy()
    return result


def make_data_fingerprint(train, validation, state_hash):
    digest = hashlib.sha256()
    for data in [train, validation]:
        for key in ["X", "y_scaled", "y_original", "observation_id", "battery_id", "feature_names"]:
            digest.update(np.ascontiguousarray(data[key]).tobytes())
    digest.update(state_hash.encode())
    return digest.hexdigest()


def pipeline_state_fingerprint(prep, target_scaler):
    """Hash learned values, not incidental pickle object/string aliasing."""
    state = {
        "version": 1,
        "base_state": joblib.hash(prep.base_preprocessor),
        "blocks": list(prep.blocks),
        "quantiles": list(prep.clip_quantiles),
        "chemistries": list(prep.chemistries_),
        "medians": {c: float(prep.medians_[c]) for c in prep.extra_columns_},
        "caps": {c: list(v) for c, v in prep.clip_bounds_.items()},
        "log_features": prep.log_features_,
        "extra_scaler_mean": prep.scaler_.mean_.tolist(),
        "extra_scaler_scale": prep.scaler_.scale_.tolist(),
        "extra_scaler_variance": prep.scaler_.var_.tolist(),
        "fit_rows": prep.fit_rows_,
        "target_mean": target_scaler.mean_.tolist(),
        "target_scale": target_scaler.scale_.tolist(),
        "output_features": prep.get_feature_names_out().tolist(),
        "source_sha256": prep.source_hash_,
    }
    return hashlib.sha256(json.dumps(state, sort_keys=True, allow_nan=False).encode()).hexdigest()


def prepare_variant(name, blocks, raw, base_data, settings, root=ROOT):
    prep = EngineeredPreprocessor(
        base_data["bundle"]["preprocessor"], blocks, settings["clip_quantiles"]
    ).fit(raw["train"])
    frozen = joblib.hash(prep)
    scaler = copy.deepcopy(base_data["bundle"]["target_scaler"])
    state_hash = pipeline_state_fingerprint(prep, scaler)
    data = {}
    directory = root / "data/processed" / name
    directory.mkdir(parents=True, exist_ok=True)
    for split in ["train", "validation"]:
        X = prep.transform(raw[split])
        base = base_data[split]
        if not np.array_equal(X[:, : base["X"].shape[1]], base["X"]):
            raise AssertionError("The 67 control features changed.")
        data[split] = {
            **base,
            "X": X,
            "feature_names": prep.get_feature_names_out(),
            "recipe": np.array(name),
            "pipeline_state_hash": np.array(state_hash),
        }
        np.savez_compressed(directory / f"{split}.npz", **data[split])
    if joblib.hash(prep) != frozen:
        raise AssertionError("Transforming validation changed fitted feature-engineering parameters.")
    provenance = {
        "fit_split": "train",
        "fit_rows": len(raw["train"]),
        "fit_batteries": len(set(base_data["train"]["battery_id"])),
        "source_sha256": prep.source_hash_,
        "settings_sha256": hashlib.sha256(
            (root / "configs/feature_engineering.json").read_bytes()
        ).hexdigest(),
        "base_pipeline_state_hash": base_data["bundle"]["state_hash"],
        "test_arrays_loaded": False,
        "test_transformed": False,
    }
    bundle = {
        "preprocessor": prep,
        "target_scaler": scaler,
        "recipe": name,
        "state_hash": state_hash,
        "feature_names": prep.get_feature_names_out(),
        "required_columns": prep.required_columns,
        "provenance": provenance,
    }
    model_dir = root / "models/feature_engineering"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_dir / f"{name}.joblib")
    reloaded = joblib.load(model_dir / f"{name}.joblib")
    if pipeline_state_fingerprint(reloaded["preprocessor"], reloaded["target_scaler"]) != state_hash:
        raise AssertionError("Parameter fingerprint changed after serialization.")
    output = root / "results/feature_engineering"
    output.mkdir(parents=True, exist_ok=True)
    prep.describe_train(raw["train"]).to_csv(
        output / f"{name}_parameters.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(
        {"index": range(prep.n_features_out_), "output_feature": prep.get_feature_names_out()}
    ).to_csv(output / f"{name}_features.csv", index=False, lineterminator="\n")
    data["bundle"] = bundle
    data["fingerprint"] = make_data_fingerprint(data["train"], data["validation"], state_hash)
    return data


def load_engineered_data(recipe, root=ROOT):
    settings = read_settings(root)
    if recipe not in settings["variants"]:
        raise ValueError("Unknown feature-engineering variant.")
    bundle = joblib.load(root / "models/feature_engineering" / f"{recipe}.joblib")
    state = pipeline_state_fingerprint(bundle["preprocessor"], bundle["target_scaler"])
    if state != bundle["state_hash"]:
        raise ValueError("Feature-engineering pipeline state differs.")
    data = {}
    for split in ["train", "validation"]:
        with np.load(root / "data/processed" / recipe / f"{split}.npz", allow_pickle=False) as arrays:
            data[split] = {k: arrays[k] for k in arrays.files}
        if (
            data[split]["pipeline_state_hash"].item() != state
            or data[split]["recipe"].item() != recipe
            or data[split]["split"].item() != split
        ):
            raise ValueError("Arrays do not match the feature-engineering pipeline.")
        if not np.array_equal(data[split]["feature_names"], bundle["feature_names"]):
            raise ValueError("Feature-engineering feature order differs.")
        if not np.isfinite(data[split]["X"]).all():
            raise ValueError("Non-finite feature-engineering values.")
    data["bundle"] = bundle
    data["fingerprint"] = make_data_fingerprint(data["train"], data["validation"], state)
    return data


def run_feature_study(root=ROOT):
    settings = read_settings(root)
    candidate = load_plan(root)["groups"]["candidate"]
    control_cfg = RunConfig(**candidate["config"])
    screening_seed = settings["screening_seed"]
    if (
        settings["base_recipe"] != control_cfg.recipe
        or settings["reference_candidate_run_id"]
        != candidate["reference_run_ids_by_seed"][str(screening_seed)]
    ):
        raise ValueError("Feature-engineering settings differ from the recorded NumPy control configuration.")
    control_data = load_training_data(control_cfg.recipe, root)
    raw = load_raw_observations(control_data, root)
    output = root / "results/feature_engineering"
    output.mkdir(parents=True, exist_ok=True)
    all_runs = {}
    screen = []
    prepared = {}

    def record(run, variant):
        all_runs[run["run_id"]] = run
        return {
            "variant": variant,
            "run_id": run["run_id"],
            "seed": run["config"]["seed"],
            "input_neurons": run["input_features"],
            "parameters": run["parameters"],
            "train_mae_pp": run["train"]["mae"],
            "validation_mae_pp": run["validation"]["mae"],
            "validation_rmse_pp": run["validation"]["rmse"],
            "validation_r2": run["validation"]["r2"],
            "best_epoch": run["best_epoch"],
            "epochs_actual": run["epochs_actual"],
        }

    control = train_run(replace(control_cfg, seed=screening_seed), control_data, root)
    screen.append(record(control, "control_no_domain_features"))
    for name, blocks in settings["variants"].items():
        data = prepare_variant(name, blocks, raw, control_data, settings, root)
        prepared[name] = data
        run = train_run(replace(control_cfg, recipe=name, seed=screening_seed), data, root)
        screen.append(record(run, name))
        pd.DataFrame(screen).to_csv(output / "screening.csv", index=False, lineterminator="\n")
    screening = pd.DataFrame(screen)
    best = (
        screening.loc[screening.variant.ne("control_no_domain_features")]
        .sort_values(["validation_mae_pp", "parameters", "variant"])
        .iloc[0]
        .variant
    )
    confirmations = []
    for variant, data, recipe in [
        ("control_no_domain_features", control_data, settings["base_recipe"]),
        (best, prepared[best], best),
    ]:
        for seed in settings["confirmation_seeds"]:
            run = train_run(replace(control_cfg, recipe=recipe, seed=seed), data, root)
            confirmations.append(record(run, variant))
    confirmation = pd.DataFrame(confirmations)
    confirmation.to_csv(output / "seed_confirmation.csv", index=False, lineterminator="\n")
    summary = (
        confirmation.groupby("variant")
        .agg(
            n_seeds=("seed", "size"),
            mae_mean_pp=("validation_mae_pp", "mean"),
            mae_sd_pp=("validation_mae_pp", "std"),
            rmse_mean_pp=("validation_rmse_pp", "mean"),
            r2_mean=("validation_r2", "mean"),
            parameters=("parameters", "first"),
        )
        .reset_index()
    )
    summary.to_csv(output / "seed_summary.csv", index=False, lineterminator="\n")
    paired = confirmation.pivot(index="seed", columns="variant", values="validation_mae_pp")
    paired["mae_improvement_pp"] = paired["control_no_domain_features"] - paired[best]
    paired.reset_index().to_csv(output / "paired_seed_improvements.csv", index=False, lineterminator="\n")
    mean_gain = float(paired.mae_improvement_pp.mean())
    wins = int(paired.mae_improvement_pp.gt(0).sum())
    promote = (
        mean_gain >= settings["minimum_mean_mae_improvement_pp"] and wins >= settings["minimum_seed_wins"]
    )
    bands = []
    for row in confirmation.itertuples():
        pred = pd.read_csv(root / "results/experiments/runs" / row.run_id / "validation_predictions.csv")
        pred["band"] = pd.cut(
            pred.actual_soh,
            [-np.inf, 70, 80, 90, 95, np.inf],
            right=False,
            labels=["<70", "70–<80", "80–<90", "90–<95", ">=95"],
        )
        pred["error"] = (pred.predicted_soh - pred.actual_soh).abs()
        tab = (
            pred.groupby("band", observed=False)
            .agg(rows=("error", "size"), batteries=("battery_id", "nunique"), mae_pp=("error", "mean"))
            .reset_index()
        )
        tab["variant"] = row.variant
        tab["seed"] = row.seed
        bands.append(tab)
    pd.concat(bands, ignore_index=True).to_csv(
        output / "validation_bands.csv", index=False, lineterminator="\n"
    )
    # Aggregate run links here, do not overwrite the historical notebook-03 study.
    records = [record(run, run["config"]["recipe"]) for run in all_runs.values()]
    pd.DataFrame(records).to_csv(output / "run_records.csv", index=False, lineterminator="\n")
    study = {
        "status": "complete",
        "reference_candidate_run_id": control["run_id"],
        "archived_candidate_run_id": settings["reference_candidate_run_id"],
        "reference_config": control["config"],
        "selected_engineered_variant": best,
        "recommendation": best if promote else "control_no_domain_features",
        "promoted_by_prespecified_gate": promote,
        "mean_validation_mae_improvement_pp": mean_gain,
        "seed_wins": wins,
        "confirmation_seeds": settings["confirmation_seeds"],
        "unique_runs_referenced": len(all_runs),
        "new_training_runs": len(all_runs) - 3,
        "test_arrays_loaded": False,
        "test_transformed": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "settings_sha256": hashlib.sha256(
            (root / "configs/feature_engineering.json").read_bytes()
        ).hexdigest(),
        "control_data_fingerprint": control_data["fingerprint"],
        "variant_data_fingerprints": {k: v["fingerprint"] for k, v in prepared.items()},
        "extra_feature_counts": {
            k: len(v["bundle"]["preprocessor"].extra_columns_) for k, v in prepared.items()
        },
        "decision_threshold_pp": settings["minimum_mean_mae_improvement_pp"],
        "train_rows": 34929,
        "validation_rows": 7499,
        "caution": "Validation reused for screening/confirmation, feature-derived gains and any tail changes require independent final test. No claim of causal battery effects.",
    }
    write_json(output / "study.json", study)
    print(
        f"FEATURE STUDY: best={best}, mean gain={mean_gain:.5f} pp, wins={wins}/3, recommendation={study['recommendation']}",
        flush=True,
    )
    return study


if __name__ == "__main__":
    from ev_battery.features import run_feature_study as run_canonical

    run_canonical()
