"""Behavioral checks for leakage prevention and inference edge cases."""

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from ev_battery.preprocessing import MISSING, BatteryPreprocessor, load_prepared_split


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame(
            {
                "cycle_count": [0.0, 10.0, 20.0, np.nan],
                "fast_charge_ratio": [0.1, 0.2, 0.3, 0.4],
                "altitude_m": [-10.0, 0.0, 10.0, 20.0],
                "recall_flag": [0.0, 0.0, 1.0, np.nan],
                "chemistry": ["LFP", "NMC", "LFP", None],
            }
        )
        self.prep = BatteryPreprocessor(
            numeric_features=["cycle_count", "fast_charge_ratio", "altitude_m"],
            binary_features=["recall_flag"],
            categorical_features=["chemistry"],
        )

    def test_statistics_only_use_fit_rows_and_transform_does_not_refit(self):
        prep = self.prep.fit(self.train)
        self.assertEqual(prep.medians_["cycle_count"], 10)
        self.assertAlmostEqual(prep.scaler_.mean_[0], 10)
        before = joblib.hash(prep)
        future = self.train.iloc[[0]].copy()
        future["cycle_count"] = 10000.0
        transformed = prep.transform(future)
        self.assertGreater(transformed[0, 0], 100)
        self.assertEqual(before, joblib.hash(prep))
        self.assertEqual(transformed.dtype, np.float32)

    def test_missing_can_appear_after_fit_on_complete_column(self):
        prep = self.prep.fit(self.train)
        future = self.train.iloc[[0]].copy()
        future["fast_charge_ratio"] = np.nan
        X = prep.transform(future)
        name = list(prep.get_feature_names_out()).index("missing__fast_charge_ratio")
        self.assertEqual(X[0, name], 1)
        self.assertTrue(np.isfinite(X).all())

    def test_unknown_category_is_distinct_from_missing_and_fixed_width(self):
        prep = self.prep.fit(self.train)
        future = pd.concat([self.train.iloc[[0]]] * 2, ignore_index=True)
        future.loc[0, "chemistry"] = "NEW_CHEMISTRY"
        future.loc[1, "chemistry"] = None
        X = prep.transform(future)
        categorical_idx = [i for i, n in enumerate(prep.get_feature_names_out()) if n.startswith("cat__")]
        self.assertEqual(X[0, categorical_idx].sum(), 0)
        self.assertEqual(X[1, categorical_idx].sum(), 1)
        self.assertEqual(X.shape[1], prep.n_features_out_)

    def test_category_missing_allocated_even_when_train_has_no_missing(self):
        complete = self.train.copy()
        complete["chemistry"] = complete.chemistry.fillna("LFP")
        prep = self.prep.fit(complete)
        future = complete.iloc[[0]].copy()
        future["chemistry"] = None
        idx = list(prep.get_feature_names_out()).index(f"cat__chemistry_{MISSING}")
        self.assertEqual(prep.transform(future)[0, idx], 1)

    def test_invalid_values_become_missing_and_original_is_unchanged(self):
        prep = self.prep.fit(self.train)
        future = self.train.iloc[[0]].copy()
        future["cycle_count"] = np.inf
        future["fast_charge_ratio"] = 3.0
        future["recall_flag"] = 9.0
        original = future.copy(deep=True)
        X = prep.transform(future)
        names = list(prep.get_feature_names_out())
        for c in ["cycle_count", "fast_charge_ratio", "recall_flag"]:
            self.assertEqual(X[0, names.index(f"missing__{c}")], 1)
            self.assertEqual(prep.diagnostic(future)["invalid_to_missing"][c], 1)
        pd.testing.assert_frame_equal(original, future)
        self.assertTrue(np.isfinite(X).all())

    def test_reorder_is_safe_but_missing_or_leaking_columns_are_rejected(self):
        prep = self.prep.fit(self.train)
        np.testing.assert_array_equal(
            prep.transform(self.train), prep.transform(self.train[list(reversed(self.train.columns))])
        )
        with self.assertRaises(ValueError):
            prep.transform(self.train.drop(columns="chemistry"))
        with self.assertRaises(ValueError):
            prep.transform(self.train.assign(state_of_health_pct=100.0))

    def test_optional_labels_do_not_influence_feature_parameters(self):
        a = clone(self.prep).fit(self.train, y=np.zeros(len(self.train)))
        b = clone(self.prep).fit(self.train, y=np.arange(len(self.train)) * 1000.0)
        self.assertEqual(joblib.hash(a), joblib.hash(b))

    def test_clipping_and_signed_log_use_frozen_bounds_and_keep_negative_altitude(self):
        prep = (
            clone(self.prep)
            .set_params(
                log_features=["altitude_m"], clip_features=["cycle_count"], clip_quantiles=[0.25, 0.75]
            )
            .fit(self.train)
        )
        np.testing.assert_allclose(prep.clip_bounds_["cycle_count"], [5.0, 15.0])
        future = self.train.iloc[[0]].copy()
        future["cycle_count"] = 500.0
        before = joblib.hash(prep)
        transformed = prep.transform(future)
        expected = (15.0 - prep.scaler_.mean_[0]) / prep.scaler_.scale_[0]
        self.assertAlmostEqual(float(transformed[0, 0]), expected, places=6)
        numeric = prep._numeric_values(prep._clean(future)[0])
        self.assertAlmostEqual(numeric.altitude_m.iloc[0], -np.log1p(10))
        self.assertEqual(before, joblib.hash(prep))

    def test_all_missing_numeric_or_binary_requires_explicit_decision(self):
        with self.assertRaises(ValueError):
            clone(self.prep).fit(self.train.assign(cycle_count=np.nan))
        with self.assertRaises(ValueError):
            clone(self.prep).fit(self.train.assign(recall_flag=np.nan))

    def test_serialized_pipeline_can_be_loaded_without_refit(self):
        prep = self.prep.fit(self.train)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prep.joblib"
            joblib.dump(prep, path)
            loaded = joblib.load(path)
            np.testing.assert_array_equal(loaded.transform(self.train), prep.transform(self.train))

    def _write_ready_fixture(self, root, bad_state=False, bad_target=False):
        for folder in ["configs", "models/preprocessing", "data/processed/probe"]:
            (root / folder).mkdir(parents=True, exist_ok=True)
        (root / "configs/feature_contract.json").write_text("{}", encoding="utf-8")
        (root / "configs/preprocessing.json").write_text(
            json.dumps({"categorical_missing_token": MISSING, "recipes": {"probe": {}}}), encoding="utf-8"
        )
        prep = clone(self.prep).fit(self.train)
        target = StandardScaler().fit(np.array([90.0, 95.0, 80.0, 70.0]).reshape(-1, 1))
        state = joblib.hash((prep, target))
        names = prep.get_feature_names_out()
        bundle = {
            "recipe": "probe",
            "preprocessor": prep,
            "target_scaler": target,
            "state_hash": state,
            "feature_names": names,
        }
        joblib.dump(bundle, root / "models/preprocessing/probe.joblib")
        y = np.array([90.0, 95.0, 80.0, 70.0])
        np.savez_compressed(
            root / "data/processed/probe/train.npz",
            X=prep.transform(self.train),
            observation_id=np.array(["o1", "o2", "o3", "o4"]),
            battery_id=np.array(["b1"] * 4),
            y_scaled=target.transform(y.reshape(-1, 1)).ravel().astype(np.float32),
            y_original=y[:2] if bad_target else y,
            recipe=np.array("probe"),
            split=np.array("train"),
            feature_names=names,
            pipeline_state_hash=np.array("wrong_state" if bad_state else state),
        )

    def test_loader_rejects_mixed_fitted_pipeline_and_array(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_ready_fixture(root, bad_state=True)
            with self.assertRaisesRegex(ValueError, "different run or state"):
                load_prepared_split("probe", "train", root)

    def test_loader_rejects_target_with_wrong_row_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_ready_fixture(root, bad_target=True)
            with self.assertRaisesRegex(ValueError, "Invalid target array"):
                load_prepared_split("probe", "train", root)


if __name__ == "__main__":
    unittest.main()
