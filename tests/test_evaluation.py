"""Final evaluation must preserve fixed selection, infer without fit, and lock files."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import StandardScaler

from ev_battery.evaluation import describe_subgroup, evaluate_locked_predictions, lock_payload
from ev_battery.followup import validate_plan
from ev_battery.model import MLP, init_weights
from ev_battery.paths import RAW_FILES, prepare_workspace


class FinalEvaluationTests(unittest.TestCase):
    def test_lock_is_idempotent_but_refuses_changed_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            first = lock_payload(path, {"model": "fixed", "sha": "abc"})
            again = lock_payload(path, {"sha": "abc", "model": "fixed"})
            self.assertEqual(first, again)
            with self.assertRaises(ValueError):
                lock_payload(path, {"model": "tuned_after_test", "sha": "new"})
            self.assertEqual(json.loads((path / "evaluation_lock.json").read_text(encoding="utf-8")), first)

    def test_plan_rejects_seed_cherry_picking_or_fe_promotion(self):
        root = Path(__file__).resolve().parents[1]
        plan = json.loads((root / "configs/numpy/followup.json").read_text(encoding="utf-8"))
        validate_plan(plan)
        changed = copy.deepcopy(plan)
        changed["seeds"] = [42]
        with self.assertRaises(ValueError):
            validate_plan(changed)
        changed = copy.deepcopy(plan)
        changed["groups"]["candidate"]["recipe"] = "fe_combined"
        with self.assertRaises(ValueError):
            validate_plan(changed)

    def test_followup_plan_matches_the_archived_validation_configs(self):
        root = Path(__file__).resolve().parents[1]
        plan = json.loads((root / "configs/numpy/followup.json").read_text(encoding="utf-8"))
        for group, spec in plan["groups"].items():
            for seed, run_id in spec["reference_run_ids_by_seed"].items():
                summary = root / "results/numpy/experiments/runs" / run_id / "summary.json"
                archived = json.loads(summary.read_text(encoding="utf-8"))
                self.assertEqual({**spec["config"], "seed": int(seed)}, archived["config"], group)
                self.assertEqual(archived["status"], "complete")

    def test_clean_workspace_does_not_copy_or_replace_a_published_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            raw = repo / "data/raw"
            configs = repo / "configs/numpy"
            split = repo / "data/splits"
            release = repo / "results/final_evaluation"
            for folder in (raw, configs, split, release):
                folder.mkdir(parents=True)
            for name in RAW_FILES:
                (raw / name).write_text(name, encoding="utf-8")
            for name in ("preprocessing.json", "mlp_study.json", "feature_engineering.json", "followup.json"):
                (configs / name).write_text(name, encoding="utf-8")
            (split / "battery_split.csv").write_text("battery_id,split", encoding="utf-8")
            published_lock = release / "evaluation_lock.json"
            published_lock.write_text("original release", encoding="utf-8")

            workspace = prepare_workspace(repo)
            self.assertEqual(workspace, repo / "data/processed/numpy_workspace")
            self.assertEqual((workspace / "configs/followup.json").read_text(), "followup.json")
            self.assertFalse((workspace / "results").exists())
            self.assertEqual(published_lock.read_text(), "original release")

    def test_inference_never_calls_backward_or_fit(self):
        model = MLP(init_weights((2, 3, 1), seed=1), "sigmoid")
        scaler = StandardScaler().fit(np.array([80.0, 90.0, 100.0]).reshape(-1, 1))
        test = {"X": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}
        before = [w.copy() for w in model.weights]
        with (
            patch(
                "ev_battery.model.backward", side_effect=AssertionError("No backward during final evaluation")
            ),
            patch(
                "ev_battery.model.bp_fit", side_effect=AssertionError("No training during final evaluation")
            ),
            patch.object(StandardScaler, "fit", side_effect=AssertionError("No scaler fit during inference")),
        ):
            a = evaluate_locked_predictions(model, test, {"target_scaler": scaler})
            b = evaluate_locked_predictions(model, test, {"target_scaler": scaler})
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (2,))
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(before, model.weights)))

    def test_subgroups_keep_sample_and_battery_denominators(self):
        result = describe_subgroup([80.0, 90.0, 100.0], [82.0, 89.0, 97.0], ["a", "a", "b"])
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["batteries"], 2)
        self.assertEqual(result["mae_pp"], 2.0)
        self.assertAlmostEqual(result["rmse_pp"], np.sqrt(14 / 3))


if __name__ == "__main__":
    unittest.main()
