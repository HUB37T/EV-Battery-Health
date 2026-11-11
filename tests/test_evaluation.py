"""Final evaluation must preserve fixed selection, infer without fit, and lock files."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import StandardScaler

from ev_battery.evaluation import (
    REPRODUCTION,
    content_sha256,
    describe_subgroup,
    evaluate_locked_predictions,
    lock_payload,
    output_directory,
)
from ev_battery.final_models import validate_plan
from ev_battery.model import MLP, init_weights

ROOT = Path(__file__).resolve().parents[1]


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
        plan = json.loads((ROOT / "configs/final_evaluation.json").read_text(encoding="utf-8"))
        validate_plan(plan)
        changed = copy.deepcopy(plan)
        changed["seeds"] = [42]
        with self.assertRaises(ValueError):
            validate_plan(changed)
        changed = copy.deepcopy(plan)
        changed["groups"]["candidate"]["config"]["recipe"] = "fe_combined"
        with self.assertRaises(ValueError):
            validate_plan(changed)
        changed = copy.deepcopy(plan)
        del changed["groups"]["candidate"]["config"]["patience"]
        with self.assertRaises(ValueError):
            validate_plan(changed)

    def test_content_hash_ignores_the_npz_timestamp_but_not_the_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            weights = np.arange(6, dtype=np.float64).reshape(2, 3)
            np.savez(folder / "a.npz", w0=weights)
            np.savez(folder / "b.npz", w0=weights.copy())
            np.savez(folder / "c.npz", w0=weights + 1e-12)
            self.assertEqual(content_sha256(folder / "a.npz"), content_sha256(folder / "b.npz"))
            self.assertNotEqual(content_sha256(folder / "a.npz"), content_sha256(folder / "c.npz"))

    def test_a_different_lock_never_overwrites_the_recorded_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorded = root / "results/final_evaluation"
            self.assertEqual(output_directory(root, "a" * 64), recorded)
            lock_payload(recorded, {"model": "recorded"})
            same = json.loads((recorded / "evaluation_lock.json").read_text(encoding="utf-8"))["fingerprint"]
            self.assertEqual(output_directory(root, same), recorded)
            self.assertEqual(output_directory(root, "b" * 64), root / REPRODUCTION / ("b" * 16))

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
