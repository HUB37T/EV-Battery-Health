"""Checks of the from-scratch backpropagation, checkpoint selection, and holdout boundaries."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ev_battery.model import (
    MLP,
    backward,
    bp_fit,
    bp_predict,
    forward,
    init_weights,
    loss_gradient,
    loss_value,
    n_parameters,
)
from ev_battery.training import (
    RunConfig,
    load_checkpoint,
    load_training_data,
    regression_metrics,
    soh_band_weights,
    train_run,
)


def course_sig(X):
    return 1 / (1 + np.exp(-np.array(X)))


def course_sigd(X):
    s = course_sig(X)
    return s * (1 - s)


def course_bp_fit(X, target, layer_conf, max_epoch, max_error=0.1, learn_rate=0.1, w=None):
    """``bp_fit`` from Modul BAB 7, copied as is apart from taking the initial weights."""
    nin = [np.empty(i) for i in layer_conf]
    n = [np.empty(j + 1) if i < len(layer_conf) - 1 else np.empty(j) for i, j in enumerate(layer_conf)]
    w = [x.copy() for x in w]
    dw = [np.empty((layer_conf[i] + 1, layer_conf[i + 1])) for i in range(len(layer_conf) - 1)]
    d = [np.empty(s) for s in layer_conf[1:]]
    din = [np.empty(s) for s in layer_conf[1:-1]]
    epoch = 0
    mse = 1
    for i in range(len(n) - 1):
        n[i][-1] = 1
    while (max_epoch == -1 or epoch < max_epoch) and mse > max_error:
        epoch += 1
        mse = 0
        for r in range(len(X)):
            n[0][:-1] = X[r]
            for L in range(1, len(layer_conf)):
                nin[L] = np.dot(n[L - 1], w[L - 1])
                n[L][: len(nin[L])] = course_sig(nin[L])
            e = target[r] - n[-1]
            mse += sum(e**2)
            d[-1] = e * course_sigd(nin[-1])
            dw[-1] = learn_rate * d[-1] * n[-2].reshape((-1, 1))
            for L in range(len(layer_conf) - 1, 1, -1):
                din[L - 2] = np.dot(d[L - 1], np.transpose(w[L - 1][:-1]))
                d[L - 2] = din[L - 2] * course_sigd(nin[L - 1])
                dw[L - 2] = (learn_rate * d[L - 2]) * n[L - 2].reshape((-1, 1))
            for i in range(len(w)):
                w[i] += dw[i]
        mse /= len(X)
    return w, epoch, mse


class BackpropagationTests(unittest.TestCase):
    def test_matches_course_bp_fit_with_course_settings(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (30, 4))
        T = rng.uniform(0, 1, (30, 2))
        w0 = [rng.uniform(-0.5, 0.5, (5, 3)), rng.uniform(-0.5, 0.5, (4, 3)), rng.uniform(-0.5, 0.5, (4, 2))]
        expected, epochs, mse = course_bp_fit(X, T, (4, 3, 3, 2), max_epoch=5, max_error=0, w=w0)
        actual, epochs_ours, mse_ours = bp_fit(
            X,
            T,
            (4, 3, 3, 2),
            max_epoch=5,
            learn_rate=0.1,
            activation="sigmoid",
            output_activation="sigmoid",
            loss="mse",
            optimizer="gd",
            batch_size=1,
            shuffle=False,
            w=w0,
        )
        self.assertEqual(epochs, epochs_ours)
        self.assertAlmostEqual(mse, mse_ours, places=12)
        for a, b in zip(expected, actual):
            np.testing.assert_allclose(a, b, atol=1e-12)

    def test_gradients_match_finite_differences(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(6, 4))
        t = rng.normal(size=(6, 1))
        weight = rng.uniform(0.5, 2, 6)
        for activation in ["sigmoid", "tanh", "relu"]:
            for loss in ["mse", "mae", "huber"]:
                w = init_weights((4, 5, 3, 1), seed=2)
                for matrix in w:
                    matrix[-1] = 0.3  # non-zero biases keep ReLU away from its kink at 0

                def objective(weights, activation=activation, loss=loss):
                    _, out = forward(X, weights, activation)
                    value = loss_value(loss, out[-1][:, 0], t[:, 0], weight)
                    return value / 2 if loss == "mse" else value  # MSE gradient uses d(0.5 e^2)

                nets, out = forward(X, w, activation)
                grads = backward(nets, out, w, loss_gradient(loss, out[-1], t, weight), activation)
                for k, matrix in enumerate(w):
                    for idx in np.ndindex(matrix.shape):
                        plus = [m.copy() for m in w]
                        minus = [m.copy() for m in w]
                        plus[k][idx] += 1e-6
                        minus[k][idx] -= 1e-6
                        numeric = (objective(plus) - objective(minus)) / 2e-6
                        self.assertAlmostEqual(numeric, grads[k][idx], places=6, msg=f"{activation}/{loss}")

    def test_manual_worked_example(self):
        # One input, one ReLU hidden unit, one linear output (docs/METHODOLOGY.md, section 6.4).
        w = [np.array([[0.5], [0.0]]), np.array([[0.4], [0.0]])]
        X = np.array([[2.0]])
        nets, out = forward(X, w, "relu")
        self.assertAlmostEqual(out[-1].item(), 0.4)
        grads = backward(nets, out, w, loss_gradient("mse", out[-1], np.array([[1.0]])), "relu")
        np.testing.assert_allclose(grads[0].ravel(), [-0.48, -0.24])
        np.testing.assert_allclose(grads[1].ravel(), [-0.6, -0.6])
        trained, _, _ = bp_fit(X, [1.0], (1, 1, 1), max_epoch=1, learn_rate=0.1, activation="relu", w=w)
        np.testing.assert_allclose(trained[0].ravel(), [0.548, 0.024])
        np.testing.assert_allclose(trained[1].ravel(), [0.46, 0.06])
        self.assertAlmostEqual(bp_predict(X, trained, "relu").item(), 0.5752)

    def test_shapes_parameters_and_validation(self):
        w = init_weights((67, 32, 1))
        self.assertEqual(n_parameters(w), 67 * 32 + 32 + 32 + 1)
        model = MLP(w, "sigmoid")
        self.assertEqual(model.predict(np.zeros((1, 67))).shape, (1,))
        self.assertEqual(model.predict(np.zeros((5, 67))).shape, (5,))
        self.assertEqual(model.layer_conf, (67, 32, 1))
        with self.assertRaises(ValueError):
            init_weights((3,))
        with self.assertRaises(ValueError):
            bp_fit(np.zeros((2, 3)), [0, 0], (3, 2, 1), activation="softmax")
        with self.assertRaises(ValueError):
            bp_fit(np.zeros((2, 3)), [0, 0], (4, 2, 1))

    def test_max_error_stops_training_like_the_course(self):
        X = np.linspace(-1, 1, 40).reshape(-1, 1)
        y = 0.5 * X[:, 0]
        _, epoch, mse = bp_fit(X, y, (1, 4, 1), max_epoch=500, max_error=0.05, learn_rate=0.1)
        self.assertLess(epoch, 500)
        self.assertLessEqual(mse, 0.05)

    def test_adam_and_gd_both_reduce_the_loss(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(200, 3))
        y = np.sin(X[:, 0]) + 0.5 * X[:, 1]
        start = loss_value("mse", bp_predict(X, init_weights((3, 8, 1), 4), "tanh"), y)
        for optimizer, lr in [("gd", 0.05), ("adam", 0.01)]:
            w, _, _ = bp_fit(
                X, y, (3, 8, 1), max_epoch=30, learn_rate=lr, activation="tanh", optimizer=optimizer, seed=4
            )
            self.assertLess(loss_value("mse", bp_predict(X, w, "tanh"), y), start / 2, optimizer)

    def test_weighted_mini_batches_average_to_the_full_weighted_gradient(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(8, 3))
        t = rng.normal(size=(8, 1))
        weight = np.array([4.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])
        weight = weight / weight.mean()
        w = init_weights((3, 4, 1), seed=6)

        def gradient(rows):
            nets, out = forward(X[rows], w, "sigmoid")
            return backward(nets, out, w, loss_gradient("mse", out[-1], t[rows], weight[rows]), "sigmoid")

        full = gradient(slice(0, 8))
        halves = [gradient(slice(0, 4)), gradient(slice(4, 8))]
        for k in range(len(w)):
            np.testing.assert_allclose(full[k], (halves[0][k] + halves[1][k]) / 2, atol=1e-12)

    def test_unit_sample_weights_change_nothing(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(50, 2))
        y = X[:, 0] - X[:, 1]
        plain, _, _ = bp_fit(X, y, (2, 3, 1), max_epoch=5, batch_size=16)
        ones, _, _ = bp_fit(X, y, (2, 3, 1), max_epoch=5, batch_size=16, sample_weight=np.full(50, 3.0))
        for a, b in zip(plain, ones):
            np.testing.assert_array_equal(a, b)

    def test_soh_band_weights_balance_bands(self):
        y = np.array([60.0] * 2 + [96.0] * 8)
        weights = soh_band_weights(y, "inverse")
        self.assertAlmostEqual(weights.mean(), 1.0)
        self.assertAlmostEqual(weights[:2].sum(), weights[2:].sum())
        self.assertIsNone(soh_band_weights(y, "none"))

    def test_metrics_and_battery_weighting(self):
        metrics = regression_metrics([1, 2, 4], [1, 3, 2])
        self.assertAlmostEqual(metrics["mae"], 1)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(5 / 3))
        self.assertAlmostEqual(metrics["r2"], 1 - 5 / (14 / 3))
        grouped = regression_metrics([0, 0, 0, 0], [0, 2, 2, 2], ["a", "b", "b", "b"])
        self.assertEqual(grouped["mae"], 1.5)
        self.assertEqual(grouped["battery_mean_mae"], 1.0)
        self.assertIsNone(grouped["r2"])
        with self.assertRaises(ValueError):
            regression_metrics([1, 2], [[1], [2]])


class TrainingRunTests(unittest.TestCase):
    def test_training_loader_never_requests_test_split(self):
        train = {
            "X": np.zeros((2, 1), dtype=np.float32),
            "y_scaled": np.zeros(2, dtype=np.float32),
            "y_original": np.array([90.0, 91.0]),
            "observation_id": np.array(["o1", "o2"]),
            "battery_id": np.array(["a", "a"]),
            "feature_names": np.array(["x"]),
        }
        val = {**train, "battery_id": np.array(["b", "b"]), "observation_id": np.array(["v1", "v2"])}
        calls = []

        def loader(recipe, split, root):
            calls.append(split)
            if split == "test":
                raise AssertionError("Test must stay locked")
            return (train if split == "train" else val), {"state_hash": "same"}

        with patch("ev_battery.training.load_prepared_split", side_effect=loader):
            load_training_data("probe")
        self.assertEqual(calls, ["train", "validation"])

    def test_checkpoint_is_best_validation_epoch_and_reloads_exactly(self):
        X = np.linspace(-1, 1, 20).reshape(-1, 1).astype(np.float32)
        original = 80 + 10 * X[:, 0].astype(np.float64)
        scaler = StandardScaler().fit(pd.DataFrame({"state_of_health_pct": original}))
        y = scaler.transform(pd.DataFrame({"state_of_health_pct": original})).ravel().astype(np.float32)
        train = {
            "X": X,
            "y_scaled": y,
            "y_original": original,
            "battery_id": np.array(["a"] * 20),
            "observation_id": np.array([f"o{i}" for i in range(20)]),
            "feature_names": np.array(["x"]),
        }
        val = {**train, "battery_id": np.array(["b"] * 20)}
        data = {"train": train, "validation": val, "bundle": {"target_scaler": scaler}, "fingerprint": "unit"}
        config = RunConfig(recipe="probe", hidden_layers=(4,), max_epochs=6, batch_size=4, patience=3, seed=8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = train_run(config, data, root)
            history = pd.read_csv(root / "results/experiments/runs" / result["run_id"] / "history.csv")
            trained = history.loc[history.epoch.gt(0)]
            self.assertEqual(
                result["best_epoch"], int(trained.loc[trained.validation_mae_pp.idxmin(), "epoch"])
            )
            model, meta = load_checkpoint(root / result["checkpoint_best"], data)
            self.assertEqual(meta["epoch"], result["best_epoch"])
            again, _ = load_checkpoint(root / result["checkpoint_best"], data)
            np.testing.assert_array_equal(model.predict(X), again.predict(X))
            cached = train_run(config, data, root)
            self.assertEqual(cached["training_seconds"], result["training_seconds"])
            self.assertGreater(result["first_batch_gradient_l2_norm"], 0)
            with self.assertRaises(ValueError):
                load_checkpoint(root / result["checkpoint_best"], {**data, "fingerprint": "wrong-data"})


if __name__ == "__main__":
    unittest.main()
