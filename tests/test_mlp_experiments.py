"""Checks of backpropagation, checkpoint selection, and holdout boundaries."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.mlp_experiments import (
    MLP, RunConfig, load_checkpoint, load_training_data, make_loss,
    predict_scaled, regression_metrics, train_run,
)


class MLPTests(unittest.TestCase):
    def test_backprop_matches_manual_chain_rule_and_sgd_update(self):
        hidden=nn.Linear(1,1);out=nn.Linear(1,1)
        with torch.no_grad():
            hidden.weight.fill_(.5);hidden.bias.zero_();out.weight.fill_(.4);out.bias.zero_()
        model=nn.Sequential(hidden,nn.ReLU(),out)
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        prediction=model(torch.tensor([[2.]]))
        loss=.5*(prediction-1).square().mean()
        loss.backward()
        self.assertAlmostEqual(hidden.weight.grad.item(),-.48,places=6)
        self.assertAlmostEqual(hidden.bias.grad.item(),-.24,places=6)
        self.assertAlmostEqual(out.weight.grad.item(),-.6,places=6)
        optimizer.step()
        self.assertAlmostEqual(hidden.weight.item(),.548,places=6)
        self.assertAlmostEqual(out.weight.item(),.46,places=6)
        self.assertAlmostEqual(model(torch.tensor([[2.]])).item(),.5752,places=6)

    def test_parameter_count_and_batch_one_shape(self):
        model=MLP(67,(64,32),"relu")
        self.assertEqual(sum(p.numel() for p in model.parameters()),6465)
        self.assertEqual(tuple(model(torch.zeros(1,67)).shape),(1,))
        self.assertEqual(tuple(model(torch.zeros(5,67)).shape),(5,))

    def test_supported_activations_losses_and_invalid_options(self):
        for activation in ["relu","tanh","sigmoid"]:
            model=MLP(3,(4,),activation)
            self.assertTrue(torch.isfinite(model(torch.ones(2,3))).all())
        for loss in ["mse","mae","huber"]:
            self.assertTrue(torch.isfinite(make_loss(loss)(torch.ones(3),torch.zeros(3))))
        with self.assertRaises(ValueError):MLP(3,(),"relu")
        with self.assertRaises(ValueError):make_loss("accuracy")

    def test_dropout_is_disabled_during_prediction(self):
        model=MLP(3,(8,),"relu",dropout=.5)
        X=torch.ones(10,3)
        a=predict_scaled(model,X);b=predict_scaled(model,X)
        np.testing.assert_array_equal(a,b)
        self.assertFalse(model.training)

    def test_metrics_and_battery_weighting(self):
        metrics=regression_metrics([1,2,4],[1,3,2])
        self.assertAlmostEqual(metrics["mae"],1)
        self.assertAlmostEqual(metrics["rmse"],np.sqrt(5/3))
        self.assertAlmostEqual(metrics["r2"],1-5/(14/3))
        grouped=regression_metrics([0,0,0,0],[0,2,2,2],["a","b","b","b"])
        self.assertEqual(grouped["mae"],1.5)
        self.assertEqual(grouped["battery_mean_mae"],1.)
        self.assertIsNone(grouped["r2"])
        with self.assertRaises(ValueError):regression_metrics([1,2],[[1],[2]])

    def test_training_loader_never_requests_test_split(self):
        train={"X":np.zeros((2,1),dtype=np.float32),"y_scaled":np.zeros(2,dtype=np.float32),"y_original":np.array([90.,91.]),
               "observation_id":np.array(["o1","o2"]),"battery_id":np.array(["a","a"]),"feature_names":np.array(["x"])}
        val={**train,"battery_id":np.array(["b","b"]),"observation_id":np.array(["v1","v2"])}
        calls=[]
        def loader(recipe,split,root):
            calls.append(split)
            if split=="test":raise AssertionError("Test must stay locked")
            return (train if split=="train" else val),{"state_hash":"same"}
        with patch("src.mlp_experiments.load_prepared_split",side_effect=loader):
            load_training_data("probe")
        self.assertEqual(calls,["train","validation"])

    def test_checkpoint_is_best_validation_epoch_and_reloads_exactly(self):
        X=np.array([[-1.],[-.5],[0.],[.5],[1.]],dtype=np.float32)
        original=np.array([80.,85.,90.,95.,100.])
        scaler=StandardScaler().fit(pd.DataFrame({"state_of_health_pct":original}))
        y=scaler.transform(pd.DataFrame({"state_of_health_pct":original})).ravel().astype(np.float32)
        train={"X":X,"y_scaled":y,"y_original":original,"battery_id":np.array(["a"]*5),"observation_id":np.array([f"o{i}" for i in range(5)]),"feature_names":np.array(["x"])}
        val={**train,"battery_id":np.array(["b"]*5)}
        data={"train":train,"validation":val,"bundle":{"target_scaler":scaler},"fingerprint":"unit-test-input"}
        config=RunConfig(recipe="probe",hidden_layers=(4,),max_epochs=4,batch_size=2,patience=2,seed=8)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            result=train_run(config,data,root,threads=1)
            history=pd.read_csv(root/"results/experiments/runs"/result["run_id"]/"history.csv")
            trained=history.loc[history.epoch.gt(0)]
            self.assertEqual(result["best_epoch"],int(trained.loc[trained.validation_mae_pp.idxmin(),"epoch"]))
            model,_=load_checkpoint(root/result["checkpoint_best"],data)
            a=predict_scaled(model,torch.from_numpy(X))
            model2,_=load_checkpoint(root/result["checkpoint_best"],data)
            np.testing.assert_array_equal(a,predict_scaled(model2,torch.from_numpy(X)))
            again=train_run(config,data,root,threads=1)
            self.assertEqual(again["training_seconds"],result["training_seconds"])
            self.assertGreater(result["first_update"]["gradient_l2_norm"],0)
            self.assertGreater(result["first_update"]["first_layer_max_weight_change"],0)
            with self.assertRaises(ValueError):load_checkpoint(root/result["checkpoint_best"],{**data,"fingerprint":"wrong-data"})


if __name__=="__main__":
    unittest.main()
