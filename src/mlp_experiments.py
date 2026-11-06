"""MLP trained by backpropagation. Only train/validation are loaded here.

python -m src.mlp_experiments --baseline
python -m src.mlp_experiments
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data_audit import ROOT
from src.preprocessing import load_prepared_split


@dataclass(frozen=True)
class RunConfig:
    recipe: str = "baseline_standard"
    hidden_layers: tuple = (64, 32)
    activation: str = "relu"
    loss: str = "mse"
    huber_delta: float = 1.
    learning_rate: float = .001
    batch_size: int = 256
    max_epochs: int = 200
    early_stopping: bool = True
    patience: int = 15
    min_delta_pp: float = .001
    weight_decay: float = 0.
    dropout: float = 0.
    seed: int = 42


class MLP(nn.Module):
    def __init__(self, input_features, hidden_layers=(64, 32), activation="relu", dropout=0.):
        super().__init__()
        activations = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
        if activation not in activations or not hidden_layers or any(int(n) != n or n < 1 for n in hidden_layers):
            raise ValueError("Aktivasi atau ukuran hidden tidak valid.")
        if input_features < 1 or not 0 <= dropout < 1:
            raise ValueError("Dimensi/dropout tidak valid.")
        layers=[]
        previous=input_features
        for units in hidden_layers:
            layers.extend([nn.Linear(previous, units), activations[activation]()])
            if dropout:
                layers.append(nn.Dropout(dropout))
            previous=units
        layers.append(nn.Linear(previous, 1))  # Linear output: scaled SOH regression.
        self.network=nn.Sequential(*layers)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.)
                nn.init.zeros_(layer.bias)

    def forward(self, X):
        return self.network(X).squeeze(-1)  # Shape (N,), including batch size one.


def make_loss(name, huber_delta=1.):
    if name == "mse": return nn.MSELoss()
    if name == "mae": return nn.L1Loss()
    if name == "huber" and huber_delta > 0: return nn.HuberLoss(delta=huber_delta)
    raise ValueError("Loss tidak didukung.")


def regression_metrics(y, pred, battery_ids=None):
    y=np.asarray(y,dtype=np.float64);pred=np.asarray(pred,dtype=np.float64)
    if y.ndim != 1 or pred.shape != y.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Target/prediksi harus vektor finite dengan shape sama.")
    error=pred-y
    denominator=float(np.square(y-y.mean()).sum())
    metrics={"mae":float(np.abs(error).mean()),"rmse":float(np.sqrt(np.square(error).mean())),
             "r2":float(1-np.square(error).sum()/denominator) if denominator else None,
             "bias_pp":float(error.mean()),"outside_0_100_pct":float(((pred<0)|(pred>100)).mean()*100)}
    if battery_ids is not None:
        if len(battery_ids)!=len(y):raise ValueError("Panjang battery IDs tidak cocok.")
        metrics["battery_mean_mae"]=float(pd.Series(np.abs(error)).groupby(np.asarray(battery_ids)).mean().mean())
    return metrics


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8",newline="\n")


def load_training_data(recipe,root=ROOT):
    # This allowlist is deliberately separate from the preprocessing loader.
    train,bundle=load_prepared_split(recipe,"train",root)
    validation,other=load_prepared_split(recipe,"validation",root)
    if bundle["state_hash"]!=other["state_hash"] or set(train["battery_id"]) & set(validation["battery_id"]):
        raise ValueError("Pipeline berbeda atau grup train/validation tumpang tindih.")
    digest=hashlib.sha256()
    for data in [train,validation]:
        for key in ["X","y_scaled","y_original","observation_id","battery_id","feature_names"]:
            digest.update(np.ascontiguousarray(data[key]).tobytes())
    digest.update(bundle["state_hash"].encode())
    return {"train":train,"validation":validation,"bundle":bundle,"fingerprint":digest.hexdigest()}


def configure_runtime(seed,threads=2):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def predict_scaled(model,X,batch_size=4096):
    model.eval()
    predictions=[]
    with torch.no_grad():
        for batch in X.split(batch_size):predictions.append(model(batch).cpu().numpy())
    result=np.concatenate(predictions)
    if not np.isfinite(result).all():raise FloatingPointError("Prediksi nonfinite.")
    return result


def original_scale(scaled,bundle):
    return bundle["target_scaler"].inverse_transform(np.asarray(scaled,dtype=np.float64).reshape(-1,1)).ravel()


def evaluate(model,X,y,data,bundle,criterion):
    scaled=predict_scaled(model,X)
    objective=float(criterion(torch.from_numpy(scaled),y).item())
    metrics=regression_metrics(data["y_original"],original_scale(scaled,bundle),data["battery_id"])
    return objective,metrics


def load_checkpoint(path,expected_data=None):
    checkpoint=torch.load(path,map_location="cpu",weights_only=True)
    cfg=RunConfig(**checkpoint["config"])
    model=MLP(checkpoint["input_features"],cfg.hidden_layers,cfg.activation,cfg.dropout)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    if expected_data is not None:
        if checkpoint["data_fingerprint"]!=expected_data["fingerprint"] or checkpoint["feature_names"]!=expected_data["train"]["feature_names"].tolist():
            raise ValueError("Checkpoint dan input preprocessing tidak cocok.")
    return model,checkpoint


def train_run(config: RunConfig, data=None, root=ROOT, threads=2):
    if config.batch_size<1 or config.max_epochs<1 or config.patience<1 or config.learning_rate<=0 or config.min_delta_pp<0 or config.weight_decay<0:
        raise ValueError("Hyperparameter training tidak valid.")
    if data is None:data=load_training_data(config.recipe,root)
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity={"config":asdict(config),"data_fingerprint":data["fingerprint"],"training_source_sha256":source_hash,
              "torch":str(torch.__version__),"threads":threads,"device":"cpu"}
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    run_id="run_"+fingerprint[:12]
    output=root/"results/experiments/runs"/run_id
    checkpoints=root/"models/mlp"/run_id
    output.mkdir(parents=True,exist_ok=True);checkpoints.mkdir(parents=True,exist_ok=True)
    summary_path=output/"summary.json"
    if summary_path.is_file():
        cached=json.loads(summary_path.read_text(encoding="utf-8"))
        if cached.get("fingerprint")!=fingerprint:raise ValueError("Run ID collision atau provenance berubah.")
        if cached.get("status")=="complete" and (checkpoints/"best.pt").is_file() and (checkpoints/"last.pt").is_file() and (output/"history.csv").is_file():
            return cached
    write_json(output/"config.json",identity)
    write_json(output/"status.json",{"status":"running","run_id":run_id,"fingerprint":fingerprint})
    configure_runtime(config.seed,threads)
    train=data["train"];val=data["validation"];bundle=data["bundle"]
    X=torch.from_numpy(train["X"]);y=torch.from_numpy(train["y_scaled"])
    VX=torch.from_numpy(val["X"]);vy=torch.from_numpy(val["y_scaled"])
    model=MLP(X.shape[1],config.hidden_layers,config.activation,config.dropout)
    criterion=make_loss(config.loss,config.huber_delta)
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    generator=torch.Generator().manual_seed(config.seed)
    history=[];first_update={};fixed_rows=[]
    best_mae=float("inf");best_epoch=0;reference=float("inf");stale=0
    started=time.perf_counter();updates=0

    def monitor(epoch,batch_loss=None):
        tr_loss,tr_metrics=evaluate(model,X,y,train,bundle,criterion)
        va_loss,va_metrics=evaluate(model,VX,vy,val,bundle,criterion)
        row={"epoch":epoch,"train_objective_scaled":tr_loss,"validation_objective_scaled":va_loss,
             "batch_objective_scaled":batch_loss,"train_mae_pp":tr_metrics["mae"],"validation_mae_pp":va_metrics["mae"],
             "train_rmse_pp":tr_metrics["rmse"],"validation_rmse_pp":va_metrics["rmse"],
             "elapsed_seconds":time.perf_counter()-started}
        history.append(row)
        return tr_metrics,va_metrics

    def save_checkpoint(filename,epoch):
        torch.save({"run_id":run_id,"config":asdict(config),"input_features":int(X.shape[1]),
                    "feature_names":train["feature_names"].tolist(),"data_fingerprint":data["fingerprint"],
                    "fingerprint":fingerprint,"epoch":epoch,"best_epoch":best_epoch,"best_validation_mae_pp":best_mae,
                    "model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),
                    "torch_rng_state":torch.get_rng_state(),"shuffle_rng_state":generator.get_state(),
                    "torch_version":str(torch.__version__)},checkpoints/filename)
    print(f"START {run_id} {config.recipe} hidden={list(config.hidden_layers)} {config.activation}/{config.loss} seed={config.seed}",flush=True)
    try:
        monitor(0)
        for epoch in range(1,config.max_epochs+1):
            model.train();weighted_loss=0.
            order=torch.randperm(len(y),generator=generator)
            for indices in order.split(config.batch_size):
                optimizer.zero_grad(set_to_none=True)
                predicted=model(X[indices])
                if predicted.shape!=y[indices].shape:raise ValueError("Shape output dan target berbeda; broadcasting dilarang.")
                loss=criterion(predicted,y[indices])
                if not torch.isfinite(loss):raise FloatingPointError("Loss nonfinite.")
                loss.backward()  # Backpropagation: chain rule through all hidden layers.
                if not first_update:
                    first_layer=next(m for m in model.modules() if isinstance(m,nn.Linear))
                    before=first_layer.weight.detach().clone()
                    norm=math.sqrt(sum(float(p.grad.detach().square().sum()) for p in model.parameters() if p.grad is not None))
                    first_update={"batch_loss_scaled":float(loss.detach()),"gradient_l2_norm":norm,
                                  "first_layer_gradient_l2_norm":float(first_layer.weight.grad.norm())}
                    if not np.isfinite(norm) or norm==0:raise FloatingPointError("Gradien pertama invalid/nol.")
                    optimizer.step()
                    first_update["first_layer_max_weight_change"]=float((first_layer.weight.detach()-before).abs().max())
                else:
                    optimizer.step()
                weighted_loss+=float(loss.detach())*len(indices);updates+=1
            _,va=monitor(epoch,weighted_loss/len(y))
            if va["mae"]<best_mae:
                best_mae=va["mae"];best_epoch=epoch
                save_checkpoint("best.pt",epoch)
            if va["mae"]<reference-config.min_delta_pp:
                reference=va["mae"];stale=0
            else:stale+=1
            if not config.early_stopping and epoch in [50,100,200]:
                save_checkpoint(f"epoch_{epoch:03d}.pt",epoch)
                tr_fixed,va_fixed=monitor_endpoint(model,X,y,VX,vy,train,val,bundle,criterion)
                fixed_rows.append({"run_id":run_id,"epoch":epoch,**{"train_"+k:v for k,v in tr_fixed.items()},**{"validation_"+k:v for k,v in va_fixed.items()}})
            if epoch%10==0:pd.DataFrame(history).to_csv(output/"history.csv",index=False,lineterminator="\n")
            if epoch%50==0:print(f"  epoch {epoch}: train MAE={history[-1]['train_mae_pp']:.4f}, validation MAE={va['mae']:.4f}",flush=True)
            if config.early_stopping and stale>=config.patience:break
        save_checkpoint("last.pt",epoch)
        pd.DataFrame(history).to_csv(output/"history.csv",index=False,lineterminator="\n")
        if fixed_rows:pd.DataFrame(fixed_rows).to_csv(output/"fixed_epoch_metrics.csv",index=False,lineterminator="\n")
        model,checkpoint=load_checkpoint(checkpoints/"best.pt",data)
        tr_loss,tr=evaluate(model,X,y,train,bundle,criterion)
        va_loss,va=evaluate(model,VX,vy,val,bundle,criterion)
        if not np.isclose(va["mae"],best_mae,atol=1e-9):raise AssertionError("Checkpoint tidak mereproduksi validation MAE terbaik.")
        predictions=original_scale(predict_scaled(model,VX),bundle)
        pd.DataFrame({"observation_id":val["observation_id"],"battery_id":val["battery_id"],"actual_soh":val["y_original"],"predicted_soh":predictions}).to_csv(output/"validation_predictions.csv",index=False,lineterminator="\n")
        summary={"status":"complete","run_id":run_id,"fingerprint":fingerprint,"config":asdict(config),
                 "input_features":int(X.shape[1]),"parameters":sum(p.numel() for p in model.parameters()),"epochs_actual":epoch,
                 "best_epoch":best_epoch,"stopped_early":epoch<config.max_epochs,"optimizer_updates":updates,
                 "training_seconds":time.perf_counter()-started,"train":tr,"validation":va,
                 "train_objective_scaled":tr_loss,"validation_objective_scaled":va_loss,"first_update":first_update,
                 "checkpoint_best":str((checkpoints/"best.pt").relative_to(root)).replace("\\","/"),
                 "checkpoint_last":str((checkpoints/"last.pt").relative_to(root)).replace("\\","/"),"identity":identity}
        write_json(summary_path,summary);write_json(output/"status.json",{"status":"complete","run_id":run_id})
        print(f"DONE {run_id}: best epoch={best_epoch}/{epoch}, val MAE={va['mae']:.4f}, train MAE={tr['mae']:.4f}, seconds={summary['training_seconds']:.1f}",flush=True)
        return summary
    except Exception as exc:
        pd.DataFrame(history).to_csv(output/"history.csv",index=False,lineterminator="\n")
        write_json(output/"status.json",{"status":"failed","run_id":run_id,"error":str(exc)})
        raise


def monitor_endpoint(model,X,y,VX,vy,train,val,bundle,criterion):
    _,tr=evaluate(model,X,y,train,bundle,criterion)
    _,va=evaluate(model,VX,vy,val,bundle,criterion)
    return tr,va


def flatten_summary(run):
    config=run["config"]
    return {"run_id":run["run_id"],**config,"hidden_layers":"-".join(map(str,config["hidden_layers"])),
            "input_neurons":run["input_features"],"parameters":run["parameters"],"epochs_actual":run["epochs_actual"],
            "best_epoch":run["best_epoch"],"seconds":run["training_seconds"],
            **{"train_"+k:v for k,v in run["train"].items()},**{"validation_"+k:v for k,v in run["validation"].items()}}


def run_study(root=ROOT):
    settings=json.loads((root/"configs/mlp_study.json").read_text(encoding="utf-8"))
    if settings["device"]!="cpu" or settings["fixed_epoch_budgets"]!=[50,100,200]:raise ValueError("Konfigurasi study tidak didukung.")
    initial=RunConfig(**settings["baseline"])
    data_cache={};runs={};stages=[]
    output=root/"results/experiments";output.mkdir(parents=True,exist_ok=True)
    def execute(cfg):
        if cfg.recipe not in data_cache:data_cache[cfg.recipe]=load_training_data(cfg.recipe,root)
        result=train_run(cfg,data_cache[cfg.recipe],root,settings["cpu_threads"])
        runs[result["run_id"]]=result
        pd.DataFrame([flatten_summary(r) for r in runs.values()]).to_csv(output/"run_summary.csv",index=False,lineterminator="\n")
        return result
    def stage(name,configs):
        results=[execute(cfg) for cfg in configs]
        winner=min(results,key=lambda r:(r["validation"]["mae"],r["parameters"],r["run_id"]))
        for result in results:
            stages.append({"stage":name,"run_id":result["run_id"],"selected":result["run_id"]==winner["run_id"],
                           "validation_mae_pp":result["validation"]["mae"],"train_mae_pp":result["train"]["mae"],
                           "recipe":result["config"]["recipe"],"hidden_layers":"-".join(map(str,result["config"]["hidden_layers"])),
                           "activation":result["config"]["activation"],"loss":result["config"]["loss"],
                           "weight_decay":result["config"]["weight_decay"],"dropout":result["config"]["dropout"]})
        pd.DataFrame(stages).to_csv(output/"stage_comparisons.csv",index=False,lineterminator="\n")
        print(f"SELECT {name}: {winner['run_id']} validation MAE={winner['validation']['mae']:.4f}",flush=True)
        return RunConfig(**winner["config"]),winner
    baseline=execute(initial)
    cfg,_=stage("preprocessing",[replace(initial,recipe=p) for p in settings["preprocessing_recipes"]])
    cfg,_=stage("hidden_layers",[replace(cfg,hidden_layers=tuple(h)) for h in settings["hidden_layers"]])
    cfg,_=stage("activation",[replace(cfg,activation=a) for a in settings["activations"]])
    cfg,_=stage("loss",[replace(cfg,loss=l) for l in settings["losses"]])
    cfg,candidate=stage("regularization",[cfg]+[replace(cfg,**p) for p in settings["regularization_probes"]])
    fixed=execute(replace(cfg,early_stopping=False,max_epochs=max(settings["fixed_epoch_budgets"])))
    fixed_table=pd.read_csv(output/"runs"/fixed["run_id"]/"fixed_epoch_metrics.csv")
    fixed_table.to_csv(output/"epoch_budget_comparison.csv",index=False,lineterminator="\n")
    confirmations=[]
    for group,selected_cfg in [("baseline",initial),("candidate",cfg)]:
        for seed in settings["confirmation_seeds"]:
            run=execute(replace(selected_cfg,seed=seed))
            confirmations.append({"group":group,"seed":seed,"run_id":run["run_id"],"validation_mae_pp":run["validation"]["mae"],
                                  "validation_rmse_pp":run["validation"]["rmse"],"validation_r2":run["validation"]["r2"],
                                  "train_mae_pp":run["train"]["mae"],"best_epoch":run["best_epoch"]})
    confirmation=pd.DataFrame(confirmations)
    confirmation.to_csv(output/"seed_confirmation.csv",index=False,lineterminator="\n")
    seed_summary=confirmation.groupby("group").agg(n_seeds=("seed","size"),validation_mae_mean=("validation_mae_pp","mean"),
        validation_mae_std=("validation_mae_pp","std"),validation_rmse_mean=("validation_rmse_pp","mean"),validation_r2_mean=("validation_r2","mean")).reset_index()
    seed_summary.to_csv(output/"seed_summary.csv",index=False,lineterminator="\n")
    val=data_cache[initial.recipe]["validation"];train=data_cache[initial.recipe]["train"]
    constants=[]
    for name,value in [("train_median",np.median(train["y_original"])),("train_mean",np.mean(train["y_original"]))]:
        constants.append({"reference":name,"constant_soh":float(value),**regression_metrics(val["y_original"],np.full(len(val["y_original"]),value),val["battery_id"])})
    pd.DataFrame(constants).to_csv(output/"constant_references.csv",index=False,lineterminator="\n")
    predictions=pd.read_csv(output/"runs"/candidate["run_id"]/"validation_predictions.csv")
    predictions["soh_band"]=pd.cut(predictions.actual_soh,[-np.inf,70,80,90,95,np.inf],right=False,labels=["<70","70–<80","80–<90","90–<95",">=95"])
    predictions["absolute_error"]=(predictions.predicted_soh-predictions.actual_soh).abs()
    predictions.groupby("soh_band",observed=False).agg(rows=("actual_soh","size"),batteries=("battery_id","nunique"),mae_pp=("absolute_error","mean")).reset_index().to_csv(output/"candidate_validation_bands.csv",index=False,lineterminator="\n")
    study={"status":"complete","baseline_run_id":baseline["run_id"],"candidate_run_id":candidate["run_id"],"fixed_epoch_run_id":fixed["run_id"],
           "unique_training_runs":len(runs),"train_rows":len(train["X"]),"validation_rows":len(val["X"]),"test_loaded":False,
           "candidate_config":asdict(cfg),"baseline_config":asdict(initial),"confirmation_seeds":settings["confirmation_seeds"],
           "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "settings_sha256":hashlib.sha256((root/"configs/mlp_study.json").read_bytes()).hexdigest(),
           "packages":{p:version(p) for p in ["torch","numpy","pandas","scikit-learn"]},
           "data_fingerprints":{recipe:data["fingerprint"] for recipe,data in data_cache.items()},
           "device":"cpu","cpu_threads":settings["cpu_threads"],"selection":"Sequential seed-42 validation MAE, three-seed confirmation; no global optimum or unbiased final test claim.",
           "total_training_seconds":sum(r["training_seconds"] for r in runs.values())}
    write_json(output/"study.json",study)
    return study


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",action="store_true")
    args=parser.parse_args()
    if args.baseline:
        settings=json.loads((ROOT/"configs/mlp_study.json").read_text(encoding="utf-8"))
        train_run(RunConfig(**settings["baseline"]),threads=settings["cpu_threads"])
    else:
        run_study()
