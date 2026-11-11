"""The two model configurations frozen for the final evaluation.

``configs/final_evaluation.json`` fixes the baseline and candidate configurations,
the seeds and the SOH bands. The training stage builds exactly these six runs, so
the final evaluation never depends on how a fresh validation search ranks
alternatives on another machine.
"""

import json
from dataclasses import asdict, replace

import pandas as pd

from ev_battery.paths import ROOT
from ev_battery.training import RunConfig, load_training_data, train_run

SEEDS = [42, 123, 2026]
SOH_BAND_EDGES = [0, 70, 80, 90, 95, 101]


def validate_plan(plan):
    """Reject a plan that drops seeds, adds groups or promotes an engineered recipe."""
    if (
        plan.get("schema_version") != 3
        or plan["primary_group_fixed_before_test"] != "candidate"
        or plan["seeds"] != SEEDS
        or set(plan["groups"]) != {"baseline", "candidate"}
    ):
        raise ValueError("The final plan differs from the agreed model families or seeds.")
    for group in plan["groups"].values():
        config = RunConfig(**group["config"])
        if config.recipe.startswith("fe_"):
            raise ValueError("Feature-engineering variants are a validation ablation, not final models.")
        if asdict(config) != {**group["config"], "hidden_layers": config.hidden_layers}:
            raise ValueError("Every final configuration must list all RunConfig fields.")
    if plan["soh_band_edges"] != SOH_BAND_EDGES or len(plan["soh_band_labels"]) != 5:
        raise ValueError("The final SOH bands differ from the agreed plan.")


def load_plan(root=ROOT):
    plan = json.loads((root / "configs/final_evaluation.json").read_text(encoding="utf-8"))
    validate_plan(plan)
    return plan


def final_config(plan, group, seed):
    return replace(RunConfig(**plan["groups"][group]["config"]), seed=seed)


def train_final_runs(root=ROOT):
    """Train (or reuse) the six frozen runs on train and validation only."""
    plan = load_plan(root)
    training = {}
    rows = []
    for group in plan["groups"]:
        for seed in plan["seeds"]:
            config = final_config(plan, group, seed)
            if config.recipe not in training:
                training[config.recipe] = load_training_data(config.recipe, root)
            record = train_run(config, training[config.recipe], root)
            rows.append({"group": group, "seed": seed, "run_id": record["run_id"]})
    output = root / "results/experiments/final_runs.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, lineterminator="\n")
    return output
