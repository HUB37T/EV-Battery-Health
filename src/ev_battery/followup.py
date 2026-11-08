"""Rebuild the original NumPy model configurations on train and validation."""

import json
from dataclasses import replace

import pandas as pd

from ev_battery.paths import ROOT
from ev_battery.training import RunConfig, load_training_data, train_run


def validate_plan(plan):
    if (
        plan["schema_version"] != 2
        or plan["evaluation_scope"] != "numpy_followup_same_previously_evaluated_test_split"
        or plan["primary_group_fixed_before_test"] != "candidate"
        or plan["seeds"] != [42, 123, 2026]
        or set(plan["groups"]) != {"baseline", "candidate"}
    ):
        raise ValueError("The NumPy follow-up plan differs from the recorded model families or seeds.")
    identifiers = []
    for group in plan["groups"].values():
        config = RunConfig(**group["config"])
        if config.recipe != group["recipe"] or config.recipe.startswith("fe_"):
            raise ValueError("The follow-up requires the recorded base preprocessing recipes.")
        if set(group["reference_run_ids_by_seed"]) != {str(seed) for seed in plan["seeds"]}:
            raise ValueError("A reference seed is missing or was added.")
        identifiers.extend(group["reference_run_ids_by_seed"].values())
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("A reference run ID is used more than once.")
    if plan["soh_band_edges"] != [0, 70, 80, 90, 95, 101] or len(plan["soh_band_labels"]) != 5:
        raise ValueError("The follow-up SOH bands differ from the recorded plan.")


def load_plan(root):
    plan = json.loads((root / "configs/followup.json").read_text(encoding="utf-8"))
    validate_plan(plan)
    return plan


def train_followup_runs(root=ROOT):
    """Resolve local run IDs without selecting families from a fresh search or test results."""
    plan = load_plan(root)
    training = {}
    rows = []
    for group, spec in plan["groups"].items():
        recipe = spec["recipe"]
        if recipe not in training:
            training[recipe] = load_training_data(recipe, root)
        for seed in plan["seeds"]:
            config = replace(RunConfig(**spec["config"]), seed=seed)
            record = train_run(config, training[recipe], root)
            rows.append({"group": group, "seed": seed, "run_id": record["run_id"]})
    output = root / "results/experiments/followup_runs.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, lineterminator="\n")
    return output
