"""Single entry point for the whole pipeline.

    python -m ev_battery all          # every stage in order
    python -m ev_battery train        # one stage
    python -m ev_battery --list       # show stages

Stages only read what earlier stages wrote, so they must run in the order below
on a fresh clone. Training and evaluation reuse finished runs whose source/data/config
fingerprints match, so re-running ``all`` is cheap once everything exists.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable


def _audit() -> None:
    from ev_battery.audit import run_audit

    run_audit()


def _eda() -> None:
    from ev_battery.eda import run_eda

    run_eda()


def _preprocess() -> None:
    from ev_battery.preprocessing import run_preprocessing

    run_preprocessing()


def _train() -> None:
    from ev_battery.reporting.experiments import build_report
    from ev_battery.training import run_study

    run_study()
    build_report()


def _features() -> None:
    from ev_battery.features import run_feature_study
    from ev_battery.reporting.features import build_report

    run_feature_study()
    build_report()


def _evaluate() -> None:
    from ev_battery.evaluation import run_final_evaluation
    from ev_battery.reporting.final import build_report

    run_final_evaluation()
    build_report()


STAGES: dict[str, tuple[Callable[[], None], str]] = {
    "audit": (_audit, "schema/quality audit, feature contract, battery split"),
    "eda": (_eda, "train-only exploratory analysis, figures and report"),
    "preprocess": (_preprocess, "fit preprocessing recipes on train, write model inputs"),
    "train": (_train, "staged MLP study (backpropagation) on train/validation"),
    "features": (_features, "domain feature-engineering ablation on validation"),
    "evaluate": (_evaluate, "score the locked checkpoints once on the test split"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ev_battery", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stage", nargs="?", choices=[*STAGES, "all"])
    parser.add_argument("--list", action="store_true", help="list stages and exit")
    args = parser.parse_args(argv)
    if args.list or args.stage is None:
        for name, (_, help_text) in STAGES.items():
            print(f"{name:<11} {help_text}")
        return
    selected = list(STAGES) if args.stage == "all" else [args.stage]
    for name in selected:
        print(f"==> {name}", flush=True)
        STAGES[name][0]()


if __name__ == "__main__":
    main()
