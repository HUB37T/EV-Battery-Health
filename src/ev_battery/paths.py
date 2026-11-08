"""Keep NumPy rebuilds separate from both published result sets."""

from pathlib import Path
from shutil import copyfile

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "data/processed/numpy_workspace"
RAW_FILES = ("ev_battery_health.csv", "README.md", "data_dictionary .csv")


def prepare_workspace(repo=REPO_ROOT, workspace=None):
    """Copy inputs into the ignored workspace without seeding it with release outputs."""
    workspace = workspace if workspace is not None else repo / "data/processed/numpy_workspace"
    inputs = {
        **{f"data/raw/{name}": repo / "data/raw" / name for name in RAW_FILES},
        "data/splits/battery_split.csv": repo / "data/splits/battery_split.csv",
        **{
            f"configs/{name}": repo / "configs/numpy" / name
            for name in ("preprocessing.json", "mlp_study.json", "feature_engineering.json", "followup.json")
        },
    }
    missing = [source.relative_to(repo).as_posix() for source in inputs.values() if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing NumPy inputs: {', '.join(missing)}. See data/README.md.")
    for relative, source in inputs.items():
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file() or destination.read_bytes() != source.read_bytes():
            copyfile(source, destination)
    return workspace
