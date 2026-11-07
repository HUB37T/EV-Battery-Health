"""Repository locations shared by every pipeline stage."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
