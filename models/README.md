# Local models

This folder is ignored by Git except for this README. The pipeline recreates its
contents:

| Path | Created by | Contents |
|---|---|---|
| `preprocessing/<recipe>.joblib` | `python -m ev_battery preprocess` | Train-fitted preprocessor, target scaler, feature order, state hash |
| `mlp/<run_id>/best.npz`, `last.npz`, `epoch_XXX.npz` | `python -m ev_battery train` and `features` | NumPy weight matrices (bias in the last row) plus config, feature names and data fingerprint |
| `feature_engineering/<variant>.joblib` | `python -m ev_battery features` | Frozen base preprocessor plus the extra feature transforms |

Load artifacts through the functions that verify compatibility instead of calling
`np.load` or `joblib.load` directly:

```python
from ev_battery.training import load_checkpoint, load_training_data

data = load_training_data("log_sensor_caps")
model, checkpoint = load_checkpoint("models/mlp/<run_id>/best.npz", data)
predictions = model.predict(data["validation"]["X"])  # standardised SOH
```

A `run_id` is derived from a hash of the configuration, the data, and the training
code (`model.py` and `training.py`), so changing any of them produces a new run.
Only load joblib files from trusted sources.
