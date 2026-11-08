# Locked battery split

`battery_split.csv` has one row per `battery_id` with its subset. Join it to the
raw CSV on `battery_id`, never by row position.

- Split unit: battery, not observation.
- Purpose: estimate current SOH for batteries not seen during training.
- Algorithm: sort battery IDs, permute with NumPy `default_rng(42)`, assign 70% train, 15% validation and 15% test of batteries.
- Hashes of the source CSV and of this manifest are in [`results/audit/provenance.json`](../../results/audit/provenance.json).

Row proportions differ slightly from battery proportions. The manifest was not
chosen using the target or any model score. The audit rejects a different CSV
version and refuses to overwrite a manifest that does not match. Never change the
seed to look for a better score.
