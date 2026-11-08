# NumPy follow-up archive

This directory preserves the first NumPy outputs from [PR #1 at `83b1b36`](https://github.com/HUB37T/EV-Battery-Health/tree/83b1b360c9e08267227d3dc0f76dc47860b33089). The [PyTorch release](../final_evaluation/FINAL_REPORT.md) evaluated these same 390 test batteries on 6 November 2026. The NumPy snapshot was scored on 8 November WIB (7 November UTC). Its results are descriptive follow-up evidence on the same split.

- [Audit](audit/AUDIT_REPORT.md), [EDA](eda/EDA_REPORT.md) and [preprocessing](preprocessing/PREPROCESSING_REPORT.md)
- [MLP experiments](experiments/EXPERIMENT_REPORT.md) and [feature engineering](feature_engineering/FEATURE_ENGINEERING_REPORT.md)
- [Follow-up scores and limitations](final_evaluation/FINAL_REPORT.md), [all test seeds](final_evaluation/metrics_by_seed.csv), and [historical lock](final_evaluation/evaluation_lock.json)

The historical lock is preserved byte for byte. Its `selection_source_commit` field points to an older PyTorch commit by mistake. The actual reference for the archived NumPy configurations is `83b1b360c9e08267227d3dc0f76dc47860b33089`, which contains the NumPy study and run summaries. This identifies the commit containing the records, whose timestamp follows the original NumPy scoring. The [local reproduction plan](../../configs/numpy/followup.json) and newly generated locks record the corrected reference and original PyTorch release separately. Paths inside the historical lock refer to the original tree at `83b1b36`. The archived config files are under `configs/numpy/` in the current tree.

The `inverse` experiment equalises total raw weight across SOH bands in the training set. Each mini-batch divides by its own total weight, so the effective gradient contribution varies with batch composition. The archived scores use that original algorithm.

`python -m ev_battery all` writes local results to `data/processed/numpy_workspace/`, including a separate fingerprinted follow-up report. See the [setup instructions](../../README.md).
