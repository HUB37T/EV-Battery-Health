# First NumPy test evaluation (archived)

The record of the first time the from-scratch NumPy models were scored on the test
split, on 7 November 2026 at 22:22 UTC (8 November WIB). The files are kept byte
for byte:

- [`evaluation_lock.json`](evaluation_lock.json): the lock written before the test predictions.
- [`evaluation.json`](evaluation.json): evaluation status and timestamps.
- [`metrics_by_seed.csv`](metrics_by_seed.csv) and [`metrics_summary.csv`](metrics_summary.csv): the test scores.
- [`FINAL_REPORT.md`](FINAL_REPORT.md): the report as it was published then.

The full tree of that moment, including every run record, is commit
[`83b1b36`](https://github.com/HUB37T/EV-Battery-Health/tree/83b1b360c9e08267227d3dc0f76dc47860b33089).
Paths inside the lock refer to that tree. Its `selection_source_commit` field
points to an older PyTorch commit by mistake. The NumPy configurations themselves
are recorded in `83b1b36`.

The current `results/final_evaluation/` scores the same two frozen configurations
again. Their checkpoints are rebuilt from scratch, so the run IDs and the lock are
new, but the test metrics are the same as recorded here. Report figures and
figure links in `FINAL_REPORT.md` refer to the original tree.
