# Archive

Read-only records of earlier versions of the project. Nothing here is produced by
the current pipeline, and nothing here should be edited or regenerated. They are
kept so that the report can compare versions and so that the history of the test
split stays transparent.

| Folder | What it records | Test scored |
|---|---|---|
| [`pytorch/`](pytorch/README.md) | The original release built with PyTorch: notebooks, results, configs and the team guide | 6 November 2026 |
| [`numpy_first_evaluation/`](numpy_first_evaluation/README.md) | The first test scoring of the from-scratch NumPy models | 8 November 2026 (WIB) |

## Timeline of the test split

1. **PyTorch release** (commit `479ed84`). Configurations were selected on
   validation, then six checkpoints were locked and scored on the 390 test
   batteries.
2. **Rewrite to NumPy** (pull request #1). The network, backpropagation and
   optimizers were rewritten from scratch to follow the course modules. This was
   a requirement of the course, not a reaction to test results. The NumPy study
   selected its configurations on validation only, and they were scored once on
   the same test batteries.
3. **Current tree.** The same two NumPy configurations are frozen in
   `configs/final_evaluation.json`. Re-running the pipeline reproduces their
   checkpoints and scores them again, which is a reproduction, not a new test.

Because the same test batteries were scored by both versions, comparisons between
PyTorch and NumPy are descriptive. Neither version was tuned on test.
