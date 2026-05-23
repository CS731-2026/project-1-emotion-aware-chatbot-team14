"""Dataset modules.

Each module exports two symbols:
  NAME            string, used in the run-dir slug
  prepare(ctx)    function returning a DatasetSpec — fetches source,
                  applies the remap, splits, writes CSVs, persists manifest

The training pipeline imports the module references in pipeline/train.py
and the prepare_dataset phase calls `module.prepare(ctx)`. Adding a new
dataset = one new file in this folder + one line in train.py.
"""
