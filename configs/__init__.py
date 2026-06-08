"""Train config modules.

Each module exports two symbols:
  NAME     string, used in the run-dir slug
  CONFIG   dict of training hyperparams (epochs, batch_size, loss,
           optimizer, augment, num_workers, read by the train phase)

Edit pipeline/train.py to add a config to the sweep.
"""
