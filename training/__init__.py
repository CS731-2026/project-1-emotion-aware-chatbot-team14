"""Universal training helpers.

Each file in this package is a tiny registry — a `get_<thing>(name, **args)`
factory that returns an instance of one of a small set of supported
options. The training loop reads the name + args from the train config
yaml and calls the factory; the model file never imports from here.

This is where the "decouplable" decisions from the brainstorm live:
  - losses        — CE, label-smoothed CE
  - optimizers    — AdamW, SGD
  - augmentations — none, mild, strong (compositions of standard
                    torchvision transforms)
  - data          — the CSV-driven Dataset that every split uses
  - loop          — train_one_epoch + evaluate inner mechanics
"""
