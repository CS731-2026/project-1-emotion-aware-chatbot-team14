"""Model architectures.

Each model is a single module under this package exporting exactly two
symbols:

  - build(num_classes: int) -> nn.Module
        the architecture
  - PREPROCESS: torchvision.transforms.Compose
        the input pipeline (resize + normalize) that's structural to the
        architecture; augmentation gets composed on top of this by the
        training loop, never inside the model file

The training loop imports the module by name (`importlib.import_module
(f"models.{cfg.model}")`) and reads those two symbols. Nothing else
about the architecture leaks into the rest of the codebase.
"""
