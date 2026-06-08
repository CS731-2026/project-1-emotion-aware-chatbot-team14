"""Training pipeline.

Three categories of file, useful framing when deciding where new code
belongs:

  state         pure data + handoff types, no orchestration
                  pipeline/framework/*, store, keys, config, context, specs
                  (registry.py lands here too)

  utility       pure helpers, called by composition
                  pipeline/ingest.py
                  training/*, losses, optimizers, augmentations, data, loop

  composition   wiring, composes utility + state into a workflow
                  pipeline/phases.py, what each phase does
                  pipeline/driver.py, how phases sequence into a run
                  train.py, entry point: registers + runs the sweep

Imports flow strictly composition → state + utility. State files don't
import composition; utility files don't import state. The
pipeline/framework/ folder hides away the plumbing so adding a new
phase / model / dataset doesn't require navigating it.
"""
