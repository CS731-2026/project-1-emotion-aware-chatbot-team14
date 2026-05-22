"""Training pipeline scaffolding.

A small framework for running one (dataset × model × config) experiment
end-to-end: prepare the dataset, build + train the model, evaluate, drop
artifacts into output/run/<slug>/.

The public surface is a handful of phase functions glued together by a
driver. Each phase takes a single `Context` arg (config + store + save
methods) and returns None — produced objects are handed forward via the
store, keyed by the constants in `pipeline.keys`.
"""
