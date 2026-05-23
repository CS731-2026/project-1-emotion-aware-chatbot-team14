"""Store-key constants.

Phases agree on these strings rather than re-typing literals at each call
site — a rename is one edit, and a typo at a `get()` raises immediately
with a helpful "available keys: …" message.

Every key here corresponds to an object some phase puts into the store
and a later phase gets back. The comment on each line names the type the
key is expected to hold; the actual type assertion happens at
`Store.get(key, expected_type)` time.
"""

RUN_DIR        = "run_dir"          # pathlib.Path  — produced by Context.create
DATASET        = "dataset"          # DatasetSpec   — produced by prepare_dataset
MODEL_MODULE   = "model_module"     # ModuleType    — the imported models/<name>.py
TRAINED_MODEL  = "trained_model"    # TrainedModel  — produced by train
EVAL_REPORT    = "eval_report"      # EvalReport    — produced by evaluate (later)
