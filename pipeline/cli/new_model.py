"""Scaffold a new model module under pipeline/models/<id>/.

Two templates:

  --template tutorial   (default), copies pipeline/models/tutorial/ with
                        names rewritten. Heavy comments throughout; the
                        right starting point when learning the framework
                        or when you want a working template you can
                        delete pieces from.

  --template simple     30-line minimum using train_classifier, no
                        custom training loop, no per-model CFG. Use
                        when your training is "CE + AdamW, that's it".

Usage:
    make new-model ID=my_model                       # tutorial template
    make new-model ID=my_model TEMPLATE=simple       # minimal template
    python -m pipeline.cli.new_model my_model --template tutorial --force

After generation, the script prints exact next-step commands. The
tutorial template is the SOURCE OF TRUTH for the scaffolder, when
the framework changes, update pipeline/models/tutorial/ and the
scaffolder picks it up automatically.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# ── simple template (hardcoded, no equivalent reference module) ─────────

SIMPLE_INIT = '''\
"""{model_id}, simple classifier using the shared train_classifier helper.

For a custom training procedure (split-LR, freeze schedules, MixUp,
focal loss, etc) scaffold with --template tutorial instead, or look at
pipeline/models/tutorial/ for the fully-documented template.
"""

from __future__ import annotations

from pipeline.framework.context import Context
from pipeline.framework.specs import DatasetSpec, TrainedModel
from pipeline.training.standard import train_classifier

from .augment import PREPROCESS
from .model import build


def train(ctx: Context, dataset: DatasetSpec) -> TrainedModel:
    return train_classifier(
        ctx, dataset,
        model=build(dataset.num_classes),
        preprocess=PREPROCESS,
    )
'''

SIMPLE_MODEL = '''\
"""Architecture for {model_id}. Edit me."""

from __future__ import annotations

import torch.nn as nn


class {class_name}(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        # TODO: replace with your real architecture
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build(num_classes: int) -> nn.Module:
    return {class_name}(num_classes=num_classes)
'''

SIMPLE_AUGMENT = '''\
"""Transforms for {model_id}. PREPROCESS is what train_classifier
prepends a configurable augmentation block to (see configs/*.py)."""

from __future__ import annotations

import torchvision.transforms as T


IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


PREPROCESS = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
'''


# ── tutorial template (sourced from pipeline/models/tutorial/) ───────────
#
# The tutorial directory is a real, runnable model. The scaffolder
# reads its files at runtime and rewrites the names. This way updating
# the tutorial = updating the template, no drift.
#
# Substitution rules, applied IN ORDER, longest match first matters:
#   "Tutorial"   → user's class name (UpperCamelCase, e.g. "MyModel")
#   "tutorial"   → user's model id (snake_case, e.g. "my_model")
#
# Files copied: __init__.py, model.py, augment.py, data.py, train_loop.py.

_TUTORIAL_FILES = ("__init__.py", "model.py", "augment.py", "data.py", "train_loop.py")


def _tutorial_source_dir(repo_root: Path) -> Path:
    return repo_root / "pipeline" / "models" / "tutorial"


def _render_tutorial(content: str, *, model_id: str, class_name: str) -> str:
    """Substitute 'Tutorial' / 'tutorial' tokens throughout the file.

    Word-boundary regex so we don't mangle unrelated substrings (e.g.
    if some teammate later names their model 'tutorial_v2').
    """
    content = re.sub(r"\bTutorial\b", class_name, content)
    content = re.sub(r"\btutorial\b", model_id, content)
    return content


def _read_tutorial_template(repo_root: Path) -> dict[str, str]:
    src = _tutorial_source_dir(repo_root)
    if not src.is_dir():
        raise FileNotFoundError(
            f"tutorial source not found at {src}. The scaffolder reads "
            f"from there. Re-clone or restore the file."
        )
    out = {}
    for name in _TUTORIAL_FILES:
        path = src / name
        if not path.exists():
            raise FileNotFoundError(f"tutorial source missing file: {path}")
        out[name] = path.read_text()
    return out


# ── helpers ─────────────────────────────────────────────────────────────

def _class_name_for(model_id: str) -> str:
    """my_model → MyModel."""
    return "".join(part.capitalize() for part in model_id.split("_"))


def _write(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists (pass --force to overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


# ── main ────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_id",
                    help="snake_case model id; becomes pipeline/models/<id>/")
    ap.add_argument("--template", choices=("tutorial", "simple"), default="tutorial")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing files (default: fail if any exist)")
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()

    model_id = args.model_id
    if not model_id.replace("_", "").isalnum() or not model_id[0].isalpha():
        print(f"✗ model_id must be snake_case ascii starting with a letter: {model_id!r}",
              file=sys.stderr)
        return 2
    if model_id == "tutorial":
        print("✗ 'tutorial' is reserved for the reference module. Pick another id.",
              file=sys.stderr)
        return 2

    repo_root = Path(args.repo_root).resolve()
    target = repo_root / "pipeline" / "models" / model_id
    if target.exists() and not args.force:
        print(f"✗ {target} already exists. Pass --force to overwrite, or pick a different id.",
              file=sys.stderr)
        return 2

    cls = _class_name_for(model_id)

    if args.template == "tutorial":
        raw = _read_tutorial_template(repo_root)
        files = {name: _render_tutorial(content, model_id=model_id, class_name=cls)
                  for name, content in raw.items()}
    else:  # simple
        files = {
            "__init__.py": SIMPLE_INIT.format(model_id=model_id),
            "model.py":    SIMPLE_MODEL.format(model_id=model_id, class_name=cls),
            "augment.py":  SIMPLE_AUGMENT.format(model_id=model_id),
        }

    for name, content in files.items():
        _write(target / name, content, args.force)
        print(f"  + {target.relative_to(repo_root)}/{name}")

    print()
    print(f"✓ scaffolded pipeline/models/{model_id}/ ({args.template} template)")
    print()
    print("Next:")
    print(f"  1. read pipeline/models/{model_id}/__init__.py, it points at the other files")
    print(f"  2. edit pipeline/models/{model_id}/model.py, your architecture")
    if args.template == "tutorial":
        print(f"  3. edit pipeline/models/{model_id}/train_loop.py CFG, hyperparameters")
    print("  4. add a line to runs.yaml:")
    print(f"       - {{ dataset: synthetic_smoke, model: {model_id}, config: fast }}")
    print(f"  5. make train RUN={model_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
