"""Copy a trained checkpoint from a pipeline run dir into the
model_service's models/ tree and register it in models.yaml.

Called by `make deploy-model`. Usage:

    python -m pipeline.cli.deploy_model \
        --run output/run/fer2013__empathbot_final__thorough__20260524-... \
        --id  empathbot_final \
      [ --checkpoint best.pth ]          # which checkpoint inside the run dir
      [ --variant   empathbot ]          # override the inferred variant

The pipeline model name (parsed from the run dir slug) maps to a service
variant:
    empathbot_{v1,v3,resnet18,final}  → empathbot
    resnet18                           → resnet18
    everything else                    → require --variant
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml


# pipeline-model-name → model_service variant
VARIANT_MAP = {
    "empathbot_v1":         "empathbot",
    "empathbot_v3":         "empathbot",
    "empathbot_resnet18":   "empathbot",
    "empathbot_final":      "empathbot",
    "resnet18":             "resnet18",
}


def infer_pipeline_model(run_dir: Path) -> str | None:
    """The pipeline writes runs as `<dataset>__<model>__<config>__<ts>/`."""
    parts = run_dir.name.split("__")
    return parts[1] if len(parts) >= 3 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run dir under output/run/")
    ap.add_argument("--id", required=True,
                    help="id to register under in models.yaml "
                         "(consumed at runtime via EMOTION_MODEL_ID)")
    ap.add_argument("--checkpoint", default="best.pth",
                    help="checkpoint filename under <run>/checkpoints/")
    ap.add_argument("--variant", default=None,
                    help="override inferred variant (one of placeholder|resnet18|empathbot)")
    ap.add_argument("--repo-root", default=".", help="repo root (default cwd)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing models/<id>/<checkpoint> instead "
                         "of refusing. Without this flag, deploy refuses to "
                         "clobber a checkpoint that already exists on disk.")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = (repo_root / run_dir).resolve()
    if not run_dir.is_dir():
        print(f"✗ run dir not found: {run_dir}", file=sys.stderr)
        return 2

    src_ckpt = run_dir / "checkpoints" / args.checkpoint
    if not src_ckpt.exists():
        print(f"✗ checkpoint not found: {src_ckpt}", file=sys.stderr)
        return 2

    variant = args.variant
    if variant is None:
        pipeline_model = infer_pipeline_model(run_dir)
        variant = VARIANT_MAP.get(pipeline_model or "")
        if variant is None:
            print(
                f"✗ can't infer service variant from run dir model "
                f"'{pipeline_model}'. Pass --variant explicitly. "
                f"Known mappings: {sorted(VARIANT_MAP)}",
                file=sys.stderr,
            )
            return 2

    # Stage into models/<id>/ at the repo root (gitignored).
    models_root = repo_root / "models"
    dest_dir = models_root / args.id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_ckpt = dest_dir / args.checkpoint
    if dest_ckpt.exists() and not args.force:
        print(
            f"✗ refusing to overwrite existing checkpoint:\n"
            f"    {dest_ckpt.relative_to(repo_root)}\n"
            f"  Use a different --id (e.g. --id {args.id}_v2), or pass "
            f"--force to overwrite.",
            file=sys.stderr,
        )
        return 2
    shutil.copy2(src_ckpt, dest_ckpt)
    print(f"✓ copied {src_ckpt.relative_to(repo_root)} → "
          f"{dest_ckpt.relative_to(repo_root)}")

    # Path stored in models.yaml is relative to repo root.
    rel_path = dest_ckpt.relative_to(repo_root)
    registry_path = repo_root / "application" / "model_service" / "models.yaml"
    with registry_path.open() as f:
        data = yaml.safe_load(f) or {}
    models = data.setdefault("models", {})
    models[args.id] = {"path": str(rel_path), "variant": variant}
    with registry_path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"✓ registered '{args.id}' (variant={variant}) in "
          f"{registry_path.relative_to(repo_root)}")

    print()
    print("Use it:")
    print(f"  EMOTION_MODEL_ID={args.id} make dev")
    print(f"  # or add to .env:  EMOTION_MODEL_ID={args.id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
