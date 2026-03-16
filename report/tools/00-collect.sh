#!/bin/sh
# 00-collect.sh <source-dir> <output-file-list> [paper.yaml]
# Writes an ordered list of .md source file paths to output-file-list.
#
# Ordering logic (first match wins):
#   1. build.content list in paper.yaml  — explicit order, supports include/exclude
#   2. Alphabetical sort of source-dir/*.md  — fallback when no list is specified
set -e

SRC_DIR="$1"
OUT_FILE="$2"
YAML="${3:-source/paper.yaml}"

if [ -z "$SRC_DIR" ] || [ -z "$OUT_FILE" ]; then
  echo "Usage: 00-collect.sh <source-dir> <output-file-list> [paper.yaml]" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_FILE")"

# Use the explicit content list from paper.yaml if present
if [ -f "$YAML" ] && python3 - "$YAML" <<'PYEOF' 2>/dev/null
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
sys.exit(0 if (d or {}).get("build", {}).get("content") else 1)
PYEOF
then
  python3 - "$YAML" "$OUT_FILE" <<'PYEOF'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
files = d["build"]["content"]
with open(sys.argv[2], "w") as f:
    f.write("\n".join(files) + "\n")
PYEOF
  echo "00-collect: using content list from $YAML"
else
  ls "$SRC_DIR"/*.md 2>/dev/null | sort > "$OUT_FILE"
  echo "00-collect: using alphabetical sort of $SRC_DIR"
fi

COUNT=$(wc -l < "$OUT_FILE" | tr -d ' ')
echo "00-collect: found $COUNT source file(s)"
cat "$OUT_FILE"
