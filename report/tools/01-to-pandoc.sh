#!/bin/sh
# 01-to-pandoc.sh <input.md> <output.pandoc>
# Converts a single markdown file to pandoc-normalised markdown.
# The output is human-readable and inspectable when debugging render issues.
set -e

INPUT="$1"
OUTPUT="$2"

if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
  echo "Usage: 01-to-pandoc.sh <input.md> <output.pandoc>" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
pandoc "$INPUT" -f markdown -t markdown --wrap=none -o "$OUTPUT"
echo "01-to-pandoc: $INPUT → $OUTPUT"
