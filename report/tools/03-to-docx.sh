#!/bin/sh
# 03-to-docx.sh <combined.pandoc> <template.docx> <refs.bib> <style.csl> <output.docx>
# Converts the combined pandoc intermediate to a Word document.
set -e

INPUT="$1"
TEMPLATE="$2"
BIB="$3"
CSL="$4"
OUTPUT="$5"

if [ -z "$INPUT" ] || [ -z "$TEMPLATE" ] || [ -z "$BIB" ] || [ -z "$CSL" ] || [ -z "$OUTPUT" ]; then
  echo "Usage: 03-to-docx.sh <combined.pandoc> <template.docx> <refs.bib> <style.csl> <output.docx>" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
pandoc "$INPUT" \
  --from markdown \
  --to docx \
  --reference-doc="$TEMPLATE" \
  --bibliography="$BIB" \
  --csl="$CSL" \
  --citeproc \
  -o "$OUTPUT"

echo "03-to-docx: wrote $OUTPUT"
