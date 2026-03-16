#!/bin/sh
# 04-to-latex.sh <combined.pandoc> <meta.yaml> <refs.bib> <style.csl> <output.tex> [template.latex]
# Converts the combined pandoc intermediate to LaTeX (IEEEtran conference format).
# Citations are processed by pandoc --citeproc so the .tex file is self-contained
# (no separate BibTeX compilation step needed).
#
# Optional 6th argument: path to a pandoc .latex template file.
# If omitted, pandoc's built-in default template is used.
# Set via latex_template: in paper.yaml, or pass LATEX_TEMPLATE= on the make command line.
set -e

INPUT="$1"
META="$2"
BIB="$3"
CSL="$4"
OUTPUT="$5"
LATEX_TEMPLATE="$6"

if [ -z "$INPUT" ] || [ -z "$META" ] || [ -z "$BIB" ] || [ -z "$CSL" ] || [ -z "$OUTPUT" ]; then
  echo "Usage: 04-to-latex.sh <combined.pandoc> <meta.yaml> <refs.bib> <style.csl> <output.tex> [template.latex]" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"

if [ -n "$LATEX_TEMPLATE" ]; then
  pandoc "$INPUT" \
    --from markdown \
    --to latex \
    --standalone \
    --metadata-file="$META" \
    --template="$LATEX_TEMPLATE" \
    -V documentclass=IEEEtran \
    -V classoption=conference \
    --bibliography="$BIB" \
    --csl="$CSL" \
    --citeproc \
    -o "$OUTPUT"
else
  pandoc "$INPUT" \
    --from markdown \
    --to latex \
    --standalone \
    --metadata-file="$META" \
    -V documentclass=IEEEtran \
    -V classoption=conference \
    --bibliography="$BIB" \
    --csl="$CSL" \
    --citeproc \
    -o "$OUTPUT"
fi

echo "04-to-latex: wrote $OUTPUT"
