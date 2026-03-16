#!/bin/sh
# 05-to-pdf-latex.sh <input.tex> <output.pdf>
# Compiles a LaTeX file to PDF.
# Prefers latexmk (handles multi-pass compilation automatically).
# Falls back to two pdflatex passes for cross-reference resolution.
#
# Checks both $PATH and /Library/TeX/texbin (BasicTeX default on macOS).
set -e

INPUT="$1"
OUTPUT="$2"

if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
  echo "Usage: 05-to-pdf-latex.sh <input.tex> <output.pdf>" >&2
  exit 1
fi

OUT_DIR="$(dirname "$OUTPUT")"
TEX_DIR="$(dirname "$INPUT")"
STEM="$(basename "$INPUT" .tex)"
mkdir -p "$OUT_DIR"

# Ensure BasicTeX binaries are on PATH so that latexmk can spawn pdflatex.
# /Library/TeX/texbin is the default install location on macOS but is not
# added to $PATH automatically until the user opens a new login shell.
[ -d "/Library/TeX/texbin" ] && export PATH="/Library/TeX/texbin:$PATH"

# Add project-local LaTeX classes (e.g. IEEEtran.cls downloaded by make deps).
# TEXINPUTS trailing : preserves the default search paths.
TOOLS_LATEX="$(cd "$(dirname "$0")/.." && pwd)/tools/latex"
[ -d "$TOOLS_LATEX" ] && export TEXINPUTS="$TOOLS_LATEX//:${TEXINPUTS:-}"

LATEXMK="$(command -v latexmk 2>/dev/null || true)"
PDFLATEX="$(command -v pdflatex 2>/dev/null || true)"

if [ -n "$LATEXMK" ]; then
  # -g: force processing even if latexmk thinks targets are up-to-date.
  # -f: force completion even if errors occur (e.g., longtable in twocolumn mode)
  # Make decides when to call this script; latexmk's own cache check is redundant.
  "$LATEXMK" -g -f -pdf -interaction=nonstopmode -outdir="$TEX_DIR" "$INPUT"
elif [ -n "$PDFLATEX" ]; then
  # Two passes so cross-references and citations resolve correctly
  "$PDFLATEX" -interaction=nonstopmode -output-directory="$TEX_DIR" "$INPUT"
  "$PDFLATEX" -interaction=nonstopmode -output-directory="$TEX_DIR" "$INPUT"
else
  echo "05-to-pdf-latex: neither latexmk nor pdflatex found." >&2
  echo "  Run: make deps" >&2
  exit 1
fi

# Move generated PDF to the expected output path if they differ
GENERATED="$TEX_DIR/$STEM.pdf"
if [ "$GENERATED" != "$OUTPUT" ]; then
  mv "$GENERATED" "$OUTPUT"
fi

echo "05-to-pdf-latex: wrote $OUTPUT"
