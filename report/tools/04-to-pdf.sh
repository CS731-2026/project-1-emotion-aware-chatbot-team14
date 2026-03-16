#!/bin/sh
# 04-to-pdf.sh <input.docx> <output.pdf>
# Converts a Word document to PDF using LibreOffice headless.
set -e

INPUT="$1"
OUTPUT="$2"

if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
  echo "Usage: 04-to-pdf.sh <input.docx> <output.pdf>" >&2
  exit 1
fi

OUT_DIR="$(dirname "$OUTPUT")"
mkdir -p "$OUT_DIR"

# Locate soffice — check common macOS path, then fall back to PATH
if [ -x "/Applications/LibreOffice.app/Contents/MacOS/soffice" ]; then
  SOFFICE="/Applications/LibreOffice.app/Contents/MacOS/soffice"
elif command -v soffice >/dev/null 2>&1; then
  SOFFICE="soffice"
else
  echo "04-to-pdf: LibreOffice not found. Install it or add soffice to PATH." >&2
  exit 1
fi

# LibreOffice places the PDF next to the input file, then we move it
"$SOFFICE" --headless --convert-to pdf "$INPUT" --outdir "$OUT_DIR"

# soffice names the output based on the input filename
GENERATED="$OUT_DIR/$(basename "$INPUT" .docx).pdf"
if [ "$GENERATED" != "$OUTPUT" ]; then
  mv "$GENERATED" "$OUTPUT"
fi

echo "04-to-pdf: wrote $OUTPUT"
