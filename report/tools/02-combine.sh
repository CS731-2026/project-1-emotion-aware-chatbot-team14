#!/bin/sh
# 02-combine.sh <file-list.txt> <output-combined.pandoc>
# Reads an ordered list of .pandoc files and concatenates them into one,
# inserting a blank line between each section.
set -e

FILE_LIST="$1"
OUTPUT="$2"

if [ -z "$FILE_LIST" ] || [ -z "$OUTPUT" ]; then
  echo "Usage: 02-combine.sh <file-list.txt> <output-combined.pandoc>" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
> "$OUTPUT"

FIRST=1
while IFS= read -r MD_FILE; do
  PANDOC_FILE="builds/$(basename "$MD_FILE" .md).pandoc"
  if [ ! -f "$PANDOC_FILE" ]; then
    echo "02-combine: missing $PANDOC_FILE — run make from project root" >&2
    exit 1
  fi
  if [ "$FIRST" = "1" ]; then
    FIRST=0
  else
    printf '\n\n' >> "$OUTPUT"
  fi
  cat "$PANDOC_FILE" >> "$OUTPUT"
done < "$FILE_LIST"

echo "02-combine: wrote $OUTPUT"
