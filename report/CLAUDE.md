# paper-writer — Claude Instructions

## Project purpose
Pipeline: Markdown files → LaTeX → PDF (primary), with optional Word .docx output.
Uses IEEEtran document class for conference paper formatting.
Automatic reference numbering via BibTeX + CSL (write `[@citekey]`, numbers auto-assigned).

## Primary pipeline
```
*.md → *.pandoc (intermediate, kept for debugging)
     → combined.pandoc
     → paper.tex  (pandoc + IEEEtran + citeproc)
     → paper.pdf  (latexmk / pdflatex)
```

## Optional DOCX pipeline
```
make docx → paper.docx  (pandoc + reference-doc + docxcompose merge)
```
DOCX is secondary — used for draft sharing with reviewers who need Word.

## Directory structure
```
source/          ← writing only: markdown files, references.bib, paper.yaml
tools/           ← numbered build scripts + templates/ + CSL files
builds/          ← gitignored, all generated output
.tmp/            ← gitignored, dynamically generated intermediates
Makefile         ← orchestrates the pipeline
```

## Tool numbering
```
00-collect.sh              → .tmp/file-list.txt (ordered manifest of source .md files)
00b-fill-template.py       → .tmp/template-ready.docx (DOCX: fills {{tags}} in Word template)
00c-yaml-to-pandoc-meta.py → .tmp/pandoc-meta.yaml (LaTeX: converts paper.yaml for pandoc)
01-to-pandoc.sh            → builds/*.pandoc (per-file md → pandoc intermediate)
02-combine.sh              → builds/combined.pandoc (merge ordered pandoc files)
03-to-docx.sh              → .tmp/body.docx (DOCX pipeline: pandoc → body.docx)
03b-merge.py               → builds/paper.docx (DOCX pipeline: merge front + body)
04-to-latex.sh             → builds/paper.tex (LaTeX pipeline: pandoc + IEEEtran)
05-to-pdf-latex.sh         → builds/paper.pdf (LaTeX pipeline: latexmk/pdflatex)
```

## Conventions
- Makefile holds static config (TEMPLATE, CSL, BIB paths at the top)
- Dynamic intermediates go to `.tmp/`; Makefile declares them as dependencies
- Each tool takes inputs/output as CLI args — no implicit path knowledge inside scripts
- Script language not locked in — choose per script based on simplest fit
- Citations: `[@author-year]` pandoc syntax; `--citeproc` processes them at build time

## Key decisions
- LaTeX (IEEEtran) is primary — this is what the IEEE community uses for conference submissions
- DOCX pipeline kept as optional `make docx` for sharing drafts in Word format
- pandoc `--reference-doc` ignores body content (only styles/page setup/headers) — that's why DOCX needs the docxcompose merge approach for front matter
- paper.yaml is the single source of truth for metadata; 00c converts it for LaTeX, 00b for DOCX
