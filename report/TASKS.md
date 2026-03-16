# Tasks

## Up next
- [ ] Test LaTeX pipeline end-to-end (`make` with LaTeX installed)
- [ ] Improve IEEEtran author block formatting (custom pandoc LaTeX template with `\IEEEauthorblockN` / `\IEEEauthorblockA`)
- [ ] Wire up reference tracking tool (dynamic .bib processing)
- [ ] `make watch` via `fswatch` for rebuild on file save

## Done
- [x] Directory structure (source/, tools/, builds/, .tmp/)
- [x] .gitignore
- [x] Makefile (with install + incremental build)
- [x] tools/00-collect.sh
- [x] tools/01-to-pandoc.sh
- [x] tools/02-combine.sh
- [x] tools/03-to-docx.sh
- [x] tools/03b-merge.py
- [x] tools/04-to-pdf.sh (LibreOffice — kept for reference)
- [x] tools/00b-fill-template.py (DOCX: docxtpl placeholder injection)
- [x] tools/00c-yaml-to-pandoc-meta.py (LaTeX: paper.yaml → pandoc metadata)
- [x] tools/04-to-latex.sh (pandoc + IEEEtran → paper.tex)
- [x] tools/05-to-pdf-latex.sh (latexmk/pdflatex → paper.pdf)
- [x] source/paper.yaml (metadata for both pipelines)
- [x] Makefile restructured: LaTeX→PDF primary, DOCX optional (`make docx`)
- [x] End-to-end test (DOCX pipeline) — PDF produced with citations [1] [2]
