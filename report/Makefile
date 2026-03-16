# ── Config (generated from source/paper.yaml build: section) ────────────────
# -include loads config.mk if it exists; Make rebuilds it automatically when
# paper.yaml changes and re-reads the Makefile with the new values.
-include .tmp/config.mk

# Defaults — used on the very first run before .tmp/config.mk exists,
# or for any field not set in the build: section of paper.yaml.
TEMPLATE       ?= tools/templates/default.docx
CSL            ?= tools/default.csl
BIB            ?= source/references.bib
DOCX_METHOD    ?= docxcompose
LATEX_TEMPLATE ?=
TEMPLATES_DIR  := tools/templates

# ── Source files ─────────────────────────────────────────────────────────────
SOURCE_FILES := $(wildcard source/*.md)
PANDOC_FILES := $(patsubst source/%.md,builds/%.pandoc,$(SOURCE_FILES))

all: builds/paper.pdf

# ── Primary pipeline: Markdown → LaTeX → PDF ──────────────────────────────

# Step 5: compile LaTeX → PDF
builds/paper.pdf: builds/paper.tex
	tools/05-to-pdf-latex.sh builds/paper.tex builds/paper.pdf

# Step 4: combined.pandoc + metadata → paper.tex (IEEEtran)
builds/paper.tex: builds/combined.pandoc .tmp/pandoc-meta.yaml $(BIB) .tmp/config.mk \
                  $(if $(LATEX_TEMPLATE),$(LATEX_TEMPLATE),)
	tools/04-to-latex.sh builds/combined.pandoc .tmp/pandoc-meta.yaml $(BIB) $(CSL) \
	    builds/paper.tex $(LATEX_TEMPLATE)

# Step 0d: convert paper.yaml → pandoc-compatible metadata YAML
.tmp/pandoc-meta.yaml: source/paper.yaml
	@mkdir -p .tmp
	python3 tools/00c-yaml-to-pandoc-meta.py source/paper.yaml .tmp/pandoc-meta.yaml

# ── Optional: DOCX pipeline ────────────────────────────────────────────────

docx: builds/paper.docx

# Step 3b: merge filled template (front matter) + pandoc body → paper.docx
builds/paper.docx: .tmp/template-ready.docx .tmp/body.docx
	python3 tools/03b-merge.py .tmp/template-ready.docx .tmp/body.docx builds/paper.docx

# Step 3: combined.pandoc → body.docx (body sections only, styled from template)
.tmp/body.docx: builds/combined.pandoc .tmp/template-ready.docx $(BIB) .tmp/config.mk
	tools/03-to-docx.sh builds/combined.pandoc .tmp/template-ready.docx $(BIB) $(CSL) .tmp/body.docx

# Step 0c: fill all {{placeholders}} in template from paper.yaml → template-ready.docx
.tmp/template-ready.docx: $(TEMPLATE) source/paper.yaml
	@mkdir -p .tmp
	python3 tools/00b-fill-template.py source/paper.yaml $(TEMPLATE) .tmp/template-ready.docx

# ── Shared steps ────────────────────────────────────────────────────────────

# Step 2: merge body .pandoc files → combined.pandoc
builds/combined.pandoc: .tmp/file-list.txt $(PANDOC_FILES)
	tools/02-combine.sh .tmp/file-list.txt builds/combined.pandoc

# Step 1: convert each source .md → builds/*.pandoc
builds/%.pandoc: source/%.md
	@mkdir -p builds
	tools/01-to-pandoc.sh $< $@

# Step 0b: extract build: config from paper.yaml → Makefile variables
.tmp/config.mk: source/paper.yaml
	@mkdir -p .tmp
	python3 tools/config.py source/paper.yaml .tmp/config.mk

# Step 0: collect + order source files into a manifest
# Depends on paper.yaml so the list rebuilds when content: order changes
.tmp/file-list.txt: $(wildcard source/*.md) source/paper.yaml
	@mkdir -p .tmp
	tools/00-collect.sh source/ .tmp/file-list.txt source/paper.yaml

# ── Convenience targets ─────────────────────────────────────────────────────
.PHONY: all docx pdf pdf-with use-template list-templates vibe-template vibe-template-from-docx clean deps install

pdf:     builds/paper.pdf
install: deps   # alias

# Build PDF with a specific template without changing paper.yaml.
# Accepts a bare name (ieeetran) or a full path (tools/templates/ieeetran.latex).
# Drops paper.tex and paper.pdf so Make always regenerates with the chosen template.
# Usage: make pdf-with TPL=ieeetran
pdf-with:
	@if [ -z "$(TPL)" ]; then \
		echo "Usage: make pdf-with TPL=<name>" >&2; \
		echo "" >&2; \
		$(MAKE) --no-print-directory list-templates >&2; \
		exit 1; \
	fi
	@RESOLVED="$(TPL)"; \
	if ! echo "$$RESOLVED" | grep -q '/'; then \
		RESOLVED="$(TEMPLATES_DIR)/$$RESOLVED.latex"; \
	fi; \
	if [ ! -f "$$RESOLVED" ]; then \
		echo "Error: template not found: $$RESOLVED" >&2; \
		$(MAKE) --no-print-directory list-templates >&2; \
		exit 1; \
	fi; \
	rm -f builds/paper.tex builds/paper.pdf; \
	$(MAKE) pdf LATEX_TEMPLATE="$$RESOLVED"

# List all .latex template files available in tools/templates/.
list-templates:
	@echo "Available templates:"
	@ls $(TEMPLATES_DIR)/*.latex 2>/dev/null \
	  | sed 's|.*/||; s|\.latex$$||; s|^|  |' \
	  || echo "  (none yet — run make vibe-template to create one)"

# Set the active LaTeX template in source/paper.yaml.
# Usage: make use-template NAME=ieeetran
use-template:
	@if [ -z "$(NAME)" ]; then \
		echo "Usage: make use-template NAME=<name>" >&2; \
		echo "" >&2; \
		$(MAKE) --no-print-directory list-templates >&2; \
		exit 1; \
	fi
	@RESOLVED="$(TEMPLATES_DIR)/$(NAME).latex"; \
	if [ ! -f "$$RESOLVED" ]; then \
		echo "Error: template not found: $$RESOLVED" >&2; \
		$(MAKE) --no-print-directory list-templates >&2; \
		exit 1; \
	fi
	@python3 tools/use-template.py source/paper.yaml "$(TEMPLATES_DIR)/$(NAME).latex"

# Generate a new pandoc LaTeX template from an image or PDF using Claude Code.
# PDFs are converted to PNG images first — visual layout is easier to reason about than raw PDF.
# Requires: poppler (brew install poppler) for PDF→PNG conversion.
# Usage: make vibe-template FILE=tools/templates/template.pdf NAME=conference-name
#        make vibe-template FILE=tools/templates/photo.png   NAME=conference-name
# Output: tools/templates/<NAME>.latex  (then use with: make pdf-with TPL=tools/templates/<NAME>.latex)
vibe-template:
	@if [ -z "$(FILE)" ] || [ -z "$(NAME)" ]; then \
		echo "Usage: make vibe-template FILE=tools/templates/template.pdf NAME=conference-name" >&2; \
		echo "       FILE can be a PDF, PNG, or JPG" >&2; \
		exit 1; \
	fi
	@if [ ! -f "$(FILE)" ]; then \
		echo "Error: file not found: $(FILE)" >&2; \
		exit 1; \
	fi
	@mkdir -p .tmp; \
	case "$(FILE)" in \
	  *.pdf|*.PDF) \
	    echo "Converting PDF to PNG images (first 2 pages) for visual analysis..."; \
	    if ! command -v pdftoppm >/dev/null 2>&1; then \
	      echo "Error: pdftoppm not found. Install with: brew install poppler" >&2; exit 1; \
	    fi; \
	    pdftoppm -r 200 -png -f 1 -l 2 "$(FILE)" ".tmp/$(NAME)-page"; \
	    IMAGES="$$(ls .tmp/$(NAME)-page*.png 2>/dev/null | sort | tr '\n' ' ')"; \
	    ;; \
	  *) \
	    IMAGES="$(FILE)"; \
	    ;; \
	esac; \
	echo "Generating tools/templates/$(NAME).latex from $$IMAGES ..."; \
	PROMPT="Read the image(s) at $$IMAGES. They show pages from a conference paper template. STEP 1 — OBSERVE AND DESCRIBE (write this out explicitly before writing any code). For each image state: (a) Column count — is the body text single-column or two-column? Look carefully: two-column means text flows in two side-by-side columns on the same page; single-column means text spans the full page width. (b) Page margins — estimate top, bottom, left, right. (c) Font family — serif (e.g. Times) or sans-serif (e.g. Arial)? (d) Title — size, weight, alignment. (e) Author block — how are names and affiliations shown? superscript numbers? separate affiliation lines? (f) Abstract — full-width or indented? heading style? (g) Section headings — exactly how are they formatted, including numbering scheme (e.g. '1. INTRODUCTION' with dot, or '1 INTRODUCTION' without, or no number) and case. (h) Paragraph formatting — first-line indent? extra spacing between paragraphs? (i) Header and footer — any running header with conference name? page number position? STEP 2 — GENERATE a pandoc LaTeX template at tools/templates/$(NAME).latex that faithfully implements what you described in Step 1. Model the file on tools/templates/ieeetran.latex — same citeproc compatibility macros, same pandoc variable syntax. Your LaTeX documentclass and geometry MUST match the column count and margins you stated. KNOWN GOTCHAS — apply unconditionally: (1) No authblk package — implement author blocks manually. (2) No Helvetica/Arial fonts (helvet/phv) — use lmodern; for sans-serif body text add renewcommand familydefault sfdefault. (3) No newunicodechar package — use DeclareUnicodeCharacter; always add DeclareUnicodeCharacter{2212}{-} for the Unicode minus sign. (4) If two-column, suppress longtable error with makeatletter def LT@err#1#2 PackageWarning longtable #1 makeatother — the #1#2 slots are critical, omitting them prints error text into the PDF. (5) Only use TeX Live Basic packages: amsmath, amssymb, xcolor, iftex, fontenc, inputenc, lmodern, microtype, longtable, booktabs, array, calc, etoolbox, graphicx, bookmark, xurl, hyperref, fancyhdr. After writing the file run: make pdf-with TPL=tools/templates/$(NAME).latex and fix any LaTeX errors until the PDF compiles cleanly."; \
	echo "$$PROMPT" | tools/stream-claude.sh --allowedTools "Read,Write,Edit,Bash"

# Generate a template from a Word .docx file by converting it to PDF first.
# PDF preserves exact layout (vector fonts, precise margins) and Claude reads it natively.
# Tries docx2pdf (uses Word on macOS, pip install docx2pdf) then libreoffice as fallback.
# Usage: make vibe-template-from-docx DOCX=tools/templates/conference.docx NAME=conference-name
vibe-template-from-docx:
	@if [ -z "$(DOCX)" ] || [ -z "$(NAME)" ]; then \
		echo "Usage: make vibe-template-from-docx DOCX=tools/templates/file.docx NAME=conference-name" >&2; \
		echo "       e.g.: make vibe-template-from-docx DOCX=tools/templates/nzcsrsc.docx NAME=nzcsrsc" >&2; \
		exit 1; \
	fi
	@if [ ! -f "$(DOCX)" ]; then \
		echo "Error: file not found: $(DOCX)" >&2; \
		exit 1; \
	fi
	@mkdir -p .tmp
	@echo "Converting $(DOCX) to PDF..."
	@if command -v docx2pdf >/dev/null 2>&1; then \
		docx2pdf "$(DOCX)" ".tmp/$(NAME).pdf"; \
	elif command -v libreoffice >/dev/null 2>&1; then \
		libreoffice --headless --convert-to pdf --outdir .tmp "$(DOCX)"; \
		mv ".tmp/$$(basename $(DOCX) .docx).pdf" ".tmp/$(NAME).pdf" 2>/dev/null || true; \
	else \
		echo "Error: no DOCX→PDF converter found." >&2; \
		echo "  Install one of:" >&2; \
		echo "    pip install docx2pdf        (uses Microsoft Word on macOS)" >&2; \
		echo "    brew install --cask libreoffice" >&2; \
		exit 1; \
	fi
	@echo "Converted to .tmp/$(NAME).pdf — launching Claude Code to generate template..."
	$(MAKE) vibe-template FILE=.tmp/$(NAME).pdf NAME=$(NAME)

clean:
	rm -rf builds/ .tmp/

# Install / verify all dependencies
deps:
	@echo "=== paper-writer dependencies ==="
	@echo ""
	@echo "--- pandoc ---"
	@if command -v pandoc >/dev/null 2>&1; then \
		echo "  ok: $$(pandoc --version | head -1)"; \
	else \
		echo "  installing pandoc via Homebrew..."; \
		brew install pandoc; \
	fi
	@echo ""
	@echo "--- LaTeX (BasicTeX + latexmk) ---"
	@if command -v pdflatex >/dev/null 2>&1 || [ -x "/Library/TeX/texbin/pdflatex" ]; then \
		echo "  ok: pdflatex found"; \
	else \
		echo "  installing BasicTeX via Homebrew (~100 MB)..."; \
		brew install --cask basictex; \
	fi
	@TLMGR=""; \
	if command -v tlmgr >/dev/null 2>&1; then TLMGR="tlmgr"; \
	elif [ -x "/Library/TeX/texbin/tlmgr" ]; then TLMGR="/Library/TeX/texbin/tlmgr"; \
	fi; \
	if [ -n "$$TLMGR" ]; then \
		if ! command -v latexmk >/dev/null 2>&1 && ! [ -x "/Library/TeX/texbin/latexmk" ]; then \
			echo "  installing latexmk..."; \
			sudo $$TLMGR install latexmk; \
		else \
			echo "  ok: latexmk found"; \
		fi; \
	else \
		echo "  WARNING: tlmgr not found — open a new terminal and re-run 'make deps'"; \
	fi
	@echo ""
	@echo "--- IEEEtran LaTeX class ---"
	@mkdir -p tools/latex
	@if [ -f tools/latex/IEEEtran.cls ]; then \
		echo "  ok: tools/latex/IEEEtran.cls"; \
	else \
		echo "  downloading IEEEtran.cls from CTAN..."; \
		curl -fsSL \
		  "https://mirrors.ctan.org/macros/latex/contrib/IEEEtran/IEEEtran.cls" \
		  -o tools/latex/IEEEtran.cls; \
		echo "  downloaded tools/latex/IEEEtran.cls"; \
	fi
	@echo ""
	@echo "--- Default Word template (for optional make docx) ---"
	@mkdir -p tools/templates
	@if [ -f tools/templates/default.docx ]; then \
		echo "  ok: tools/templates/default.docx"; \
	else \
		pandoc --print-default-data-file reference.docx > tools/templates/default.docx; \
		echo "  created tools/templates/default.docx"; \
	fi
	@echo ""
	@echo "--- IEEE CSL citation style ---"
	@if [ -f tools/default.csl ]; then \
		echo "  ok: tools/default.csl"; \
	else \
		curl -fsSL \
		  "https://raw.githubusercontent.com/citation-style-language/styles/master/ieee.csl" \
		  -o tools/default.csl; \
		echo "  downloaded tools/default.csl (IEEE style)"; \
	fi
	@echo ""
	@echo "--- Python packages (docxtpl, docxcompose, pyyaml, docx2pdf) ---"
	@pip3 install --quiet docxtpl docxcompose pyyaml docx2pdf
	@echo "  ok: Python packages ready"
	@echo ""
	@echo "--- docx2pdf (DOCX→PDF for vibe-template-from-docx) ---"
	@if command -v docx2pdf >/dev/null 2>&1; then \
		echo "  ok: docx2pdf found (uses Microsoft Word on macOS)"; \
	elif command -v libreoffice >/dev/null 2>&1; then \
		echo "  ok: libreoffice found (fallback DOCX→PDF converter)"; \
	else \
		echo "  WARNING: neither docx2pdf nor libreoffice found."; \
		echo "  'make vibe-template-from-docx' will fail without one."; \
		echo "  Fix: pip3 install docx2pdf  (needs Microsoft Word installed)"; \
		echo "    or: brew install --cask libreoffice"; \
	fi
	@echo ""
	@echo "--- poppler/pdftoppm (PDF→PNG for vibe-template) ---"
	@if command -v pdftoppm >/dev/null 2>&1; then \
		echo "  ok: pdftoppm found ($$(pdftoppm -v 2>&1 | head -1))"; \
	else \
		echo "  installing poppler via Homebrew..."; \
		brew install poppler; \
	fi
	@echo ""
	@echo "=== All done. Run 'make' to build. ==="
