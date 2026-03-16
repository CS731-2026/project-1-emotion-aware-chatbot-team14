# paper-writer

A pipeline for writing academic papers in Markdown and producing a properly formatted PDF via LaTeX (IEEEtran). DOCX output is also available as an optional target for draft sharing.

## Commands

| Command | What it does |
|---|---|
| `make deps` | Install all dependencies (pandoc, BasicTeX, latexmk, IEEEtran.cls, CSL, Python packages) |
| `make` | Build `builds/paper.pdf` using the default or configured LaTeX template |
| `make pdf` | Same as `make` |
| `make docx` | Build `builds/paper.docx` using the Word template pipeline |
| `make clean` | Delete `builds/` and `.tmp/` |
| `make list-templates` | List all `.latex` template files in `tools/templates/` |
| `make pdf-with TPL=<path>` | Build PDF with a specific LaTeX template without changing `paper.yaml` |
| `make vibe-template FILE=<pdf/png> NAME=<name>` | Generate a new pandoc `.latex` template from an image or PDF using Claude |
| `make vibe-template-from-docx DOCX=<path> NAME=<name>` | Convert a `.docx` to PDF first, then run `vibe-template` |

### Examples

```sh
make deps
make
make pdf-with TPL=tools/templates/ieeetran.latex
make pdf-with TPL=tools/templates/eswc2019.latex
make vibe-template FILE=tools/templates/myconf.pdf NAME=myconf
make vibe-template-from-docx DOCX=tools/templates/myconf.docx NAME=myconf
make list-templates
make clean
```

## Primary pipeline

```
source/*.md
   ↓  01-to-pandoc.sh
builds/*.pandoc          ← inspectable intermediates
   ↓  02-combine.sh
builds/combined.pandoc
   ↓  04-to-latex.sh  (pandoc + IEEEtran + citeproc)
builds/paper.tex
   ↓  05-to-pdf-latex.sh  (latexmk / pdflatex)
builds/paper.pdf
```

Citations are written as `[@citekey]` in Markdown. Pandoc processes them via `--citeproc` using IEEE style, so reference numbers update automatically.

## Optional DOCX pipeline

```
make docx   →   builds/paper.docx
```

Uses a Word template (`tools/templates/default.docx`) styled via pandoc `--reference-doc`. Metadata (title, authors, abstract) from `source/paper.yaml` is injected into the template via [docxtpl](https://docxtpl.readthedocs.io) before the merge step.

## Project structure

```
source/          ← your markdown files, references.bib, paper.yaml, figures/
tools/           ← build scripts, templates, CSL styles, latex/
builds/          ← generated output (gitignored)
.tmp/            ← dynamic intermediates (gitignored)
Makefile         ← orchestrates the pipeline
```

## Configuration

All build settings live in `source/paper.yaml` under the `build:` section — no Makefile edits needed.

```yaml
build:
  docx_template: tools/templates/default.docx  # swap templates here
  docx_method: docxcompose
  csl: tools/default.csl
  bib: source/references.bib
  content:                        # explicit section order (omit for alphabetical)
    - source/01-introduction.md
    - source/02-methods.md
    - source/03-related-work.md
```

`tools/config.py` reads this section and writes `.tmp/config.mk`, which the Makefile includes automatically. If a field is absent the Makefile's built-in defaults apply.

### Switching LaTeX templates (PDF)

The `latex_template` field in `source/paper.yaml` selects the pandoc `.latex` template used when building the PDF.

```yaml
build:
  latex_template: tools/templates/ieeetran.latex  # IEEEtran two-column format (default)
  # latex_template: tools/templates/myconf.latex  # any other template in tools/templates/
```

- **Omit the field** (or comment it out) to use pandoc's built-in default template.
- **`tools/templates/ieeetran.latex`** — hand-crafted IEEEtran template with proper `\IEEEauthorblockN` / `\IEEEauthorblockA` author blocks. Uses the author fields (`name`, `dept`, `org`, `city`, `email`) from `source/paper.yaml`.

**Permanent switch** — edit `latex_template` in `source/paper.yaml`, then run `make`.

**One-off switch** — use `make pdf-with` without touching `paper.yaml`:

```sh
make pdf-with TPL=tools/templates/ieeetran.latex
make pdf-with TPL=tools/templates/myconf.latex
```

**Generating a new template from an image/PDF** — use `make vibe-template`:

```sh
make vibe-template FILE=tools/templates/confstyle.pdf NAME=myconf
# → generates tools/templates/myconf.latex
```

Then test it with `make pdf-with TPL=tools/templates/myconf.latex` before setting it permanently.

### Switching templates (DOCX)

1. Add your `.docx` template to `tools/templates/`
2. Update `docx_template` in `source/paper.yaml`
3. Run `make docx` — the new template is picked up automatically

### DOCX template method

`docx_method` controls how the template is applied:

| Value | Behaviour |
|---|---|
| `docxcompose` | Fills `{{placeholders}}` via docxtpl, appends pandoc body via docxcompose. **(current)** |
| `custom` | Reserved — placeholder for a future custom DOCX generation approach. |

## Paper metadata

Edit `source/paper.yaml` with your paper details:

```yaml
title: "Your Paper Title"
authors:
  - name: "First Last"
    dept: "Dept. of Computer Science"
    org: "University of Auckland"
    city: "Auckland, New Zealand"
    email: "user@auckland.ac.nz"
abstract: >
  Your abstract here.
keywords: "keyword1, keyword2"
```

This YAML feeds both pipelines:
- **LaTeX**: `tools/00c-yaml-to-pandoc-meta.py` converts it to pandoc metadata (`title`, `author`, `abstract`)
- **DOCX**: `tools/00b-fill-template.py` fills `{{placeholder}}` tags in the Word template

## Figures

Place image files in `source/figures/`. Reference them in Markdown as:

```markdown
![Caption text.](source/figures/my-figure.png)
```

Supported formats: PNG, JPEG, PDF. The LaTeX pipeline wraps the image in a `figure` environment automatically.

## LaTeX dependencies

`make deps` handles everything automatically:
- Installs **BasicTeX** via Homebrew if no LaTeX is found
- Installs **latexmk** via `tlmgr` if missing
- Downloads **IEEEtran.cls** from CTAN to `tools/latex/` (no sudo needed)

To install manually if needed:
```sh
brew install --cask basictex
sudo /Library/TeX/texbin/tlmgr install latexmk
# IEEEtran.cls is downloaded automatically to tools/latex/ by make deps
```

---

## Configuring the DOCX template

See below only if you use `make docx`.

Pandoc copies **named styles** from the reference doc — not content. Open your `.docx` in Word and ensure these paragraph styles exist:

| Style name | Used for |
|---|---|
| `Normal` | Default body text |
| `Body Text` | Body paragraphs |
| `Heading 1` | `#` section headings |
| `Heading 2` | `##` subsection headings |
| `Heading 3` | `###` sub-subsection headings |
| `Bibliography` | Reference list entries |
| `Caption` | Figure/table captions |
| `Verbatim` | Fenced code blocks |

Character styles: `Verbatim Char`, `Footnote Reference`.

### Injecting metadata into the DOCX template

Place `{{placeholder}}` tags in your Word template; `source/paper.yaml` provides the values:

| Placeholder | Source |
|---|---|
| `{{title}}` | `title` |
| `{{abstract}}` | `abstract` |
| `{{keywords}}` | `keywords` |
| `{{author_1_name}}` | `authors[0].name` |
| `{{author_1_dept}}` | `authors[0].dept` |
| `{{author_1_org}}` | `authors[0].org` |
| `{{author_1_city}}` | `authors[0].city` |
| `{{author_1_email}}` | `authors[0].email` |
| *(repeat for author_2 … author_6)* | |
