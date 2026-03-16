#!/usr/bin/env python3
# 00b-fill-template.py <paper.yaml> <template.docx> <output.docx>
# Fills {{placeholders}} in a Word template using values from paper.yaml.
# Uses docxtpl (https://docxtpl.readthedocs.io)
#
# NOTE: pandoc's --reference-doc ignores template body content — it only takes
# styles, page setup, and headers/footers. This script is therefore useful for:
#   - Filling {{placeholders}} in headers/footers (journal name, paper title, etc.)
#   - Filling {{author_N_*}} in the author table (complex layout that can't come from markdown)
#
# Title, abstract, and keywords are injected as body content via
# 00b-generate-frontmatter.py instead.

import sys
import yaml
from docxtpl import DocxTemplate

def main():
    if len(sys.argv) != 4:
        print("Usage: 00b-fill-template.py <paper.yaml> <template.docx> <output.docx>", file=sys.stderr)
        sys.exit(1)

    yaml_path, template_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    # Flatten authors into individually named keys for simple templates
    # e.g. {{author_1_name}}, {{author_2_email}}, etc.
    # Also keep the full list available for loop-based templates: {{authors}}
    authors = meta.get("authors", [])
    for i, author in enumerate(authors, start=1):
        for field, value in author.items():
            meta[f"author_{i}_{field}"] = value or ""

    doc = DocxTemplate(template_path)
    doc.render(meta)
    doc.save(output_path)

    print(f"00b-fill-template: wrote {output_path}")

if __name__ == "__main__":
    main()
