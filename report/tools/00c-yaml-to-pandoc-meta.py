#!/usr/bin/env python3
# 00c-yaml-to-pandoc-meta.py <paper.yaml> <output-meta.yaml>
# Converts paper.yaml into pandoc-compatible metadata YAML for LaTeX output.
#
# paper.yaml uses a nested authors: [{name:, dept:, org:, city:, email:}] format.
# This script emits two forms for pandoc:
#   author:  [name strings]           — used by pandoc's built-in default template
#   authors: [{name:, dept:, ...}]    — used by custom templates via $for(authors)$
# Blank author entries (name: "") are filtered out before either list is written.

import sys
import yaml


def main():
    if len(sys.argv) != 3:
        print("Usage: 00c-yaml-to-pandoc-meta.py <paper.yaml> <output-meta.yaml>", file=sys.stderr)
        sys.exit(1)

    yaml_path, output_path = sys.argv[1], sys.argv[2]

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    pandoc_meta = {}

    if meta.get("title"):
        pandoc_meta["title"] = meta["title"]

    if meta.get("abstract"):
        pandoc_meta["abstract"] = meta["abstract"]

    if meta.get("keywords"):
        pandoc_meta["keywords"] = meta["keywords"]

    # Filter out blank author entries (paper.yaml ships with empty placeholder rows)
    real_authors = [a for a in meta.get("authors", []) if a.get("name")]

    # Flat name list — consumed by pandoc's built-in template ($for(author)$)
    if real_authors:
        pandoc_meta["author"] = [a["name"] for a in real_authors]

    # Full nested list — consumed by custom templates ($for(authors)$...$authors.name$...)
    if real_authors:
        pandoc_meta["authors"] = [
            {k: a.get(k, "") for k in ("name", "dept", "org", "city", "email")}
            for a in real_authors
        ]

    with open(output_path, "w") as f:
        yaml.dump(pandoc_meta, f, default_flow_style=False, allow_unicode=True)

    print(f"00c-yaml-to-pandoc-meta: wrote {output_path}")


if __name__ == "__main__":
    main()
