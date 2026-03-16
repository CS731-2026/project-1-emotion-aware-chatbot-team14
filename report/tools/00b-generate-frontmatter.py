#!/usr/bin/env python3
# 00b-generate-frontmatter.py <paper.yaml> <output.pandoc>
# Generates a pandoc markdown frontmatter file from paper.yaml.
# This becomes the first section of the document — title, abstract, keywords.
# Uses pandoc custom-style divs so Word applies the correct template styles.
#
# The author block is NOT generated here — it is complex table layout that
# must be maintained manually in the Word template.

import sys
import yaml

def main():
    if len(sys.argv) != 3:
        print("Usage: 00b-generate-frontmatter.py <paper.yaml> <output.pandoc>", file=sys.stderr)
        sys.exit(1)

    yaml_path, output_path = sys.argv[1], sys.argv[2]

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    lines = []

    # Title
    title = meta.get("title", "")
    if title:
        lines += [
            '::: {custom-style="paper title"}',
            title,
            ":::",
            "",
        ]

    # Subtitle
    subtitle = meta.get("subtitle", "")
    if subtitle:
        lines += [
            '::: {custom-style="paper subtitle"}',
            subtitle,
            ":::",
            "",
        ]

    # Abstract
    abstract = meta.get("abstract", "")
    if abstract:
        # Strip leading/trailing whitespace from block scalar
        abstract = abstract.strip()
        lines += [
            '::: {custom-style="Abstract"}',
            f"**Abstract**\\u2014{abstract}",
            ":::",
            "",
        ]

    # Keywords
    keywords = meta.get("keywords", "")
    if keywords:
        lines += [
            '::: {custom-style="Keywords"}',
            f"**Keywords**\\u2014{keywords}",
            ":::",
            "",
        ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"00b-generate-frontmatter: wrote {output_path}")

if __name__ == "__main__":
    main()
