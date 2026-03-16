#!/usr/bin/env python3
# 03b-merge.py <front.docx> <body.docx> <output.docx>
# Appends body.docx onto the end of front.docx and saves as output.docx.
# Uses docxcompose (https://docxcompose.readthedocs.io)
#
# front.docx  = template-ready.docx (filled by docxtpl: title, authors, abstract)
# body.docx   = pandoc output (body sections with styles from the same template)
# output.docx = merged result

import sys
from docx import Document
from docxcompose.composer import Composer

def main():
    if len(sys.argv) != 4:
        print("Usage: 03b-merge.py <front.docx> <body.docx> <output.docx>", file=sys.stderr)
        sys.exit(1)

    front_path, body_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

    front = Document(front_path)
    composer = Composer(front)
    body = Document(body_path)
    composer.append(body)
    composer.save(output_path)

    print(f"03b-merge: wrote {output_path}")

if __name__ == "__main__":
    main()
