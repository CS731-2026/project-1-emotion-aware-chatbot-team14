#!/usr/bin/env python3
"""Set latex_template in source/paper.yaml to the given path."""
import re, sys

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <paper.yaml> <template-path>", file=sys.stderr)
    sys.exit(1)

yaml_path, tpl = sys.argv[1], sys.argv[2]
with open(yaml_path) as f:
    content = f.read()
content = re.sub(
    r'[ \t]*#?[ \t]*latex_template:.*',
    '  latex_template: ' + tpl,
    content,
    flags=re.MULTILINE,
)
with open(yaml_path, 'w') as f:
    f.write(content)
print('Active template: ' + tpl)
print('Run: make pdf')
