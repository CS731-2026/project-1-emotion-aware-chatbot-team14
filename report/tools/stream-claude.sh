#!/bin/sh
# tools/stream-claude.sh
# Runs `claude -p --output-format stream-json` and prints human-readable progress.
# Usage: echo "prompt" | tools/stream-claude.sh [extra claude args...]
#
# stream-json events handled:
#   assistant  → text blocks printed inline; tool_use printed as "[ToolName] arg1, arg2"
#   result     → final result text printed at end
exec claude -p --verbose --output-format stream-json "$@" \
  | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        ev = json.loads(line)
        t = ev.get('type', '')
        if t == 'assistant':
            for blk in ev.get('message', {}).get('content', []):
                if blk.get('type') == 'text' and blk.get('text'):
                    sys.stdout.write(blk['text'])
                    sys.stdout.flush()
                elif blk.get('type') == 'tool_use':
                    name = blk.get('name', '?')
                    inp = blk.get('input', {})
                    if 'file_path' in inp:
                        detail = inp['file_path']
                    elif 'command' in inp:
                        cmd = inp['command']
                        detail = cmd if len(cmd) <= 80 else cmd[:77] + '...'
                    else:
                        detail = ', '.join(inp.keys())
                    sys.stdout.write('\n[' + name + '] ' + detail + '\n')
                    sys.stdout.flush()
        elif t == 'result':
            r = ev.get('result', '')
            if r:
                sys.stdout.write(r)
                sys.stdout.flush()
    except Exception:
        pass
"
