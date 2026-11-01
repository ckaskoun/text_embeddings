"""
Claude Code PreToolUse hook: asks before any shell command that touches the data folders.

raw_data/ and saved_embeddings/ hold student writing (PEPPER is not anonymized), and .pkl
embedding files also store the sentence text. Anything Claude reads is sent to the model,
so a command that names these paths needs a person's approval first.
"""

import json
import re
import sys

PATTERN = re.compile(r'(raw_data|saved_embeddings)(/|\b)|\.(xlsx|xls|csv|pkl|pickle)\b', re.IGNORECASE)

payload = json.load(sys.stdin)
command = payload.get('tool_input', {}).get('command', '')

if PATTERN.search(command):
    print(json.dumps({
        'hookSpecificOutput': {
            'hookEventName': 'PreToolUse',
            'permissionDecision': 'ask',
            'permissionDecisionReason': (
                'This command touches the research data folders or data files '
                '(raw_data/, saved_embeddings/, .xlsx/.csv/.pkl). They can contain '
                'non-anonymized student text. Approve only if no real data will be read.'
            ),
        }
    }))
