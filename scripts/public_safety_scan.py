#!/usr/bin/env python3
import re, sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
patterns = [
    ('home-path', re.compile(r'/home/[A-Za-z0-9_.-]+')),
    ('github-token', re.compile(r'(?:ghp|github_pat)_[A-Za-z0-9_\-]{20,}')),
    ('hf-token', re.compile(r'\bhf_[A-Za-z0-9]{24,}\b')),
    ('agentmail-token', re.compile(r'\bam_[A-Za-z0-9]{24,}\b')),
    ('email-address', re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')),
    ('lan-ip', re.compile(r'\b(?:10|192\.168|172\.(?:1[6-9]|2[0-9]|3[01]))\.\d+\.\d+\b')),
    ('secret-key-name', re.compile(r'(GITHUB_TOKEN|HF_ACCESS_TOKEN|AGENTMAIL_API_KEY|OPENAI_API_KEY|XAI_API_KEY)')),
]
allow_files = {Path('scripts/public_safety_scan.py')}
allowed_exact = {'@mr-r0b0t'}
failures = []
for p in ROOT.rglob('*'):
    if not p.is_file() or '.git' in p.parts:
        continue
    rel = p.relative_to(ROOT)
    if rel in allow_files:
        continue
    try:
        text = p.read_text(errors='ignore')
    except Exception:
        continue
    for name, rx in patterns:
        for m in rx.finditer(text):
            val = m.group(0)
            if val in allowed_exact:
                continue
            failures.append(f'{rel}: {name}: {val[:120]}')
if failures:
    print('PUBLIC SAFETY SCAN FAILED')
    for f in failures[:200]: print(f)
    if len(failures) > 200: print(f'... {len(failures)-200} more')
    sys.exit(1)
print(f'PUBLIC SAFETY SCAN PASSED: {ROOT}')
