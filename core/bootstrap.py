"""Import this first in any CLI to make output safe on Windows terminals.

Windows' legacy console defaults to cp1252, which can't encode characters like
'→' or '—' that rich/our output use, causing UnicodeEncodeError. Reconfiguring
stdout/stderr to UTF-8 fixes it without touching every print.
"""
from __future__ import annotations

import sys

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")   # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass
