"""Component 5 — Email Parsing.

Extract the branch (and city) from a hospital's reply. Heuristics first:
a reply like "Dr. John Smith is associated with our Dallas Medical Center branch"
is easy to pattern-match. Qwen3 handles complicated / multi-sentence replies.
"""
from __future__ import annotations

import re
from typing import Optional

from core import llm

# Patterns are case-INSENSITIVE and tolerant of casual phrasing, because answers
# come from typed text and speech-to-text ("he works at dallas", "dallas branch"),
# not just polished email prose. Ordered most-specific first so a full name like
# "Dallas Medical Center" wins over just "Dallas". Each captures the branch name.
_WORD = r"[A-Za-z][\w&.\-]*"                       # one name word
_NAME = rf"{_WORD}(?:\s+{_WORD}){{0,3}}?"           # up to 4 words, non-greedy
_KEYWORD = r"(?:branch|location|center|centre|campus|hospital|clinic|office)"

_BRANCH_PATTERNS = [
    # "at our Dallas Medical Center branch", "in the Houston location"
    re.compile(rf"\b(?:at|in|with)\s+(?:our\s+|the\s+)?({_NAME})\s+{_KEYWORD}\b", re.I),
    # "works at Dallas", "working at New York", "based in Houston", "practicing at..."
    re.compile(rf"\b(?:affiliated with|associated with|work(?:s|ing|ed)?\s+(?:at|in|from)|"
               rf"practice?s?\s+(?:at|in)|based\s+(?:at|in|out of)|stationed\s+(?:at|in)|"
               rf"located\s+(?:at|in)|operates?\s+(?:at|in|from)|serves?\s+(?:at|in)|"
               rf"currently\s+(?:at|in)|moved?\s+to|transferred?\s+to)\s+(?:our\s+|the\s+)?({_NAME})\b", re.I),
    # "Dallas branch" / "New York campus" (no preposition)
    re.compile(rf"\b({_NAME})\s+{_KEYWORD}\b", re.I),
    # "it's New York City" / "that is the branch place" — caller restating
    re.compile(rf"\bit['\s]s\s+(?:the\s+)?({_NAME})\b", re.I),
    # bare "at dallas" / "in austin" at the end of a short reply
    re.compile(rf"\b(?:at|in)\s+(?:our\s+|the\s+)?({_WORD}(?:\s+{_WORD})?)\s*[.,!?]*\s*$", re.I),
]

_STOP = {"a", "an", "the", "our", "their", "your", "his", "her"}


def heuristic_parse(body: str) -> Optional[str]:
    for pat in _BRANCH_PATTERNS:
        m = pat.search(body)
        if m:
            branch = m.group(1).strip(" .,-")
            words = [w for w in branch.split() if w.lower() not in _STOP]
            branch = " ".join(words)
            if 2 <= len(branch) <= 40:
                # Normalize casual lowercase ("dallas" -> "Dallas") but keep
                # already-capitalized multi-word names intact.
                return branch if any(c.isupper() for c in branch) else branch.title()
    return None


_LLM_SYSTEM = (
    "Extract the hospital branch a doctor is affiliated with from an email reply. "
    'Return JSON: {"branch": str|null, "city": str|null}. Null if not stated.'
)


def llm_parse(body: str, doctor_name: str) -> tuple[Optional[str], Optional[str]]:
    if not llm.is_available():
        return None, None
    prompt = f"Doctor: {doctor_name}\n\nReply email:\n{body[:2000]}"
    try:
        data = llm.chat_json(prompt, system=_LLM_SYSTEM)
    except Exception:
        return None, None
    return data.get("branch") or None, data.get("city") or None


def parse_reply(
    body: str, doctor_name: str, use_llm: bool = True
) -> tuple[Optional[str], Optional[str]]:
    """Return (branch, city). Heuristics first; Qwen only if they fail."""
    branch = heuristic_parse(body)
    if branch:
        return branch, None
    if use_llm:
        return llm_parse(body, doctor_name)
    return None, None
