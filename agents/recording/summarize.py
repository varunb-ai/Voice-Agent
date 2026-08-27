"""Generate a short summary for a completed call using LLM or heuristic fallback."""
from __future__ import annotations

from typing import cast


def summarize(snap: dict, use_llm: bool = True) -> str:
    doctor   = snap.get("doctor", "Unknown")
    hospital = snap.get("hospital", "Unknown Hospital")
    branch   = snap.get("branch")
    turns    = snap.get("transcript", [])

    if use_llm:
        try:
            from core import llm
            if llm.is_available():
                transcript_text = "\n".join(
                    f"{t['role'].capitalize()}: {t['text']}"
                    for t in turns
                    if isinstance(t, dict)
                )
                prompt = (
                    f"Summarize this hospital call in one sentence. "
                    f"Doctor: {doctor}, Hospital: {hospital}.\n\n{transcript_text}"
                )
                resp = llm.client().chat.completions.create(
                    model=llm.settings.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=100,
                )
                # content is Optional; a None here raises and the except below
                # falls through to the heuristic, which is the intended path.
                # The cast keeps that behaviour exactly.
                return cast(str, resp.choices[0].message.content).strip()
        except Exception:
            pass

    # Heuristic fallback
    if branch:
        return f"Called {hospital} to verify Dr. {doctor}'s branch. Branch confirmed: {branch}."
    return f"Called {hospital} to verify Dr. {doctor}'s branch. Branch could not be confirmed."
