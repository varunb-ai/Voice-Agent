"""Tool definitions the Voice Agent's LLM can call (Layer 4 — tool calling).

Three tools:
  save_branch  — doctor's branch/location confirmed → marks call resolved
  note_info    — capture supplementary info (website, email, phone, return date …)
  escalate     — call cannot be completed → records reason

Framework-neutral: works in the offline brain test and in all telephony workers.
"""
from __future__ import annotations

from typing import Any

from agents.voice.memory import CallMemory

# ── Tool schemas (OpenAI-compatible) ─────────────────────────────────────────

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "save_branch",
            "description": (
                "Record the confirmed branch, city, or location where the doctor works. "
                "Call this the moment you have a real place name. "
                "Multiple locations: call once with all of them in the branch field, comma-separated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "branch": {
                        "type": "string",
                        "description": "Branch / campus / clinic / city name (e.g. 'North Campus', 'Downtown, East Branch')",
                    },
                    "city": {
                        "type": "string",
                        "description": "City name if mentioned separately from branch",
                    },
                    "schedule": {
                        "type": "string",
                        "description": "Days/times if receptionist mentioned them (e.g. 'Mon-Wed Downtown, Fri North Campus')",
                    },
                },
                "required": ["branch"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "note_info",
            "description": (
                "Save any supplementary information the receptionist provides that is NOT the branch location: "
                "a website URL, an email address, a phone number, the doctor's return date from leave, "
                "a new hospital the doctor moved to, that a voicemail was left, or anything else worth recording. "
                "Do NOT use this instead of save_branch — use it IN ADDITION to save_branch when extra info is given, "
                "or alone when no branch is available but the info is still useful."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": (
                            "Category of information. Use one of: "
                            "website | email | email_pending | phone | return_date | new_hospital | "
                            "voicemail | address | closed | renamed | callback_time | other"
                        ),
                    },
                    "value": {
                        "type": "string",
                        "description": "The actual information (URL, email, date, etc.)",
                    },
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate",
            "description": (
                "End the call when the branch cannot be obtained. "
                "Always include a clear reason so the follow-up team knows what happened."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": (
                            "Why branch could not be obtained. Examples: "
                            "'wrong number', 'doctor no longer at hospital', 'doctor retired', "
                            "'doctor on leave until [date]', 'doctor relocated to [hospital]', "
                            "'doctor deceased', 'receptionist refused to provide info', "
                            "'referred to website [url]', 'referred to email [addr]', "
                            "'reached voicemail', 'branch permanently closed', "
                            "'could not confirm after max attempts'"
                        ),
                    },
                },
                "required": ["reason"],
            },
        },
    },
]


# ── Tool implementations ──────────────────────────────────────────────────────

_SPECIALIZATIONS = {
    "cardiology", "neurology", "orthopedics", "pediatrics", "dermatology",
    "oncology", "radiology", "surgery", "gynecology", "ophthalmology",
    "psychiatry", "urology", "endocrinology", "nephrology", "gastroenterology",
    "pulmonology", "rheumatology", "hematology", "anesthesiology", "pathology",
    "dentistry", "general medicine", "internal medicine", "obstetrics",
    "icu", "opp", "opd", "emergency", "ward",
}

_INVALID_BRANCH_WORDS = {
    "this", "that", "here", "there", "it", "its", "them", "they", "we",
    "our", "the", "a", "an", "he", "she", "him", "her", "look", "see",
    "yes", "no", "ok", "okay", "sure", "unknown", "none", "nothing",
    "nobody", "nowhere", "everywhere", "anywhere", "somewhere",
    "nothing over", "over", "goodbye", "bye", "hello", "hi",
    "someone", "anyone", "everyone", "no one", "somebody", "anybody",
    "middle", "last", "first", "next", "both", "all", "any",
    # Vague location words that are only valid WITH a qualifier (e.g. "North Campus")
    # Both singular AND plural forms — plural was the root cause of "locations" being saved
    "campus", "campuses",
    "branch", "branches",
    "office", "offices",
    "building", "buildings",
    "hospital", "hospitals",
    "clinic", "clinics",
    "centre", "centres", "center", "centers",
    "location", "locations",
    "place", "places",
    "site", "sites",
    "facility", "facilities",
    "institute", "institutes",
    "unit", "units",
    "area", "areas",
    "city", "cities",
    "region", "regions",
    "sector", "sectors",
}


_PROMPT_ECHO_PHRASES = (
    # STT hallucinations of the initial_prompt
    "doctor name branch", "branch location city", "hospital branch location",
    "branch city area", "doctor name branch city", "name branch location",
    # Agent's own script lines — LLM sometimes saves these as the branch
    "healthcare professionals", "medical directory", "forage ai",
    "update our", "calling from", "practice locations", "directory for",
    "database of doctors", "not shared or sold", "used internally",
    # 8b fallback system prompt phrases leaking as branch name
    "never share personal data", "not shared with", "share personal",
    "only collect branch", "comply with regulation", "legitimate medical",
    "real place name", "from these instructions", "the caller says",
    "currently working at", "directory service", "directory company",
    # Caller utterances mistakenly saved as branch
    "location is all i need", "location is all", "all i need",
    "answer from you", "answer from", "from you",
    "just answer", "that's all", "thats all", "is all i",
    "can i ask you", "ask you another", "another question",
)


_CONJUNCTION_STARTS = {
    "or", "and", "but", "so", "nor", "yet", "which", "that", "if",
    "where", "when", "what", "why", "how", "who", "whom", "whose",
}

def save_branch(
    memory: CallMemory,
    branch: str,
    city: str | None = None,
    schedule: str | None = None,
) -> dict:
    cleaned = branch.lower().strip().rstrip(".,!?")
    if any(p in cleaned for p in _PROMPT_ECHO_PHRASES):
        return {"ok": False, "error": f"'{branch}' looks like an STT hallucination, not a real location"}
    if cleaned in _SPECIALIZATIONS:
        return {"ok": False, "error": f"'{branch}' is a specialization/department, not a location"}
    if len(cleaned) < 3:
        return {"ok": False, "error": "branch value too short"}
    if cleaned in _INVALID_BRANCH_WORDS:
        return {"ok": False, "error": f"'{branch}' is not a valid branch name"}
    words = cleaned.split()
    if words and words[0] in _CONJUNCTION_STARTS:
        return {"ok": False, "error": f"'{branch}' starts with a conjunction — not a real location name"}
    if words and len(words[0]) == 1:
        return {"ok": False, "error": f"'{branch}' starts with a single letter — not a real location name"}
    if words and all(w in _INVALID_BRANCH_WORDS for w in words):
        return {"ok": False, "error": f"'{branch}' contains only filler words"}
    memory.update(branch=branch, city=city, schedule=schedule, resolved=True)
    return {"ok": True, "branch": branch, "city": city, "schedule": schedule}


def note_info(memory: CallMemory, key: str, value: str) -> dict:
    memory.update(**{f"note_{key}": value})
    return {"ok": True, "key": key, "value": value}


def escalate(memory: CallMemory, reason: str) -> dict:
    memory.update(resolved=False, escalated=True, escalate_reason=reason)
    return {"ok": True, "escalated": True}


TOOL_IMPLS = {
    "save_branch": save_branch,
    "note_info":   note_info,
    "escalate":    escalate,
}


def run_tool(name: str, memory: CallMemory, arguments: dict[str, Any]) -> dict:
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        return {"ok": False, "error": f"unknown tool: {name}"}
    return impl(memory, **arguments)
