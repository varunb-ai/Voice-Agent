"""Tool definitions the Voice Agent's LLM can call (Layer 4 — tool calling).

Seven tools:
  save_branch  — records the doctor's branch/location (does NOT decide the
                 call's outcome; see run_tool and agents/voice/objectives.py)
  save_doctor_identity    — did we reach the right doctor at the right
                 practice? Asked FIRST; everything else is gated on it.
                 not_here and wrong_number are DIFFERENT outcomes
  save_new_patient_status — records whether they are taking new patients, as one
                 of four states (yes | no | waitlist | unsure), not a boolean
  save_scheduling_status  — can a new patient actually get booked in; same four
  save_referral_requirement — always | depends | no | unsure, plus what it
                 depends on. Its own vocabulary: "depends" is the answer the
                 client acts on and yes/no would throw it away
  note_info    — capture supplementary info (website, email, phone, return date …)
  escalate     — call cannot be completed → records reason

Which of these a given call may use is the TEMPLATE's business, not this
module's: a template declares the fields it collects (CallTemplate.objective)
and the tool schemas it needs. Every tool is defined here regardless.

Framework-neutral: works in the offline brain test and in all telephony workers.
"""
from __future__ import annotations

import re
from typing import Any

from core.memory import CallMemory

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
                    "notes": {
                        "type": "object",
                        # CARRIES THE INSTRUCTION BECAUSE THE PROMPT CANNOT.
                        # The tool-list hint in patient_discovery was ignored on
                        # its first live outing (call-20260903-1126 made two
                        # separate note_info calls a second apart), and that
                        # template is pinned at exactly 5,900/5,900 tokens, so
                        # there is no room to argue harder there. Schemas are
                        # outside the prompt ceiling and are what actually
                        # drives call shape.
                        # THE EXAMPLE NAMES NO ENDING LABEL, deliberately. The
                        # suite asserts tools.py never spells that key, because
                        # a template may rename it and a hand-copied literal
                        # here would go stale silently. Two ordinary keys show
                        # the shape just as well.
                        "description": (
                            "Key-value pairs of metadata to record together in "
                            "ONE call, e.g. {\"waitlist\": \"number 15 in the "
                            "queue\", \"phone\": \"555-0100\"}. ALWAYS use this "
                            "instead of key/value when you have more than one "
                            "thing to record — each separate note_info call is "
                            "another silence the caller sits through. Same keys "
                            "as `key` above."
                        ),
                        "additionalProperties": {"type": "string"},
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_doctor_identity",
            "description": (
                "Record whether you have reached the right doctor at the right "
                "practice. ASK THIS FIRST — nothing else on the call means "
                "anything until it is settled. Four states: "
                "confirmed (right doctor, right practice), "
                "not_here (you reached the practice but the doctor is not "
                "there — left, never was, or a different site; the phone "
                "number is fine and the listing is wrong), "
                "wrong_number (you did not reach the practice at all), "
                "unsure (whoever answered does not know)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "identity": {
                        "type": "string",
                        "enum": ["confirmed", "not_here", "wrong_number",
                                 "unsure"],
                        "description": ("One of: confirmed | not_here | "
                                        "wrong_number | unsure"),
                    },
                    "heard": {
                        "type": "string",
                        "description": (
                            "What the caller ACTUALLY SAID, quoted as closely "
                            "as you can. It is REPLACED with the caller turn "
                            "the transcript shows, so a paraphrase is "
                            "discarded."
                        ),
                    },
                    "detail": {
                        "type": "string",
                        "description": (
                            "The specialty as they confirmed it, and anything "
                            "qualifying the answer in their words — 'we have a "
                            "Dr. Smith but he's a dermatologist', 'she moved to "
                            "the north site last year'. The specialty belongs "
                            "HERE: it is how they tell two doctors of the same "
                            "name apart, so it is part of establishing WHICH "
                            "doctor, not a separate fact."
                        ),
                    },
                },
                "required": ["identity", "heard"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_new_patient_status",
            "description": (
                "Record whether the doctor is currently accepting new patients. "
                "Call this the moment they tell you, in THEIR words. "
                "Four possible states — do not force an answer into yes or no: "
                "yes (taking new patients), no (not taking them), "
                "waitlist (full, but a list or queue exists — including when they "
                "give a queue position like 'you'd be number 21'), "
                "unsure (the person you are speaking to does not know)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["yes", "no", "waitlist", "unsure"],
                        "description": "One of: yes | no | waitlist | unsure",
                    },
                    "heard": {
                        "type": "string",
                        "description": (
                            "What the caller ACTUALLY SAID, quoted as closely as you "
                            "can. Not your summary of it. It is REPLACED with the "
                            "caller turn the transcript shows, so a paraphrase here "
                            "is discarded rather than recorded."
                        ),
                    },
                    "detail": {
                        "type": "string",
                        "description": (
                            "Anything qualifying the status, in their words: a queue "
                            "position, whether a referral is needed and what it depends "
                            "on, how to request an appointment, when to call back."
                        ),
                    },
                },
                "required": ["status", "heard"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_scheduling_status",
            "description": (
                "Record whether a new patient can actually get an appointment "
                "scheduled right now. Ask this ONLY when they are accepting new "
                "patients. Four states: yes (they can book), no (they cannot), "
                "waitlist (not now, but there is a list or queue), "
                "unsure (the person you are speaking to does not know)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["yes", "no", "waitlist", "unsure"],
                        "description": "One of: yes | no | waitlist | unsure",
                    },
                    "heard": {
                        "type": "string",
                        "description": (
                            "What the caller ACTUALLY SAID, quoted as closely as "
                            "you can. It is REPLACED with the caller turn the "
                            "transcript shows, so a paraphrase is discarded."
                        ),
                    },
                    "detail": {
                        "type": "string",
                        "description": (
                            "Anything qualifying it, in their words: how far out "
                            "the next opening is, who to call, when to try again."
                        ),
                    },
                },
                "required": ["status", "heard"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_referral_requirement",
            "description": (
                "Record whether a new patient needs a referral. Ask this ONLY "
                "when they are accepting new patients. Four states, and the "
                "conditionality is the point: always (every new patient needs "
                "one), depends (only for some insurers or situations — put what "
                "it depends on in `depends_on`), no (none needed), "
                "unsure (the person you are speaking to does not know)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "requirement": {
                        "type": "string",
                        "enum": ["always", "depends", "no", "unsure"],
                        "description": "One of: always | depends | no | unsure",
                    },
                    "heard": {
                        "type": "string",
                        "description": (
                            "What the caller ACTUALLY SAID, quoted as closely as "
                            "you can. It is REPLACED with the caller turn the "
                            "transcript shows, so a paraphrase is discarded."
                        ),
                    },
                    "depends_on": {
                        "type": "string",
                        "description": (
                            "What it depends on, in THEIR words — the insurer, "
                            "the plan, the reason for the visit. Required in "
                            "practice when requirement is 'depends': that "
                            "qualifier is the answer the client acts on."
                        ),
                    },
                },
                "required": ["requirement", "heard"],
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


# Filler the CALLER says that has been mistakenly saved as a branch. Not
# derivable from anything we send — these are ordinary English, so they stay
# enumerated. A short closed-ish list is the right shape for this half.
_CALLER_FILLER_ECHOES = (
    "location is all i need", "location is all", "all i need",
    "answer from you", "answer from", "from you",
    "just answer", "that's all", "thats all", "is all i",
    "can i ask you", "ask you another", "another question",
)


def _ngrams(text: str, n: int) -> set:
    words = re.findall(r"[a-z']+", (text or "").lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _derive_prompt_echoes() -> frozenset:
    """Sequences from the text WE send, so an echo of it can be recognised.

    This used to be 41 phrases copied by hand out of the prompt — including
    "forage ai", which was still in the list days after the organisation was
    renamed. A duplicated list rots silently every time the original changes,
    and the prompt has been edited eleven times this week.

    Derived from the actual instructions, greeting and transcription hint of
    every template, so it cannot drift from what is really being sent.

    Four-word sequences only. A real branch name is a short proper noun and will
    essentially never contain four consecutive words of our own instructions,
    so this cannot reject a genuine answer — whereas two-word sequences like
    "medical directory" plausibly could.
    """
    from agents.voice.templates import TEMPLATES
    grams: set = set()
    for tpl in TEMPLATES.values():
        for source in (tpl.instructions, tpl.greeting, tpl.transcribe_hint):
            grams |= _ngrams(source, 4)
    return frozenset(grams)


_PROMPT_ECHO_PHRASES: frozenset = frozenset()


def _prompt_echoes() -> frozenset:
    """Derived once, lazily — templates import at module load would cycle."""
    global _PROMPT_ECHO_PHRASES
    if not _PROMPT_ECHO_PHRASES:
        _PROMPT_ECHO_PHRASES = _derive_prompt_echoes() | frozenset(_CALLER_FILLER_ECHOES)
    return _PROMPT_ECHO_PHRASES


_CONJUNCTION_STARTS = {
    "or", "and", "but", "so", "nor", "yet", "which", "that", "if",
    "where", "when", "what", "why", "how", "who", "whom", "whose",
}


# ── Bare-city rejection ───────────────────────────────────────────────────────
# A city is not a branch. A hospital group with five offices in one city is
# precisely the case this project exists to resolve, so "New York" tells the
# directory nothing it did not already have.
#
# This slipped through before: a live call saved branch="New York branch",
# city="New York" as a RESOLVED result. Every individual word passed, because
# "new" and "york" are not filler, so the all-filler check never fired.
#
# The distinction being drawn below is between a word that names a site
# ("Northgate Campus" — a proper name that happens to contain a generic noun)
# and a word that just means "our presence in <city>" ("New York branch").
# The former is a usable answer; the latter is the question restated.

_US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine",
    "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
    "missouri", "montana", "nebraska", "nevada", "new hampshire", "new jersey",
    "new mexico", "new york", "north carolina", "north dakota", "ohio",
    "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina",
    "south dakota", "tennessee", "texas", "utah", "vermont", "virginia",
    "washington", "west virginia", "wisconsin", "wyoming",
    "district of columbia", "washington dc", "dc",
}

_MAJOR_CITIES = {
    # US — the metros a provider directory actually hits
    "new york", "new york city", "los angeles", "chicago", "houston",
    "phoenix", "philadelphia", "san antonio", "san diego", "dallas",
    "austin", "jacksonville", "fort worth", "columbus", "charlotte",
    "san francisco", "indianapolis", "seattle", "denver", "boston",
    "nashville", "detroit", "portland", "memphis", "louisville",
    "milwaukee", "baltimore", "albuquerque", "tucson", "fresno",
    "sacramento", "kansas city", "atlanta", "miami", "raleigh", "omaha",
    "minneapolis", "tulsa", "cleveland", "wichita", "arlington",
    "new orleans", "tampa", "honolulu", "pittsburgh", "cincinnati",
    "st louis", "saint louis", "orlando", "san jose", "el paso",
    "oklahoma city", "las vegas", "long beach", "virginia beach",
    "colorado springs", "st petersburg", "salt lake city", "buffalo",
    "richmond", "birmingham", "rochester", "des moines", "spokane",
    "madison", "boise", "hartford", "charleston", "savannah",
    # India — used during quality testing
    "hyderabad", "mumbai", "delhi", "new delhi", "chennai", "bengaluru",
    "bangalore", "pune", "kolkata", "ahmedabad", "ongole", "vijayawada",
    "visakhapatnam", "london",
}

# Words meaning "our presence in <somewhere>" rather than naming a place.
# "the New York branch" restates the city; "Northgate Campus" is a name.
_PRESENCE_WORDS = {
    "branch", "branches", "office", "offices", "location", "locations",
    "city", "site", "sites", "area", "region", "unit", "units",
}

# Words that can legitimately form part of a site's proper name.
_SITE_NAME_WORDS = {
    "campus", "campuses", "clinic", "clinics", "center", "centre",
    "centers", "centres", "hospital", "pavilion", "tower", "building",
    "wing", "annex", "institute", "practice", "medical",
}

_ALL_PLACE_NOUNS = _PRESENCE_WORDS | _SITE_NAME_WORDS


def _strip_place_nouns(text: str) -> str:
    """Remove generic place nouns, leaving whatever actually names somewhere."""
    words = [w for w in text.split() if w not in _ALL_PLACE_NOUNS
             and w not in {"the", "our", "a", "an", "in", "at", "of"}]
    return " ".join(words).strip()


def _looks_like_presence_in_a_place(cleaned: str) -> bool:
    """True for "<Placename> branch" / "the Newark office" and similar.

    The city list cannot keep up. "New York branch" was caught because New York
    is in it; "Newark branch" was saved on a live call because Newark is not,
    and there will always be another. The SHAPE is the reliable signal: one or
    two words naming somewhere, plus a word meaning "our presence there".

    Deliberately narrow. "Mercy General South Campus" and "1420 Beacon Street"
    do not match, because 'campus' names a site and a street address has more
    parts. Only the "<name> branch" pattern trips it.
    """
    words = cleaned.split()
    if not (2 <= len(words) <= 3):
        return False
    if words[-1] not in _PRESENCE_WORDS:
        return False
    lead = [w for w in words[:-1] if w not in {"the", "our", "a", "an"}]
    return 1 <= len(lead) <= 2


def _is_bare_city(cleaned: str, city: str | None) -> str | None:
    """Return a rejection reason if `cleaned` names no more than a city."""
    core = _strip_place_nouns(cleaned)
    if not core:
        return None          # handled by the all-filler check

    # The branch adds nothing the city field did not already say.
    if city and core == _strip_place_nouns(city.lower().strip().rstrip(".,!?")):
        return (f"'{city}' is already the city — the branch field needs the "
                f"specific office or site within it, not the city again")

    # A city or state with nothing else attached. Hard reject: there is no site
    # information here at all, so there is nothing a retry could confirm.
    if (core in _MAJOR_CITIES or core in _US_STATES) and core == cleaned:
        return (f"'{cleaned}' is a city or state, not a specific office. "
                f"Ask which office or site within it.")

    # "the New York branch" is NOT handled here. It used to be hard-rejected
    # when the city happened to be in the list above, which meant "Boston
    # office" was refused outright while "Newark branch" sailed through — the
    # behaviour depended on list coverage rather than on the input. Both now
    # fall through to the ask-once-then-accept path in save_branch, which
    # treats them the same regardless of whether we recognise the place.
    return None

def _malformed_args(impl, arguments: dict) -> tuple:
    """(missing required, unexpected) for a tool call about to be splatted.

    Read off the implementation's own signature rather than a second table of
    argument names, which would go out of step the first time a tool changes.
    `memory` is the bound first parameter and is never supplied by the model.
    """
    import inspect
    try:
        params = list(inspect.signature(impl).parameters.values())[1:]
    except (TypeError, ValueError):
        return set(), set()
    names = {p.name for p in params}
    required = {p.name for p in params
                if p.default is inspect.Parameter.empty
                and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                               inspect.Parameter.KEYWORD_ONLY)}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return required - set(arguments), set()
    return required - set(arguments), set(arguments) - names


def _reject(reason: str, need: str = "") -> dict:
    """A rejection the model can act on but cannot read out.

    On a live call the agent said, out loud, to a receptionist:

        "If there are several sites there, I need the specific site name or
         street address. If that's the only site, tell me that and I'll take it."

    which is this module's rejection text, lightly paraphrased. The old messages
    were written as fluent English imperatives — "Ask whether that is their only
    office there, or get the site name or street address" — so relaying them
    verbatim produced a grammatical sentence, and the model duly relayed one. No
    person says "I'll take it" about a branch name; it was the most machine-like
    turn in the best call we had.

    Rejections are therefore written as terse fragments in a register nobody
    speaks. Paraphrasing one now yields something visibly wrong rather than
    something plausible, which pushes the model to say the next thing in its own
    words. Paired with the "tool results are internal" rule in the prompt.
    """
    return {"ok": False, "error": reason + (f" | NEED: {need}" if need else "")}


def save_branch(
    memory: CallMemory,
    branch: str,
    city: str | None = None,
    schedule: str | None = None,
) -> dict:
    cleaned = branch.lower().strip().rstrip(".,!?")
    NEED = "site name or street address"
    if any(p in cleaned for p in _prompt_echoes()):
        return _reject(f"REJECTED {branch!r}: transcription artefact", NEED)
    if cleaned in _SPECIALIZATIONS:
        return _reject(f"REJECTED {branch!r}: department, not a place", NEED)
    if len(cleaned) < 3:
        return _reject(f"REJECTED {branch!r}: too short", NEED)
    if cleaned in _INVALID_BRANCH_WORDS:
        return _reject(f"REJECTED {branch!r}: generic word", NEED)
    words = cleaned.split()
    if words and words[0] in _CONJUNCTION_STARTS:
        return _reject(f"REJECTED {branch!r}: leading conjunction", NEED)
    if words and len(words[0]) == 1:
        return _reject(f"REJECTED {branch!r}: leading single letter", NEED)
    if words and all(w in _INVALID_BRANCH_WORDS for w in words):
        return _reject(f"REJECTED {branch!r}: filler only", NEED)
    bare_city = _is_bare_city(cleaned, city)
    if bare_city:
        return _reject(f"REJECTED {branch!r}: names a city, not a site", NEED)

    # "<Placename> branch" — probably the city restated, but not certainly: a
    # group with one Newark office really does call it the Newark branch.
    # So push back ONCE and accept whatever comes back. Rejecting outright
    # would loop forever when the caller confirms it is the only one.
    if _looks_like_presence_in_a_place(cleaned):
        if not memory.get("branch_clarification_asked"):
            memory.update(branch_clarification_asked=True)
            return _reject(
                f"NOT SAVED {branch!r}: possibly the city restated",
                "confirmation this is their only location there, OR a site "
                "name/street address. Same value on retry is accepted.",
            )
        memory.update(branch_needed_clarification=True)

    # RECORDS THE FIELD. DOES NOT DECIDE THE CALL.
    #
    # This line used to also mark the whole call a success, and it was the only
    # place in the programme that did. That made "the call succeeded" a claim
    # one tool was entitled to make about the whole call: a call that
    # established accepting-new-patients and referral requirements and no
    # branch recorded as a failure, and every metric and guard reading that
    # flag inherited the mistake.
    #
    # Success is now derived from the template's declared objective, in
    # objectives.record_outcome, after ANY tool call — see run_tool below.
    memory.update(branch=branch, city=city, schedule=schedule)
    return {"ok": True, "branch": branch, "city": city, "schedule": schedule}


def save_doctor_identity(
    memory: CallMemory,
    identity: str,
    heard: str = "",
    detail: str | None = None,
) -> dict:
    """Did we reach the right doctor at the right practice? Ask this FIRST.

    From the client-side contact: "If we don't know which doctor they're
    talking about, accepting new patients makes no sense." Everything else the
    script collects is gated on this coming back confirmed.

    THE TWO NEGATIVES ARE NOT THE SAME OUTCOME. not_here means the number is
    good and the doctor-to-hospital association on file is wrong — a directory
    correction, and the most valuable negative result this programme can
    produce. wrong_number means the number itself is wrong. Recording one as the
    other sends somebody to re-verify a number that was never the problem.
    """
    from agents.voice.objectives import (
        IDENTITY_STATUS_KEY, IdentityAnswer, classify_identity,
    )
    return _save_state(
        memory, identity, heard, detail,
        key=IDENTITY_STATUS_KEY,
        valid={s.value for s in IdentityAnswer},
        classifier=classify_identity,
        options="confirmed | not_here | wrong_number | unsure",
        detail_suffix="detail",
    )


def save_new_patient_status(
    memory: CallMemory,
    status: str,
    heard: str = "",
    detail: str | None = None,
) -> dict:
    """Record whether the doctor is taking new patients. Four states, not two.

    THE VALUE IS VALIDATED THE SAME WAY A BRANCH IS, and for the same reason.
    save_branch rejects a value the process cannot recognise rather than
    writing it and hoping; a status field is a smaller target but the failure
    is identical — Field.present() treats an unclassifiable CHOICE value as NOT
    collected, so a tool that accepted "I'll have to check" would leave the
    objective permanently PARTIAL while the log said it had been saved.

    GROUNDING IS NOT DONE HERE. It needs the call transcript and the per-turn
    audio measurements, neither of which this module has — same split as
    save_branch, whose grounding lives in realtime_worker._ungrounded_terms.
    See _ungrounded_status there for why a CHOICE field needs MORE than the
    location check, not less.
    """
    from agents.voice.objectives import (
        NEW_PATIENT_STATUS_KEY, ChoiceAnswer, classify_choice,
    )
    return _save_state(
        memory, status, heard, detail,
        key=NEW_PATIENT_STATUS_KEY,
        valid={s.value for s in ChoiceAnswer},
        classifier=classify_choice,
        options="yes | no | waitlist | unsure",
        detail_suffix="detail",
    )


def save_scheduling_status(
    memory: CallMemory,
    status: str,
    heard: str = "",
    detail: str | None = None,
) -> dict:
    """Can a new patient actually get on the books? Same four states as accepting.

    Deliberately NOT a narrower yes/no. "Not until the new year" and "there's a
    wait but you can get on the list" are the same shape of answer as the
    accepting field's waitlist, and the client acts on both the same way.
    """
    from agents.voice.objectives import (
        SCHEDULING_STATUS_KEY, ChoiceAnswer, classify_choice,
    )
    return _save_state(
        memory, status, heard, detail,
        key=SCHEDULING_STATUS_KEY,
        valid={s.value for s in ChoiceAnswer},
        classifier=classify_choice,
        options="yes | no | waitlist | unsure",
        detail_suffix="detail",
    )


def save_referral_requirement(
    memory: CallMemory,
    requirement: str,
    heard: str = "",
    depends_on: str | None = None,
) -> dict:
    """Is a referral needed, and is that unconditional?

    ITS OWN VOCABULARY, not the accepting field's. The client's question is
    "always, or does it depend?", so DEPENDS is a first-class state and the
    qualifier beside it is the part their team acts on — squeezing this into
    yes/no would discard the distinction the question exists to draw, the same
    way recording a queue position as "no" would.
    """
    from agents.voice.objectives import (
        REFERRAL_STATUS_KEY, ReferralAnswer, classify_referral,
    )
    return _save_state(
        memory, requirement, heard, depends_on,
        key=REFERRAL_STATUS_KEY,
        valid={s.value for s in ReferralAnswer},
        classifier=classify_referral,
        options="always | depends | no | unsure",
        detail_suffix="depends_on",
    )


def _save_state(memory: CallMemory, value: str, heard: str,
                detail: str | None, *, key: str, valid: set,
                classifier, options: str, detail_suffix: str) -> dict:
    """Shared body for the closed-set save tools.

    One implementation because all three do exactly the same three things —
    canonicalise, reject what cannot be canonicalised, store the state beside
    the wording it came from — and three copies of that would drift the way the
    hand-copied prompt phrases above drifted. The vocabulary is the parameter.

    GROUNDING IS NOT DONE HERE. It needs the transcript and the per-turn audio
    measurements, neither of which this module has — same split as save_branch.

    THE ONE THING IT DOES REFUSE ON ITS OWN IS A REGRESSION TO `unsure`, and
    that is a fact about the VALUES rather than about the evidence, so it
    belongs here and not in the guard. See the block below.
    """
    text = (value or "").strip().lower()
    if text not in valid:
        recognised = classifier(value or "")
        if recognised is None:
            return _reject(f"REJECTED {value!r}: not one of the states", options)
        text = recognised.value

    # THE REGRESSION LOCK. `unsure` is not a correction — it is the ABSENCE of
    # an answer — so it must never replace one the call already has.
    #
    # call-20260831-1048: identity was saved `confirmed` at 10:49:25 on "Yes.
    # [She] is one of a cardiologist." Twenty seconds later the caller answered
    # a question about the BRANCH with "I don't know the branch name", the
    # model called this tool again with `unsure`, and the write went straight
    # through. `memory.update` ignores Nones, so the second call did not even
    # replace the row — it merged into it, leaving a state from turn 9 beside a
    # `detail` from turn 3 and a quote from turn 9. A composite nobody uttered.
    #
    # That single write then ended the call: identity gates every other field
    # (see _IF_RIGHT_DOCTOR in templates.py), so `unsure` made branch,
    # accepting, scheduling and referral all not-required, the objective read
    # COMPLETE, and the teardown fired on a question the caller was still being
    # asked. The window fix in evidence.py stops that particular turn reaching
    # here; this stops the class of it, from any turn, on any of the four
    # fields — and it stops it in the layer that owns the value, so a future
    # caller of run_tool cannot route around the guard.
    #
    # A REAL CORRECTION IS STILL ALLOWED. confirmed -> not_here, yes -> no,
    # always -> depends: those carry information and are exactly what a caller
    # putting us right sounds like. Only the move to `unsure` is barred, and
    # only when there is something definite to lose. unsure -> anything is an
    # upgrade and passes untouched.
    current = str(memory.get(key) or "").strip().lower()
    if text == "unsure" and current in valid and current != "unsure":
        return _reject(
            f"REJECTED 'unsure': already {current!r}, and 'unsure' does not "
            f"correct it",
            f"nothing further on this; {current!r} stands")

    memory.update(**{
        key: text,
        f"{key}_heard": (heard or "").strip() or None,
        f"{key}_{detail_suffix}": (detail or "").strip() or None,
    })
    return {"ok": True, "status": text, "detail": detail}


def note_info(memory: CallMemory, key: str, value: str) -> dict:
    memory.update(**{f"note_{key}": value})
    return {"ok": True, "key": key, "value": value}


def escalate(memory: CallMemory, reason: str) -> dict:
    # `resolved=False` was here and is gone for the same reason it left
    # save_branch: escalating is a fact about how the call ENDED, not a verdict
    # on what it collected. A call that saved the branch and then escalated
    # because the accepting status could not be had is exactly the partial the
    # objective now expresses — stamping resolved=False here would delete the
    # half that worked, which is the failure direction this codebase pays for.
    memory.update(escalated=True, escalate_reason=reason)
    return {"ok": True, "escalated": True}


TOOL_IMPLS = {
    "save_branch":               save_branch,
    "save_doctor_identity":      save_doctor_identity,
    "save_new_patient_status":   save_new_patient_status,
    "save_scheduling_status":    save_scheduling_status,
    "save_referral_requirement": save_referral_requirement,
    "note_info":                 note_info,
    "escalate":                  escalate,
}


# Which FIELD each save tool writes, named by the tool's own argument, so a
# gate can be looked up from a tool call. Spelled as memory keys imported from
# objectives rather than as string literals here, for the reason
# unwritable_fields() exists: a key hand-copied into a second module is a key
# that drifts the first time somebody renames it.
#
# note_info and escalate are absent on purpose. note_info writes an arbitrary
# note_* key and escalate writes no field at all, so neither has a field whose
# gate could be consulted; both stay unconditionally dispatchable, which is what
# lets a call that cannot confirm the doctor still record WHY.
def _tool_field(objective: Any, name: str):
    """The objective's Field that this tool writes, or None."""
    from agents.voice.objectives import (
        IDENTITY_STATUS_KEY, NEW_PATIENT_STATUS_KEY, SCHEDULING_STATUS_KEY,
        REFERRAL_STATUS_KEY,
    )
    key = {
        "save_branch":               "branch",
        "save_doctor_identity":      IDENTITY_STATUS_KEY,
        "save_new_patient_status":   NEW_PATIENT_STATUS_KEY,
        "save_scheduling_status":    SCHEDULING_STATUS_KEY,
        "save_referral_requirement": REFERRAL_STATUS_KEY,
    }.get(name)
    if key is None:
        return None
    return next((f for f in objective.fields if f.memory_key == key), None)


def _append(memory: CallMemory, key: str, item: Any) -> None:
    """Append to a list held in memory. update() has no append."""
    memory.update(**{key: list(memory.get(key) or []) + [item]})


def _defer_save(memory: CallMemory, name: str, field_name: str,
                arguments: dict, gate_name: str) -> dict:
    """Hold a value whose gate has not been settled yet, and say so.

    NOT A DISCARD, and the distinction is the whole design. The caller really
    did say "he's at the east side clinic"; refusing that outright would throw
    away a real answer, which this project treats as its expensive direction of
    failure — a wrong row can be found later, a discarded answer looks exactly
    like a receptionist who would not say. So the value is held and applied the
    moment the gate opens, and the model is TOLD it is held, so it does not go
    back and ask a question that has already been answered.

    Only the most recent value per field is kept. A second attempt at the same
    field is the model correcting itself, not two answers to be replayed in
    order, and replaying a superseded value would write the one they retracted.
    """
    held = [d for d in (memory.get("deferred_saves") or [])
            if d.get("field") != field_name]
    held.append({"tool": name, "field": field_name, "arguments": arguments,
                 "gate": gate_name})
    memory.update(deferred_saves=held)
    return _reject(
        f"NOT SAVED: {gate_name} unsettled — {field_name} cannot be filed "
        f"against a doctor nobody has confirmed",
        f"settle {gate_name} first. Value HELD, applied automatically. "
        f"Nothing further needed from them on {field_name}.",
    )


def _flush_deferred(memory: CallMemory, objective: Any) -> None:
    """Apply or drop held values now that a gate field has been answered.

    Runs after every successful save, because any save may be the one that
    opens a gate. A held value whose gate has CLOSED is dropped rather than
    kept: identity=not_here means the doctor is not at this practice, so the
    branch collected there belongs to nobody and no later turn can change that.

    Both outcomes are RECORDED. A value that silently evaporates between the
    caller saying it and the artifact being written is the same invisibility
    every guard in this codebase has had to be retrofitted with — see
    suppressed_echoes, dropped_second_items, and the emptied-qualifier note.
    """
    from agents.voice.objectives import GateVerdict, gate_state
    held = list(memory.get("deferred_saves") or [])
    if not held:
        return
    keep: list = []
    for item in held:
        field = objective.field_named(item.get("field") or "")
        verdict, _gate, _value = gate_state(objective, memory, field)
        if verdict is GateVerdict.PENDING:
            keep.append(item)
            continue
        if verdict is GateVerdict.CLOSED:
            _append(memory, "deferred_dropped",
                    {**item, "why": f"{item.get('gate')}={_value}"})
            continue
        impl = TOOL_IMPLS.get(item.get("tool") or "")
        if impl is None:
            continue
        # Validated NOW, not when it was held — the value never reached the
        # tool, so save_branch's own checks have not run on it yet. A rejection
        # here is recorded rather than swallowed: a held value that fails
        # validation on replay is exactly as invisible as one that evaporates.
        replayed = impl(memory, **(item.get("arguments") or {}))
        if replayed.get("ok"):
            _append(memory, "deferred_applied", item.get("field"))
        else:
            _append(memory, "deferred_dropped",
                    {**item, "why": replayed.get("error") or "rejected on replay"})
    memory.update(deferred_saves=keep)


def _ending_label_owed(objective: Any, memory: CallMemory, name: str,
                       arguments: dict) -> tuple:
    """What the call still owes, if this call is the model stamping the ending.

    Empty means write it. Non-empty is the ordering the prompt asks for and
    does not get: the outcome label last, after everything it is a label ON.

    NOT KEYED ON A TOOL NAME. `note_info` writes note_<key> for anything, so
    what makes a call an ending label is the FIELD the key lands on — declared
    by the template as `records_the_ending`, read here. A second template
    calling its label something else is covered by construction, and one that
    marks a field the guard cannot see is reported by unenforceable_endings()
    before a call is ever placed.

    THE MODEL'S SPELLING IS NORMALISED HERE AND NOWHERE ELSE, deliberately.
    `note_info` stores the key it is handed, so `Call_Outcome` writes
    note_Call_Outcome and reads back as "not collected" — the safe direction,
    documented where the field is declared. This guard lowercases before
    looking up because it is judging INTENT, not the value: a model stamping
    the ending in the wrong case is still a model that has stopped working the
    call, and letting the typo through the ordering check would be a bypass
    nobody could see in an artifact.

    ESCALATING OPENS IT, and that is the one exit. escalate() is the model
    saying, on the record and with a reason, that the rest of this call cannot
    be had — which is exactly the fact the label is a label on. Refusing after
    that would keep the label off precisely the calls that failed, which is
    where a reader most needs it. The refusal text below does not mention this:
    the model's job on a refusal is to go and get the missing answer, and
    offering the exit in the same breath is offering it instead.
    """
    if name != "note_info":
        return ()
    if memory.get("escalated"):
        return ()
    key = str(arguments.get("key") or "").strip().lower()
    if not key:
        return ()
    from agents.voice.objectives import premature_ending_label
    _field, owed = premature_ending_label(objective, memory, f"note_{key}")
    return owed


def _run_note_batch(memory: CallMemory, arguments: dict[str, Any],
                    objective: Any) -> dict:
    """Apply ``notes={key: value, ...}`` as N ordinary note_info calls.

    WHY IT EXISTS: latency, measured. Three sequential tool calls in one turn
    cost three inference round trips before the agent makes a sound —
    call-20260902-2207 spent 4.69s that way and call-20260902-1541 spent 7.39s.
    Batching the metadata writes collapses those into one. It is a TAIL fix and
    is worth stating as one: only 3.1% of instrumented turns carry more than a
    single tool, so this moves p90 and leaves the median where it was.

    NOTHING IS ADJUDICATED HERE. Each key re-enters run_tool alone, so it meets
    the same guards in the same order; this function only splits and collects.

    OK IS NOT PARTIAL. A batch with any refusal returns ok=False while still
    reporting what landed in `written` — the model has to see that the label
    was refused (or it stops working the call), and the notes that succeeded
    must not be re-asked (or a real answer gets thrown away twice).
    """
    notes = arguments.get("notes")
    extra = sorted(k for k in arguments if k != "notes")
    if extra:
        return _reject(
            f"REJECTED: the call carried both notes and {', '.join(extra)}",
            "note_info again: EITHER key and value, OR notes alone")
    if not isinstance(notes, dict) or not notes:
        return _reject(
            "REJECTED: notes must be a non-empty object of key -> value",
            'note_info again with notes={"key": "value", ...}')

    written: dict[str, Any] = {}
    refused: list[str] = []
    for key, value in notes.items():
        _r = run_tool("note_info", memory,
                      {"key": str(key),
                       "value": "" if value is None else str(value)},
                      objective)
        if _r.get("ok"):
            written[str(key)] = _r.get("value")
        else:
            refused.append(f"{key}: {_r.get('error') or 'refused'}")

    if not refused:
        return {"ok": True, "written": written}
    _err = " | ".join(refused)
    if written:
        _err += f" | RECORDED, do not repeat: {', '.join(sorted(written))}"
    return {"ok": False, "written": written, "error": _err}


def run_tool(name: str, memory: CallMemory, arguments: dict[str, Any],
             objective: Any = None) -> dict:
    """Dispatch a tool call, then re-derive what the call has achieved.

    THE SUCCESS CONDITION LIVES HERE, ONCE, and it is evaluated after every
    tool — not asserted inside save_branch. `objective` defaults to the
    configured template's, so the classic pipeline in brain.py and any test
    that does not thread one through still get a correct verdict.

    The re-derivation runs even when the tool REJECTED the value. That is
    deliberate: a rejected save must not leave a stale `resolved` standing from
    an earlier one, and `note_info` can complete a field the objective declares
    against a note_* key.

    THE GATE IS ENFORCED HERE TOO, and it had to move somewhere. `RequiredWhen`
    decided whether a field was REQUIRED and nothing decided whether it could be
    WRITTEN, so on call-20260825-1437 the branch and the accepting status were
    filed for a doctor the call never confirmed — with `missing: ["identity"]`
    recorded in the same artifact. This is the single point every tool call
    passes through and the only one that has both the objective and the memory,
    so it is where the gate can be read.
    """
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        return {"ok": False, "error": f"unknown tool: {name}"}
    from agents.voice.objectives import (
        GateVerdict, default_objective, gate_state, record_outcome,
    )
    objective = objective or default_objective()

    # ── A BATCH OF NOTES IS N SINGLE NOTES, NOT A SECOND CODE PATH ──────────
    # Every note in a batch re-enters this function one key at a time, so the
    # ending-label refusal, the malformed check, _flush_deferred and
    # record_outcome all apply per key exactly as they always did. Writing the
    # loop any other way would fork the adjudication, and the fork is where a
    # guard quietly stops running on the newer shape.
    #
    # PER-KEY VERDICTS, because note_info is NOT the failure-free logging tool
    # it looks like: _ending_label_owed refuses precisely the ending-label key
    # that a multi-note turn tends to finish on. A batch carrying that key
    # alongside a real answer can therefore be half-written, and collapsing
    # that to one ok/failed would either discard the answer or claim the label
    # landed. The key is never spelled here — see the guard, which reads it off
    # the template's `records_the_ending` declaration.
    if name == "note_info" and "notes" in arguments:
        return _run_note_batch(memory, arguments, objective)

    # Read BEFORE the dispatch below, because `missing` is what it tests and
    # the tool about to run can change it.
    _owed_label = _ending_label_owed(objective, memory, name, arguments)
    field = _tool_field(objective, name)
    verdict, gate, gate_value = gate_state(objective, memory, field)
    if verdict is GateVerdict.PENDING and field is not None and gate is not None:
        result = _defer_save(memory, name, field.name, arguments, gate.name)
    elif verdict is GateVerdict.CLOSED and field is not None and gate is not None:
        # Nothing to hold. The gate is answered and answered against this
        # field, so no later turn on this call can make the value belong to
        # anybody — recording it would put a real practice's details under a
        # doctor who is not there.
        _append(memory, "deferred_dropped",
                {"tool": name, "field": field.name, "arguments": arguments,
                 "gate": gate.name, "why": f"{gate.name}={gate_value}"})
        result = _reject(
            f"REJECTED: {gate.name} came back {gate_value!r} — {field.name} "
            f"does not apply",
            f"nothing on {field.name}. Do not ask for it.",
        )
    elif _owed_label:
        # ── THE LABEL IS NOT A WAY PAST A QUESTION ──────────────────────────
        # A REFUSAL, NOT A HOLD, and this is the one write on the call where
        # that is the right shape. _defer_save holds a value because the caller
        # said it and discarding a real answer is what this project pays for.
        # Nobody said this one: it is the model's own summary, and holding it
        # would apply a verdict formed before the call finished to a call that
        # then went differently. call-20260902-2005 logged `unresolved` at
        # 20:06:15 out of three failed tool calls in a row, and at 20:07:30 the
        # identity landed, the held branch and waitlist flushed behind it, and
        # the call resolved COMPLETE. Replaying that label would have stamped
        # `unresolved` on a call that worked.
        #
        # RECORDED, because a guard that fires invisibly is this codebase's
        # recurring defect and because this one is the ordering evidence: an
        # artifact with entries here is a call where the model tried to stop
        # working before it was done, whether or not it recovered.
        _label = str(arguments.get("value") or "")
        _append(memory, "ending_label_refusals",
                {"value": _label, "owed": list(_owed_label)})
        result = _reject(
            f"REJECTED {_label!r}: the call is not finished — "
            f"{', '.join(_owed_label)} still unrecorded",
            f"{_owed_label[0]} first: ask for it, get its tool to succeed. "
            f"The label goes last.")
    else:
        # ── THE CALL ITSELF CAN BE MALFORMED ────────────────────────────────
        # `impl(memory, **arguments)` splats whatever the model sent straight
        # into a typed function, so a missing required argument is a TypeError
        # out of the event loop rather than a refusal the model can act on.
        # call-20260902-1822 died exactly there: a barge-in cut a
        # save_doctor_identity mid-arguments, the truncated JSON became {}, the
        # grounding guard had no value to object to and passed, and the call
        # was cut short at 53s with the caller mid-sentence.
        #
        # A REFUSAL, not a repair. Filling in a default would invent an answer,
        # which is the one thing this file exists to stop; the model is told
        # what was missing and may call again.
        _missing, _unexpected = _malformed_args(impl, arguments)
        if _missing or _unexpected:
            _why = []
            if _missing:
                _why.append("arrived without " + ", ".join(sorted(_missing)))
            if _unexpected:
                _why.append("carried unknown "
                            + ", ".join(sorted(_unexpected)))
            _append(memory, "malformed_tool_calls",
                    {"tool": name, "arguments": arguments,
                     "missing": sorted(_missing),
                     "unexpected": sorted(_unexpected)})
            result = _reject(
                f"REJECTED: the call {' and '.join(_why)}",
                f"{name} again, complete, in one call")
        else:
            result = impl(memory, **arguments)
            if result.get("ok"):
                # This save may be the one that opened a gate.
                _flush_deferred(memory, objective)
    record_outcome(memory, objective)
    return result
