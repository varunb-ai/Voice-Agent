"""OpenAI Realtime API bridge for Twilio voice calls — speech-to-speech only.

Architecture:
    Twilio WS  ←→  FastAPI  ←→  OpenAI Realtime WS
   (μ-law 8kHz)               (PCM16 24kHz)

One persistent WebSocket carries the whole conversation. Caller audio goes in,
agent audio comes out, and the model never round-trips through a separate STT
or TTS service. Inline `input_audio_transcription` runs alongside purely to
produce the written transcript — it is not in the conversational path, and
nothing waits on it.

Two things this module deliberately does NOT do, because both broke the
speech-to-speech guarantee:
  * no out-of-band whisper-1 HTTP transcription per caller turn
  * no fallback to the classic VAD→STT→LLM→TTS pipeline

Prompt caching: the session is configured once with a template's STATIC
instructions, and per-call facts are sent as the first conversation item. Never
put the doctor or hospital into `instructions`, and never pass a per-response
`instructions` override — either one moves the cache boundary and the whole
prefix is re-billed on every turn. See agents/voice/templates.py.

Enable with USE_REALTIME=true in .env.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import threading
import time
import traceback
from datetime import datetime, timezone
from statistics import median
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import websockets
import websockets.exceptions
from fastapi import WebSocket

from core.config import settings, persona_for_voice
from core.models import Doctor, DoctorStatus, Source, TranscriptTurn
from agents.voice.memory import CallMemory
from agents.voice.templates import get_template
from agents.voice.tools import run_tool, TOOL_SCHEMAS
from agents.voice.audio_utils import resample, _mulaw_decode, _mulaw_encode

log = logging.getLogger(__name__)

# Caller-utterance RMS below which the line is treated as too faint to trust.
# Clear phone speech measures roughly 0.03-0.08; a live call that produced a
# fabricated answer measured 0.004-0.012 throughout.
_LOW_AUDIO_RMS = 0.015

# Where call artefacts land. Indirected through functions so tests can point
# them at a temp directory — the protocol suite used to write real WAVs and
# JSON into data/ on every run, polluting the actual call records.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def audio_dir() -> Path:
    return _PROJECT_ROOT / "data" / "3 cases voice"


def json_dir() -> Path:
    return _PROJECT_ROOT / "data" / "3 cases jsons"


REALTIME_URL = "wss://api.openai.com/v1/realtime?model={model}"
# Per-attempt ceiling on the OpenAI handshake. Deliberately below the
# websockets default of 10s: the callee is already on the line and every
# second here is silence they hear, so a stall must be caught early enough to
# retry inside their patience rather than after it. Measured healthy: 1.7s.
_OAI_CONNECT_TIMEOUT_S = 6.0
# Serialises the read-modify-write of master.json. See save().
_MASTER_LOCK = threading.Lock()
# Same, for the doctor directory. A separate lock rather than reusing
# _MASTER_LOCK: the two files are independent and never written nested, so
# sharing one would only couple them.
_DOCTORS_LOCK = threading.Lock()

_TWILIO_SR = 8_000
_OAI_SR    = 24_000


# ── Audio format conversion ───────────────────────────────────────────────────

def _convert_oai_to_twilio(pcm16_b64: str) -> str:
    """OpenAI base64(PCM16 24kHz) → base64(μ-law 8kHz) for Twilio."""
    raw     = base64.b64decode(pcm16_b64)
    f32_24k = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    f32_8k  = resample(f32_24k, _OAI_SR, _TWILIO_SR)
    return base64.b64encode(_mulaw_encode(f32_8k)).decode()


# ── μ-law passthrough ─────────────────────────────────────────────────────────
# With REALTIME_AUDIO_FORMAT=pcmu the session speaks the same g711 μ-law Twilio
# already sends, so frames cross untouched in both directions: no μ-law decode,
# no 8k→24k resample inbound, no 24k→8k resample and re-encode outbound. That is
# two resamples removed from each 20ms frame, 50 frames a second, each way.
#
# Recording still needs linear PCM, but only to write a WAV — decoding μ-law at
# 8kHz is cheap and skips the resample entirely, so the recording is written at
# 8kHz rather than 24kHz. It is a phone call; 8kHz is the true bandwidth anyway.

def _passthrough_enabled() -> bool:
    return settings.realtime_audio_format == "pcmu"


def _wire_sample_rate() -> int:
    """Sample rate of whatever is stored in the recording buffers."""
    return _TWILIO_SR if _passthrough_enabled() else _OAI_SR


def _wire_to_pcm16(raw: bytes) -> np.ndarray:
    """Decode a recording-buffer chunk to float32, whatever format it holds."""
    if _passthrough_enabled():
        return _mulaw_decode(raw)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _wire_samples(raw: bytes) -> int:
    """Number of audio samples in a chunk. μ-law is 1 byte/sample, PCM16 is 2."""
    return len(raw) if _passthrough_enabled() else len(raw) // 2


def _loudest_window_rms(arr: np.ndarray, window_s: float = 0.3) -> float:
    """RMS of the LOUDEST window in an utterance, not the mean across it.

    Mean RMS over a whole utterance is dominated by the gaps between words. On
    a live call a perfectly audible caller — peak 0.098, twelve windows above
    0.02, every turn transcribing cleanly — measured 0.0016 by the mean and was
    told "you're coming through faint". Telling an audible person they cannot
    be heard is its own way to lose the call.

    The loudest window answers the question actually being asked: when they
    were speaking, was there enough signal? Utterances too short to judge
    return 0.0, which the caller treats as "no opinion".
    """
    if arr.size == 0:
        return 0.0
    sr = _wire_sample_rate()
    win = int(window_s * sr)
    if arr.size < win:
        return 0.0     # too short to judge — do not guess
    best = 0.0
    for start in range(0, arr.size - win + 1, max(1, win // 2)):
        seg = arr[start:start + win]
        best = max(best, float(np.sqrt(np.mean(seg ** 2))))
    return best


# A caller turn that carries nothing to work with: a greeting, an
# acknowledgement, or a request to repeat. Not an answer, and — crucially —
# not evidence that the caller HEARD the question either.
_FILLER_REPLY = re.compile(
    r"^(?:\W*(?:hello|hullo|hi|hey|yes|yeah|yep|yup|ok|okay|sure|right|alright|"
    r"mm+|hm+|uh+|um+|er+|ah+|oh+|go ahead|that'?s fine|i see|fine|"
    r"sorry|pardon|come again|say again|what|huh|"
    r"are you there|still there|can you hear me)\W*)+$", re.I)


def _is_filler_reply(text: str, agent_name: str = "") -> bool:
    """True if this caller turn answers nothing and asks nothing.

    "Hello." on call-20260818-1338 was treated as a non-answer and the agent
    re-asked — but it is more than a non-answer, it is a signal the caller did
    not hear. Their previous turn had been truncated to 750ms by a barge-in, so
    they genuinely had not.
    """
    t = (text or "").strip()
    if not t:
        return True
    if agent_name:
        # "Hello, David." is still just hello. Strip the name we introduced
        # ourselves with before judging, or every greeting reads as content.
        t = re.sub(rf"\b{re.escape(agent_name)}\b", " ", t, flags=re.I)
    return bool(_FILLER_REPLY.match(t))


def _caller_answered_since(sess: "RealtimeSession", since_idx: int) -> bool:
    """Did the caller say anything substantive after turn `since_idx`?"""
    for t in sess.turns[since_idx:]:
        if (t.role == "caller" and t.text.strip() != "[...]"
                and not _is_filler_reply(t.text, sess.agent_name)):
            return True
    return False


# How many times the agent may re-ask into silence before those re-asks start
# costing budget again. Without a bound, a caller who only ever says "hello"
# would let the agent ask forever: the budget would never advance and nothing
# would end the call.
_MAX_UNANSWERED_REASKS = 2

# How long after a truncation the next caller turn is read as a repair signal
# rather than an answer. Generous: the caller has to notice the line went odd,
# decide to say something, and be transcribed. Bounded so a truncation early in
# the call cannot colour an unrelated turn a minute later.
_REPAIR_WINDOW_S = 12.0

# Truncations shorter than this mean they heard essentially nothing. Above it
# they heard most of a sentence and may have interrupted deliberately, which is
# a normal conversational move needing no repair. Measured reference: the
# truncation on call-20260818-1338 was 750ms and the caller plainly had not
# followed it.
_CUT_SHORT_MS = 1500

# Shortest gap allowed between two location asks. On call-20260811-1649 the
# agent asked at 16:49:31, the caller said "Yes, speaking" while it was still
# talking, and it asked again 0.14s after its own audio ended — three asks in
# the first thirteen seconds. Nothing stopped it: back_to_back_asks is computed
# and printed but never acted on, and templates.py's "do NOT ask again" is a
# phrasing rule the model ignored. The ask budget already proved that a rule
# the code enforces beats a rule the prompt requests.
#
# This cannot unsay the ask that trips it — the agent has already spoken by the
# time its transcript arrives — but it stops the run continuing, which is what
# turned one re-ask into a burnt budget and a dead call.
_MIN_REASK_GAP_S = 6.0


# The caller asking who they are talking to. This must be answered, and on
# call-20260811-1649 it was not: "Hello, may I ask who is speaking?" came back
# "Sorry, I didn't catch that — could you say the branch name again?" The
# faint-line path did not fire (it requires an EMPTY transcript and this one
# transcribed perfectly), so the model simply chose to deflect — dodging the
# question AND spending an ask from the budget to do it. On a cold call this is
# the worst possible moment to sound evasive: it is precisely when the person
# is deciding whether to keep talking to you.
def _is_reintroduction(text: str, agent_name: str, org: str) -> bool:
    """True if this turn re-delivers the greeting: self-identification + org.

    templates.py has the rule already — "Do NOT answer it by re-introducing
    yourself. Your name and your employer are the answer to WHO, not to WHY" —
    and on call-20260813-1409 the agent broke it on turn TWO: "Sure, let me
    explain who I am and why I'm calling. I'm David, calling on behalf of
    Definitive Healthcare." That is the greeting again. The callee learned
    nothing about what was wanted, said nothing further, and the next forty
    seconds of the call were watchdog prompts recovering from it.

    Worth noting what triggered it: the caller had not asked anything. The
    transcript was "Hi, Ms. Mage" — a mis-transcription — and the agent
    inferred an identity question from it and then answered that phantom
    question wrongly. So the guard cannot key off "did they ask who I am".

    Deliberately NOT "contains the org name". Naming the org is correct when
    someone genuinely asks who is calling; that is what the org name is FOR.
    What is wrong is redelivering the whole introduction — the self-naming AND
    the org together, which is the greeting formula and nothing else. That
    keeps "I'm an automated system from {org}" out of scope here: it is a
    different failure (a false employment claim) and wants its own check.
    """
    if not text or not org:
        return False
    low = text.lower()
    if org.lower() not in low:
        return False
    name = (agent_name or "").strip().lower()
    if not name:
        return False
    return bool(re.search(rf"\b(i'?m|i am|this is|my name is)\s+{re.escape(name)}\b",
                          low))


def _claims_employment(text: str, org: str) -> bool:
    """True if this turn says the agent is FROM/WITH/AT the client org.

    The agent calls ON BEHALF OF a client; it is not employed by them. "I'm an
    automated system from Definitive Healthcare" is a false statement about who
    is on the phone, made to a medical office, and it does not survive the
    receptionist checking later. Removing it from the greetings was the whole
    point of the "on behalf of" work — and on call-20260813-1409 it came back
    out of the model mid-call anyway, at 14:11:33, because the tests assert on
    build_greeting() and nothing watched what the model actually said.

    Same three forms the greeting test already treats as the employment claim,
    so the runtime check and the artifact check cannot disagree about what the
    claim is. "on behalf of {org}" is untouched: none of from/with/at precede
    the org there.
    """
    if not text or not org:
        return False
    return bool(re.search(rf"\b(from|with|at)\s+{re.escape(org)}\b",
                          text, re.I))


_IDENTITY_ASK = re.compile(
    r"(who (is|are|am i|'s) (this|you|speaking|calling|i speaking)|"
    r"who am i (speaking|talking)|may i ask who|who gave you|"
    r"what company|which company|where are you calling from|"
    r"are you (a )?(robot|bot|ai|human|real))", re.I)


# Turns that MENTION the location without asking for it: acknowledging a value
# just given, or signing off. Everything else that names a location is a request,
# whatever shape it takes.
_NOT_AN_ASK = re.compile(
    r"\b(thanks|thank you|got it|perfect|great|appreciate|have a (good|great)|"
    r"take care|goodbye|bye now|i'?ll (note|record|pass)|i have that|"
    r"that'?s all|no (problem|worries))\b", re.I)

# Reading back a value the caller already gave.
_CONFIRMS_VALUE = re.compile(
    r"\b(i have that as|i'?ve got that|i'?ll note|noted as|recorded as|"
    r"i'?ll put (that|it) down|so that'?s)\b", re.I)

# Reporting that the location was NOT obtained. Names a location noun and reads
# as an ask to the inverted detector, but it is the opposite — it is the agent
# giving up. On call-20260818-1338 "I wasn't able to get the specific branch
# today" was counted as an ask, so a closing line spent a slot of the ask
# budget. Only checked on statements: "I couldn't find the branch — do you know
# it?" carries a question mark and is a genuine ask.
_REPORTS_FAILURE = re.compile(
    r"\b(was ?n'?t able|were ?n'?t able|was not able|could ?n'?t|could not|"
    r"can'?t|cannot|unable|did ?n'?t manage|no luck)\b", re.I)

_LOCATION_NOUN = re.compile(
    r"\b(branch|location|office|campus|site|address|practis\w*|practic\w*)\b", re.I)


# The model writes TYPOGRAPHIC apostrophes — "wasn’t", "it’s", "that’s" — and
# every pattern in this file spells them ASCII ("n'?t", "that'?s"). So the
# detectors were blind to the agent's own most common output. Found on
# call-20260818-1338: "I wasn’t able to get the specific branch today" was
# counted as a location ask because _REPORTS_FAILURE could not see the word
# "wasn’t". Ten patterns in this file contain an apostrophe.
_SMART_QUOTES = str.maketrans({"’": "'", "‘": "'",
                               "“": '"', "”": '"'})


def _norm_quotes(text: str) -> str:
    """ASCII-ise typographic quotes so the patterns can match what is said."""
    return (text or "").translate(_SMART_QUOTES)


# The agent telling the caller the location is recorded, or that the call is
# finished. Both are false the moment save_branch returns a rejection, and the
# second is worse: it invites them to hang up.
#
# Two families, because they fail differently. "I'll save that" is a claim
# about the tool; "we'll be all set" is a claim about the call. The model
# produced BOTH in one sentence on call-20260818-1613.
_CLAIMS_SAVED = re.compile(
    r"\b(i'?ll (save|note|record|log|put|get) (that|it|this|them)"
    r"|i'?ve (saved|noted|recorded|logged|got) (that|it|this)"
    r"|got (that|it) (saved|noted|recorded|down)"
    r"|that'?s (saved|noted|recorded|logged|in)"
    r"|i('| a)m saving (that|it)"
    r"|we'?(ll be|re) all set|we'?re (done|all done|set|good)"
    r"|that'?s (everything|us|it) (done|sorted)?"
    r"|all (set|sorted|done))\b", re.I)


def _claims_saved(text: str) -> bool:
    """Did this agent turn tell the caller the location is recorded, or done?"""
    return bool(_CLAIMS_SAVED.search(_norm_quotes(text or "")))


def _is_location_ask(text: str) -> bool:
    """Is this agent turn asking where the doctor practises?

    Counts statement-form asks as well as questions. A request phrased politely
    is still a request, and the person on the other end experiences it as one.

    This used to be a whitelist of phrasings requiring a question mark, and it
    scored 0 asks on a call that asked four times — the agent had simply picked
    wordings that were not on the list ("trying to confirm" where the list held
    "trying to find out"). Enumerating phrasings cannot work: the model has more
    ways to ask than anyone can list.

    So it is inverted. Naming a location IS an ask unless the turn is plainly
    acknowledging or closing. This over-counts a little, which is the safe
    direction for a budget whose purpose is to stop the agent pestering people.
    """
    text = _norm_quotes(text)
    if not _LOCATION_NOUN.search(text):
        return False
    # Reading a value back is not asking for one.
    if "?" not in text and (_CONFIRMS_VALUE.search(text)
                            or _REPORTS_FAILURE.search(text)):
        return False
    if "?" in text:
        return True
    # An acknowledgement that goes on to ask for something is still an ask, so
    # only a turn that is ENTIRELY acknowledgement is exempt.
    stripped = _NOT_AN_ASK.sub("", text)
    return bool(_LOCATION_NOUN.search(stripped))


def _ask_budget_outcome(turns: list, sent_at: Optional[int],
                        sent: bool, escalated: bool) -> dict:
    """What happened after the give-up directive was injected.

    The count alone is not enough. Thanking them and escalating in one turn is
    the directive working. Taking two turns where the first contains another
    question is the directive landing but not taking effect — a soft version of
    the model ignoring it outright, and worth telling apart, because the fix
    differs: a wording tweak versus enforcing the budget at the response level
    instead of asking nicely in a user turn.
    """
    if not sent or sent_at is None:
        return {"limit": settings.realtime_max_location_asks,
                "directive_sent": False, "verdict": "not needed"}

    after = [t for t in turns[sent_at:] if t.role == "agent"]
    asked_again = sum(1 for t in after if _is_location_ask(t.text))

    if not escalated:
        verdict = "IGNORED — directive sent, agent never escalated"
    elif asked_again:
        verdict = f"OBEYED LATE — asked {asked_again} more time(s) first"
    elif len(after) <= 1:
        verdict = "OBEYED — closed on the next turn"
    else:
        verdict = f"OBEYED — took {len(after)} turns, no further asks"

    return {
        "limit": settings.realtime_max_location_asks,
        "directive_sent": True,
        "agent_turns_after": len(after),
        "asked_again_after": asked_again,
        "escalated": escalated,
        "verdict": verdict,
        "turns_after": [t.text[:90] for t in after],
    }


# Abbreviations whose full stop does not end a sentence. Without this, "Which
# branch is Dr. Okafor at?" splits into "Which branch is Dr." + "Okafor at?",
# which reads as a statement-request followed by a question — so the double-ask
# detector fired on nearly every turn, since almost every turn names the doctor.
_ABBREV = re.compile(
    r"\b(Dr|Mr|Mrs|Ms|Prof|St|Ave|Blvd|Rd|Ste|Dept|Inc|Co|approx|no)\.\s",
    re.I)
# A visible sentinel. The empty string is wrong (replacing "" inserts
# the replacement between every character) and a control byte is worse:
# invisible in source, and a literal 0x08 has landed in this file twice.
_ABBREV_MARK = "@@DOT@@"


def _sentences(text: str) -> list:
    """Split into sentences without treating "Dr." as the end of one."""
    protected = _ABBREV.sub(lambda m: m.group(0).replace(".", _ABBREV_MARK), text)
    parts = re.split(r"(?<=[.!?])\s+", protected.strip())
    return [p.replace(_ABBREV_MARK, ".").strip() for p in parts if p.strip()]


def _clauses(text: str) -> list:
    """Sentences, split again at dashes, semicolons and colons.

    The repeat detector counted SENTENCES and reported 0 for a call containing
    a 45-character exact repeat. call-20260818-1613:

        turn 1: "...about a doctor listing — which branch is Dr. Okafor
                 working out of?"
        turn 3: "I can hear you now — which branch is Dr. Okafor working
                 out of?"

    Neither turn has an internal sentence break, so each was one "sentence",
    the two differed, and nothing was counted. But the repeated part is the
    clause after the dash — and that is not a coincidence of this call. The
    prompt's own turn shape is "React, THEN say the thing, folded into ONE
    sentence", which produces exactly `reaction — ask`. The ask is therefore
    the unit that gets repeated, and it almost never sits at a sentence
    boundary.

    A metric that reports a clean number for a dirty call is worse than no
    metric: repeated_sentences is one of the figures used to compare calls, so
    every comparison drawn from it was weaker than it looked.
    """
    out = []
    for s in _sentences(text):
        for part in re.split(r"\s*[—–-]{1,2}\s+|\s*[;:]\s+", s):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _norm_clause(text: str) -> str:
    """Normalised form for equality: case, quotes, whitespace, edge punctuation.

    "...working out of?" and "...working out of." are the same thing said
    twice, and a detector that treats them as different is measuring
    punctuation rather than repetition.
    """
    t = _norm_quotes(text).lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t.strip(".,!?;:—–- ")


# Asking for time to go and look something up. Matched by shape — a first-person
# or please-wait construction plus a checking/waiting word — rather than a list
# of phrasings, because "ways to ask for a minute" is an open set.
#
# An imperfect match is safe in one direction only, which is why a heuristic is
# acceptable here: a false positive delays the give-up by a turn or two, while a
# false negative just restores the behaviour we already have.
_HOLD_REQUEST = re.compile(
    r"\b(?:(?:let me|i'?ll|i will|i need to|i have to|i'?m going to|gonna)\s+"
    r"(?:just\s+)?(?:check|look|see|find|ask|grab|pull)"
    r"|(?:give|gimme)\s+me\s+a\s+(?:minute|moment|sec|second)"
    r"|(?:can|could|would)\s+you\s+(?:just\s+|please\s+)*(?:wait|hold|hang on)"
    r"|(?:hold on|hang on|one moment|just a (?:minute|moment|sec|second)"
    r"|bear with me|one sec))\b", re.I)


def is_hold_request(text: str) -> bool:
    """Is the caller asking for time to go and find the answer?

    This is the opposite of refusing. A live call ended because the give-up
    directive had already fired, and the caller's very next words were "can you
    please give me a minute? I just need to check" — the most cooperative thing
    said on that call. The agent thanked them and hung up while they were on
    their way to look it up.
    """
    return bool(_HOLD_REQUEST.search(text or ""))


def _content_words(text: str) -> set:
    """Words for comparing one caller turn against another.

    Deliberately NOT _UNGROUNDED_STOPWORDS: that list drops street, campus,
    branch and centre because they are not evidence of a specific place. Here
    they are exactly the signal — two turns that both say "Street" and
    "California" are the same answer. Only very short function words go.
    """
    return {w for w in re.findall(r"[a-z']+", (text or "").lower()) if len(w) > 1}


def caller_repeated_answer(text: str, sess: "RealtimeSession") -> str:
    """Has the caller now given substantially the same answer twice?

    A person who repeats themselves is telling you that is all they have. On a
    live call:

        CALLER: "He is working in Lombard Street in California."
        AGENT : "which city is that Lambert Street location in?"
        CALLER: "He is working in Lambert Street in California."
        AGENT : "which city is that Lambert Street site in?"

    A street and a state is a location — it is exactly what the validator asks
    for. The call ran 135 seconds, they answered twice, and save_branch was
    never called. Nothing was recorded.

    Compared by content-word overlap, so it survives the transcription drifting
    ("Lombard" -> "Lambert") and needs no vocabulary of its own. Returns the
    earlier wording, or "" if this is not a repeat.
    """
    # Only repeated ANSWERS count. "What do you want?" asked twice is a repeat
    # too, and nudging the agent to save it would be nonsense.
    if "?" in (text or ""):
        return ""
    now = _content_words(text)
    if len(now) < 4:          # "hello", "yes" — too short to mean anything
        return ""
    for turn in reversed(sess.turns):
        if turn.role != "caller" or not turn.text or turn.text == "[...]":
            continue
        if "?" in turn.text:
            continue
        prev = _content_words(turn.text)
        if len(prev) < 4:
            continue
        overlap = len(now & prev) / max(len(now | prev), 1)
        if overlap >= 0.7:
            return turn.text
    return ""


def _double_ask(text: str) -> bool:
    """Two requests for the same thing inside one turn.

    Counted by requests, not by question marks. "I need the specific branch name
    or street address where Dr. Okafor sees patients. Which one is it?" carries
    one "?" and asks twice — a statement-form request followed by a question.
    The trailing question is also vaguer than the statement it repeats, so it
    reads as asking the caller to choose between the options just listed.
    """
    parts = _sentences(text)
    # "Which one is it?" names nothing, so it is not an ask by content — it is an
    # ask by context, pointing back at the request before it. That is precisely
    # what makes it vague. So the shape to catch is a statement-form request
    # followed by a separate question, not two recognisable location asks.
    statement_asks = [p for p in parts if "?" not in p and _is_location_ask(p)]
    questions      = [p for p in parts if "?" in p]
    if statement_asks and questions:
        return True
    return sum(1 for p in questions if _is_location_ask(p)) > 1


def conversation_metrics(turns: list) -> dict:
    """Count the conversational failures that prose rules keep failing to stop.

    Three attempts at fixing the same behaviour by writing more forceful
    instructions, each one ignored, is evidence the marginal rule is doing
    less each time. These are constraints on the SHAPE of a turn rather than
    its content, which makes them detectable in code even though the prompt
    cannot reliably enforce them.

    Nothing here changes behaviour — you cannot unsay a turn. The point is to
    have a number, so the next prompt edit can be evaluated against the last
    one instead of by reading a transcript and forming an impression.

      stapled_questions  — agent asked a question in the same turn it answered
                           one of theirs. Six of these in one 111s call.
      back_to_back_asks  — agent asked in two consecutive turns.
      repeated_sentences — same agent sentence said more than once.
    """
    agent = [t for t in turns if t.role == "agent"]
    stapled = back_to_back = 0
    prev_agent_asked = False

    for i, turn in enumerate(turns):
        if turn.role != "agent":
            continue
        asks = "?" in turn.text
        prev_caller = next((turns[j] for j in range(i - 1, -1, -1)
                            if turns[j].role == "caller"), None)
        # Did the caller's most recent turn, immediately before this one, ask
        # something? Then answering AND asking in one breath is the failure.
        if asks and prev_caller is not None and i > 0 and turns[i - 1] is prev_caller \
                and "?" in prev_caller.text:
            stapled += 1
        if asks and prev_agent_asked:
            back_to_back += 1
        prev_agent_asked = asks

    seen: dict[str, int] = {}
    for t in agent:
        for sentence in _sentences(t.text):
            key = _norm_clause(sentence)
            if len(key.split()) >= 4:
                seen[key] = seen.get(key, 0) + 1

    # Sentence-level repeats first, then clause-level ones that are NOT already
    # inside a sentence counted above. Saying one sentence twice is ONE
    # repetition, not one for the sentence plus one for each of its clauses.
    #
    # Counting clauses INSTEAD of sentences was the first attempt and it lost
    # repeats: a five-word sentence splits into two sub-threshold clauses and
    # vanishes. Checked against the whole call history — that swap silently
    # dropped a real repeat on call-20260806-2029 while fixing three others.
    # Both levels, largest unit wins.
    #
    # NOTE: values are NOT comparable with calls analysed before 2026-08-18.
    # The old figure counted sentences only and was structurally too low, so a
    # rise across that date is the metric being fixed, not the agent worsening.
    repeated = sum(n - 1 for n in seen.values() if n > 1)
    _covered = {k for k, n in seen.items() if n > 1}
    clause_seen: dict = {}
    for t in agent:
        for sentence in _sentences(t.text):
            if _norm_clause(sentence) in _covered:
                continue    # already counted as a whole-sentence repeat
            for clause in _clauses(sentence):
                key = _norm_clause(clause)
                if len(key.split()) >= 4:
                    clause_seen[key] = clause_seen.get(key, 0) + 1
    repeated += sum(n - 1 for n in clause_seen.values() if n > 1)

    # Denominators, so counts can be compared across calls of different
    # difficulty. A hostile caller who answers nothing gives the agent six
    # chances to staple; a cooperative one gives it one. Raw counts make the
    # easy call look better when it may simply have had fewer opportunities.
    caller = [t for t in turns if t.role == "caller"]
    caller_questions = sum(1 for t in caller if "?" in t.text)
    return {
        # How many times it asked where the doctor practises. On the call that
        # exposed this it was six, with no location offered between any of
        # them — the number that says "it would not let go".
        "location_asks": sum(1 for t in agent if _is_location_ask(t.text)),
        # Two requests for the same fact inside one turn. The prompt's rule was
        # "EXACTLY ONE question mark per turn", which a statement-form request
        # followed by a question passes with one "?" — the same blind spot the
        # ask DETECTOR had. On a live call: "I need the specific branch name or
        # street address where Dr. Okafor sees patients. Which one is it?"
        "double_asks": sum(1 for t in agent if _double_ask(t.text)),
        # Moves stacked into one turn, counted as sentences and needing no
        # vocabulary at all — the banned-phrase list for thinking-narration
        # missed 2 of the 3 wordings actually used, because "ways to narrate"
        # is an open set. Sentence count is structural and cannot rot.
        # The greeting is excluded: it is a fixed line, not a pile-up, and it
        # would otherwise dominate the count.
        "piled_turns": sum(1 for t in agent[1:] if len(_sentences(t.text)) >= 3),
        "longest_turn_sentences": max((len(_sentences(t.text)) for t in agent[1:]),
                                      default=0),
        "longest_turn_words": max((len(t.text.split()) for t in agent[1:]), default=0),
        "agent_turns": len(agent),
        "caller_turns": len(caller),
        "caller_questions": caller_questions,
        "question_turns": sum(1 for t in agent if "?" in t.text),
        "stapled_questions": stapled,
        # The rate is the comparable figure: of the times they asked something,
        # how often did the agent answer and ask back in the same breath?
        "staple_rate": round(stapled / caller_questions, 2) if caller_questions else None,
        "back_to_back_asks": back_to_back,
        "repeated_sentences": repeated,
    }


# Escalation reasons that assert a FACT about the doctor rather than describing
# how the call went. "declined to share" is an observation about the call and
# needs no evidence; "doctor deceased" is a claim about a real person and does.
_FACTUAL_ESCALATIONS = {
    "deceased": ("deceased", "died", "passed away", "passed", "late "),
    "retired":  ("retired", "retirement"),
    "left":     ("left", "no longer", "moved on", "resigned", "quit"),
    "relocated": ("relocated", "transferred", "moved to"),
    "on leave": ("on leave", "maternity", "sabbatical", "sick leave"),
}


def _ungrounded_escalation(reason: str, sess: "RealtimeSession") -> str:
    """Reject an escalation reason asserting something the caller never said.

    A live call ended with escalate(reason="doctor deceased") after the caller
    said only "actually, he's not working right now". Nobody said died, passed
    away, or deceased. save_branch was guarded against exactly this and
    escalate was not — so a fabricated claim about a named real person went
    into the record, where a reviewer would read it as fact.

    Only claims ABOUT THE DOCTOR are checked. Reasons describing the call
    itself ("declined to share", "wrong number", "no response") are the agent's
    own observation and need no corroboration.
    """
    heard = " ".join(t.text.lower() for t in sess.turns
                     if t.role == "caller" and t.text.strip() != "[...]")
    if not heard.strip():
        return ""
    low = reason.lower()
    for claim, markers in _FACTUAL_ESCALATIONS.items():
        if claim in low and not any(m in heard for m in markers):
            return (f"reason {reason!r} states the doctor is {claim}, which "
                    f"nobody said on this call")
    return ""


# ── The inverse guard: an answer the caller GAVE and the call threw away ─────
#
# Everything else here blocks false positives — saving a location the caller
# never said. On call-20260818-1112 the system failed the other way and nothing
# noticed. The caller said "office Abadan branch" on their second turn; the
# model called save_branch("Northside Branch"), reshaped from the hospital name
# in its own context; the grounding guard correctly rejected it; the ask budget
# correctly ran out; and the call escalated with
# reason="caller engaged but never provided a location".
#
# Every guard did its job and the reason is false. It is now in the record as
# fact, and a reviewer reading it has no way to tell.
#
# That asymmetry is the expensive one for a data-collection product. A resolved
# call that should not have resolved shows up as a wrong row someone can find.
# A real answer discarded shows up as nothing at all — indistinguishable from a
# receptionist who genuinely would not say.

# Words that anchor a location. A distinctive word sitting next to one of these
# is a candidate place name; the same word anywhere else is just a word. The
# adjacency requirement is what keeps this from firing on every proper noun in
# the call — "Hello, David" has no anchor near it.
_LOCATION_ANCHORS = frozenset({
    "branch", "branches", "campus", "campuses", "clinic", "clinics",
    "office", "offices", "center", "centre", "centers", "centres",
    "hospital", "hospitals", "location", "locations", "site", "sites",
    "building", "tower", "wing", "block", "street", "road", "avenue",
    "boulevard", "lane", "drive", "parkway", "suite", "floor", "area",
})

# Conversational words that will happily sit next to an anchor while naming no
# place at all: "the main branch", "our other office", "which location".
_NON_PLACE = frozenset({
    "main", "other", "another", "same", "this", "that", "these", "those",
    "our", "their", "his", "her", "its", "one", "two", "both", "all", "any",
    "some", "each", "every", "which", "what", "where", "when", "who", "why",
    "here", "there", "yes", "yeah", "yep", "no", "not", "but", "for", "with",
    "from", "about", "only", "just", "also", "still", "sorry", "please",
    "thanks", "thank", "hello", "hey", "okay", "sure", "right", "well",
    "you", "your", "yours", "we", "our", "they", "them", "him", "she", "he",
    "are", "was", "were", "have", "has", "had", "does", "did", "can",
    "could", "will", "would", "should", "need", "needs", "want", "know",
    "tell", "say", "said", "give", "gave", "get", "got", "see", "sees",
    "working", "works", "work", "patients", "patient", "doctor", "doctors",
    "emergency", "call", "calling", "called", "number", "details", "detail",
    "information", "anything", "something", "nothing", "everything",
    "speaking", "moment", "minute", "second", "wait", "hold", "checking",
    # Capitalisation is doing the heavy lifting, so this list only has to
    # cover words that survive it — sentence-initial ones, where every word is
    # capitalised whatever it is.
    "closed", "open", "sorry", "sure", "try", "let", "hang", "just", "look",
    "there's", "thats", "yeah", "well", "actually", "maybe", "probably",
})


def _candidate_location(sess: "RealtimeSession") -> str:
    """A place the CALLER named that was never saved. Empty if there is none.

    The mirror image of _ungrounded_terms: that one asks "did the caller say
    this?", this one asks "did the caller say ANYTHING, when we are about to
    record that they said nothing?".

    Deliberately conservative — it gates an escalation, and a detector that
    fires on ordinary conversation would trap the agent on a call it cannot
    end. A word counts only if it is distinctive (not a stopword, not a filler,
    not already on our own record), it sits within two words of a location
    anchor, and it survives the same hint-echo test a saved branch has to pass.
    """
    usable = [t for t in sess.turns
              if t.role == "caller" and t.text.strip() != "[...]"]
    if not usable:
        return ""

    # Words we already had before the call started cannot be an answer FROM the
    # call. The hospital on record is the whole reason the fabrication happened
    # — the model reshaped it into a branch name — so hearing it echoed back is
    # not the caller naming a site.
    known: set[str] = set()
    known |= _distinctive(getattr(sess.doctor, "hospital_name", "") or "")
    known |= _distinctive(sess.org_name or "")
    known |= {w for w in re.findall(r"[a-z]+", (sess.doctor.doctor_name or "").lower())
              if len(w) > 2}
    if sess.agent_name:
        known.add(sess.agent_name.lower())

    for t in usable:
        raw = [w.strip(".,!?-—'\"") for w in t.text.split()]
        words = [w.lower() for w in raw]
        # A place name is a PROPER NOUN, and the transcriber capitalises it.
        # That is the strongest signal available and lowercasing throws it
        # away: without it "the office is closed" and "hospital, how can I
        # help" both read as candidates, because "closed" and "how" are simply
        # words no stoplist thought to name. Enumerating English is not a
        # strategy. Capitalisation cuts the space in one move.
        #
        # It is a CONJUNCTION with the stoplists, never a replacement —
        # sentence-initial words are capitalised regardless of what they are,
        # which is what the stoplists are still for.
        #
        # If a turn came back with no case information at all (all lower, all
        # upper), capitalisation says nothing about any word in it, so fall
        # back to the stoplists alone rather than silently detecting nothing.
        # Same rule the grounding check follows: absence of a signal is not
        # evidence, and a degraded transcript must not quietly disable a guard.
        cased = t.text != t.text.lower() and t.text != t.text.upper()
        for i, w in enumerate(words):
            if len(w) <= 2 or not w.isalpha():
                continue
            if (w in _UNGROUNDED_STOPWORDS or w in _NON_PLACE
                    or w in _ORG_STOPWORDS or w in known):
                continue
            if cased and not raw[i][:1].isupper():
                continue
            near = words[max(0, i - 2):i] + words[i + 1:i + 3]
            if not any(n in _LOCATION_ANCHORS for n in near):
                continue
            # Same defence a real save has to clear: a bare term on dead air is
            # the transcriber echoing its own hint, not the caller speaking.
            if _is_hint_echo(t, [w], _caller_speech_level(sess)):
                continue
            return f"{raw[i]!r} — they said: {t.text.strip()!r}"
    return ""


# Escalation reasons that ASSERT no location was given. Only these are checked:
# "wrong number", "voicemail", "declined to share" describe the call and are the
# agent's own observation, so a stray place name in the transcript says nothing
# about whether they are true.
_NO_LOCATION_CLAIMS = (
    "never provided a location", "never gave a location", "no location",
    "did not provide", "didn't provide", "never provided", "never gave",
    "does not know", "doesn't know", "did not know", "could not obtain",
    "couldn't obtain", "unable to obtain", "never answered",
)


def _discarded_location(reason: str, sess: "RealtimeSession") -> str:
    """Block an escalation claiming nothing was given when something was.

    Returns a rejection description, or "" to allow the escalation.
    """
    if not any(m in reason.lower() for m in _NO_LOCATION_CLAIMS):
        return ""
    return _candidate_location(sess)


async def _create_response(oai_ws, sess: "RealtimeSession", *, why: str,
                           allow_when_done: bool = False,
                           allow_when_active: bool = False) -> bool:
    """The one place `response.create` is sent. Returns True if it was.

    There are six call sites and each carried its own guard conditions. Two
    shipped without checking `_response_active` and both produced dead air on
    live calls: 97ff46d fixed the silence watchdog, and the empty-response
    re-request was fixed on 2026-08-11 after a rejected response was read as
    dead air, prompting another that collided and failed in turn. That is one
    missing abstraction, not two bugs — guard logic duplicated per call site
    cannot be made correct by review, and the seventh site would have had the
    same coin-flip.

    THE SITES DO NOT SHARE ONE POLICY. A helper that simply refused when
    `sess.done` would silently kill the goodbye and the goodbye retry, which
    fire *because* the call is done — reintroducing the exact silent no-op this
    exists to prevent. So the policy is declared per site rather than assumed:

      default                  in-flight? refuse.  call over? refuse.
      allow_when_done=True     the closing goodbye and its retry
      allow_when_active=True   the goodbye, which is sent from inside the
                               tool-call handler while that response is still
                               open (see its call site — this one is load-
                               bearing, not caution)

    `why` is logged on refusal. A guard that silently does nothing looks
    exactly like a guard that works, and this module has been bitten by that
    three times.
    """
    if sess._response_active and not allow_when_active:
        log.info("[Realtime] response.create skipped (%s): one already in flight", why)
        return False
    if sess.done and not allow_when_done:
        log.info("[Realtime] response.create skipped (%s): call is closing", why)
        return False
    await oai_ws.send(json.dumps({"type": "response.create"}))
    return True


def _echo_gate_allows(raw: bytes) -> bool:
    """Should this caller frame reach OpenAI while the agent is speaking?

    Governs whether the caller can interrupt at all. See REALTIME_ECHO_GATE.
    """
    mode = settings.realtime_echo_gate
    if mode == "pass":
        return True
    if mode == "energy":
        arr = _wire_to_pcm16(raw)
        if arr.size == 0:
            return False
        return float(np.sqrt(np.mean(arr ** 2))) >= settings.realtime_echo_rms
    return False   # "drop"


# ── Tool schema conversion ────────────────────────────────────────────────────

def _realtime_tools() -> list[dict]:
    """Convert TOOL_SCHEMAS (chat format) → OpenAI Realtime flat format."""
    result = []
    for s in TOOL_SCHEMAS:
        if s["type"] == "function":
            fn = s["function"]
            result.append({
                "type": "function",
                "name": fn["name"],
                "description": fn["description"],
                "parameters": fn["parameters"],
            })
    return result


# ── Session audio configuration ───────────────────────────────────────────────

def build_audio_config(*, transcribe_model: str, transcribe_hint: str,
                       audio_format: str, noise_reduction: str,
                       turn_detection: str, eagerness: str,
                       voice: str, silence_ms: int = 500,
                       interrupt_response: bool = True) -> dict:
    """Assemble the session.update `audio` block.

    Split out so check_realtime.py can probe variants against the live API
    without duplicating the shape — the settings below are empirical questions,
    not things to settle by reading.

    ``interrupt_response`` was never sent, so it ran on the API default and
    nobody had decided it. It is declared now at the value that default was
    (True), so this is not a behaviour change — it is the same behaviour,
    written down and probeable. True means OpenAI cancels an in-flight response
    when it hears the caller, in ADDITION to this module's own barge-in
    handler; the two race, and response.done logs which one won.
    """
    fmt: dict = ({"type": "audio/pcmu"} if audio_format == "pcmu"
                 else {"type": "audio/pcm", "rate": _OAI_SR})

    if turn_detection == "semantic_vad":
        td: dict = {"type": "semantic_vad", "eagerness": eagerness,
                    "interrupt_response": interrupt_response}
    else:
        td = {
            "type": "server_vad",
            "threshold": 0.55,
            "prefix_padding_ms": 300,
            "silence_duration_ms": silence_ms,
            "interrupt_response": interrupt_response,
        }

    audio_in: dict = {
        "format": fmt,
        "transcription": {
            "model": transcribe_model,
            "language": "en",
            "prompt": transcribe_hint,
        },
        "turn_detection": td,
    }
    if noise_reduction and noise_reduction != "off":
        audio_in["noise_reduction"] = {"type": noise_reduction}

    return {"input": audio_in, "output": {"format": fmt, "voice": voice}}


# ── Grounding: a saved location must be one the caller actually said ─────────

# Words that carry no identifying information, so their presence in the
# transcript proves nothing about whether the caller named a real place.
_UNGROUNDED_STOPWORDS = {
    "the", "a", "an", "of", "at", "in", "on", "our", "their", "and",
    "branch", "branches", "office", "offices", "campus", "campuses",
    "clinic", "clinics", "center", "centre", "centers", "centres",
    "hospital", "location", "locations", "site", "sites", "medical",
    "building", "unit", "practice", "city", "street", "road", "avenue",
}


# Words that appear in almost every healthcare organisation's name. Matching on
# these would make "Methodist Medical Center" look like "Northside Medical
# Group", which is exactly the confusion this check exists to catch.
_ORG_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "for", "at", "st", "saint",
    "hospital", "hospitals", "clinic", "clinics", "medical", "medicine",
    "health", "healthcare", "center", "centre", "group", "practice",
    "associates", "physicians", "care", "services", "system", "systems",
    "institute", "department", "dept", "office", "offices", "campus",
})

# "thank you for calling X" / "you've reached X" — X can only be the place.
_SELF_ID = re.compile(
    r"(?:thank(?:s| you) for calling|you'?ve reached|you have reached|"
    r"welcome to)\s+(.{3,60}?)(?:[,.!?]|$)", re.I)

# "this is X" is how people give their OWN NAME — "Northside, this is Amy." So
# it only counts as naming the organisation when the phrase carries an
# organisational word. Without this, Amy reads as a rival hospital.
_SELF_ID_WEAK = re.compile(r"this is\s+(.{3,60}?)(?:[,.!?]|$)", re.I)
_ORG_WORD = re.compile(
    r"\b(hospital|clinic|medical|health|centre|center|group|practice|"
    r"associates|physicians|institute|system)\b", re.I)


def _distinctive(name: str) -> set:
    """The tokens in an organisation name that actually identify it."""
    return {w for w in re.findall(r"[a-z]+", (name or "").lower())
            if w not in _ORG_STOPWORDS and len(w) > 2}


def hospital_mismatch(sess: "RealtimeSession") -> str:
    """The caller answered as a DIFFERENT organisation than the one on record.

    A branch saved against the wrong hospital is corrupt data, and it is the one
    failure the grounding guard cannot see: every word can be genuinely quoted
    from the caller and the record still ends up wrong, because the call reached
    the wrong place.

    On a live call the record said "Northside Medical Group" and the caller
    answered "Thank you for calling the Methodist Medical Center." Nothing
    noticed, and the agent went on to invent an address for it.

    Fires only on a POSITIVE mismatch — a recognisable different name in an
    answering phrase. Silence is the norm, not a signal: most people answer
    without naming the place, and treating that as suspicion would block almost
    every call. Empty string means no conflict found.
    """
    recorded = getattr(getattr(sess, "doctor", None), "hospital_name", "") or ""
    on_record = _distinctive(recorded)
    if not on_record:
        return ""
    for turn in sess.turns:
        if turn.role != "caller" or not turn.text:
            continue
        # If the recorded name appears anywhere in this turn, they are the right
        # place however else they phrase it. "Northside, this is Amy."
        if on_record & _distinctive(turn.text):
            continue
        claims = list(_SELF_ID.findall(turn.text))
        claims += [c for c in _SELF_ID_WEAK.findall(turn.text) if _ORG_WORD.search(c)]
        for claimed in claims:
            said = _distinctive(claimed)
            # Overlap of even one distinctive token means the same place under a
            # slightly different name — "Northside Medical Center" vs "Group".
            if said and not (said & on_record):
                return (f"caller answered as {claimed.strip()!r}, but this call "
                        f"is recorded against {recorded!r}")
    return ""


def _ungrounded_terms(args: dict, sess: "RealtimeSession") -> str:
    """Return a description of any branch/city term the caller never said.

    Empty string means everything checks out. Compares against the caller's
    own transcribed words only — the agent's words are excluded, or the model
    could ground a fabrication in its own earlier hallucination.

    If no caller speech was transcribed at all (every turn still a `[...]`
    placeholder) the check is skipped rather than blocking every save, since
    absence of transcript is not evidence of fabrication.
    """
    # A caller turn is not the caller — it is a model's guess at the caller, and
    # the transcription hint is a prompt to that model. On call-20260813-1409
    # "Yes, speaking" (the second phrase in the hint's old "Likely phrases"
    # list) came back four times. The hint still names health systems and
    # location words, because those are what make REAL answers transcribe
    # correctly — which means the vocabulary that constitutes a valid branch
    # answer is exactly the vocabulary that can be echoed. If that echo lands in
    # a caller turn it becomes grounding evidence, and a fabricated location
    # gets written to the directory looking verified. That is worse than any
    # wasted turn: the check built to stop fabrication would be certifying it.
    #
    # Two independent signals have to fail before a turn is discounted, because
    # neither is sufficient alone:
    #
    #   1. The turn is EXACTLY the term and nothing else. A hint echo arrives
    #      bare; a real answer usually comes with surrounding words ("she's at
    #      the Mercy campus", "that'd be north campus I think"). But
    #      "Northgate" on its own is a perfectly good answer, so this cannot
    #      stand alone.
    #   2. The audio carried no real signal. Loudest-300ms window, never the
    #      mean — the mean is dominated by the gaps between words and once told
    #      an audible caller they were faint.
    #
    # A bare one-word answer on strong audio still grounds. A bare one-word
    # answer on near-silence does not.
    _usable = []
    for t in sess.turns:
        if t.role != "caller" or t.text.strip() == "[...]":
            continue
        _usable.append(t)
    heard = " ".join(t.text.lower() for t in _usable)
    if not heard.strip():
        return ""   # nothing transcribed — cannot judge, do not block

    missing = []
    for field in ("branch", "city"):
        value = (args.get(field) or "").strip()
        if not value:
            continue
        terms = [w.strip(".,!?-—'\"") for w in value.lower().split()]
        content = [w for w in terms if w and w not in _UNGROUNDED_STOPWORDS]
        if not content:
            continue
        # One content word appearing is enough — transcription is imperfect and
        # we would rather let a real answer through than block it.
        if not any(w in heard for w in content):
            missing.append(f"{field}={value!r}")
            continue
        # It appears. Check it did not appear ONLY as a bare echo on dead air.
        _support = [t for t in _usable
                    if any(w in t.text.lower() for w in content)]
        _level = _caller_speech_level(sess)
        if _support and all(_is_hint_echo(t, content, _level) for t in _support):
            missing.append(
                f"{field}={value!r} (only heard as a bare term on silent audio)")
    return " and ".join(missing)


# A caller turn is "quiet" relative to how loudly THIS caller has been
# speaking, not against a fixed number. _LOW_AUDIO_RMS alone is an absolute
# threshold on a quantity that has no absolute meaning: line gain, handset,
# carrier and distance all move it, so one constant cannot be right for two
# different calls.
#
# Measured on call-20260818-1338, where the transcriber emitted "Mercy Medical
# Center" — a phrase assembled from _US_TRANSCRIBE_HINT, which names Mercy
# first among health systems and "medical center" among location words. The
# caller never said it:
#
#     real  "why are you collecting"    0.0954
#     real  "Los Angeles, California"   0.1532
#     real  "It is Los Angeles only."   0.0465
#     FAKE  "Mercy Medical Center."     0.0174     <- cleared _LOW_AUDIO_RMS (0.015)
#
# The hallucination sat just above the constant while being a quarter of this
# caller's own median level. Every fraction from 0.25 to 0.50 separates the
# four cleanly; 0.35 is the middle of that band. Checked against
# call-20260818-1112, where all four caller turns are believed genuine: none
# is flagged.
#
# RE-DERIVED 2026-08-18 against the Twilio recordings, after the accusation
# this was built on turned out to be false and after audio_rms itself was found
# to be under-reporting. Method: for each of 30 calls with a dual-channel
# recording, take the N loudest caller-channel bursts where N is the number of
# transcribed caller turns, and compute min/median over them — i.e. how quiet a
# GENUINE turn gets relative to that caller's own typical level.
#
#     lowest 0.291   p10 0.458   p25 0.662   median 0.766
#     calls with a genuine turn below median*0.35 :  2/30
#     calls with a genuine turn below median*0.20 :  0/30
#
# 0.35 was too aggressive: on ~7% of calls it would classify a real caller turn
# as quiet, and a bare one-word branch name is exactly the shape that then gets
# rejected — "'Northgate' on its own is a perfectly good answer".
#
# BE CLEAR ABOUT WHAT THIS NOW BUYS. The case it was written for (the "Mercy
# Medical Center" turn) was retracted — that audio is real. With no confirmed
# positive case and a safe calibration, the adaptive term only acts on turns
# between the absolute floor and median*0.20, which is a narrow band. It is
# kept because the reasoning still holds — an absolute constant on a
# level-dependent quantity cannot be right for two different lines — not
# because it is known to catch anything. Do not widen it without a confirmed
# fabrication to widen it against.
_QUIET_FRACTION = 0.20

# Below this many measured turns the median is not a median. One turn's
# "median" is itself, which can never be a fraction of itself, so the adaptive
# test would silently never fire — the failure mode this file keeps relearning.
_MIN_TURNS_FOR_ADAPTIVE = 3


def _caller_speech_level(sess: "RealtimeSession") -> Optional[float]:
    """This caller's typical loudest-300ms level, or None if not yet knowable.

    Median, not mean: it survives one hallucinated near-silent turn among
    several real ones, which is precisely the population being judged.
    """
    vals: list[float] = []
    for t in sess.turns:
        if t.role != "caller" or t.text.strip() == "[...]":
            continue
        r = getattr(t, "audio_rms", None)
        if r is not None:
            vals.append(float(r))
    if len(vals) < _MIN_TURNS_FOR_ADAPTIVE:
        return None
    return float(median(vals))


def _is_hint_echo(turn, content_words: list, speech_level: Optional[float] = None) -> bool:
    """True if this caller turn looks like the transcriber echoing its own hint.

    Both signals must fail: the turn is nothing but the term, AND the audio it
    came from carried no real signal. See _ungrounded_terms for why neither is
    sufficient on its own. An unmeasured turn (audio_rms None) is given the
    benefit of the doubt — absence of measurement is not evidence of
    fabrication, the same rule the transcript check already follows.
    """
    rms = getattr(turn, "audio_rms", None)
    # Absolute floor OR a fraction of this caller's own level, whichever is
    # higher. The absolute alone let a hallucination through at 0.0174; the
    # relative alone would collapse toward zero on a call where every turn is
    # quiet, since a fraction of nothing is nothing.
    quiet_below = _LOW_AUDIO_RMS
    if speech_level is not None:
        quiet_below = max(quiet_below, speech_level * _QUIET_FRACTION)
    if rms is None or rms >= quiet_below:
        return False
    bare = [w.strip(".,!?-—'\"") for w in turn.text.lower().split()]
    bare = [w for w in bare if w and w not in _UNGROUNDED_STOPWORDS]
    return bool(bare) and set(bare) <= set(content_words)


# ── Per-call session ──────────────────────────────────────────────────────────

class RealtimeSession:
    def __init__(self, call_sid: str, doctor: Doctor):
        ts               = datetime.now().strftime("%Y%m%d-%H%M")
        self.call_id     = f"call-{ts}-{call_sid[-4:]}"
        self.call_sid    = call_sid
        self.doctor      = doctor
        self.start_dt    = datetime.now()
        self.memory      = CallMemory(call_id=self.call_id)
        self.memory.clear()
        self.memory.update(doctor=doctor.doctor_name, hospital=doctor.hospital_name)
        self.turns: list[TranscriptTurn] = []
        self.stream_sid  = ""
        self.done        = False
        # Gate: don't forward caller audio until the greeting finishes playing
        self.listen_enabled = asyncio.Event()
        # Gate: don't forward caller audio while the agent is speaking, plus an
        # echo cooldown after. Frames arriving in this window are dropped.
        self.agent_speaking = False

        # Agent PCM: list of (time_offset_from_stream_start_seconds, pcm16_bytes)
        # Timestamps let us place agent audio at the right position on the timeline
        self._agent_pcm: list[tuple[float, bytes]] = []
        # Caller PCM: continuous stream from Twilio — already timeline-aligned
        self._caller_pcm: list[bytes] = []
        # When the Twilio stream started (set on "start" event)
        self._stream_start_time: Optional[datetime] = None
        # Set when response.create for the greeting is sent; cleared once the
        # first audio delta arrives, so we measure the callee's dead air.
        self._greeting_requested_at: Optional[float] = None
        # Warn the model about a faint line at most once per call.
        self._low_audio_warned: bool = False
        self._repeat_nudged: bool = False
        # When the agent last stopped talking, and how many times we have
        # prompted a silent callee. Both greetings now end on a statement rather
        # than a question, which is the right shape — it hands the turn over —
        # but it means a callee who simply waits produces no speech, so server
        # VAD never fires and nothing creates a response. Without a watchdog the
        # call sits in silence until Twilio times it out.
        self._agent_quiet_since: Optional[float] = None
        # Budget for the WHOLE call, not per silence. Resetting it whenever the
        # caller spoke meant someone who says "hello?" and nothing else could be
        # prompted indefinitely — the cap held inside one silence run and not
        # across the call, which is not what a cap is for.
        #
        # Split by PHASE, because one shared budget was spent in the wrong
        # place. On call-20260813-1409 both prompts went at 25.6s and 34.8s,
        # before the callee had said a word; a genuine 40-second gap opened at
        # 55.4s and the watchdog had nothing left, so the line sat dead until
        # the caller happened to speak. Opening silence and mid-conversation
        # silence are different failures — a callee who has not spoken may not
        # have picked up properly, one who has gone quiet is thinking or
        # checking — and draining the second budget on the first is what left
        # two thirds of the call unprotected.
        #
        # Still bounded, which was the point of the original cap: worst case is
        # _MAX_SILENCE_PROMPTS per phase, and neither counter is ever reset.
        self._silence_prompts_opening: int = 0
        self._silence_prompts_midcall: int = 0
        self._response_active: bool = False
        # RMS of the last utterance, held until its transcript arrives. Low
        # energy alone never means "we cannot hear you" — only low energy with
        # nothing transcribed does.
        self._pending_low_rms: Optional[float] = None
        # Loudest-window RMS of the utterance currently awaiting transcription,
        # attached to the caller turn when its text arrives. Unlike
        # _pending_low_rms this is set for every utterance, not only faint ones.
        self._pending_utterance_rms: Optional[float] = None
        # How many VAD segments accumulated under the transcript now pending.
        # >1 means the VAD split the caller's turn, which is the condition that
        # used to lose the measurement.
        self._utterance_segments: int = 0
        # Asks for the location so far, and whether we have already told the
        # model to stop. See realtime_max_location_asks.
        self._location_asks: int = 0
        self._give_up_sent: bool = False
        # When the last location ask finished, so a re-ask fired seconds later
        # can be caught. See _MIN_REASK_GAP_S. Nudge at most once — a second
        # copy of the same directive is context the model has already ignored.
        self._last_location_ask_at: Optional[float] = None
        self._reask_nudged: bool = False
        # Turn index at the last location ask, so the next one can look at what
        # the caller said in between rather than at a clock. See
        # _caller_answered_since. -1 means NO ask has been made yet, which is
        # not the same as "an ask nobody answered": the greeting is the first
        # ask and has no predecessor to be unanswered, so it must always count.
        # Initialising this to 0 made the opener score as an unanswered re-ask
        # and the budget started a turn behind.
        self._last_ask_turn_idx: int = -1
        # Consecutive asks the caller never answered. Bounded by
        # _MAX_UNANSWERED_REASKS so a caller who only ever says "hello" cannot
        # keep the call alive forever.
        self._unanswered_reasks: int = 0
        # When the deferred goodbye retry is due, or None. Set by the
        # response.done handler, acted on by the watchdog — the handler must not
        # sleep, because sleeping there stops the event pump.
        self._goodbye_retry_at: Optional[float] = None
        # Interruption repair. When the agent was last truncated, and how much
        # of its turn the caller actually heard. The next caller turn is read
        # against these rather than classified on its words alone — "Hello"
        # after a 750ms cut is a repair signal, not filler.
        self._truncated_at: Optional[float] = None
        self._truncated_heard_ms: int = 0
        # One-shot, like every other injected directive here.
        self._repair_nudged: bool = False
        # Told the caller the branch was saved when the tool then rejected it.
        self._false_save_nudged: bool = False
        # Answered-identity nudge, also one-shot for the same reason.
        self._identity_nudged: bool = False
        # Said the same sentence twice in a row, one-shot for the same reason.
        self._self_repeat_nudged: bool = False
        # Spoken persona and client org, set once the template is resolved.
        # _oai_to_twilio needs both to spot a re-introduction, and deriving
        # them again there would let the detector and the greeting disagree
        # about who the agent claims to be.
        self.agent_name: str = ""
        self.org_name: str = ""
        self._reintro_nudged: bool = False
        # Blocked one escalation for discarding an answer the caller gave.
        # One-shot: see the call site for why a permanent block is worse than
        # the false record it prevents.
        self._discard_blocked: bool = False
        # Said it works FOR the client rather than on their behalf. Recorded as
        # well as nudged: a false employment claim was made to a real medical
        # office and that belongs in the call record, not only in the console.
        self._employment_claimed: bool = False
        # Turn index when the give-up directive was injected, so we can tell
        # afterwards whether the agent actually acted on it. The directive is
        # appended to the conversation and there is no second lever, so its
        # effectiveness has to be measured rather than assumed.
        self._give_up_at_turn: Optional[int] = None

        # Token usage tracking (from response.done events).
        # Cached tokens are counted SEPARATELY and billed at the cached rate —
        # they are the only direct evidence that prompt caching is working, so
        # they are tracked even though the totals include them.
        self._input_audio_tokens:        int = 0
        self._input_audio_cached_tokens: int = 0
        self._output_audio_tokens:       int = 0
        self._input_text_tokens:         int = 0
        self._input_text_cached_tokens:  int = 0
        self._output_text_tokens:        int = 0
        self._responses:                 int = 0

    def note_utterance_rms(self, rms: float) -> None:
        """Record one VAD segment's loudest-window RMS, keeping the loudest.

        Segments accumulate until a transcript consumes them, because one
        transcript can cover several VAD segments and the question this answers
        is "did a human speak during the audio under this transcript".
        """
        if rms is None or rms <= 0.0:
            return
        self._pending_utterance_rms = max(self._pending_utterance_rms or 0.0, rms)
        self._utterance_segments += 1

    def take_utterance_rms(self) -> tuple[Optional[float], int]:
        """Consume the accumulated measurement for the transcript that arrived."""
        rms, n = self._pending_utterance_rms, self._utterance_segments
        self._pending_utterance_rms, self._utterance_segments = None, 0
        return rms, n

    def add_turn(self, role: str, text: str,
                 audio_rms: Optional[float] = None) -> None:
        self.turns.append(TranscriptTurn(
            role=role,
            text=text,
            timestamp=datetime.now().strftime("%H:%M:%S"),
            audio_rms=audio_rms,
        ))

    def _cost_lines(self, duration_seconds: int) -> tuple[list[tuple[str, str, float]], float]:
        """Itemise the call cost. Returns ([(label, detail, usd)], total_usd).

        Cached input is billed at the cached rate, so the totals reflect whether
        prompt caching actually engaged rather than assuming it did or didn't.
        Rates come from core.config — verify them against current OpenAI pricing.
        """
        s = settings
        audio_in_fresh = max(0, self._input_audio_tokens - self._input_audio_cached_tokens)
        text_in_fresh  = max(0, self._input_text_tokens  - self._input_text_cached_tokens)

        rows = [
            ("Audio in",        f"{audio_in_fresh:,} tok",
             audio_in_fresh / 1_000_000 * s.price_audio_in),
            ("Audio in cached", f"{self._input_audio_cached_tokens:,} tok",
             self._input_audio_cached_tokens / 1_000_000 * s.price_audio_in_cached),
            ("Audio out",       f"{self._output_audio_tokens:,} tok",
             self._output_audio_tokens / 1_000_000 * s.price_audio_out),
            ("Text in",         f"{text_in_fresh:,} tok",
             text_in_fresh / 1_000_000 * s.price_text_in),
            ("Text in cached",  f"{self._input_text_cached_tokens:,} tok",
             self._input_text_cached_tokens / 1_000_000 * s.price_text_in_cached),
            ("Text out",        f"{self._output_text_tokens:,} tok",
             self._output_text_tokens / 1_000_000 * s.price_text_out),
            ("Telephony",       f"{duration_seconds/60:.2f} min",
             duration_seconds / 60.0 * s.price_telephony_per_min),
        ]
        return rows, sum(usd for _, _, usd in rows)

    def _calc_cost(self, duration_seconds: int) -> float:
        return self._cost_lines(duration_seconds)[1]

    def _cache_hit_rate(self) -> float:
        """Fraction of input tokens served from cache. 0.0 means caching never hit."""
        total = self._input_text_tokens + self._input_audio_tokens
        if not total:
            return 0.0
        cached = self._input_text_cached_tokens + self._input_audio_cached_tokens
        return cached / total

    def _print_cost(self, duration_seconds: int) -> None:
        rows, total = self._cost_lines(duration_seconds)
        hit = self._cache_hit_rate()

        print("\n" + "─" * 56, flush=True)
        print(f"  💰  CALL COST — {self.call_id}", flush=True)
        print(f"      model={settings.realtime_model}  template={settings.call_template}", flush=True)
        print("─" * 56, flush=True)
        print(f"  {'Duration':<17}{duration_seconds}s ({duration_seconds/60:.1f} min), "
              f"{self._responses} responses", flush=True)
        for label, detail, usd in rows:
            print(f"  {label:<17}{detail:>16}  → ${usd:.4f}", flush=True)
        print("─" * 56, flush=True)
        print(f"  {'TOTAL':<17}{'':>16}  → ${total:.4f}", flush=True)
        if duration_seconds:
            print(f"  {'per minute':<17}{'':>16}  → ${total / (duration_seconds/60):.4f}", flush=True)
        print(f"  {'cache hit rate':<17}{'':>16}    {hit:.1%}", flush=True)
        if hit < 0.20 and self._responses > 2:
            print("  ⚠  Low cache hit rate. Something per-call is leaking into the", flush=True)
            print("     prefix — check that `instructions` is static and that no", flush=True)
            print("     response.create carries an `instructions` override.", flush=True)
        print("─" * 56 + "\n", flush=True)

    def _enrich_doctor(self, branch: Optional[str], resolved: bool) -> dict:
        """Apply what this call learned to self.doctor, and describe the result.

        Mirrors the email agent's node_parse_done, which is the only other
        place a Doctor is enriched — same fields, same intent, so a record
        touched by voice and one touched by email stay comparable.

        Status is NOT set to COMPLETE the way the email path does. COMPLETE
        means "all required fields present", and is_complete() requires a
        specialization that run_twilio.py never supplies, so claiming COMPLETE
        would be a claim the record itself contradicts. VERIFIED — "confirmed
        by >=1 extra source" — is what a successful call actually establishes,
        and it is downgraded to PARTIALLY_VERIFIED, with the missing fields
        named, when the record is not otherwise usable. That keeps the
        specialization question visible in the data instead of resolving it by
        guessing.
        """
        doc = self.doctor
        was = doc.status
        if resolved and branch:
            doc.branch = branch
            city = self.memory.get("city")
            if city:
                doc.city = city
            # The first assignment of Source.VOICE anywhere in the programme.
            doc.source = Source.VOICE
            doc.status = (DoctorStatus.VERIFIED if doc.is_complete()
                          else DoctorStatus.PARTIALLY_VERIFIED)
            doc.enriched_at = datetime.now(timezone.utc)
        elif not doc.branch:
            # The call did not get one and the record still has none. Says
            # nothing about WHY — the reason lives in the call artifact — only
            # that this record still needs a branch.
            doc.status = DoctorStatus.MISSING_BRANCH

        missing = doc.missing_for_complete()
        return {
            "doctor_name":    doc.doctor_name,
            "hospital_name":  doc.hospital_name,
            "specialization": doc.specialization,
            "branch":         doc.branch,
            "city":           doc.city,
            "source":         doc.source.value,
            "status":         doc.status.value,
            "status_before":  was.value,
            # Deliberately NOT set here. models.py assigns confidence to the
            # validation agent, and inventing a number in this file would put
            # two different scoring schemes in the directory. The evidence a
            # scorer needs is already recorded: `grounding` on the call record
            # says whether the branch was checked against caller speech.
            "confidence":     doc.confidence,
            # Why this is not COMPLETE. Empty list means it is.
            "missing_for_complete": missing,
            "enriched_at":    doc.enriched_at.isoformat() if doc.enriched_at else None,
            "enriched_by":    self.call_id,
        }

    def _write_doctor_directory(self, doctor_record: dict) -> None:
        """Upsert the enriched record into doctors.json.

        Without this the enrichment lives only on an in-memory object that is
        discarded when the call ends — which is exactly the state the email
        agent is in, and why Source.VOICE had never been written to disk.

        Keyed on (doctor_name, hospital_name): the same doctor at two hospitals
        is two directory rows, and re-calling the same one must update the row
        rather than append a duplicate. Locked for the same reason master.json
        is — read-modify-write, and the module global that currently prevents
        concurrency is on its way out.
        """
        path = json_dir() / "doctors.json"
        key = (doctor_record.get("doctor_name"), doctor_record.get("hospital_name"))
        with _DOCTORS_LOCK:
            try:
                rows = json.loads(path.read_text()) if path.exists() else []
            except Exception:
                rows = []
            for i, row in enumerate(rows):
                if (row.get("doctor_name"), row.get("hospital_name")) == key:
                    rows[i] = doctor_record
                    break
            else:
                rows.append(doctor_record)
            path.write_text(json.dumps(rows, indent=2))

    async def save(self) -> None:
        import soundfile as sf
        # Recording buffers hold whatever the wire format is: 8kHz μ-law
        # under passthrough, 24kHz PCM16 otherwise.
        _SR = _wire_sample_rate()
        duration    = int((datetime.now() - self.start_dt).total_seconds())
        audio_path: Optional[str] = None

        # Build WAV from accumulated streams
        try:
            base_dir = audio_dir()
            base_dir.mkdir(parents=True, exist_ok=True)
            wav_path = base_dir / f"{self.call_id}.wav"

            # Caller: continuous stream from Twilio — already timeline-aligned (t=0 = stream start)
            caller_raw = b"".join(self._caller_pcm)
            caller = (_wire_to_pcm16(caller_raw)
                      if caller_raw else np.zeros(_SR, dtype=np.float32))

            # Agent: place each response block at its timestamp position.
            # Each block is a complete response — all deltas joined in order, so
            # PCM at 24 kHz naturally spans the correct duration from t_offset.
            n = max(len(caller), int(duration * _SR), _SR)
            agent = np.zeros(n, dtype=np.float32)
            print(f"[Realtime] Recording: caller={len(caller)} samples, "
                  f"agent_responses={len(self._agent_pcm)}, n={n}", flush=True)
            for (t_offset, pcm_bytes) in self._agent_pcm:
                start = int(t_offset * _SR)
                arr   = _wire_to_pcm16(pcm_bytes)
                end   = min(start + len(arr), n)
                if end > start:
                    agent[start:end] += arr[:end - start]
                print(f"  agent block: t={t_offset:.2f}s, dur={_wire_samples(pcm_bytes)/_SR:.2f}s, "
                      f"samples={len(arr)}", flush=True)

            # Pad to same length
            n = max(len(caller), len(agent))
            if len(caller) < n: caller = np.pad(caller, (0, n - len(caller)))
            if len(agent)  < n: agent  = np.pad(agent,  (0, n - len(agent)))

            # Soft-gate the caller channel:
            # During the agent's speaking windows, reduce caller volume to 10% so the echo
            # is barely audible. Outside those windows, keep caller at full volume.
            # This is better than hard-gating (zeroing) which silenced the caller entirely
            # when they responded immediately after the agent stopped speaking.
            soft_gate = np.ones(n, dtype=np.float32)
            for (t_off, pcm_b) in self._agent_pcm:
                s = int(t_off * _SR)
                # 0.3s tail after agent audio ends — covers residual phone echo
                e = min(s + _wire_samples(pcm_b) + int(0.30 * _SR), n)
                if e > s:
                    soft_gate[s:e] = 0.10   # 10% = echo barely audible, speech still recorded

            gated_caller = caller * soft_gate

            # Debug: log raw caller RMS in 3s chunks to see if voice is present
            chunk = 3 * _SR
            raw_rms = [float(np.sqrt(np.mean(caller[i:i+chunk]**2)))
                       for i in range(0, len(caller) - chunk, chunk)]
            print(f"[Realtime] Caller raw RMS per 3s: {[f'{r:.4f}' for r in raw_rms]}", flush=True)

            # Normalise each channel independently then mix to MONO.
            def _normalise(arr: np.ndarray, target: float = 0.7) -> np.ndarray:
                peak = np.max(np.abs(arr))
                return arr * (target / peak) if peak > 0.01 else arr
            agent_norm  = _normalise(agent)
            caller_norm = _normalise(gated_caller)
            mono = np.clip(agent_norm + caller_norm, -1.0, 1.0)
            sf.write(str(wav_path), mono, _SR)
            audio_path = str(wav_path)
            log.info("Realtime recording saved (mono, soft-gated caller): %s", wav_path)
        except Exception as e:
            log.error("Failed to save audio: %s", e, exc_info=True)

        # Save per-call JSON
        data_dir = json_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        branch  = self.memory.get("branch")
        resolved = bool(self.memory.get("resolved"))
        # Write what the call learned back onto the record it came from. Until
        # now nothing did: a resolved call wrote a CallRecord and the Doctor
        # that started it was never touched, so Source.VOICE was assigned to
        # nothing anywhere in the repo. The programme's purpose is enriching a
        # client directory, and the enrichment was ending at the call log.
        doctor_record = self._enrich_doctor(branch, resolved)
        # clean_doctor_name strips "Dr." so we don't get "Dr. Dr. John"
        from agents.voice.templates import clean_doctor_name
        doctor_display = clean_doctor_name(self.doctor.doctor_name)
        summary = (
            f"Called {self.doctor.hospital_name} to verify Dr. {doctor_display}'s branch. "
            + (f"Branch confirmed: {branch}." if resolved else "Branch could not be confirmed.")
        )

        # Clean up transcript:
        # 1. Drop [...]  — placeholder never replaced (too short/quiet to transcribe)
        # 2. Drop short agent fragments (< 4 words, e.g. "Right, right —", "Got it,")
        #    that appear between caller turns — these are filler utterances, not full turns
        # 3. Merge consecutive agent turns within 5s (fragment + real response collapsed into one)
        # 4. Merge consecutive caller turns within 4s (VAD split one sentence into two)
        # 5. Drop duplicate agent turns (same text within 3s — barge-in double-fire)
        merged: list[TranscriptTurn] = []
        for turn in self.turns:
            # Drop untranscribed placeholders
            if turn.text.strip() == "[...]":
                continue

            if merged:
                prev = merged[-1]
                # timestamp is Optional[str], so this has to handle a missing
                # one BEFORE parsing: strptime(None) raises TypeError, which
                # the old `except ValueError` did not catch — one untimed turn
                # would have taken down save() and cost the record of a call
                # that had already succeeded. A missing or unparseable time is
                # treated as a large gap, so the turns simply are not merged.
                if prev.timestamp and turn.timestamp:
                    try:
                        prev_t = datetime.strptime(prev.timestamp, "%H:%M:%S")
                        cur_t  = datetime.strptime(turn.timestamp, "%H:%M:%S")
                        gap    = (cur_t - prev_t).total_seconds()
                    except ValueError:
                        gap = 99
                else:
                    gap = 99

                # Merge consecutive agent turns within 5s (fragment collapsed into next real turn)
                if prev.role == "agent" == turn.role and gap <= 5:
                    prev_words = len(prev.text.strip().split())
                    if prev_words <= 4:
                        # prev is the fragment — drop it, keep the fuller current turn
                        merged[-1] = turn
                    else:
                        # both are substantial — merge (rare, but handles multi-sentence responses)
                        merged[-1] = TranscriptTurn(
                            role=prev.role,
                            text=prev.text.rstrip() + " " + turn.text.lstrip(),
                            timestamp=prev.timestamp,
                        )
                    continue

                # Merge split caller sentences (same role, within 4s)
                if prev.role == "caller" == turn.role and gap <= 4:
                    merged[-1] = TranscriptTurn(
                        role=prev.role,
                        text=prev.text.rstrip() + " " + turn.text.lstrip(),
                        timestamp=prev.timestamp,
                        # Carry the LOUDER of the two. audio_rms is the only
                        # evidence separating a real answer from the
                        # transcriber echoing its own hint, and this merge
                        # dropped it — every caller turn in the artifact read
                        # audio_rms=null while the console reported "caller
                        # turns measured 7 of 7", because the live turns had it
                        # and the saved ones did not. Louder, not first: the
                        # merged turn contains both utterances, so the evidence
                        # that anyone spoke at all is the loudest part of it.
                        audio_rms=max(
                            (x for x in (getattr(prev, "audio_rms", None),
                                         getattr(turn, "audio_rms", None))
                             if x is not None), default=None),
                    )
                    continue

                # Drop duplicate agent turns (same text within 3s)
                if (prev.role == "agent" == turn.role and gap <= 3
                        and prev.text.strip() == turn.text.strip()):
                    continue

            merged.append(turn)

        # Final pass: drop lone agent fragments (≤ 3 words ending in — or ,)
        # e.g. "Right, right —" or "Got it," that weren't adjacent to another agent turn
        def _is_fragment(t: TranscriptTurn) -> bool:
            if t.role != "agent":
                return False
            txt = t.text.strip()
            return len(txt.split()) <= 3 and txt[-1:] in ("—", ",", "–")
        merged = [t for t in merged if not _is_fragment(t)]

        # Calculate cost before saving so it goes into JSON/DB
        cost_usd = self._calc_cost(duration)

        record = {
            "call_id":        self.call_id,
            "doctor_name":    self.doctor.doctor_name,
            "hospital_name":  self.doctor.hospital_name,
            "branch":         branch,
            "resolved":       resolved,
            # The enrichment this call produced, as applied to the Doctor. Kept
            # in the call artifact as well as doctors.json so a row in the
            # directory can always be traced to the call that wrote it.
            "doctor_record":  doctor_record,
            "duration_seconds": duration,
            "cost_usd":       round(cost_usd, 6),
            "template":       settings.call_template,
            # How much to trust `branch`. "SKIPPED" means the caller's speech
            # never transcribed, so nothing verified the saved location against
            # what they actually said — filter on this before treating a batch
            # of results as clean.
            "grounding":      self.memory.get("grounding"),
            # Countable conversational failures. Prose rules against these have
            # been ignored across three prompt versions; measuring them makes
            # the next edit evaluable instead of impressionistic.
            "conversation":   conversation_metrics(merged),
            # Did the ask-budget directive actually work? It is injected as a
            # conversation item with no follow-up lever, so if the agent
            # acknowledges it and asks again there is nothing else to pull.
            # Recorded so the budget is evaluated, not trusted.
            "ask_budget": _ask_budget_outcome(
                self.turns, self._give_up_at_turn,
                self._give_up_sent, bool(self.memory.get("escalated"))),
            # Recorded even when it does not fire, so there is data on how often
            # a callee names themselves at all — the check is untestable until
            # real hospital numbers are dialled.
            "hospital_mismatch": hospital_mismatch(self) or None,
            "branch_needed_clarification":
                bool(self.memory.get("branch_needed_clarification")),
            # A false statement was made to a real medical office. That belongs
            # in the record, not just the console, so it is auditable later.
            "false_employment_claim": self._employment_claimed,
            # How much of the caller audio the hint-echo guard could actually
            # judge. _is_hint_echo exempts turns with no measurement, which is
            # right in principle — absence of measurement is not evidence — but
            # the exemption is only safe if it is rare. If unmeasured turns turn
            # out to be the common path, the guard is weaker than its tests
            # suggest and this is the number that says so.
            "caller_turns_unmeasured": sum(
                1 for t in self.turns
                if t.role == "caller" and t.text.strip() != "[...]"
                and getattr(t, "audio_rms", None) is None),
            "caller_turns_measured": sum(
                1 for t in self.turns
                if t.role == "caller" and t.text.strip() != "[...]"
                and getattr(t, "audio_rms", None) is not None),
            "model":          settings.realtime_model,
            # Recorded so latency across calls can be attributed to the settings
            # that produced it, instead of reconstructed from memory afterwards.
            "audio_settings": {
                "turn_detection": settings.realtime_turn_detection,
                "silence_ms":     settings.realtime_silence_ms,
                "eagerness":      settings.realtime_vad_eagerness,
                "voice":          settings.realtime_voice,
                "noise_reduction": settings.realtime_noise_reduction,
            },
            "usage": {
                "responses":         self._responses,
                "input_audio":       self._input_audio_tokens,
                "input_audio_cached": self._input_audio_cached_tokens,
                "output_audio":      self._output_audio_tokens,
                "input_text":        self._input_text_tokens,
                "input_text_cached": self._input_text_cached_tokens,
                "output_text":       self._output_text_tokens,
                "cache_hit_rate":    round(self._cache_hit_rate(), 4),
            },
            "transcript": [
                # audio_rms is persisted deliberately. It is the ONLY evidence
                # that a caller turn came from a human rather than from the
                # transcription hint being echoed back, and without it in the
                # artifact a suspected hallucination cannot be adjudicated
                # after the call — which is exactly what happened when
                # "Mercy Medical Center" appeared on call-20260818-1338 and
                # every rms in the JSON was null.
                {"role": t.role, "text": t.text, "timestamp": t.timestamp,
                 "audio_rms": getattr(t, "audio_rms", None)}
                for t in merged
            ],
            "summary":     summary,
            "audio_path":  audio_path,
            "recorded_at": self.start_dt.isoformat(),
        }
        json_path = data_dir / f"{self.call_id}.json"
        json_path.write_text(json.dumps(record, indent=2))

        # Update master.json.
        #
        # Read-modify-write with no lock: two calls finishing together both read
        # the same list, both append their own entry, and the second write
        # silently discards the first. No error, no warning — a completed call
        # simply is not in the index.
        #
        # Paired deliberately with the CallSid routing fix in twilio_worker.
        # Fixing only that one would remove the module global that currently
        # makes concurrency impossible, turning this from dormant into live.
        master = data_dir / "master.json"
        with _MASTER_LOCK:
            try:
                existing = json.loads(master.read_text()) if master.exists() else []
            except Exception:
                existing = []
            existing.append({
                "call_id":          self.call_id,
                "time":             self.start_dt.isoformat(),
                "doctor":           self.doctor.doctor_name,
                "hospital":         self.doctor.hospital_name,
                "branch":           branch,
                "resolved":         resolved,
                "grounding":        self.memory.get("grounding"),
                "duration_seconds": duration,
                "cost_usd":         round(cost_usd, 6),
                "summary":          summary,
                "audio_path":       audio_path,
                "json_path":        f"data/3 cases jsons/{self.call_id}.json",
            })
            master.write_text(json.dumps(existing, indent=2))

        # The directory the whole programme exists to build. Written last, so a
        # failure here cannot cost us the call record — the call is evidence of
        # what happened and the directory row is derived from it, never the
        # other way round.
        try:
            self._write_doctor_directory(doctor_record)
        except Exception as e:
            log.error("Failed to write doctor directory: %s", e, exc_info=True)
        log.info("Realtime call saved: %s (resolved=%s branch=%s)", self.call_id, resolved, branch)

        # ── End-of-call summary ────────────────────────────────────────
        _W = 60
        print("\n" + "═" * _W, flush=True)
        print(f"  CALL ENDED  —  {self.call_id}", flush=True)
        print("─" * _W, flush=True)
        print(f"  Doctor   : {self.doctor.doctor_name}", flush=True)
        print(f"  Hospital : {self.doctor.hospital_name}", flush=True)
        print(f"  Duration : {duration}s", flush=True)
        if resolved:
            print(f"  Result   : ✅ RESOLVED — Branch: {branch}", flush=True)
        else:
            reason = self.memory.get("escalate_reason", "unknown")
            print(f"  Result   : ⚠️  NOT RESOLVED — {reason}", flush=True)
        print("─" * _W, flush=True)
        print("  TRANSCRIPT:", flush=True)
        for t in merged:
            role_label = "🤖 AGENT " if t.role == "agent" else "👤 Caller"
            print(f"  [{t.timestamp}] {role_label}: {t.text}", flush=True)
        print("─" * _W, flush=True)
        if audio_path:
            print(f"  Recording: {audio_path}", flush=True)
        print(f"  JSON     : data/3 cases jsons/{self.call_id}.json", flush=True)
        print("═" * _W + "\n", flush=True)

        m = conversation_metrics(merged)
        print(f"  CONVERSATION SHAPE", flush=True)
        print(f"    agent turns          {m['agent_turns']}", flush=True)
        print(f"    of which questions   {m['question_turns']}", flush=True)
        rate = f"{m['staple_rate']:.0%}" if m['staple_rate'] is not None else "n/a"
        print(f"    caller turns         {m['caller_turns']} "
              f"({m['caller_questions']} of them questions)", flush=True)
        print(f"    stapled onto answers {m['stapled_questions']} of "
              f"{m['caller_questions']}  ({rate})", flush=True)
        print(f"    asked twice running  {m['back_to_back_asks']}", flush=True)
        print(f"    asked twice in a turn {m['double_asks']}", flush=True)
        print(f"    turns stacking moves {m['piled_turns']}"
              f"   (longest {m['longest_turn_sentences']} sentences, "
              f"{m['longest_turn_words']} words)", flush=True)
        print(f"    repeated sentences   {m['repeated_sentences']}"
              f"{'   <- this is the one that correlates with a bad call' if m['repeated_sentences'] else ''}",
              flush=True)
        # Is the hint-echo guard's benefit-of-the-doubt exemption a corner case
        # or the common path? Only counting answers that.
        _meas = sum(1 for t in self.turns if t.role == "caller"
                    and t.text.strip() != "[...]"
                    and getattr(t, "audio_rms", None) is not None)
        _unmeas = sum(1 for t in self.turns if t.role == "caller"
                      and t.text.strip() != "[...]"
                      and getattr(t, "audio_rms", None) is None)
        print(f"    caller turns measured {_meas} of {_meas + _unmeas}"
              f"{'   <- unmeasured turns bypass the hint-echo check' if _unmeas else ''}",
              flush=True)
        if self._employment_claimed:
            print(f"    ⚠️  FALSE EMPLOYMENT CLAIM made on this call",
                  flush=True)

        self._print_cost(duration)


# ── Main handler ──────────────────────────────────────────────────────────────

async def handle_realtime(twilio_ws: WebSocket, call_sid: str, doctor: Doctor) -> None:
    """Bridge Twilio WebSocket ↔ OpenAI Realtime API for a single call."""
    sess     = RealtimeSession(call_sid, doctor)
    template = get_template(settings.call_template)

    # Never let configured settings be silently ignored — someone set them for a
    # reason, and a call going out under the wrong org name or in the wrong
    # language is not recoverable once the callee has heard it.
    for warning in template.config_warnings(agent_language=settings.agent_language,
                                            org_name=settings.org_name):
        log.warning("[Realtime] %s", warning)
        print(f"\n  ⚠  {warning}\n", flush=True)

    # The organisation is a runtime value: it names whichever client's campaign
    # this call belongs to, and it reaches the model through the per-call
    # context item, never through the cached instructions.
    # The spoken name must match the voice the callee hears. These were two
    # independent settings until a cedar (male) call introduced itself as Sarah
    # and the caller spent three turns on it instead of the branch.
    persona = persona_for_voice(settings.realtime_voice)
    # Same two values the greeting is built from, so the re-introduction check
    # is judging against exactly what the callee was told.
    sess.agent_name = persona
    sess.org_name   = settings.org_name
    greeting = template.build_greeting(doctor, org=settings.org_name,
                                       agent_name=persona)
    context  = template.build_context(
        doctor,
        callback_number=settings.callback_number,
        callback_email=settings.callback_email,
        org=settings.org_name,
        agent_name=persona,
    )

    # Let /recording_ready name the downloaded MP3 after this call_id so audio,
    # JSON and transcript all share one identifier.
    from agents.voice import twilio_worker
    twilio_worker._call_id_by_sid[call_sid] = sess.call_id

    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
    model   = settings.realtime_model

    # One model, named in config. The old fallback list meant the cost breakdown
    # could describe a different model than the one that actually served the call.
    # The callee has already picked up by the time we get here, so every second
    # spent connecting is dead air on a live line. On 2026-08-18 this handshake
    # stalled past the websockets default (10s) and the call died having played
    # nothing at all — the callee answered, heard seventeen seconds of silence,
    # and was hung up on. A probe from the same machine minutes later connected
    # in 1.7s, so the stall was transient, which is precisely the failure a
    # retry is for.
    #
    # Budget: two attempts at _OAI_CONNECT_TIMEOUT_S. Shorter than the default
    # so a stall is caught while the callee is still listening, and retried
    # once rather than surrendering the call to one bad TCP setup.
    ws_obj = None
    for _attempt in (1, 2):
        try:
            # Both the constructor and the handshake are inside the try: the
            # library raises from __aenter__, but building the connector can
            # fail too (bad URL, bad header), and a retry that only covers one
            # of the two is the same half-a-guard this module keeps relearning.
            conn = websockets.connect(REALTIME_URL.format(model=model),
                                      additional_headers=headers,
                                      open_timeout=_OAI_CONNECT_TIMEOUT_S)
            ws_obj = await conn.__aenter__()
            break
        except Exception as e:
            log.warning("[Realtime] OpenAI handshake attempt %d/2 failed: %s: %s",
                        _attempt, type(e).__name__, e)
            print(f"[Realtime] Handshake attempt {_attempt}/2 failed "
                  f"({type(e).__name__}) — {'retrying' if _attempt == 1 else 'giving up'}",
                  flush=True)
            if _attempt == 2:
                # Re-raised so media_stream closes the Twilio socket and the
                # call ends now, rather than holding a silent line open.
                raise
    if ws_obj is None:      # unreachable: the loop above breaks or re-raises
        raise RuntimeError("realtime handshake returned no socket")
    try:
        raw   = await asyncio.wait_for(ws_obj.recv(), timeout=10.0)
        first = json.loads(raw)
        if first.get("type") == "error":
            err = first.get("error", {})
            raise RuntimeError(f"{model} rejected the connection: {err.get('message')}")
        print(f"[Realtime] Connected: {model}", flush=True)
    except Exception:
        # Close the socket we opened — the old code leaked it on the timeout path.
        await conn.__aexit__(None, None, None)
        raise

    oai_ws_ctx = conn

    try:
        oai_ws = ws_obj

        # ── 2. Configure session — ONE message, everything in it ──────
        # Splitting this across two session.update calls churned the cached
        # prefix. `instructions` is the template's STATIC text: no doctor, no
        # hospital, no time of day. Those go in the conversation item at step 4.
        await oai_ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type":         "realtime",
                "instructions": template.instructions,
                "tools":        _realtime_tools(),
                "audio": build_audio_config(
                    transcribe_model=settings.realtime_transcribe_model,
                    transcribe_hint=template.transcribe_hint,
                    audio_format=settings.realtime_audio_format,
                    noise_reduction=settings.realtime_noise_reduction,
                    turn_detection=settings.realtime_turn_detection,
                    eagerness=settings.realtime_vad_eagerness,
                    voice=settings.realtime_voice,
                    silence_ms=settings.realtime_silence_ms,
                ),
                "max_output_tokens": settings.realtime_max_response_tokens,
            },
        }))

        # ── 2b. Drain messages until session.updated or error ─────────
        for _ in range(10):
            session_confirmed = await asyncio.wait_for(oai_ws.recv(), timeout=10.0)
            sc = json.loads(session_confirmed)
            ev = sc.get("type", "")
            if ev == "error":
                err = sc.get("error", {})
                raise RuntimeError(f"session.update rejected: {err.get('code')} {err.get('message')}")
            if ev == "session.updated":
                print(f"[Realtime] Session configured — template={template.name} "
                      f"voice={settings.realtime_voice}", flush=True)
                break
        else:
            print("[Realtime] session.updated not received — continuing anyway", flush=True)

        # ── 3. Wait for Twilio stream to start ────────────────────────
        print("[Realtime] Waiting for Twilio stream start...", flush=True)
        async for raw_msg in twilio_ws.iter_text():
            msg = json.loads(raw_msg)
            if msg.get("event") == "start":
                sess.stream_sid = msg["start"]["streamSid"]
                sess._stream_start_time = datetime.now()
                print(f"[Realtime] Twilio stream started: {sess.stream_sid}", flush=True)
                # Start Twilio recording NOW — audio stream just opened, agent is about to speak.
                # Starting here (not in /answer) skips the ringing/setup gap entirely.
                async def _start_twilio_recording(csid=call_sid):
                    from twilio.rest import Client as TwilioClient
                    tw = TwilioClient(settings.twilio_account_sid, settings.twilio_auth_token)

                    # Trial accounts reject the richer recording parameters.
                    # Fall back to a bare recording rather than losing it: the
                    # audio is the point, dual-channel and the callback are not.
                    try:
                        rec = await asyncio.to_thread(
                            lambda: tw.calls(csid).recordings.create(
                                recording_channels="dual",
                                trim="trim-silence",
                                recording_status_callback=settings.server_public_url + "/recording_ready",
                                recording_status_callback_method="POST",
                            )
                        )
                        print(f"[Recording] Started (dual channel): {rec.sid}", flush=True)
                        return
                    except Exception as e:
                        print(f"[Recording] Full options rejected ({e}) — "
                              f"retrying bare", flush=True)
                    try:
                        rec = await asyncio.to_thread(
                            lambda: tw.calls(csid).recordings.create()
                        )
                        print(f"[Recording] Started (mono, no callback): {rec.sid}. "
                              f"Fetch it from the Twilio console, or rely on the "
                              f"local mix written at the end of the call.", flush=True)
                    except Exception as e:
                        print(f"[Recording] Could not start: {e}. The local WAV "
                              f"mix will still be written.", flush=True)
                asyncio.create_task(_start_twilio_recording())
                break

        # ── Call start banner ──────────────────────────────────────────
        _W = 60
        print("\n" + "═" * _W, flush=True)
        print(f"  CALL STARTED  {datetime.now().strftime('%H:%M:%S')}", flush=True)
        print(f"  Doctor  : {doctor.doctor_name}", flush=True)
        print(f"  Hospital: {doctor.hospital_name}", flush=True)
        print(f"  Call ID : {sess.call_id}", flush=True)
        print("═" * _W, flush=True)
        print(f"  Greeting → {greeting}", flush=True)
        print("─" * _W + "\n", flush=True)

        # ── 4. Send per-call context, then ask for the opening line ───
        # The context item carries the doctor, hospital and the exact greeting.
        # It lands AFTER the cached instructions prefix, so varying it between
        # calls costs ~110 tokens instead of re-billing the whole prompt.
        # response.create deliberately carries NO `instructions` override — an
        # override replaces the session instructions for that response and puts
        # it on a different, uncacheable prefix.
        await oai_ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": context}],
            },
        }))
        # First response of the call: nothing can be in flight and the call
        # cannot be over, so the default policy is satisfied by construction.
        await _create_response(oai_ws, sess, why="greeting")
        # First-audio latency is the dead air the callee hears after picking up.
        # Measured separately from mid-call latency because the first response
        # pays for an uncached prompt and any connection warm-up.
        sess._greeting_requested_at = time.monotonic()
        print("[Realtime] Context sent, greeting requested — starting audio loops", flush=True)

        # ── 5. Run both directions concurrently ───────────────────────
        # First leg to finish ends the call; the other is cancelled explicitly.
        # asyncio.gather() would leave the surviving leg orphaned, and the
        # Twilio leg only notices done_event when a new frame arrives — so if
        # the far end goes quiet it can hang and sess.save() never runs.
        done_event = asyncio.Event()
        legs = [
            asyncio.create_task(_twilio_to_oai(twilio_ws, oai_ws, sess, done_event),
                                name="twilio->oai"),
            asyncio.create_task(_oai_to_twilio(oai_ws, twilio_ws, sess, done_event),
                                name="oai->twilio"),
            asyncio.create_task(_silence_watchdog(oai_ws, sess, done_event),
                                name="silence-watchdog"),
        ]
        try:
            # The finished leg is not inspected — whichever finishes first ends
            # the call and the other is cancelled below regardless of which it
            # was. Named `_` so that stays a statement rather than an omission.
            _, pending = await asyncio.wait(
                legs, return_when=asyncio.FIRST_COMPLETED
            )
            done_event.set()
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.wait(pending, timeout=5.0)
        finally:
            for task in legs:
                if not task.done():
                    task.cancel()
        for task in legs:
            if task.done() and not task.cancelled() and task.exception():
                log.error("[Realtime] %s leg failed: %s", task.get_name(), task.exception())

    except websockets.exceptions.ConnectionClosed as e:
        log.info("[Realtime] OAI WebSocket closed: %s", e)
    except Exception as e:
        log.error("[Realtime] handle_realtime error: %s", e, exc_info=True)
    finally:
        try:
            await oai_ws_ctx.__aexit__(None, None, None)
        except Exception:
            pass
        await sess.save()


# ── Twilio → OpenAI ───────────────────────────────────────────────────────────

async def _twilio_to_oai(
    twilio_ws: WebSocket,
    oai_ws,
    sess: RealtimeSession,
    done_event: asyncio.Event,
) -> None:
    """Forward Twilio inbound audio to OpenAI Realtime input buffer."""
    try:
        async for raw in twilio_ws.iter_text():
            if done_event.is_set():
                break
            msg   = json.loads(raw)
            event = msg.get("event", "")

            if event == "media":
                if msg["media"].get("track") != "inbound":
                    continue
                payload = msg["media"]["payload"]
                try:
                    if _passthrough_enabled():
                        # Twilio already speaks the session's format. Store the
                        # μ-law bytes as-is and forward the payload untouched.
                        raw_bytes = base64.b64decode(payload)
                        sess._caller_pcm.append(raw_bytes)
                        oai_payload = payload
                    else:
                        raw_bytes = base64.b64decode(payload)
                        pcm_24k = (resample(_mulaw_decode(raw_bytes), _TWILIO_SR, _OAI_SR) * 32767).astype(np.int16)
                        sess._caller_pcm.append(pcm_24k.tobytes())
                        oai_payload = base64.b64encode(pcm_24k.tobytes()).decode()
                    if not sess.listen_enabled.is_set():
                        continue
                    if sess.agent_speaking and not _echo_gate_allows(raw_bytes):
                        # Only reached under REALTIME_ECHO_GATE=drop|energy.
                        #
                        # Dropping every frame here made the agent
                        # uninterruptible. OpenAI's VAD can only fire on audio
                        # it receives, so with the gate shut
                        # input_audio_buffer.speech_started never arrives and
                        # the barge-in handler below is unreachable: a
                        # receptionist who starts talking gets talked over
                        # until the agent finishes its turn. That is the most
                        # robotic thing a voice agent can do, and no prompt
                        # wording fixes it.
                        #
                        # Default is now "pass". near_field noise reduction and
                        # semantic_vad both post-date this gate and may handle
                        # line echo on their own — an empirical question one
                        # call answers.
                        continue
                    await oai_ws.send(json.dumps({
                        "type":  "input_audio_buffer.append",
                        "audio": oai_payload,
                    }))
                except Exception as e:
                    log.debug("[Realtime] Twilio→OAI audio error: %s", e)

            elif event == "stop":
                log.info("[Realtime] Twilio stream stopped")
                break

    except Exception as e:
        log.info("[Realtime] Twilio→OAI loop ended: %s", e)


# ── OpenAI → Twilio ───────────────────────────────────────────────────────────

# How long to let a silence run before saying something.
#
# Two thresholds, because the two cases are not alike. Mid-conversation, someone
# who has just been asked something is usually thinking, and seven seconds of
# thinking room is right — that is the whole reason silence_duration_ms sits at
# 700ms rather than 360. But straight after the opening line there is nothing to
# think about: a confused callee reacts in two or three seconds, and seven
# seconds of dead air on a cold call is the point at which people hang up.
_SILENCE_PROMPT_FIRST = 3.5
_SILENCE_PROMPT_AFTER = 7.0
_MAX_SILENCE_PROMPTS = 2


async def _silence_watchdog(oai_ws, sess: "RealtimeSession",
                            done_event: asyncio.Event) -> None:
    """Speak again if the callee never does.

    Both greetings end on a statement now, which is the right shape — it hands
    the turn over instead of spending the opener on a question nobody answered.
    But it means a callee who simply waits produces no speech at all, so server
    VAD never fires, no response is ever created, and nothing in either pump
    runs again. The call sits silent until Twilio times it out.

    Nothing else can cover this. Every other recovery in this file is triggered
    by an event — a transcript, a response, a tool call — and the failure here is
    the absence of events.
    """
    while not done_event.is_set():
        await asyncio.sleep(0.5)
        # Deferred goodbye retry. Owned here, not by the response.done handler
        # that schedules it, because this is a separate task: the event pump
        # keeps reading while we wait, so `_response_active` reflects any
        # response OpenAI's VAD created in the meantime instead of a value read
        # before an in-handler sleep. See where _goodbye_retry_at is set.
        _retry_at = sess._goodbye_retry_at
        if _retry_at is not None and time.time() >= _retry_at:
            sess._goodbye_retry_at = None
            # Fires BECAUSE sess.done — this is the goodbye retry (6f0930a).
            # A helper refusing when done would drop the line in silence, which
            # is the bug that site was written to fix.
            if not await _create_response(oai_ws, sess, why="goodbye retry",
                                          allow_when_done=True):
                # Refused because a response is already in flight — which now
                # means OpenAI is already answering the caller, so the line is
                # not silent and there is nothing to retry.
                print("[Realtime] Goodbye retry unnecessary — a response is "
                      "already in flight", flush=True)
        quiet_since = sess._agent_quiet_since
        if sess.done or quiet_since is None or not sess.listen_enabled.is_set():
            continue
        # Nothing has been said yet, so the greeting is the only thing they have
        # heard and there is nothing for them to be thinking about.
        heard_from_them = any(t.role == "caller" and t.text
                              and t.text != "[...]" for t in sess.turns)
        wait_for = _SILENCE_PROMPT_AFTER if heard_from_them else _SILENCE_PROMPT_FIRST
        if time.time() - quiet_since < wait_for:
            continue
        # A response the VAD started in the same tick is already on its way, and
        # a second response.create raises conversation_already_has_active_response
        # — logged, swallowed, and invisible. Let the real one run.
        if sess._response_active:
            sess._agent_quiet_since = None
            continue
        sess._agent_quiet_since = None          # don't re-fire while it speaks
        # Same phase test that picked the threshold picks the budget, so the
        # two can never disagree about which kind of silence this is.
        _phase = "mid-call" if heard_from_them else "opening"
        _used = (sess._silence_prompts_midcall if heard_from_them
                 else sess._silence_prompts_opening)
        if _used >= _MAX_SILENCE_PROMPTS:
            continue
        if heard_from_them:
            sess._silence_prompts_midcall += 1
            _used = sess._silence_prompts_midcall
        else:
            sess._silence_prompts_opening += 1
            _used = sess._silence_prompts_opening
        print(f"[Realtime] {wait_for:.1f}s of silence — prompting "
              f"the callee ({_phase} {_used}/{_MAX_SILENCE_PROMPTS})",
              flush=True)
        try:
            await oai_ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": (
                             "(system: they have not said anything since you "
                             "stopped speaking. Check they are still there in a "
                             "few words — 'still with me?' — or ask your question "
                             "again more simply. Say ONE short thing.)")}]},
            }))
            await _create_response(oai_ws, sess, why="silence watchdog")
        except Exception:
            return


class _ToolOutcome(NamedTuple):
    """What a tool call changed in the event loop's own state.

    None means "not touched", which is NOT the same as False: the loop must
    not clobber _closing_sent with False just because a tool call that had
    nothing to say about it happened to run. None is safe as the sentinel
    because "" and False are the meaningful values here and both are distinct
    from it.

    Typed concretely rather than as `object`. The first version used an
    `object()` sentinel, which widened _agent_text_buf to `object` all the way
    back into the loop and broke the call into _handle_agent_transcript(...,
    _agent_text_buf: str). Pyright caught that the moment the split brought the
    function back under its analysis ceiling — a type error that had been
    sitting there invisible.
    """
    agent_text_buf: Optional[str]
    closing_sent: Optional[bool]
    pending_response_create: Optional[bool]
    stop: bool


async def _handle_tool_call(msg: dict, sess: "RealtimeSession", oai_ws,
                            _pending_tools: dict,
                            _response_had_audio: bool) -> _ToolOutcome:
    """Run one tool call and its guards. Extracted from _oai_to_twilio.

    Pyright refused to analyse that function at all —

        Code is too complex to analyze; reduce complexity by refactoring
        into subroutines or reducing conditional code paths

    — and when it gives up it can no longer prove any local inside is read, so
    the editor greyed out ~60 names as unused and stopped seeing the calls the
    function makes. Raising maxCodeComplexity does NOT help; the ceiling is
    not the binding constraint. The only fix is the one the message names.

    That mattered beyond the noise. Every recurring bug this week lived in
    that unanalysed function: the barge-in pre-audio race, the six
    response.create sites, the five-clause dead-air condition, the audio_rms
    overwrite, and a dead assignment. Most bugs, least tooling.

    This handler is the largest self-contained piece — 290 lines, 34 branch
    points, a quarter of the function's total — and its coupling to the loop
    is three flags and one `continue`, which is why it goes first.
    """
    _agent_text_buf: Optional[str] = None
    _closing_sent: Optional[bool] = None
    _pending_response_create: Optional[bool] = None
    call_id  = msg.get("call_id", "")
    name     = msg.get("name", "")
    args_str = msg.get("arguments") or _pending_tools.get(call_id, {}).get("args", "{}")
    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        args = {}

    # Grounding check. On a live call the model called save_branch
    # with {'branch': 'Riverside Clinic', 'city': 'Atlanta'} when
    # the caller had said only "Hello" and "Okay, next slide,
    # please". "Riverside Campus" was an EXAMPLE in the prompt; the
    # model reshaped it into a fabricated result and hung up.
    # Nothing downstream could tell that record from a real one.
    #
    # So a location may only be saved if the caller actually said
    # it. Verified against the transcript, not the model's claim.
    if name == "save_branch":
        # The check switches itself off when nothing was
        # transcribed — correct, since absence of transcript is not
        # evidence of fabrication and blocking would kill genuine
        # saves on a bad line. But that is exactly the condition
        # that produces fabrications: bad line -> no transcript ->
        # guard off -> a location the model may have inferred gets
        # written as fact. And with the out-of-band whisper
        # fallback removed there is no second path to a transcript.
        #
        # So record it. A save that could not be verified must not
        # be indistinguishable downstream from one that was.
        heard_any = any(t.role == "caller" and t.text.strip() != "[...]"
                        for t in sess.turns)
        sess.memory.update(
            grounding="verified against caller transcript" if heard_any
            else "SKIPPED — no caller speech was transcribed on this "
                 "call, so the saved location could not be checked "
                 "against anything the caller actually said"
        )
        ungrounded = _ungrounded_terms(args, sess)
        if ungrounded:
            # Terse fragment, not an English imperative. The old
            # wording ("Never save a location you were not told.
            # Ask them for it...") was fluent prose, so relaying it
            # produced a grammatical sentence — and on
            # call-20260818-1112 the agent said, out loud, "Sorry,
            # I can't use that unless you've actually said the
            # place name" to a caller who HAD just said one.
            #
            # RE-READ comes first because that is the actual fix
            # nine times in ten: the caller said "office Abadan
            # branch" and the model tried to save "Northside
            # Branch", reshaped from the hospital name on its own
            # record. The answer was already on the call. Telling
            # it to ask is what sent that call to escalation with
            # the location sitting in the transcript.
            result = {
                "ok": False,
                "error": (
                    "REJECTED — absent from caller transcript "
                    "| RE-READ: caller turns, verbatim; a valid "
                    "location is often already among them "
                    "| NEED: wording the caller used out loud"
                ),
            }
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                  f"🚫 HALLUCINATED BRANCH BLOCKED: {args}", flush=True)
        elif (mismatch := hospital_mismatch(sess)):
            # Every word can be genuinely quoted from the caller and
            # the record still be wrong, because the call reached
            # the wrong organisation. Grounding cannot see this.
            sess.memory.update(hospital_mismatch=mismatch)
            result = {
                "ok": False,
                "error": (
                    f"NOT SAVED — wrong organisation: {mismatch} "
                    f"| NEED: which place this call actually reached"
                ),
            }
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                  f"🏥 WRONG ORGANISATION: {mismatch}", flush=True)
        else:
            result = run_tool(name, sess.memory, args)
    elif name == "escalate":
        # Clearing sess._give_up_sent stops us RE-SENDING the
        # directive; it cannot unsay it. Once injected, the model has
        # "stop asking and escalate" in its context and will act on
        # it whatever the caller says next — which on a live call was
        # "can you please give me a minute? I just need to check".
        # So the block has to be here, at the tool call, the same way
        # a fabricated branch is blocked.
        last_caller = next((t.text for t in reversed(sess.turns)
                            if t.role == "caller" and t.text
                            and t.text != "[...]"), "")
        if is_hold_request(last_caller) and not sess.memory.get("branch"):
            result = {"ok": False, "error": (
                "NOT ESCALATED — caller is mid-lookup, not refusing "
                "| NEED: a two-word hold acknowledgement, then "
                "silence until they return")}
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                  f"⏳ ESCALATION BLOCKED — caller is checking: "
                  f"{last_caller[:60]!r}", flush=True)
            await oai_ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": (
                             "(system: disregard the earlier "
                             "instruction to stop and escalate. They "
                             "are looking the branch up right now. "
                             "Wait for them.)")}]},
            }))
            _pending_tools.pop(call_id, None)
            await oai_ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "function_call_output",
                         "call_id": call_id,
                         "output": json.dumps(result)},
            }))
            _agent_text_buf = ""
            return _ToolOutcome(_agent_text_buf, _closing_sent,
                                 _pending_response_create, True)
        _reason = args.get("reason", "")
        # The inverse guard. Recorded whether or not it blocks:
        # blocking is one-shot, but a discarded answer must never
        # leave the call invisible. Without this the artifact says
        # only "never provided a location", which is the false
        # claim itself, and nothing downstream can tell.
        discarded = _discarded_location(_reason, sess)
        if discarded:
            sess.memory.update(discarded_location=discarded)
        bad = _ungrounded_escalation(_reason, sess)
        if discarded and not sess._discard_blocked:
            # ONE-SHOT, like every other injected directive here. A
            # guard that can refuse forever is a call that cannot be
            # ended: the detector is deliberately conservative, but
            # "conservative" is not "never wrong", and the failure
            # mode of blocking twice is an agent stuck on the phone
            # with a receptionist it has already thanked.
            sess._discard_blocked = True
            result = {"ok": False, "error": (
                f"NOT ESCALATED — reason asserts no location was "
                f"given; the transcript has one "
                f"| CALLER SAID: {discarded} "
                f"| NEED: save_branch with THEIR wording, or an "
                f"escalation reason that is true")}
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                  f"↩️  DISCARDED ANSWER — escalation blocked: "
                  f"{discarded[:80]}", flush=True)
        elif bad:
            result = {"ok": False, "error": (
                f"REJECTED — {bad} | NEED: a reason drawn from this "
                f"call's events, not an inference about the doctor "
                f"| FALLBACK: 'could not obtain the location'")}
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                  f"🚫 UNGROUNDED ESCALATION BLOCKED: {args}",
                  flush=True)
        else:
            result = run_tool(name, sess.memory, args)
    else:
        result = run_tool(name, sess.memory, args)
    # Report what the tool ACTUALLY did. This used to print
    # "✅ BRANCH SAVED" unconditionally, without looking at the
    # result — so a live call logged
    #     🚫 HALLUCINATED BRANCH BLOCKED: {'branch': 'Downtown'}
    #     ✅ BRANCH SAVED : {'branch': 'Downtown'}
    # one line apart. The guard had worked and nothing was saved,
    # but the log said otherwise. A safeguard that reports itself as
    # having failed is worse than no log at all: it sends you
    # hunting a bug that isn't there and hides the one that is.
    ts = datetime.now().strftime("%H:%M:%S")
    ok = bool(result.get("ok"))
    if name == "save_branch":
        if ok:
            print(f"\n[{ts}] ✅ BRANCH SAVED   : {args}", flush=True)
        else:
            print(f"\n[{ts}] ⛔ BRANCH REJECTED: {args}", flush=True)
            print(f"          reason: {result.get('error', '')}", flush=True)
            # The agent may already have TOLD them it was saved.
            # On call-20260818-1613:
            #   "Thanks for checking — I'll save that and then
            #    we'll be all set."          <- spoken
            #   ⛔ BRANCH REJECTED                <- 0.0s later
            # The caller was told the job was done. It was not.
            # That call recovered because the next turn happened to
            # ask a follow-up; the same shape on a rejection that
            # does not recover leaves a receptionist hanging up
            # believing a location was recorded when nothing was
            # written.
            #
            # Same class as the lying console log fixed in 0c28baa:
            # a success message emitted before the operation that
            # decides success. That was fixed in the print; the
            # model does it on the wire.
            #
            # Not fixable by prompt — the model cannot know the
            # result before the tool returns, so no rule makes it
            # reliable. The prompt already carries "Never claim to
            # have noted, saved, or recorded a location you were
            # not given" and it did not hold. But the PROCESS knows
            # both halves: what was said, and that it was rejected.
            _said = next((t.text for t in reversed(sess.turns)
                          if t.role == "agent"), "")
            if (_claims_saved(_said) and not sess._false_save_nudged
                    and not sess.done):
                sess._false_save_nudged = True
                print(f"[{ts}] ⚠️  FALSE SAVE CLAIM — they were told "
                      f"it was saved; correcting", flush=True)
                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text", "text": (
                                 "(system: you just told them the "
                                 "location was saved, or that you "
                                 "were finished. Neither is true — "
                                 "nothing has been recorded. Do not "
                                 "imply it has been. Do not thank "
                                 "them as though the call is over. "
                                 "Say you need one more detail, and "
                                 "ask for it.)")}]},
                }))
    elif name == "escalate":
        label = "⚠️  ESCALATED     " if ok else "⛔ ESCALATE FAILED"
        print(f"\n[{ts}] {label}: {args}", flush=True)
    elif name == "note_info":
        print(f"[{ts}] {'📝 NOTE           ' if ok else '⛔ NOTE REJECTED  '}: {args}",
              flush=True)
    else:
        print(f"[{ts}] 🔧 TOOL           : {name}({args}) → {result}", flush=True)

    if name in ("save_branch", "escalate") and result.get("ok"):
        sess.done = True

    await oai_ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {
            "type":    "function_call_output",
            "call_id": call_id,
            "output":  json.dumps(result),
        },
    }))

    if sess.done:
        # "_response_had_audio" was being read as "the agent said
        # goodbye", so the call hung up on whatever it happened to
        # be saying. On a live call it asked "which office is Dr.
        # Okafor working out of?", called save_branch in the same
        # response, and hung up — leaving the caller answering a
        # question to a dead line.
        #
        # An utterance ending in a question mark is not a farewell.
        last_agent = next((t.text for t in reversed(sess.turns)
                           if t.role == "agent"), "")
        sounded_like_a_goodbye = bool(last_agent) and not last_agent.rstrip().endswith("?")

        if _response_had_audio and sounded_like_a_goodbye:
            # Model already said goodbye in its audio — don't inject another line
            # The current response.done will trigger the close
            _closing_sent = False
        else:
            # Tool fired with no spoken goodbye. Ask for one via a
            # conversation item rather than a per-response
            # `instructions` override — an override swaps out the
            # session instructions and lands this response on a
            # different, uncacheable prefix.
            await oai_ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": "(say a brief warm goodbye now, then stop)",
                    }],
                },
            }))
            # BOTH overrides, and both are load-bearing:
            #  - done: sess.done was set 40 lines up, by the very
            #    tool call this goodbye belongs to.
            #  - active: we are inside the tool-call handler, so
            #    the response carrying that tool call has not
            #    emitted response.done yet. Before the barge-in fix
            #    _response_active was set on the first AUDIO delta,
            #    and a tool-only response emits none — so this read
            #    False by accident. Setting it on response.created
            #    made it correctly True, which would have made a
            #    naive helper eat the goodbye. The call is left
            #    unguarded exactly as it was, deliberately.
            await _create_response(oai_ws, sess, why="closing goodbye",
                                   allow_when_done=True,
                                   allow_when_active=True)
            _closing_sent = True  # skip tool-call response.done, close on closing's
    else:
        _pending_response_create = True
    return _ToolOutcome(_agent_text_buf, _closing_sent,
                        _pending_response_create, False)


async def _handle_caller_transcript(msg: dict, sess: "RealtimeSession", oai_ws) -> None:
    """One completed caller transcript: log it, and run the turn-level guards.

    Extracted from _oai_to_twilio to bring that function back under pyright's
    analysis ceiling — see _handle_tool_call for why that matters. This one
    shares NO mutable state with the event loop: everything it changes lives on
    `sess`, and its only `break` is a local for-loop break. That is what made it
    the safest of the large handlers to move.
    """
    text = msg.get("transcript", "").strip()
    # The faint-line decision belongs here, not at speech_stopped.
    # Words that arrived are proof the line carried them, whatever
    # the RMS says; nothing arriving on a quiet slice is the only
    # combination that actually means "we cannot hear you".
    _low = getattr(sess, "_pending_low_rms", None)
    sess._pending_low_rms = None
    if _low is not None and not text and not sess._low_audio_warned:
        sess._low_audio_warned = True
        print(f"[Realtime] Caller audio faint AND nothing "
              f"transcribed (RMS {_low:.4f}) — asking them to "
              f"speak up", flush=True)
        await oai_ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": ("(system: that came through too faint to "
                             "make out and nothing was transcribed. Ask "
                             "them to repeat it. Do not guess at "
                             "anything you did not clearly hear.)"),
                }],
            },
        }))
    # ── Repair after an interruption ────────────────────────────
    # The interruption path finally fired on call-20260818-1338 and
    # made the call worse. The agent was truncated to 750ms, the
    # caller heard three-quarters of a second, lost the thread, and
    # said "Hello." The agent classified that as filler and asked
    # its question again.
    #
    # "Hello" after being cut off is not filler. It is a REPAIR
    # SIGNAL — the caller checking the line is alive. The same word
    # arriving cold means something else entirely, and no amount of
    # prose gets the model to tell those apart, because the
    # distinguishing fact is not in the transcript. It is in this
    # process: we know we truncated, and we know to how many
    # milliseconds.
    #
    # So this is code, not a Conversation Flow rule. The section is
    # already twice the size of everything about how to sound, and
    # a state-dependent rule is exactly the kind that reads fine in
    # prose and gets applied inconsistently.
    #
    # Restate, do NOT re-ask: they did not decline to answer, they
    # never heard the question. Re-asking spends an ask on a turn
    # that was never delivered.
    if (text and sess._truncated_at is not None
            and time.time() - sess._truncated_at <= _REPAIR_WINDOW_S
            and sess._truncated_heard_ms < _CUT_SHORT_MS
            and not sess._repair_nudged):
        sess._repair_nudged = True
        sess._truncated_at = None
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"🔁 REPAIR — they were cut off mid-sentence; "
              f"telling the agent to restate, not re-ask",
              flush=True)
        await oai_ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": (
                         "(system: your last turn was cut off after "
                         "well under a second, so they almost "
                         "certainly did not hear it. Whatever they "
                         "just said is them checking the line, not "
                         "an answer. Do NOT ask anything new and do "
                         "not treat this as them declining. Say the "
                         "same thing again, shorter and simpler.)")}]},
        }))

    if text:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] 👤 CALLER : {text}", flush=True)
        # They asked who they are talking to. Reaching this line at
        # all proves the question transcribed, so "I didn't catch
        # that" is not available — that is exactly the dodge that
        # happened on call-20260811-1649, and it cost an ask from
        # the budget on top of sounding evasive.
        if (_IDENTITY_ASK.search(text) and not sess.done
                and not sess._identity_nudged):
            sess._identity_nudged = True
            print(f"[Realtime] Caller asked who is speaking — "
                  f"telling the agent to answer it first", flush=True)
            await oai_ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": (
                            "(system: they just asked who you are. "
                            "Answer that directly and truthfully "
                            "before anything else — your name and "
                            "who you are calling on behalf of. It "
                            "transcribed clearly, so do NOT say you "
                            "did not catch it, and do not answer it "
                            "with a question about the branch.)"
                        ),
                    }],
                },
            }))
        # Someone going to look it up has not refused. The give-up
        # directive is a one-shot: once sent, the agent escalates on
        # its next turn whatever they say in between — and on a live
        # call that next turn was "can you please give me a minute?
        # I just need to check". It thanked them and hung up.
        # They have said it twice — that is all they have. Asking a
        # third time gets the same words back and eventually a
        # hang-up, and on a live call it ended with nothing saved
        # despite a street and a state having been given.
        _again = caller_repeated_answer(text, sess)
        if _again and not sess.done and not sess.memory.get("branch") \
                and not sess._repeat_nudged:
            sess._repeat_nudged = True
            print(f"[Realtime] Caller has repeated their answer — "
                  f"telling the agent to take what it has", flush=True)
            await oai_ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": (
                             "(system: they have now given you the "
                             "same answer twice. That is all they "
                             "have — asking again will return the "
                             "same words. Save it with save_branch "
                             "exactly as they said it, then close.)")}]},
            }))
        if is_hold_request(text) and not sess.done:
            if sess._give_up_sent or sess._location_asks:
                print(f"[Realtime] Caller is going to check — "
                      f"give-up cancelled, ask count reset "
                      f"(was {sess._location_asks})", flush=True)
            sess._give_up_sent = False
            sess._give_up_at_turn = None
            sess._location_asks = 0
        # Replace the most recent "[...]" placeholder with real text
        _utt_rms, _utt_segs = sess.take_utterance_rms()
        if _utt_segs > 1:
            # Visible because this is the condition that used to
            # lose the measurement outright. If it turns out to be
            # the common path, the grounding guards are resting on
            # a number that is routinely reconstructed rather than
            # read, and that is worth knowing from the log rather
            # than from a recording three days later.
            print(f"[Realtime] transcript covered {_utt_segs} VAD "
                  f"segments — using the loudest "
                  f"(rms {_utt_rms:.4f})", flush=True)
        for i in range(len(sess.turns) - 1, -1, -1):
            if sess.turns[i].role == "caller" and sess.turns[i].text == "[...]":
                sess.turns[i] = TranscriptTurn(
                    role="caller", text=text,
                    timestamp=sess.turns[i].timestamp,
                    audio_rms=_utt_rms,
                )
                break
        else:
            sess.add_turn("caller", text, audio_rms=_utt_rms)

async def _handle_agent_transcript(msg: dict, sess: "RealtimeSession", oai_ws,
                                   _agent_text_buf: str,
                                   _barge_in_pending: bool) -> tuple:
    """One completed agent transcript: record the turn, run the turn guards.

    Extracted from _oai_to_twilio for the analysis ceiling — see
    _handle_tool_call. Unlike the caller-side handler this one does share loop
    state, so it takes the two flags in and hands them back rather than using
    nonlocal: a returned value is visible at the call site, where a nonlocal
    write into a 1,200-line function was not visible to anything, including the
    type checker.
    """
    if _barge_in_pending:
        # This transcript was cancelled — never fully heard, skip it
        _barge_in_pending = False
        _agent_text_buf = ""
        return _agent_text_buf, _barge_in_pending
    text = (msg.get("transcript") or _agent_text_buf).strip()
    if text:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] 🤖 AGENT  : {text}", flush=True)
        # Verbatim repeat of the turn just spoken. On
        # call-20260813-1409 the agent said "could I just get the
        # exact branch name or address so I don't save the wrong
        # place?" twice, both inside ONE 10.65s response — two
        # transcript items from a single generation, so the re-ask
        # gap guard measured 0.0s and had no next turn to correct.
        # conversation_metrics already counts repeated_sentences
        # after the fact and the printout calls it "the one that
        # correlates with a bad call"; this is the same detection
        # moved to where it can still act.
        #
        # It cannot unsay this one either — the audio is already on
        # the wire by the time the transcript lands. What it buys is
        # a visible marker in the log and a directive that stops the
        # pattern continuing across the following turns.
        # False employment claim — "from/with/at {org}" instead of
        # "on behalf of {org}". Checked before the re-introduction
        # test because the two are deliberately disjoint: naming the
        # org while self-naming is a re-introduction, claiming to
        # work there is this.
        if (not sess._employment_claimed
                and _claims_employment(text, sess.org_name)):
            sess._employment_claimed = True
            print(f"[{ts}] ⚠️  FALSE EMPLOYMENT CLAIM — said it is "
                  f"from/with/at {sess.org_name}, not on their "
                  f"behalf", flush=True)
            if not sess.done:
                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text", "text": (
                                 f"(system: you just said you are "
                                 f"from or with {sess.org_name}. You "
                                 f"are not employed by them — you "
                                 f"are calling ON BEHALF OF them, "
                                 f"and saying otherwise is a false "
                                 f"claim about who is on this call. "
                                 f"If it comes up again, say 'on "
                                 f"behalf of {sess.org_name}'.)")}]},
                }))
        # Re-introduction: the greeting delivered a second time.
        # Exempt the first agent turn, which IS the greeting.
        _agent_turns = sum(1 for t in sess.turns if t.role == "agent")
        if (_agent_turns >= 1 and not sess._reintro_nudged
                and not sess.done
                and _is_reintroduction(text, sess.agent_name,
                                       sess.org_name)):
            sess._reintro_nudged = True
            print(f"[{ts}] 🔂 RE-INTRODUCTION — said the greeting "
                  f"again instead of saying what it wants",
                  flush=True)
            await oai_ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": (
                             "(system: you just introduced yourself "
                             "again. They already heard your name "
                             "and who you are calling for in the "
                             "opening line, so repeating it tells "
                             "them nothing and leaves them still not "
                             "knowing what you want. Say what you "
                             "need FROM THEM instead, concretely, in "
                             "one short sentence.)")}]},
            }))
        _prev_agent = next((t.text for t in reversed(sess.turns)
                            if t.role == "agent" and t.text), "")
        _norm = lambda s: re.sub(r"[^a-z0-9 ]", "",
                                 s.lower()).strip()
        if (_prev_agent and _norm(text) == _norm(_prev_agent)
                and not sess._self_repeat_nudged and not sess.done):
            sess._self_repeat_nudged = True
            print(f"[{ts}] 🔁 REPEATED SENTENCE — agent said the "
                  f"same thing twice verbatim", flush=True)
            await oai_ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": (
                             "(system: you just said the same "
                             "sentence twice in a row, word for "
                             "word. They heard it the first time. "
                             "Never repeat a sentence you have "
                             "already said — say the next thing, or "
                             "say nothing and wait.)")}]},
            }))
        sess.add_turn("agent", text)

        # Enforce the exit condition the prompt never had.
        #
        # A live call asked for the location six times in 111
        # seconds. The caller engaged throughout but never refused,
        # never said they did not know, was not a wrong number and
        # was not voicemail — so none of the prompt's escalation
        # triggers matched, and "never close until you have saved a
        # location or escalated" left asking again as the only move.
        # The repetition was a symptom of having no way out, not a
        # phrasing failure, and no wording of a phrasing rule fixes
        # it. A budget does.
        if _is_location_ask(text) and not sess.done:
            # An ask the caller never answered must not spend the
            # budget. On call-20260818-1338 the agent asked three
            # times in twenty seconds with only "Hello." in
            # between — the caller had been cut off by a barge-in
            # and had not heard the question. All three counted.
            # The budget then fired on ask four, which was the
            # first productive one ("which branch is that in Los
            # Angeles?"), and the caller said "Mercy Medical
            # Center" eleven seconds later, into a call that had
            # already given up.
            #
            # _MIN_REASK_GAP_S measured the wrong thing. It gates
            # on elapsed SECONDS — 7s cleared its 6s threshold — but
            # the defect was never speed. It was asking a question
            # again that nobody had answered. Time is a proxy;
            # "did they answer" is the actual question, and the
            # transcript already knows.
            _first_ask = sess._last_ask_turn_idx < 0
            _answered = _first_ask or _caller_answered_since(
                sess, sess._last_ask_turn_idx)
            _forced = sess._unanswered_reasks >= _MAX_UNANSWERED_REASKS
            if _answered or _forced:
                sess._location_asks += 1
                sess._unanswered_reasks = 0
            else:
                sess._unanswered_reasks += 1
                print(f"[Realtime] Re-ask into an unanswered question "
                      f"({sess._unanswered_reasks}/{_MAX_UNANSWERED_REASKS}) "
                      f"— not spending budget "
                      f"({sess._location_asks}/"
                      f"{settings.realtime_max_location_asks} used)",
                      flush=True)
            sess._last_ask_turn_idx = len(sess.turns)
            # Two asks inside _MIN_REASK_GAP_S is badgering, not
            # persistence. This fires after the fact — the agent has
            # already said it — so it cannot prevent the re-ask that
            # trips it, only stop the run from continuing. That is
            # still the difference between one clumsy turn and the
            # three-in-thirteen-seconds that burnt the budget on
            # call-20260811-1649.
            _prev_ask = sess._last_location_ask_at
            _now_ask  = time.time()
            sess._last_location_ask_at = _now_ask
            if (_prev_ask is not None
                    and _now_ask - _prev_ask < _MIN_REASK_GAP_S
                    and not sess._reask_nudged
                    and not sess._give_up_sent):
                sess._reask_nudged = True
                print(f"[Realtime] Re-asked the location "
                      f"{_now_ask - _prev_ask:.1f}s after the last ask "
                      f"— telling the agent to give them room",
                      flush=True)
                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{
                            "type": "input_text",
                            "text": (
                                "(system: you just asked for the "
                                "location twice within a few seconds. "
                                "They have not had a chance to answer. "
                                "Do not ask again on your next turn — "
                                "respond to what they actually said, "
                                "or answer their question, and then "
                                "wait.)"
                            ),
                        }],
                    },
                }))
            if (sess._location_asks >= settings.realtime_max_location_asks
                    and not sess._give_up_sent):
                sess._give_up_sent = True
                sess._give_up_at_turn = len(sess.turns)
                print(f"[Realtime] {sess._location_asks} asks with no "
                      f"location — telling the agent to stop and "
                      f"escalate", flush=True)
                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{
                            "type": "input_text",
                            "text": (
                                # "Thank them briefly, say goodbye"
                                # produced exactly that: "Thanks for
                                # your time, goodbye." The callee is
                                # never told the call is ending
                                # because the agent could not get
                                # what it came for, so they get no
                                # last chance to supply it — and
                                # people often do, once they hear
                                # something was missed. Name the
                                # outcome, own it rather than blame
                                # them, then close.
                                f"(system: you have now asked for the "
                                f"location {sess._location_asks} times "
                                f"and have not been given one. Stop "
                                f"asking. Say plainly that you were not "
                                f"able to get the branch today — phrase "
                                f"it as something you could not do, not "
                                f"as something they failed to give — "
                                f"then thank them and say goodbye. Do "
                                f"not ask again, and do not sound "
                                f"annoyed. Call escalate with reason "
                                f"'caller engaged but never provided a "
                                f"location'.)"
                            ),
                        }],
                    },
                }))
    _agent_text_buf = ""
    return _agent_text_buf, _barge_in_pending


async def _end_speaking_gate(sess: "RealtimeSession", delay: float) -> None:
    """Clear agent_speaking once the audio we sent has finished playing out.

    Was a closure redefined inside the event loop on every response, with its
    arguments smuggled in as default values (`s=sess, delay=_echo_cooldown`).
    Pyright could not resolve its type at all — "refers to itself" — which is
    the last thing that stayed unanalysed after the loop was split. Rebuilding
    a coroutine function per response was also pure waste.

    Module level, arguments passed explicitly. Same behaviour, and now typed.
    """
    await asyncio.sleep(delay)
    sess.agent_speaking = False
    # Under REALTIME_ECHO_GATE=pass this window gates nothing — frames flow
    # throughout — so announcing it as "now listening" was misleading output,
    # implying the caller had been unheard for 6.91s when they had not.
    if settings.realtime_echo_gate != "pass":
        print(f"[Realtime] Echo cooldown done ({delay:.2f}s) — "
              f"listening for caller", flush=True)


async def _oai_to_twilio(
    oai_ws,
    twilio_ws: WebSocket,
    sess: RealtimeSession,
    done_event: asyncio.Event,
) -> None:
    """Forward OpenAI Realtime events to Twilio + handle tool calls."""
    _pending_tools: dict[str, dict] = {}
    _agent_text_buf       = ""
    # _response_active lives on the SESSION, not here: the silence watchdog
    # runs in a different task and must not create a response while one is
    # already generating. As a local it was invisible to it.
    sess._response_active = False
    _response_had_audio   = False   # True if current response included any audio (model spoke)
    _barge_in_pending     = False   # True when we cancelled a response — skip its transcript
    _closing_sent         = False   # True after we send closing response.create — wait for its response.done
    _closing_retries      = 0       # a goodbye the caller talked over is not a goodbye
    _empty_responses      = 0       # responses that completed without saying anything
    # Tool results arrive on response.function_call_arguments.done, which fires
    # BEFORE response.done for the same response. Creating a response there
    # raises conversation_already_has_active_response, so defer it to response.done.
    _pending_response_create = False
    _caller_speaking       = False
    _speech_start_pcm_pos  = 0       # position in sess._caller_pcm when caller speech started
    _samples_this_response = 0       # PCM16 samples sent this response — used for dynamic echo cooldown
    _current_response_pcm: list[bytes] = []    # accumulate all deltas for one response
    _current_response_start: Optional[float] = None  # stream-relative time of first delta
    # monotonic clock when this response's first audio chunk went to Twilio;
    # playback ends at this + audio duration, which is what the echo gate needs
    _first_delta_sent_at: Optional[float] = None
    # id of the assistant item currently being spoken — needed to truncate it
    # to what the caller actually heard when they interrupt
    _current_item_id: Optional[str] = None
    # response ids already accounted for, so a repeated response.done cannot
    # double-count its tokens into the cost figure
    _counted_responses: set[str] = set()

    try:
        async for raw in oai_ws:
            msg        = json.loads(raw)
            event_type = msg.get("type", "")

            # ── Caller barge-in: cancel current response immediately ───────
            if event_type == "input_audio_buffer.speech_started":
                sess._agent_quiet_since = None    # they are talking; stand down
                _caller_speaking = True
                _speech_start_pcm_pos = len(sess._caller_pcm)  # mark start of this utterance
                if sess.done:
                    continue  # don't interrupt the closing farewell
                if sess._response_active and not _barge_in_pending:
                    # Only cancel once per active response — prevents inflation
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✋ BARGE-IN  : caller interrupted agent", flush=True)
                    _barge_in_pending = True
                    sess._response_active = False
                    sess.agent_speaking = False
                    # Flush whatever agent audio arrived before the cancel
                    if _current_response_pcm and _current_response_start is not None:
                        sess._agent_pcm.append((_current_response_start, b"".join(_current_response_pcm)))
                    _current_response_pcm.clear()
                    _current_response_start = None
                    try:
                        await oai_ws.send(json.dumps({"type": "response.cancel"}))
                        await twilio_ws.send_text(json.dumps({
                            "event": "clear", "streamSid": sess.stream_sid,
                        }))
                        # Cancelling stops generation but leaves the FULL
                        # response in OpenAI's conversation history, while the
                        # caller only heard the part that had played. The model
                        # then reasons as though it said things nobody heard —
                        # it won't repeat them, and may refer back to them
                        # ("as I mentioned") about words never spoken.
                        #
                        # Truncate the item to what actually reached the ear.
                        # Twilio plays what we send at realtime speed, so
                        # elapsed-since-first-chunk is the audio they heard,
                        # capped at what was generated.
                        if _current_item_id and _first_delta_sent_at is not None:
                            heard_ms = int((time.monotonic() - _first_delta_sent_at) * 1000)
                            generated_ms = int(_samples_this_response / _wire_sample_rate() * 1000)
                            audio_end_ms = max(0, min(heard_ms, generated_ms))
                            await oai_ws.send(json.dumps({
                                "type": "conversation.item.truncate",
                                "item_id": _current_item_id,
                                "content_index": 0,
                                "audio_end_ms": audio_end_ms,
                            }))
                            print(f"[Realtime] Truncated to {audio_end_ms}ms — "
                                  f"the model's context now matches what was heard",
                                  flush=True)
                            # Remember that this happened, and how little they
                            # got. The next caller turn has to be read in that
                            # light — see the repair handler at the caller
                            # transcript. The process knows it was truncated;
                            # the model can only guess from the transcript, and
                            # on call-20260818-1338 it guessed wrong.
                            sess._truncated_at = time.time()
                            sess._truncated_heard_ms = audio_end_ms
                    except Exception:
                        pass

            # ── Caller finished speaking ───────────────────────────────────
            elif event_type == "input_audio_buffer.speech_stopped":
                if _caller_speaking and sess.listen_enabled.is_set():
                    _caller_speaking = False
                    # Placeholder — filled in by the session's own inline
                    # transcription when conversation.item.input_audio_
                    # transcription.completed arrives. Nothing else fills it:
                    # the out-of-band whisper-1 HTTP fallback was removed to
                    # keep this path pure speech-to-speech. Placeholders that
                    # never resolve are dropped from the transcript in save().
                    sess.add_turn("caller", "[...]")

                    # Tell the model when the line is genuinely too quiet to
                    # trust. Left to its own judgement it does not report
                    # difficulty — on a live call a caller at roughly a tenth
                    # of normal phone level produced a confident fabrication
                    # rather than "sorry, I didn't catch that". Measured level
                    # is evidence the model does not otherwise have.
                    # If ANY caller turn has transcribed cleanly, the line is
                    # audible by definition — never tell them otherwise.
                    #
                    # This alarm has now fired falsely on three consecutive
                    # calls, twice in one of them, telling a caller measuring
                    # 0.0335 RMS that they were "coming through really faint".
                    # Measuring the loudest window instead of the mean was not
                    # enough: server VAD sometimes fires speech_started on
                    # noise, and the resulting slice contains no speech at all,
                    # so any energy measure of it reads as silence.
                    #
                    # Working transcription is the evidence that matters.
                    heard_clearly = any(
                        t.role == "caller" and t.text.strip() not in ("", "[...]")
                        for t in sess.turns)
                    #
                    # Fourth false fire, and the "wait until something has
                    # transcribed" guard could never have helped: the alarm goes
                    # off on the FIRST utterance, when nothing has transcribed
                    # yet by definition. The slice measured came from a VAD
                    # trigger during the greeting, before the caller had said a
                    # word — so it measured silence and called it a faint line.
                    # The agent then spent a whole turn asking a perfectly
                    # audible person to speak up, instead of answering the
                    # question they had just asked.
                    #
                    # RMS cannot decide this. Whether the words came through is
                    # the only evidence that matters, and that is not known until
                    # the transcript arrives. So measure here, decide there.
                    utterance = b"".join(sess._caller_pcm[_speech_start_pcm_pos:])
                    if utterance:
                        arr = _wire_to_pcm16(utterance)
                        rms = _loudest_window_rms(arr)
                        # Measured for EVERY utterance now, not only when the
                        # faint-line warning is still available. The faint
                        # warning wants "is this too quiet to trust"; grounding
                        # wants "did a human actually say this", and the second
                        # question is asked on turns that are perfectly audible.
                        # Loudest-300ms window, not the mean — the mean is
                        # dominated by gaps between words and told an audible
                        # caller they were faint.
                        # ACCUMULATE, do not overwrite. This is set at every
                        # speech_stopped but only consumed when the transcript
                        # arrives, and transcription lags the VAD. If the VAD
                        # segments again — on a trailing breath, on room noise,
                        # on the second half of "yes, yes" — before the
                        # transcript lands, a plain assignment throws away the
                        # measurement of the real speech and keeps the silence.
                        #
                        # Observed on call-20260818-1613: the caller's "Yes,
                        # yes." recorded audio_rms=0.0025 while Twilio's own
                        # caller channel shows that utterance at ~0.13 peak.
                        # A 50x under-report, on the single number the
                        # hint-echo guard depends on — and it errs toward
                        # calling real speech silence, which is the direction
                        # that throws away genuine answers.
                        #
                        # The transcript covers whatever audio accumulated
                        # under it, so the evidence that a human spoke is the
                        # LOUDEST part of that audio, not the last fragment of
                        # it.
                        sess.note_utterance_rms(rms)
                        if not sess._low_audio_warned and not heard_clearly:
                            _acc = sess._pending_utterance_rms or 0.0
                            sess._pending_low_rms = (
                                _acc if 0.0 < _acc < _LOW_AUDIO_RMS else None)

            # ── Response created: it exists from this moment, not from audio ─
            elif event_type == "response.created":
                # This event was not handled at all, and _response_active was
                # set on the FIRST AUDIO DELTA instead. Those are two different
                # facts. Between response.create and the first delta there is
                # real latency — 1.19s measured on call-20260818-1112 — and for
                # that whole window a response existed that nothing could see.
                #
                # A caller who began speaking inside the window reached the
                # barge-in handler with _response_active still False, so it
                # skipped entirely: no response.cancel, no Twilio `clear`, no
                # truncate, and no ✋ BARGE-IN line. OpenAI's own VAD then
                # cancelled the response server-side. That is exactly the
                # signature in the log — two `[cancelled]` responses with
                # out_audio=0 and no barge-in line anywhere on the call.
                #
                # The consequence is not stale audio (nothing had been
                # generated yet) but a LOST TURN: the agent was asked a
                # question, its response was killed before it made a sound, and
                # the dead-air guards fired afterwards trying to explain the
                # silence.
                #
                # "A response is in flight" and "audio is reaching the ear" are
                # separate questions. sess.agent_speaking answers the second.
                # This answers the first — and the silence watchdog and the
                # empty-response guard, which both read _response_active, wanted
                # the first all along.
                sess._response_active = True

            # ── Audio → Twilio ─────────────────────────────────────────────
            # gpt-realtime-2 uses response.output_audio.delta (not response.audio.delta)
            elif event_type == "response.output_audio.delta":
                delta = msg.get("delta", "")
                if delta:
                    sess.agent_speaking  = True
                    sess._response_active = True
                    _response_had_audio  = True
                if delta and sess.stream_sid:
                    try:
                        raw_pcm = base64.b64decode(delta)
                        _samples_this_response += _wire_samples(raw_pcm)
                        if _first_delta_sent_at is None:
                            _first_delta_sent_at = time.monotonic()
                        _current_item_id = msg.get("item_id") or _current_item_id
                        # Buffer for recording: stamp only the first delta of this response.
                        # All deltas arrive fast (~0.2s for a 2s response) so we must NOT
                        # timestamp each chunk individually — they'd all pile up at the same
                        # position and overlap into distorted audio.  Instead we collect them
                        # and flush the whole block on response.done.
                        _current_response_pcm.append(raw_pcm)
                        if _current_response_start is None and sess._stream_start_time:
                            _current_response_start = (datetime.now() - sess._stream_start_time).total_seconds()
                        # Dead air the callee hears before the agent speaks.
                        if sess._greeting_requested_at is not None:
                            gap = time.monotonic() - sess._greeting_requested_at
                            sess._greeting_requested_at = None
                            print(f"[Realtime] First audio {gap:.2f}s after "
                                  f"response.create", flush=True)
                            if gap > 2.0:
                                print(f"[Realtime]   ^ that is dead air on the "
                                      f"callee's end before the greeting starts",
                                      flush=True)
                        twilio_payload = (delta if _passthrough_enabled()
                                          else _convert_oai_to_twilio(delta))
                        await twilio_ws.send_text(json.dumps({
                            "event":    "media",
                            "streamSid": sess.stream_sid,
                            "media":    {"payload": twilio_payload},
                        }))
                    except Exception as e:
                        log.error("[Realtime] audio send error: %s", e)
                elif delta and not sess.stream_sid:
                    log.warning("[Realtime] Audio delta received but stream_sid empty — dropped")

            # ── Agent transcript ───────────────────────────────────────────
            elif event_type == "response.output_audio_transcript.delta":
                _agent_text_buf += msg.get("delta", "")

            elif event_type == "response.output_audio_transcript.done":
                _agent_text_buf, _barge_in_pending = await _handle_agent_transcript(
                    msg, sess, oai_ws, _agent_text_buf, _barge_in_pending)

            # ── Caller transcript — replace placeholder if transcription enabled ──
            elif event_type == "conversation.item.input_audio_transcription.completed":
                await _handle_caller_transcript(msg, sess, oai_ws)

            # ── Tool call arguments streaming ──────────────────────────────
            elif event_type == "response.function_call_arguments.delta":
                call_id = msg.get("call_id", "")
                name    = msg.get("name", "")
                if call_id not in _pending_tools:
                    _pending_tools[call_id] = {"name": name, "args": ""}
                _pending_tools[call_id]["args"] += msg.get("delta", "")

            # ── Tool call complete ─────────────────────────────────────────
            elif event_type == "response.function_call_arguments.done":
                _out = await _handle_tool_call(
                    msg, sess, oai_ws, _pending_tools, _response_had_audio)
                # Only the flags the handler actually set travel back — see
                # _ToolOutcome on why None is not False.
                if _out.agent_text_buf is not None:
                    _agent_text_buf = _out.agent_text_buf
                if _out.closing_sent is not None:
                    _closing_sent = _out.closing_sent
                if _out.pending_response_create is not None:
                    _pending_response_create = _out.pending_response_create
                if _out.stop:
                    continue

            # ── Response done: extract token usage + check resolution ────
            elif event_type == "response.done":
                sess._response_active = False
                # `_response_spoke = _response_had_audio` stood here, assigned
                # and never read. It came in with c443356 (the 8.2s dead-air
                # fix) and was orphaned when that check moved to the model's
                # own `_out_audio_tokens` from the usage block, which is the
                # honest measure — our delta flag cannot see a response whose
                # audio we gated. Removed 2026-08-18.
                _response_had_audio = False   # reset for next response
                sess._responses    += 1
                # "completed" | "cancelled" | "incomplete" | "failed". This was
                # never read, so a closing response the caller talked over was
                # indistinguishable from one that actually played, and the call
                # hung up on a goodbye nobody heard.
                _resp_status = ((msg.get("response") or {}).get("status")
                                or "completed")
                # The model's own count of audio it produced. Zero on a
                # completed response means it said nothing at all, which on a
                # phone line is indistinguishable from the call having dropped.
                # Read from usage rather than from our local audio-delta flag so
                # that a response carrying a tool call, or one whose deltas we
                # gated, is judged by what the model actually emitted.
                _out_audio_tokens = (((msg.get("response") or {}).get("usage") or {})
                                     .get("output_token_details", {})
                                     .get("audio_tokens", 0))
                # Input tokens this response consumed. A response that was
                # REJECTED before it ran — conversation_already_has_active_response
                # is the one that matters — comes back failed having read
                # nothing, so both of these are zero. A response that genuinely
                # ran and simply produced no audio has read the conversation and
                # reports input tokens. That difference is the only way to tell
                # "say something, the line is dead" apart from "you already have
                # a response in flight", and re-requesting on the latter is what
                # produced the 25s of dead air on call-20260811-1640.
                _resp_in = (((msg.get("response") or {}).get("usage") or {})
                            .get("input_token_details", {}))
                _in_tokens = ((_resp_in.get("text_tokens")  or 0)
                              + (_resp_in.get("audio_tokens") or 0))
                # A response can be cancelled by US (the barge-in handler above,
                # which sets _barge_in_pending) or by OPENAI, whose server VAD
                # interrupts on caller speech on its own. Until now the second
                # kind was completely silent: status came back "cancelled",
                # nothing had logged a barge-in, and no `clear` was ever sent to
                # Twilio, so any audio already buffered there kept playing after
                # generation had stopped.
                #
                # Closing the response.created race above should make this rare
                # — our handler now fires first in the common case. It is kept
                # because "rare" is not "never": the server can still win the
                # race on a slow link, and an interruption path that only works
                # when we win a race is the thing that has been invisible for
                # eight sessions. Logged distinctly so the two are told apart in
                # the transcript rather than inferred.
                # A response that completed and made a sound means the agent has
                # since been heard, so any earlier truncation is no longer the
                # thing to read the next caller turn against. _REPAIR_WINDOW_S
                # bounds this by time; this bounds it by events, which is the
                # tighter of the two and the one that is actually the reason.
                if _resp_status == "completed" and _out_audio_tokens > 0:
                    sess._truncated_at = None
                if _resp_status == "cancelled" and not _barge_in_pending:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"✋ BARGE-IN  : cancelled by OpenAI's VAD "
                          f"(audio_out={_out_audio_tokens} tok)", flush=True)
                    if sess.stream_sid:
                        try:
                            await twilio_ws.send_text(json.dumps({
                                "event": "clear", "streamSid": sess.stream_sid,
                            }))
                        except Exception:
                            pass
                    sess.agent_speaking = False
                # A cancelled response may never emit transcript.done. Clearing
                # the flag only there meant it leaked into the NEXT response and
                # silently swallowed a real transcript line.
                _barge_in_pending   = False
                # Flush buffered agent audio as one contiguous block.
                # Placing it all at _current_response_start means the PCM runs at the correct
                # sample rate (24 kHz) from that point — no overlap, no gaps.
                if _current_response_pcm and _current_response_start is not None:
                    sess._agent_pcm.append((_current_response_start, b"".join(_current_response_pcm)))
                    print(f"[Realtime] Flushed agent response: {len(_current_response_pcm)} chunks, "
                          f"start={_current_response_start:.2f}s, "
                          f"dur={_samples_this_response/_wire_sample_rate():.2f}s", flush=True)
                _current_response_pcm.clear()
                _current_response_start = None
                # Dynamic echo cooldown: wait until audio finishes playing on the phone +
                # echo travel time.  response.done fires when the SERVER finishes generating
                # (fast), but the audio is still playing on the handset.  Using a fixed 0.5s
                # caused the agent to hear its own echo and generate a duplicate response.
                # Formula: playback_duration + 0.65s echo margin (min 0.5s for very short clips).
                # Wait until the audio has finished PLAYING on the handset,
                # then a small margin — and no longer, because caller audio is
                # dropped for this whole window.
                #
                # The old formula measured the wait from response.done, which
                # fires when the SERVER finishes generating. Generation runs
                # faster than realtime, so response.done lands well before
                # playback ends, and adding the full clip duration on top of it
                # over-waited by roughly the generation time — about 2s of
                # deafness added to every single turn, directly inflating the
                # measured 2.5-4s response latency.
                #
                # Playback ends at (first chunk sent) + (audio duration), since
                # Twilio plays what we send at realtime speed.
                _audio_seconds = _samples_this_response / _wire_sample_rate()
                if _first_delta_sent_at is not None:
                    _playback_ends_at = _first_delta_sent_at + _audio_seconds
                    _echo_cooldown = max(0.3, _playback_ends_at + 0.25 - time.monotonic())
                    # How much of this clip the callee has STILL not heard. The
                    # echo gate already reasons in these terms; the silence
                    # watchdog did not, and that was the bug — see the comment
                    # where _agent_quiet_since is set below.
                    _playback_remaining = max(0.0, _playback_ends_at - time.monotonic())
                else:
                    _echo_cooldown = max(0.3, _audio_seconds + 0.25)
                    # No delta was ever sent, so nothing is playing out.
                    _playback_remaining = 0.0
                _first_delta_sent_at = None
                _current_item_id = None
                _samples_this_response = 0
                asyncio.create_task(_end_speaking_gate(sess, _echo_cooldown))
                # Account each response's tokens ONCE. A live call logged the
                # same usage line twice, identical to the token
                # (in_text=4572 cached=4416 in_audio=372 out_audio=108), and
                # counted 6 responses against 4 audio blocks. Every duplicate
                # inflates the cost figure — the one number this project has
                # been trying to get honest.
                _resp_id = msg.get("response", {}).get("id")
                if _resp_id and _resp_id in _counted_responses:
                    log.debug("[Realtime] duplicate response.done for %s — "
                              "usage already counted", _resp_id)
                    usage = {}
                else:
                    if _resp_id:
                        _counted_responses.add(_resp_id)
                    usage = msg.get("response", {}).get("usage", {})
                if usage:
                    details_in  = usage.get("input_token_details",  {})
                    details_out = usage.get("output_token_details", {})
                    sess._input_audio_tokens  += details_in.get("audio_tokens",  0)
                    sess._input_text_tokens   += details_in.get("text_tokens",   0)
                    sess._output_audio_tokens += details_out.get("audio_tokens", 0)
                    sess._output_text_tokens  += details_out.get("text_tokens",  0)
                    # Cached tokens — the only direct evidence that the prompt
                    # cache is engaging. Shape varies by API version: a flat
                    # `cached_tokens` plus an optional per-modality breakdown.
                    cached = details_in.get("cached_tokens_details") or {}
                    c_audio = cached.get("audio_tokens", 0)
                    c_text  = cached.get("text_tokens",  0)
                    if not (c_audio or c_text):
                        # No breakdown available — attribute the flat total to
                        # text, which is where the static prompt prefix lives.
                        c_text = details_in.get("cached_tokens", 0)
                    sess._input_audio_cached_tokens += c_audio
                    sess._input_text_cached_tokens  += c_text
                    print(f"[Realtime] usage: in_text={details_in.get('text_tokens', 0)} "
                          f"(cached {c_text})  in_audio={details_in.get('audio_tokens', 0)} "
                          f"(cached {c_audio})  out_audio={details_out.get('audio_tokens', 0)}"
                          f"  [{_resp_status}]",
                          flush=True)
                # The agent has stopped talking; the ball is with the callee. If
                # they never speak, no VAD event fires and nothing else in this
                # loop will ever run again.
                #
                # The clock starts when the callee STOPS HEARING us, not when
                # response.done arrives. response.done fires when the server
                # finishes generating, and generation runs faster than realtime,
                # so this used to start counting while the agent was still
                # talking — the agent's own voice was counted as the callee's
                # silence. Measured on call-20260811-1649: the watchdog reported
                # 3.5s before "Are you still with me?" when the real gap was
                # 1.41s, and 7.0s before the goodbye when the real gap was 2.45s.
                # The error scales with clip length, so the longest turns were
                # cut off hardest — the call was hung up 2.45s after a handover
                # line, while the callee was still drawing breath.
                #
                # Pointing this at a moment in the FUTURE is intentional: the
                # watchdog compares time.time() - quiet_since, which simply goes
                # negative until playback ends.
                sess._agent_quiet_since = time.time() + _playback_remaining
                # Enable caller audio forwarding after first response (greeting) finishes
                if not sess.listen_enabled.is_set():
                    print("[Realtime] Greeting done — now listening to caller", flush=True)
                    sess.listen_enabled.set()
                # Deferred response.create from a tool result — safe now that the
                # previous response has completed.
                if _pending_response_create and not sess.done:
                    _pending_response_create = False
                    await _create_response(oai_ws, sess, why="deferred tool result")
                elif (not sess.done and _resp_status != "cancelled"
                      and _out_audio_tokens == 0 and _empty_responses < 2
                      and not (_resp_status == "failed" and _in_tokens == 0)
                      and not sess._response_active):
                    # A response that COMPLETED without producing any audio is
                    # dead air: nothing is queued behind it, so the line stays
                    # silent until the caller gives up and speaks. On a live
                    # call this ran 8.2 seconds and the caller asked "are you
                    # there?" — exactly what a person says to a dropped line.
                    # Only 'cancelled' is excluded — those are barge-ins, where
                    # silence is correct because the caller is talking. This
                    # used to require status == 'completed', so an 'incomplete'
                    # or 'failed' response producing no audio slipped through
                    # and became 10s of dead air on a live call. The status was
                    # not logged either, so there was no way to tell which.
                    #
                    # Widening it to 'failed' then caused the opposite failure.
                    # This is the sixth response.create call site and the second
                    # to be written without checking _response_active — the same
                    # bug 97ff46d fixed in the watchdog. A rejected response
                    # comes back failed, this handler read that as dead air and
                    # created another, which collided and failed in turn. Two
                    # guards, because the two causes are different: skip when a
                    # response is already in flight, and skip a failure that
                    # never consumed input, which is what a rejection looks like.
                    _empty_responses += 1
                    print(f"[Realtime] Response produced no audio — "
                          f"re-requesting to avoid dead air "
                          f"({_empty_responses}/2)", flush=True)
                    await _create_response(oai_ws, sess, why="empty response")
                if sess.done:
                    if _closing_sent:
                        # This is the tool-call response.done — closing response is being generated, wait for it
                        _closing_sent = False
                    elif _resp_status != "completed" and _closing_retries < 1:
                        # The goodbye was cancelled — the caller was still
                        # talking, so barge-in killed it. Hanging up here is
                        # what drops the line in silence. The goodbye item is
                        # still in the conversation; ask for it once more, after
                        # a beat so we are not talking over them again.
                        _closing_retries += 1
                        print(f"[Realtime] Closing response was {_resp_status} — "
                              f"caller talked over it. Retrying the goodbye once.",
                              flush=True)
                        # Hand the retry to the watchdog instead of sleeping
                        # here. This block runs INSIDE the event loop, so an
                        # `await asyncio.sleep(0.8)` stops us reading the
                        # socket for 0.8s — and OpenAI's server VAD creates its
                        # own response the moment the caller speaks. On
                        # call-20260818-1338 the caller said "Mercy Medical
                        # Center" during that sleep, `response.created` sat
                        # unread so `_response_active` was still False, and the
                        # retry went out against stale state:
                        #     conversation_already_has_active_response
                        # Sleeping inside an event handler means acting on a
                        # snapshot of the world taken before the nap.
                        #
                        # The watchdog is a separate task, so events keep being
                        # processed while it waits and `_response_active` is
                        # true by the time it fires.
                        sess._goodbye_retry_at = time.time() + 0.8
                        continue
                    else:
                        # This is the closing response.done.
                        # Wait for the FULL audio to finish playing on the caller's phone before hanging up.
                        # _echo_cooldown = audio_duration + 0.65s, computed just above from _samples_this_response.
                        # Sleeping only 1s was cutting off the goodbye mid-sentence.
                        hangup_wait = max(_echo_cooldown, 1.5)
                        print(f"[Realtime] Closing done — waiting {hangup_wait:.1f}s for audio to finish playing", flush=True)
                        await asyncio.sleep(hangup_wait)
                        print("[Realtime] Hanging up now", flush=True)
                        done_event.set()
                        try:
                            await twilio_ws.close()
                        except Exception:
                            pass
                        break

            elif event_type == "error":
                err  = msg.get("error", {})
                code = err.get("code", "")
                msg_text = err.get("message", "")
                # Suppress harmless errors
                if code == "response_cancel_not_active":
                    pass
                elif "input_audio_transcription" in msg_text or "unknown_parameter" in code:
                    print(f"[Realtime] Transcription not supported on this model — caller turns will show as '[...]'", flush=True)
                else:
                    # print, not log.error, for consistency with the rest of
                    # this module and for flush=True — an unflushed error that
                    # arrives after the call has ended is nearly as useless as
                    # no error.
                    #
                    # CORRECTION to an earlier version of this comment, which
                    # claimed these errors "went nowhere": they did NOT. With no
                    # logging config Python's lastResort handler prints WARNING
                    # and above to stderr, so log.error was visible all along.
                    # Only INFO and DEBUG are dropped — which is what actually
                    # hid the call outcome in twilio_worker's /status handler.
                    # The evidence for the 25s of dead air on call-20260811-1640
                    # is the [failed] responses reporting in_text=0 in_audio=0,
                    # not the absence of an error line.
                    print(f"[Realtime] API ERROR: {code} {msg_text}", flush=True)

    except websockets.exceptions.ConnectionClosed:
        log.info("[Realtime] OAI WebSocket closed normally")
    except Exception as e:
        # This used to log at INFO with no traceback, so a bug in the event loop
        # silently ended the call and looked, from the caller's side, exactly
        # like a dropped line. A NameError here cost 12 test failures that all
        # pointed somewhere else.
        log.exception("[Realtime] OAI→Twilio loop CRASHED: %s", e)
        print(f"\n[Realtime] ❌ EVENT LOOP CRASHED — the call was cut short.\n"
              f"           {type(e).__name__}: {e}\n"
              f"{traceback.format_exc()}", flush=True)
