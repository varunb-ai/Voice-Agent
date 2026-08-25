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
from agents.voice.objectives import (
    ACCEPTING_ASK,
    AnswerKind,
    IDENTITY_ASK,
    states_in_its_own_right,
    REFERRAL_ASK,
    SCHEDULING_ASK,
    CallObjective,
    LOCATION_NOUN as _LOCATION_NOUN,
    Outcome,
    clauses as _clauses,
    default_objective,
    describe as _describe_objective,
    expected_answers,
    norm_quotes as _norm_quotes,
    sentences as _sentences,
)
from agents.voice.templates import get_template
from agents.voice.tools import run_tool, TOOL_SCHEMAS
from agents.voice.audio_utils import resample, _mulaw_decode, _mulaw_encode
from agents.voice import backchannel

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


def _wire_bytes_per_ms() -> float:
    """Bytes of recording buffer per millisecond of audio."""
    return _wire_sample_rate() / 1000.0 * (1 if _passthrough_enabled() else 2)


def _utterance_slice(sess: "RealtimeSession",
                     start_ms: Optional[int],
                     end_ms: Optional[int],
                     fallback_chunk_pos: int) -> bytes:
    """The caller audio for one utterance, cut by OpenAI's own timestamps.

    THIS IS THE FIX FOR THE MEASUREMENT, and the bug it replaces was subtle.

    The old code marked the start as `len(sess._caller_pcm)` at the moment
    `input_audio_buffer.speech_started` ARRIVED. But that event is generated on
    a US server and travels to India — half a second to a second — and by the
    time it lands, the caller's audio is already sitting in our buffer. For a
    SHORT utterance the whole thing is buffered before the marker is set, so
    the slice contains only the silence that follows.

    The signature is unmistakable and appeared twice: audio_rms exactly
    0.000244140625, which is what a buffer of mu-law 0xFF — digital silence —
    decodes to. On call-20260819-2006 that number was recorded for a turn where
    the caller channel of the Twilio recording measures 0.2425, and on
    call-20260819-1847 for a "Yes, yes." Varun confirmed he said.

    Both times a guard then acted on it: the quarantine dropped the turn as
    "audio carried nothing" while the caller was audibly speaking. Right answer
    once, wrong answer once, right reason never.

    OpenAI already tells us where the speech was. `speech_started` carries
    `audio_start_ms` and `speech_stopped` carries `audio_end_ms`, both indexed
    into the very buffer we have been feeding it. Using those removes the
    arrival-time guess entirely.

    THE BUFFERS DO NOT START AT THE SAME PLACE — found 2026-08-20, and the
    reason the fix above did not work. `_caller_pcm` is appended for EVERY
    inbound frame, from the moment the Twilio stream starts, because save()
    needs the whole call for the recording. OpenAI's buffer starts later: the
    forward is behind `if not sess.listen_enabled.is_set(): continue`, and
    listening is only enabled once the greeting has finished playing. So
    OpenAI's ms indices are zeroed at "greeting done" while ours are zeroed at
    "stream start", and indexing one with the other reads that far too EARLY.

    Measured on call-20260820-1154, solving for the offset that reproduces the
    recorded audio_rms of all six caller turns against the Twilio recording:
    best fit 9.6s, against a greeting that ended at 9.50s. Offset 0 — what
    this function assumed — predicts 0.13-0.19 for every turn and matches none
    of them. The old 0.000244140625 signature came back for four turns of
    perfectly audible speech, because 9.6s before each of them the line was
    silent. One turn came back at 0.123, the LOUDEST on the call, because 9.6s
    before it the caller happened to be mid-sentence — and that turn was a
    transcription of silence, which the quarantine then waved through as the
    clearest speech on the call.

    So the fix is to shift by where OpenAI's buffer actually begins.

    CAVEAT worth knowing: this assumes that from `_listen_start_bytes` onward
    our buffer and OpenAI's hold the same audio. True while REALTIME_ECHO_GATE
    is "pass", because from there we append every frame and forward every
    frame. Under "energy" or "drop" frames are dropped mid-call, the two
    diverge again, and no fixed offset can express it — so it falls back to
    the chunk position if the timestamps land out of range.
    """
    if start_ms is None or end_ms is None or end_ms <= start_ms:
        return b"".join(sess._caller_oai_pcm[fallback_chunk_pos:])
    buf = b"".join(sess._caller_oai_pcm)
    bpms = _wire_bytes_per_ms()
    # Where OpenAI's input buffer begins inside ours. Zero until the greeting
    # finishes, which is also the only window in which no caller turn exists.
    base = sess._listen_start_bytes
    lo = base + int(start_ms * bpms)
    hi = base + int(end_ms * bpms)
    # Out of range means the buffers have drifted; the fallback is wrong too,
    # but it is wrong in the direction of measuring MORE audio rather than none.
    if lo >= len(buf) or hi <= lo:
        return b"".join(sess._caller_oai_pcm[fallback_chunk_pos:])
    return buf[lo:min(hi, len(buf))]


# The pause inserted between two replies that would otherwise be delivered as
# one unbroken run of speech. Matched to realtime_silence_ms (0.7s), which is
# already the project's answer to "how long is a gap that reads as a turn
# ending" — the callee needs at least as long to recognise their opening as
# OpenAI's VAD needs to recognise theirs.
_STACK_BREATH_S = 0.7

# 8kHz mu-law, one byte per sample, 0xFF is silence. Twilio's media frames are
# 20ms, so 160 bytes each.
_TWILIO_SILENCE_FRAME = base64.b64encode(b"\xff" * 160).decode()


async def _send_breath(twilio_ws, sess: "RealtimeSession", seconds: float) -> None:
    """Queue `seconds` of silence to Twilio so the callee gets a gap to speak.

    Sending silence rather than sleeping is deliberate. This runs inside the
    OpenAI event pump, and that task must keep reading: barge-in cancellation,
    response.done and the tool calls all arrive on it. Awaiting a sleep here
    would stall every one of them for the length of the pause — the pattern
    this module already refuses elsewhere (see the goodbye retry, which is
    owned by the watchdog task for exactly this reason).

    Twilio plays queued media in order, so appending silence lands the gap in
    the caller's ear without blocking anything on ours.
    """
    if not sess.stream_sid or seconds <= 0:
        return
    for _ in range(int(seconds * 1000 / 20)):
        await twilio_ws.send_text(json.dumps({
            "event": "media", "streamSid": sess.stream_sid,
            "media": {"payload": _TWILIO_SILENCE_FRAME},
        }))


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
#
# SPLIT IN TWO on 2026-08-24, and the split is the fix. The single list held
# "yes|yeah|yep|yup" alongside "hello|okay|mm", so `_is_filler_reply("Yes.")`
# was True and `_caller_answered_since` skipped that turn: the ask budget kept
# counting as though nobody had spoken and the give-up directive fired. "No."
# was not in the list and returned False. Only the POSITIVE answer was
# discarded — the one the client is calling to collect.
#
# That is correct for a location ask (a bare "yes" is not a place) and wrong for
# a yes/no field, and no template can change it because the judgement is made
# here, on the words alone. It now depends on what was ASKED — see
# objectives.expected_answers.
_ACK_WORDS = (r"hello|hullo|hi|hey|ok|okay|sure|right|alright|"
              r"mm+|hm+|uh+|um+|er+|ah+|oh+|go ahead|that'?s fine|i see|fine|"
              r"sorry|pardon|come again|say again|what|huh|"
              r"are you there|still there|can you hear me")
# ACKNOWLEDGEMENT ONLY, and these stay filler even when a yes/no answer is
# what we asked for. "Mm-hm" to "are you accepting new patients?" is an
# affirmative in real speech, and it is ALSO the exact text of our own
# backchannel clips — mm-hm, okay, right, sure — coming back up the line off a
# speakerphone. _BACKCHANNEL_ECHO_MARGIN_S says why that cannot be told apart
# after the fact: "There would be no way to tell our own echo from a real
# backchannel". Letting one of those four strings satisfy a field would let the
# agent answer its own question. So the affirmative set below is the EXPLICIT
# one only.
_ACK_REPLY = re.compile(rf"^(?:\W*(?:{_ACK_WORDS})\W*)+$", re.I)
# A bare affirmative, possibly padded with acknowledgements ("Yes, okay."). The
# four words are exactly the ones the old single list held, so the location-ask
# verdict on any given turn is unchanged.
_AFFIRM_REPLY = re.compile(
    rf"^(?:\W*(?:yes|yeah|yep|yup|{_ACK_WORDS})\W*)+$", re.I)
_HAS_AFFIRM = re.compile(r"\b(yes|yeah|yep|yup)\b", re.I)


def _is_filler_reply(text: str, agent_name: str = "",
                     expects: Optional[frozenset] = None) -> bool:
    """True if this caller turn answers nothing and asks nothing.

    `expects` is the set of answer kinds the pending ask entitles them to give
    (`objectives.expected_answers`). None means no ask is in view, and then this
    behaves exactly as it always did — a bare "yes" is filler — because the
    only ask this agent has ever made is for a place.

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
    if _ACK_REPLY.match(t):
        return True
    if _AFFIRM_REPLY.match(t) and _HAS_AFFIRM.search(t):
        # A bare "Yes." IS the answer to a closed-set ask and is NOT a place.
        return not (expects and AnswerKind.CHOICE in expects)
    # "No.", "Nope.", "Not sure.", "We're full — you'd be number twenty-one."
    # were never filler and still are not. They are answers to a yes/no field
    # and a refusal to a location ask, and both of those are information.
    return False


def _pending_expectation(sess: "RealtimeSession",
                         before_idx: int) -> Optional[frozenset]:
    """What the agent's most recent ask, at or before `before_idx`, asked for.

    None when there is no agent turn to read — a caller turn arriving before we
    have said anything is judged as it always was.
    """
    for t in reversed(sess.turns[:max(before_idx, 0)]):
        if t.role == "agent" and t.text.strip():
            return expected_answers(t.text, _objective_of(sess))
    return None


def _objective_of(sess: "RealtimeSession") -> CallObjective:
    """The objective this call is working to.

    getattr, because the guards in this module are routinely handed a namespace
    carrying only the four attributes they read — see `double()` in the test
    suite — and a guard that raises on a test double is a guard that stops being
    tested.
    """
    obj = getattr(sess, "objective", None)
    return obj if isinstance(obj, CallObjective) else default_objective()


def _caller_answered_since(sess: "RealtimeSession", since_idx: int) -> bool:
    """Did the caller say anything substantive after turn `since_idx`?

    Substantive is relative to the ask. The ask lives at `since_idx - 1` (the
    budget records the index AFTER appending the agent turn), so the pending
    expectation is read from there rather than from the words in isolation.
    """
    expects = _pending_expectation(sess, since_idx)
    for t in sess.turns[since_idx:]:
        if (t.role == "caller" and t.text.strip() != "[...]"
                and not _is_filler_reply(t.text, sess.agent_name, expects)):
            return True
    return False



# _MAX_UNANSWERED_REASKS is GONE, and its disappearance is the shape of the
# 2026-08-24 budget change rather than a deletion.
#
# It existed because the budget counted asks the caller HAD answered, so an ask
# nobody answered spent nothing and a caller who only ever said "hello" would
# have kept the call alive forever. Two unanswered re-asks were therefore forced
# to count anyway, purely for liveness.
#
# The budget now counts the unanswered ones — that is the whole change — so the
# forcing has nothing left to do: an unanswered re-ask spends budget on the
# first one, not the third. See settings.realtime_max_unanswered_asks.

# How many times the caller may question the agent back before those exchanges
# start costing budget again. Three because a front desk screening a cold call
# reasonably asks who you are, what it concerns, and whether it is urgent —
# call-20260819-2121 asked exactly those three and got hung up on. Bounded for
# the same reason as _MAX_UNANSWERED_REASKS: without it, a caller who only ever
# asks questions would keep the call alive indefinitely.
_MAX_VETTING_REASKS = 3


def _caller_vetted_since(sess: "RealtimeSession", since_idx: int) -> bool:
    """Since turn `since_idx`, did the caller ONLY question the agent back?

    Every substantive caller turn must be a vetting turn. One real answer, or
    one refusal, and this is False — the budget should advance normally then.
    """
    seen = False
    expects = _pending_expectation(sess, since_idx)
    for t in sess.turns[since_idx:]:
        if t.role != "caller" or t.text.strip() == "[...]":
            continue
        if _is_filler_reply(t.text, sess.agent_name, expects):
            continue
        if not _caller_is_vetting(t.text, sess):
            return False
        seen = True
    return seen

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

# ── Backchannels ─────────────────────────────────────────────────────────────
# How long the caller must be mid-utterance before a listener would make a
# noise. Under ~2s a person is still just listening; past it, silence starts to
# read as absence. Deliberately conservative: a badly-timed "mm-hm" is worse
# than none, and this fires on elapsed speech rather than on a detected pause
# because the pause is not observable from the events we get.
_BACKCHANNEL_AFTER_S = 2.8
# At most one per caller utterance, and never twice inside this window — two in
# quick succession is a tic, not listening.
_BACKCHANNEL_COOLDOWN_S = 9.0

# How long after a backchannel finishes playing its echo may still arrive back
# up the line. A callee on speakerphone hears our "mm-hm" out of their handset
# speaker and their own mic picks it up, delayed by the acoustic path plus
# Twilio's buffering.
#
# This window exists because realtime_echo_gate CANNOT cover it. That gate is
# consulted only under `sess.agent_speaking`, and a backchannel deliberately
# does not set that flag — it must not, or it would break barge-in and turn
# detection. So during a clip there is no gate in the path at all, whatever
# REALTIME_ECHO_GATE is set to.
#
# The failure it prevents is invisible from the transcript, which is why it is
# a guard and not something to watch for: the clips are "mm-hm", "okay",
# "right", "sure", and a caller genuinely saying "Okay." is the same string.
# There would be no way to tell our own echo from a real backchannel after the
# fact — so it has to be stopped at the audio, not detected in the text.
_BACKCHANNEL_ECHO_MARGIN_S = 0.4

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


# NOTE THE `\s*` AFTER `who`. It used to be a literal space, so the pattern
# needed "who 's" and never matched the contraction — "who's calling?" and
# "who's this?", which are how the question is actually asked. On
# call-20260820-1440 the caller said "Sorry, who's calling again?", this did
# not fire, the identity nudge never went out, and _is_reintroduction then
# flagged the perfectly correct answer as a re-introduction. The detector that
# should have fired did not; the one that should not have, did.
#
# "who is calling" still matches: \s* allows the space, it does not require it.
_IDENTITY_ASK = re.compile(
    r"(who\s*(is|are|am i|'s) (this|you|speaking|calling|i speaking)|"
    r"who\s*'s (this|calling|speaking)|"
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

# An acknowledgement together with the location noun it takes as its OBJECT.
#
# _NOT_AN_ASK strips the acknowledgement WORD and leaves the noun it governs,
# so "Thanks for the location." became " for the location." — still a location
# noun, still counted as an ask. Observed on call-20260820-1915: seven
# location_asks against a limit of four, and the verbatim-ask nudge firing to
# tell the agent to "stop stapling it on" about a sentence that asks nothing.
# It cost nothing that call — holds had reset the budget — but an inflated
# count ends a call early on a call without holds.
#
# The distinguishing feature is grammatical, not vocabulary: in the failing
# family the noun is the acknowledgement's object, not part of a fresh request.
# So consume the phrase whole, before the residue test runs.
#
# THE NEGATIVE LOOKAHEAD IS LOAD-BEARING. Without it the two-word gap jumps a
# clause boundary — "Great — and which campus is that" had "and which" eaten
# and the real question with it, which is the expensive direction: a missed ask
# lets the agent pester someone. Words that open a new clause end the object.
_ACK_TAKES_VALUE = re.compile(
    r"\b(thanks|thank you|appreciate|got it|perfect|great)\b"
    r"[,\s—\-]*"
    r"(?:for|on|about)?[,\s]*"
    r"(?:the|that|this|your|those)?\s*"
    r"(?:(?!(?:and|but|so|which|what|where|who|if|when|still|need)\b)\w+\s+){0,2}"
    r"(?:branch|location|office|campus|site|address)\b", re.I)

# Reading back a value the caller already gave.
# READING A VALUE BACK IS NOT ASKING FOR ONE, and the list has to cover the
# agent QUOTING the caller as well as the agent filing the value. On
# call-20260824-2014 the agent said "I heard you say she's taking the new
# patients." — a read-back by any reading — and because that phrasing was
# missing here it scored as an ASK. The grounding anchor moved past every
# caller turn that had answered, the evidence window emptied, and the guard
# stood down and accepted a status it had refused three times. The agent talked
# its own claim into the record.
_CONFIRMS_VALUE = re.compile(
    r"\b(i have that as|i'?ve got that|i'?ll note|noted as|recorded as|"
    r"i'?ll put (that|it) down|so that'?s|i heard you say|"
    r"you said|what i heard|let me read (that|it) back|"
    # NOT a bare "to confirm". "I'm trying to confirm which branch she works
    # out of" is an ASK, and swallowing it would stop the budget counting the
    # commonest phrasing the agent has. The read-back sense always quotes
    # THEM — "I heard you say", "you said" — and that is the load-bearing part.
    r"i'?ll record (that|it))\b", re.I)

# Reporting that the location was NOT obtained. Names a location noun and reads
# as an ask to the inverted detector, but it is the opposite — it is the agent
# giving up. On call-20260818-1338 "I wasn't able to get the specific branch
# today" was counted as an ask, so a closing line spent a slot of the ask
# budget. Only checked on statements: "I couldn't find the branch — do you know
# it?" carries a question mark and is a genuine ask.
_REPORTS_FAILURE = re.compile(
    r"\b(was ?n'?t able|were ?n'?t able|was not able|could ?n'?t|could not|"
    r"can'?t|cannot|unable|did ?n'?t manage|no luck)\b", re.I)

# _LOCATION_NOUN, _norm_quotes, _sentences and _clauses now live in
# agents/voice/objectives.py and are imported at the top of this file under
# these same private names. They moved because the ask-shape detection there
# has to recognise a location noun and split a turn into clauses EXACTLY as the
# detectors here do — the branch field's probe and `_is_location_ask` are two
# readings of one pattern, and a second copy would drift the way tools.py's 41
# hand-copied prompt phrases drifted before they were derived instead.
#
# Why _norm_quotes exists at all, kept here because it is the reason not to
# "simplify" it away: the model writes TYPOGRAPHIC apostrophes — "wasn’t",
# "it’s" — and every pattern in this file spells them ASCII ("n'?t"). On
# call-20260818-1338 "I wasn’t able to get the specific branch today" was
# counted as a location ask because _REPORTS_FAILURE could not see "wasn’t".


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
    r"|that'?s all i (need|needed)|that'?s (what|all) i needed"
    r"|i have (everything|what) i need"
    r"|all (set|sorted|done))\b", re.I)


def _claims_saved(text: str) -> bool:
    """Did this agent turn tell the caller the location is recorded, or done?"""
    return bool(_CLAIMS_SAVED.search(_norm_quotes(text or "")))


def _is_ask_for(text: str, probe) -> bool:
    """Is this agent turn asking for the thing `probe` recognises?

    The body of what used to be _is_location_ask, with the noun pattern passed
    in. Parametrised rather than copied when a second field arrived: the
    acknowledgement, read-back and closing exemptions below were each added
    after a live call miscounted, and a second copy of them would have to
    relearn every one of those calls.

    Counts statement-form asks as well as questions. A request phrased politely
    is still a request, and the person on the other end experiences it as one.

    This used to be a whitelist of phrasings requiring a question mark, and it
    scored 0 asks on a call that asked four times — the agent had simply picked
    wordings that were not on the list ("trying to confirm" where the list held
    "trying to find out"). Enumerating phrasings cannot work: the model has more
    ways to ask than anyone can list.

    So it is inverted. Naming the thing IS an ask unless the turn is plainly
    acknowledging or closing. This over-counts a little, which is the safe
    direction for a budget whose purpose is to stop the agent pestering people.
    """
    text = _norm_quotes(text)
    if not probe.search(text):
        return False
    # Reading a value back is not asking for one.
    if "?" not in text and (_CONFIRMS_VALUE.search(text)
                            or _REPORTS_FAILURE.search(text)):
        return False
    if "?" in text:
        return True
    # An acknowledgement that goes on to ask for something is still an ask, so
    # only a turn that is ENTIRELY acknowledgement is exempt. Take the
    # acknowledgement's own object with it first — see _ACK_TAKES_VALUE — or
    # "Thanks for the location." leaves a location noun behind and reads as a
    # request for the thing it is thanking them for.
    stripped = _ACK_TAKES_VALUE.sub("", text)
    stripped = _NOT_AN_ASK.sub("", stripped)
    return bool(probe.search(stripped))


def _is_location_ask(text: str) -> bool:
    """Is this agent turn asking where the doctor practises?"""
    return _is_ask_for(text, _LOCATION_NOUN)


def _is_objective_ask(text: str, sess: "RealtimeSession") -> bool:
    """Is this agent turn asking for ANY field the call is trying to collect?

    THE GATE THAT FEEDS THE ASK BUDGET, and the one part of that budget that was
    NOT objective-agnostic. The counters never knew about branches — they count
    asks and answers — but nothing reached them except through
    `_is_location_ask`, so on a template collecting a second field every ask
    about that field was invisible: not counted as progress, not counted as
    unanswered, and not bounded by anything. A caller who stonewalled the
    new-patient question specifically would have kept the call alive with no
    exit, which is the exact failure the budget was built for.

    Each field's probe already exists — it is what tells expected_answers which
    kind of answer an ask entitles the caller to give — so this reads the
    objective rather than adding a second list of nouns to keep in step.
    """
    if _is_location_ask(text):
        return True
    for f in _objective_of(sess).fields:
        # PLACE is covered above, with the pattern the guards all share.
        if f.kind is not AnswerKind.PLACE and _is_ask_for(text, f.probe):
            return True
    return False


# The escalate reason for each give-up trigger. SEPARATE STRINGS, because the
# old single condition had a single reason — "caller engaged but never provided
# a location" — and it was already the wrong sentence half the time it went
# out: a caller who said nothing at all did not "engage", and a reason in the
# record is read later as fact by someone with no way to check it. That is the
# failure _discarded_location exists to catch, and it is checked against BOTH
# of these (neither appears in _CALL_SHAPE_EXITS, so both are examined).
GIVE_UP_REASONS = {
    "no_progress": "caller engaged but never provided a location",
    "unanswered":  "caller did not answer after repeated asks",
}

# The invariant fragment of each directive — no counts, no wording that a
# rewrite would move. It exists so the test suite can assert this directive is
# ABSENT from a call that must not have given up, without holding a copy of the
# sentence: an absence assertion against a hand-copied literal starts passing
# for free the moment the real text changes, and that is the one failure this
# check is for.
#
# NOT interpolated into the directive below — the source-directive scanner
# (test_realtime_protocol.py's "every injected directive is found") locates
# every open-quote-paren-system-colon literal by regex and requires real
# lowercase TEXT immediately after it; an f-string placeholder sitting right
# after the quote gives it nothing to match and the directive goes uncounted.
# So these stay written out as literal words in give_up_directive, and the
# test suite proves the two copies still agree — a drift between them fails
# LOUDLY there instead of the marker silently going stale.
GIVE_UP_MARKERS = {
    "no_progress": "you have now asked for the location",
    "unanswered":  "they have not answered",
}


def give_up_directive(sess: "RealtimeSession", trigger: str) -> str:
    """The mid-call directive that ends a call the budget has run out on.

    A FUNCTION, not an inline f-string, so the test suite asserts on the text
    that actually goes out instead of on a copy of it. The check that the budget
    did not burn on a front desk's screening questions is an assertion that this
    directive is ABSENT, and an absence assertion against a hand-copied literal
    passes for free the day the wording changes — see the find/prove/judge rule.

    "Thank them briefly, say goodbye" produced exactly that: "Thanks for your
    time, goodbye." The callee is never told the call is ending because the
    agent could not get what it came for, so they get no last chance to supply
    it — and people often do, once they hear something was missed. So: name the
    outcome, own it rather than blame them, then close.
    """
    reason = GIVE_UP_REASONS.get(trigger, GIVE_UP_REASONS["no_progress"])
    _mem = getattr(sess, "memory", None)
    missing = (_objective_of(sess).missing_spoken(_mem) if _mem is not None
               else "") or "the branch"
    if trigger == "unanswered":
        opening = (f"(system: you have asked {sess._unanswered_asks} times "
                   f"and they have not answered. ")
    else:
        opening = (f"(system: you have now asked for the location "
                   f"{sess._asks_without_progress} times and have not been "
                   f"given one. ")
    return (
        opening
        + f"Stop asking. Say plainly that you were not able to get "
          f"{missing} today — phrase it as something you could not do, not as "
          f"something they failed to give — then thank them and say goodbye. "
          f"Do not ask again, and do not sound annoyed. Call escalate with "
          f"reason '{reason}'.)"
    )


def _ask_budget_outcome(turns: list, sent_at: Optional[int],
                        sent: bool, escalated: bool,
                        trigger: str = "no_progress") -> dict:
    """What happened after the give-up directive was injected.

    The count alone is not enough. Thanking them and escalating in one turn is
    the directive working. Taking two turns where the first contains another
    question is the directive landing but not taking effect — a soft version of
    the model ignoring it outright, and worth telling apart, because the fix
    differs: a wording tweak versus enforcing the budget at the response level
    instead of asking nicely in a user turn.
    """
    if not sent or sent_at is None:
        return {"unanswered_limit": settings.realtime_max_unanswered_asks,
                "no_progress_limit": settings.realtime_max_asks_without_progress,
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
        "unanswered_limit": settings.realtime_max_unanswered_asks,
        "no_progress_limit": settings.realtime_max_asks_without_progress,
        # WHICH ceiling ended the call, in the artifact. Without it the two
        # failures — they stopped talking, versus they talked and never told —
        # are one number afterwards, and they call for opposite fixes.
        "trigger": trigger,
        "directive_sent": True,
        "agent_turns_after": len(after),
        "asked_again_after": asked_again,
        "escalated": escalated,
        "verdict": verdict,
        "turns_after": [t.text[:90] for t in after],
    }


# _sentences and _clauses moved to objectives.py (imported above). The reason
# _clauses exists is worth keeping where the detectors that depend on it live:
# the repeat detector counted SENTENCES and reported 0 for a call containing a
# 45-character exact repeat. call-20260818-1613:
#
#     turn 1: "...about a doctor listing — which branch is Dr. Okafor
#              working out of?"
#     turn 3: "I can hear you now — which branch is Dr. Okafor working
#              out of?"
#
# Neither turn has an internal sentence break, so each was one "sentence", the
# two differed, and nothing was counted. The repeated part is the clause after
# the dash, and that is not a coincidence of this call: the prompt's own turn
# shape is "React, THEN say the thing, folded into ONE sentence", which produces
# exactly `reaction — ask`. The ask is the unit that gets repeated, and it almost
# never sits at a sentence boundary. It is also the unit whose SHAPE decides
# what answer the caller is entitled to give — see objectives.expected_answers.


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


# The caller announcing that THEY will go and do something. This is what
# distinguishes "hang on, let me check" from "hang on, who are you?" — the
# first promises an answer, the second demands one.
_CALLER_WILL_ACT = re.compile(
    r"(?:\b(?:i|we)\b|let me|lemme)[^.?!]{0,24}"
    r"\b(?:check|look|see|find|ask|grab|pull|get|confirm)\b", re.I)


def is_hold_request(text: str) -> bool:
    """Is the caller asking for time to go and find the answer?

    This is the opposite of refusing. A live call ended because the give-up
    directive had already fired, and the caller's very next words were "can you
    please give me a minute? I just need to check" — the most cooperative thing
    said on that call. The agent thanked them and hung up while they were on
    their way to look it up.

    "HANG ON" IS NOT ALWAYS A HOLD. On call-20260819-1915 the caller said
    "Hang on, are you a real person or is this a recording?" and this returned
    True. She was challenging the agent, not going to look anything up, and the
    console duly printed "Caller is going to check".

    That was harmless before _HOLD_GRACE_S existed. It is not harmless now:
    a hold silences the watchdog for 45 seconds, so a caller who says "hang on,
    who is this?" and then waits for an answer would be met with 45 seconds of
    nothing. A regression introduced by the hold fix itself, on the very next
    call.

    The discriminator is who is being asked to do something. A hold says the
    CALLER will act — "let me check", "give me a minute". A challenge asks the
    AGENT — "are you a real person?", "who did you say you were?". So a turn
    that puts a question to the agent is not a hold, unless it is the ordinary
    "can you hold on a moment?" form, which asks the agent to WAIT rather than
    to answer.
    """
    t = _norm_quotes(text or "")
    if not _HOLD_REQUEST.search(t):
        return False
    # The caller saying THEY will go and do something settles it, whatever
    # else is in the turn. "can you please give me a minute? I just need to
    # check" is a question, addresses the agent as "you", and is the most
    # cooperative sentence on that call — a second-person test alone rejects
    # it, which is the mistake this replaced.
    if _CALLER_WILL_ACT.search(t):
        return True
    # "can you hold on a moment?" asks the agent to WAIT, not to answer.
    if re.search(r"(?:can|could|would)\s+you\s+(?:just\s+|please\s+)*"
                 r"(?:wait|hold|hang on)", t, re.I):
        return True
    # Otherwise a question put to the agent wants an answer, not time.
    if "?" in t and (_IDENTITY_ASK.search(t)
                     or re.search(r"\b(you|your|you'?re)\b", t, re.I)):
        return False
    return True


# "Is this about a patient?" — asked twice in two calls, half-answered both
# times. On call-20260819-1847 and again on -1915 the caller asked "is this
# about a patient, or something urgent?" and the agent answered only the second
# half: "No, nothing urgent — it's just a listing check."
#
# At a medical office that omission is not a nicety. Whether a call concerns a
# patient decides whether they pull a record, route to clinical staff, or start
# thinking about PHI. Leaving it to be inferred from "listing check" is exactly
# the ambiguity a front desk is trained not to accept.
#
# The prompt already says "Several questions at once -> Answer EVERY one of
# them", and it did not hold twice running. So the process asks instead: this
# question is predictable and high-frequency for a medical cold call, the same
# way _IDENTITY_ASK is, and gets the same treatment.
_PATIENT_ASK = re.compile(
    r"\b(?:is|it'?s|this is|are you)\b[^?]{0,40}\babout\s+(?:a\s+|any\s+)?"
    r"patient\b|\bpatient(?:'?s)?\s+(?:related|matter|issue|record)\b"
    r"|\babout\s+(?:one of\s+)?(?:our|my|a)\s+patients?\b", re.I)


def _asks_about_patient(text: str) -> bool:
    """Did the caller ask whether this concerns a patient?"""
    return bool(_PATIENT_ASK.search(_norm_quotes(text or "")))


# The caller putting a question TO the agent instead of answering theirs.
#
# call-20260819-2121, in sixty seconds:
#   "Sorry, who's calling again?"
#   "Um, is this about a patient or something urgent?"
#   "Is this about patient related?"
#   "How can I help you?"
# Four turns, four questions, no refusal anywhere — a front desk deciding
# whether this call is safe to engage with, which is their job. The ask budget
# counted every one of them as an ask that went unanswered, hit its limit of
# four, and told the agent to escalate. The agent then hung up on "How can I
# help you?" — an open door, and the clearest invitation on the whole call.
#
# `_caller_answered_since` was the wrong instrument to lean on here: it asks
# "did they say something substantive", and a question IS substantive. It just
# is not a refusal, and the budget exists to end calls that are going nowhere,
# not calls where the other person is still working out who they are talking
# to.
#
# Matched by SHAPE, not by a phrase list. Interrogative opener, or an offer of
# help, in a turn that contains no location — an open set of wordings with a
# closed set of shapes.
_VETTING_OPENER = re.compile(
    r"^\W*(?:um+|uh+|er+|so|sorry|okay|ok|alright|yeah|well|hi|hello)?[\s,]*"
    r"(?:who|what|why|which|where|how|is|are|was|were|do|does|did|can|could|"
    r"would|will|may|might|should|sorry)\b", re.I)

# An explicit offer to keep going. Stronger than a screening question: they are
# not deciding whether to engage, they have decided and are waiting on you.
_INVITATION = re.compile(
    r"\bhow\s+(?:can|may|could)\s+i\s+(?:help|assist)\b"
    r"|\bwhat\s+can\s+i\s+(?:do|help)\b"
    r"|\bwhat\s+(?:do|did)\s+you\s+need\b"
    r"|\bwhat(?:'?s| is)\s+(?:this|it)\s+(?:regarding|about|in regard)\b"
    r"|\bgo\s+ahead\b|\bhow\s+can\s+i\s+help\b", re.I)


def _invites_continuation(text: str) -> bool:
    """The caller asking what the agent wants — an open door, not a refusal.

    Blocking escalation on this is the same move as blocking it on a hold
    request. A caller who says "How can I help you?" has told you they are
    willing; ending the call there throws away the one turn most likely to
    produce an answer.
    """
    return bool(_INVITATION.search(_norm_quotes(text or "")))


def _caller_is_vetting(text: str, sess: "RealtimeSession") -> bool:
    """The caller questioning the agent rather than answering, or declining.

    NOT a refusal and NOT an answer — a third thing the budget had no category
    for. Requires the turn to carry no location: "Which branch? The Mission Bay
    one." opens with an interrogative and is plainly an answer, so a shape test
    alone would misread it.
    """
    t = _norm_quotes(text or "").strip()
    if not t:
        return False
    if _invites_continuation(t):
        return True
    if not ("?" in t or _VETTING_OPENER.match(t)):
        return False
    # A turn that NAMES something is an answer however it is phrased. "Which
    # one — the Mission Bay clinic?" opens with an interrogative and is plainly
    # an answer, so the shape test alone would misread it. Same capitalisation
    # signal the grounding checks use, and the same caveat: skip the first word
    # (always capitalised) and skip what we brought to the call ourselves.
    known: set[str] = set()
    known |= _distinctive(getattr(sess.doctor, "hospital_name", "") or "")
    known |= _distinctive(sess.org_name or "")
    known |= {w for w in re.findall(r"[a-z]+",
                                    (getattr(sess.doctor, "doctor_name", "") or "").lower())
              if len(w) > 2}
    if sess.agent_name:
        known.add(sess.agent_name.lower())
    #
    # A proper noun alone is not enough: the first live case was "This is
    # Northside Medical Group and I'm Varun. Sorry, who's calling again?" —
    # which is vetting, and "Varun" is the caller's own name, not a branch. So
    # the word must also sit within two words of a location anchor, the same
    # conjunction _candidate_location uses.
    raw = [w.strip(".,!?-—'\"") for w in t.split()]
    words = [w.lower() for w in raw]
    for i, w in enumerate(words):
        if i == 0 or len(w) <= 2 or not w.isalpha():
            continue
        if (w in known or w in _UNGROUNDED_STOPWORDS or w in _NON_PLACE
                or w in _ORG_STOPWORDS):
            continue
        if not raw[i][:1].isupper():
            continue
        near = words[max(0, i - 2):i] + words[i + 1:i + 3]
        if any(n in _LOCATION_ANCHORS for n in near):
            return False
    return True


def _turn_asserts(text: str, sess: "RealtimeSession") -> bool:
    """Is this caller turn TELLING us something, or ASKING us?

    Grounding compares a saved value against the caller's own words, and until
    2026-08-20 it did that over one blob of every caller turn — so a value the
    caller ASKED about grounded exactly as well as one they stated.

    call-20260820-1703: the caller said "She's in San Francisco, right?" and
    never confirmed it afterwards. `city: "San Francisco"` was written to the
    directory stamped "verified against caller transcript". They had asked us.
    A receptionist seeking OUR confirmation is not evidence, and we had none to
    give — the record holds an organisation, not a city.

    The distinction already existed in this file. `_caller_is_vetting` was
    built for the ask budget, after the agent hung up on "How can I help you?",
    and it carries the hard part: a capitalised proper noun within two words of
    a _LOCATION_ANCHORS word makes the turn an ANSWER however interrogative its
    shape. That is what keeps "Which one — the Mission Bay clinic?" and "It's
    Mission Bay Clinic, right?" usable. Grounding simply never consulted it.

    THE "?" CONJUNCT IS NOT DECORATION. _caller_is_vetting also fires on
    _VETTING_OPENER alone, with no question mark, and the first cut of this
    predicate threw away "Sorry, Northgate." — a bare answer opening with an
    opener word. That is the case the rule right below defends in its own
    comment: "'Northgate' on its own is a perfectly good answer". Losing a real
    answer is the expensive direction, so an actual question mark is required
    before a turn can be discounted.

    NEGATION IS NOT HANDLED HERE and is tracked separately: "We're not
    Northside Medical Group" still grounds "Northside Medical Group". It is a
    different axis — a denial, not a question — and wants its own predicate.
    """
    return not ("?" in (text or "") and _caller_is_vetting(text, sess))


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
      back_to_back_asks  — agent asked again WITHOUT BEING ANSWERED in between.
      repeated_sentences — same agent sentence said more than once.
    """
    agent = [t for t in turns if t.role == "agent"]
    stapled = back_to_back = 0
    prev_agent_asked = False

    for i, turn in enumerate(turns):
        if turn.role != "agent":
            # A CALLER ANSWER BREAKS THE RUN, and until 2026-08-24 it did not:
            # this loop skipped straight past caller turns, so prev_agent_asked
            # carried across them and any two AGENT turns that both asked
            # something counted — however well the call was going.
            #
            # call-20260824-1604 scored 1 on a flawless exchange: greeting asked
            # for the branch, the caller gave it, the agent asked the second
            # scripted question. That was tolerable while the script had one
            # question, because a second ask usually WAS a re-ask. On a
            # four-question script every healthy call trips it on every adjacent
            # pair, and a metric that fires on the good case is worse than no
            # metric — it is the number people stop reading.
            #
            # The defect actually worth counting is asking again into silence,
            # so silence is what has to persist the run. Filler is silence:
            # "Hello." after a barge-in truncation is the case the ask budget
            # was built around.
            if turn.role == "caller" and turn.text.strip() != "[...]" \
                    and not _is_filler_reply(turn.text):
                prev_agent_asked = False
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

    # Adjacent agent turns that are word-for-word identical.
    #
    # Separate from `repeated_sentences` because it needs no length floor. The
    # ≥4-word floor above is there so "Got it." said in six different turns is
    # not counted as five repetitions — across a call, a short stock phrase
    # recurring is normal speech. Back to back it is not: "Sure, no rush. Sure,
    # no rush." has no innocent reading, and the floor was the only reason
    # call-20260819-2044 scored zero on a repeat the live detector had already
    # flagged in the console.
    back_to_back_repeats = sum(
        1 for a, b in zip(agent, agent[1:])
        if _norm_clause(a.text) and _norm_clause(a.text) == _norm_clause(b.text))

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
        "back_to_back_repeats": back_to_back_repeats,
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


# Reasons that describe the SHAPE of the call rather than what the caller said.
# A place name in the transcript says nothing about whether they are true — a
# voicemail greeting names the practice, a wrong number names the bakery — and
# blocking these strands the agent on a call it must be able to end.
#
# NOTE THE POLARITY, because it is the whole point. _NO_LOCATION_CLAIMS was an
# INCLUSION list: check only these wordings, and a wording not on it means a
# discarded answer and a lost call. This is an EXEMPTION list: a wording not on
# it means we CHECK, and the cost of a miss is one blocked turn against a
# one-shot flag. Same shape of list, opposite direction of failure.
_CALL_SHAPE_EXITS = (
    "wrong number", "voicemail", "declined to share", "no response",
    "non-medical", "not a medical",
)


def _discarded_location(reason: str, sess: "RealtimeSession") -> str:
    """Block an escalation claiming nothing was given when something was.

    Returns a rejection description, or "" to allow the escalation.

    THE TRANSCRIPT DECIDES, NOT THE MODEL'S WORDING. This used to run
    _candidate_location only when the reason matched _NO_LOCATION_CLAIMS — a
    phrase whitelist checked against text the model composes freely. On
    call-20260821-1152 the caller said "She works at Mission Bay clinic in San
    Francisco, but I'm not sure which location that is", the model escalated
    with "caller could not provide...", and the list holds "did not provide"
    but not "could not provide". One word, guard silent, and a branch that
    grounds cleanly — it saved on the previous call — was thrown away.
    Enumerating the model's phrasings cannot work; _is_location_ask was
    inverted for the same reason and says so in its own docstring.
    """
    if any(m in reason.lower() for m in _CALL_SHAPE_EXITS):
        return ""
    # Reaching the WRONG ORGANISATION is a legitimate exit even when a place
    # was named, and the place named is usually the wrong organisation itself.
    # Detected structurally rather than by another phrase list.
    if hospital_mismatch(sess):
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
    # STILL PLAYING is not the same as STILL GENERATING, and _response_active
    # only knows the second. OpenAI produces a reply far faster than realtime —
    # a 6.25s turn arrives in about a second — and we forward every delta to
    # Twilio immediately, so the rest sits in Twilio's queue long after OpenAI
    # calls the response done.
    #
    # Creating the next one then does not talk over the caller; it APPENDS.
    # They hear one unbroken monologue with no gap to speak into. On
    # call-20260819-2006 that surfaced as three identical questions inside a
    # single 50-word turn, and the callee hung up.
    #
    # The closing sites are exempt: a goodbye that waits for the queue to drain
    # is a goodbye that arrives after the line is already being torn down.
    _left = sess._playback_ends_at - time.monotonic()
    if _left > 0 and not allow_when_done:
        log.info("[Realtime] response.create skipped (%s): %.1fs of audio is "
                 "still playing out to the caller", why, _left)
        return False
    await oai_ws.send(json.dumps({"type": "response.create"}))
    return True


# ── The caller gave more than we recorded ────────────────────────────────────
# call-20260819-1847: she said "it's the Mission Bay clinic, 1825 Fourth
# Street" and the agent saved just "Mission Bay Clinic". Nothing blocked the
# fuller value — grounding accepts "Mission Bay Clinic, 1825 Fourth Street" —
# the model simply left it out, despite the prompt saying "Several: pass them
# all, comma-separated".
#
# The mirror image of the same morning's failure, where it INVENTED a street
# number. Both are one question asked in opposite directions: does the record
# match what the caller said? _ungrounded_terms asks whether we recorded too
# MUCH. This asks whether we recorded too LITTLE.
#
# A street number is the most specific thing a receptionist can give and the
# hardest to recover afterwards — "Mission Bay Clinic" may be one of several
# sites; 1825 Fourth Street is not.
_STREET_SUFFIX = (r"street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|"
                  r"lane|ln|way|parkway|pkwy|court|ct|place|pl|terrace|"
                  r"circle|cir|highway|hwy|suite|ste|floor")

# A house number followed within a few words by a street-type word. BOTH parts
# are required: a bare number is a suite count, a year, a number of branches,
# or noise.
_STREET_ADDRESS = re.compile(
    r"\b(\d{1,6})\s+((?:[A-Za-z0-9'\-\.]+\s+){0,3}?(?:" + _STREET_SUFFIX + r"))\b",
    re.I)


def _address_offered(sess: "RealtimeSession") -> Optional[str]:
    """A street address the caller gave, or None. The latest one wins."""
    found = None
    for t in sess.turns:
        if t.role != "caller" or t.text.strip() == "[...]":
            continue
        for m in _STREET_ADDRESS.finditer(t.text):
            found = f"{m.group(1)} {' '.join(m.group(2).split())}"
    return found


def _address_dropped(args: dict, sess: "RealtimeSession") -> Optional[str]:
    """The caller gave a street address and this save leaves it out."""
    addr = _address_offered(sess)
    if not addr:
        return None
    saved = " ".join(str(args.get(f) or "") for f in ("branch", "city")).lower()
    number = addr.split()[0]
    # Keyed on the NUMBER, not the words: "Fourth Street" can legitimately be
    # absent from a value that already names the site, but the house number is
    # either recorded or it is lost.
    return None if number in saved else addr


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


def _above_echo_floor(raw: bytes) -> bool:
    """Is this frame loud enough to be a person rather than our own echo?

    Deliberately NOT routed through realtime_echo_gate. That setting decides
    whether a caller may interrupt the agent mid-sentence, which is a product
    question; this decides whether a frame is our own noise coming back, which
    is an acoustic one. Tying them together would mean REALTIME_ECHO_GATE=pass
    — the shipped default, chosen so callers can always interrupt — silently
    switched the echo guard off too.
    """
    arr = _wire_to_pcm16(raw)
    if arr.size == 0:
        return False
    return float(np.sqrt(np.mean(arr ** 2))) >= settings.realtime_echo_rms


def _is_own_backchannel_echo(sess: "RealtimeSession", raw: bytes) -> bool:
    """Withhold this inbound frame as our own backchannel coming back?

    Split out of the media loop so it can be driven directly. The loop cannot:
    a source-level check on the call site keeps passing when the branch is
    wrapped in `if False`, which is the shape that has hidden five disabled
    guards on this codebase already.

    Both conditions are load-bearing. The window alone would eat real speech —
    the caller is mid-utterance by construction, since a clip only fires
    _BACKCHANNEL_AFTER_S into their turn. The floor alone would run for the
    whole call.
    """
    return (time.time() < sess._backchannel_mute_until
            and not _above_echo_floor(raw))


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
    # LATER EVIDENCE CAN CORRECT EARLIER EVIDENCE. This used to return on the
    # first differing claim, so a mismatch raised at pickup could never be
    # resolved however the rest of the call went.
    #
    # call-20260820-1440: the caller answered "Hi, this is North Medical Group",
    # the record said "Northside Medical Group", and the save was blocked with
    # "NEED: which place this call actually reached". The agent asked. The
    # caller answered "This is Northside Medical Group." Nothing consumed it —
    # the agent escalated one second later with a reason the caller had just
    # contradicted, and a genuine branch (Mission Bay Clinic, 1825 Fourth
    # Street, grounding clean) was thrown away.
    #
    # Note what this does NOT do: it never decides whether "North" and
    # "Northside" are the same name. That question is unanswerable from a
    # transcript and normalising them would be inventing data. It answers the
    # question the rejection actually asked, and only that one.
    #
    # NOT "the last utterance wins" either. The clear requires a positive
    # SELF-IDENTIFICATION as the recorded organisation — "this is X", "you've
    # reached X" — which _SELF_ID/_SELF_ID_WEAK already distinguish from merely
    # naming it. "We're not Northside Medical Group" and "Dr. Okafor isn't at
    # Northside any more" both contain the name and neither qualifies.
    #
    # And it must come AFTER the differing claim. A confirmation at pickup
    # followed by a different organisation later is a transfer, not a
    # correction, and that mismatch must stand.
    _mismatch = ""
    for turn in sess.turns:
        if turn.role != "caller" or not turn.text:
            continue
        claims = list(_SELF_ID.findall(turn.text))
        claims += [c for c in _SELF_ID_WEAK.findall(turn.text) if _ORG_WORD.search(c)]
        if _mismatch:
            # Only a positive self-ID as the place on record clears it.
            if any(_distinctive(c) & on_record for c in claims):
                return ""
            continue
        # If the recorded name appears anywhere in this turn, they are the right
        # place however else they phrase it. "Northside, this is Amy."
        if on_record & _distinctive(turn.text):
            continue
        for claimed in claims:
            said = _distinctive(claimed)
            # Overlap of even one distinctive token means the same place under a
            # slightly different name — "Northside Medical Center" vs "Group".
            if said and not (said & on_record):
                _mismatch = (f"caller answered as {claimed.strip()!r}, but this "
                             f"call is recorded against {recorded!r}")
                break
    return _mismatch


# Numbers written as words, mapped to their value. The value is needed to tell
# RENDERING from SUBSTITUTION, which is the whole difficulty here:
#
#   caller "1825 4th Street"   -> "1825 Fourth Street"     rendering. Fine.
#   caller "1844th Street"     -> "eighteen forty fourth"  substitution. Not.
#
# Both replace digits with words. The first keeps a digit the caller gave and
# spells an ordinal that traces back to one ("4th" -> "fourth"); the second
# erases the number entirely and nothing in it traces anywhere. A test that
# just looked for number-words blocked both, and blocking the first throws
# away a correct address — the expensive direction.
#
# NOT a general parser. "eighteen forty fourth" is genuinely ambiguous between
# 1844th, 18 44th and 1840 4th, and picking one would be inventing an address.
# Each word is checked on its own: did the caller say this word, or the digit
# it stands for? That question has an answer without resolving the ambiguity.
#
# "a" and "an" are absent on purpose: articles far more often than quantities,
# and treating them as numbers would reject half of all real branch names.
_NUMBER_WORD_VALUE: dict[str, int] = {
    **{w: i for i, w in enumerate("""
        zero one two three four five six seven eight nine ten eleven twelve
        thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty
    """.split())},
    **{w: v for w, v in zip(
        "thirty forty fifty sixty seventy eighty ninety".split(),
        range(30, 100, 10))},
    "hundred": 100, "thousand": 1000,
    **{w: i + 1 for i, w in enumerate("""
        first second third fourth fifth sixth seventh eighth ninth tenth
        eleventh twelfth thirteenth fourteenth fifteenth sixteenth
        seventeenth eighteenth nineteenth twentieth
    """.split())},
    **{w: v for w, v in zip(
        "thirtieth fortieth fiftieth sixtieth seventieth eightieth "
        "ninetieth".split(), range(30, 100, 10))},
    "hundredth": 100, "thousandth": 1000,
}


def _drop_lost_substance(spoken: str, dropped: str) -> bool:
    """Did muting the second item cost the caller something they needed?

    The one-item guard keeps the FIRST item to produce audio and mutes the
    rest, and it has to: by the time a second item appears the first is already
    on the wire, so there is no choosing between them. But the model does not
    reliably put the substance first, and when it does not the guard deletes
    the answer and keeps the throat-clearing.

    call-20260820-1421, the caller having asked "can you repeat that question
    please?":

        spoken  : "Sure, I'll repeat it clearly."
        muted   : "I'm trying to find out which branch Dr. Okafor works out of."

    They asked for the question and got a promise to give it. Seven seconds of
    silence, then the watchdog. They asked again, the answer was muted again,
    and they hung up at 88s.

    So this is NOT a test of whether to mute — that decision is forced. It is a
    test of whether anything is now OWED, so it can be said in the next turn
    rather than lost. Content words the muted item had and the spoken one did
    not: two or more, and at least half of what it was carrying.

    Deliberately conservative, because the failure of saying it anyway is
    repeating yourself, which this project treats as the thing that makes
    people hang up. "Sure, no rush." muted behind "Sure, no rush." owes
    nothing, and neither does a rephrasing of the ask that was already spoken.
    """
    # The model often REGENERATES the spoken half and appends the rest, so the
    # muted item is a superset rather than a different sentence. Judging the
    # whole thing then dilutes the new part with the repeated one: on
    # call-20260820-1421 the muted item repeated the identity answer and added
    # "Could you tell me which branch Dr. Okafor sees patients at?", and the
    # ask — the only thing the caller had not heard — scored 0.47 and was
    # written off. Strip the repeated head first and judge what is left.
    _n = lambda t: re.sub(r"[^a-z0-9 ]", " ", (t or "").lower()).split()
    _sp, _dr = _n(spoken), _n(dropped)
    owed = _dr[len(_sp):] if _dr[:len(_sp)] == _sp else _dr

    d = {w for w in owed if len(w) > 2 and w not in _UNGROUNDED_STOPWORDS}
    if not d:
        return False
    # If the half they HEARD already asked for the location, a muted second ask
    # owes them nothing — they have the question, and saying it again is the
    # repetition this project treats as what makes people hang up. Word overlap
    # cannot see this on its own: "do you know which branch she works out of
    # these days?" behind "which branch is she at?" shares little vocabulary
    # and is the same request.
    if _is_location_ask(spoken) and _is_location_ask(dropped):
        return False
    said = set(_sp)
    new = d - said
    # Two or more words they have not heard, and at least half of what the
    # muted part was carrying. Both halves matter: the count stops a one-word
    # difference counting, and the fraction stops a REPHRASING of what was
    # already said — "do you know which branch she works out of these days?"
    # behind "which branch is she at?" owes nothing but three stray words.
    return len(new) >= 2 and len(new) / len(d) >= 0.5


def _ungrounded_detail(args: dict, sess: "RealtimeSession", key: str) -> list:
    """Content words in the model's free-text qualifier that nobody said.

    `detail` (and `depends_on`) cannot be fixed by SELECTION the way `heard`
    can: there is no single caller turn that is the qualifier, because the
    field is a summary by construction — "you'd be number 21", "book online or
    call the front desk". Nothing in the transcript is the right thing to copy
    in wholesale.

    So this one is the fallback the selection avoided: check it. Word level,
    not substring, because a summary legitimately reorders and drops words —
    demanding a verbatim substring would reject every honest summary. What it
    catches is the failure actually observed: the model inserting a NOUN nobody
    said. On call-20260824-2116 the qualifier read "Book online or call the
    front desk" when the caller had said only "you need to book through online
    or call" — "desk" appears nowhere in the call.

    Same collapse as the branch check, so "front-desk" and "front desk" are the
    same word to it, and the same stopword list, so ordinary English does not
    have to be grounded.
    """
    value = str(args.get(key) or "").strip()
    if not value:
        return []
    heard = _asserted_caller_text(sess)
    if not heard.strip():
        return []
    out: list = []
    # SPLIT ON NON-LETTERS, not on whitespace. Stripping punctuation off a
    # whitespace token leaves "front-desk" whole, and `.isalpha()` is False for
    # it, so a hyphenated invention was skipped without ever being compared \u2014
    # and "front-desk" is exactly the shape of the word this has to catch.
    for w in re.findall(r"[a-z']+", value.lower()):
        w = w.strip("'")
        if (not w or len(w) <= 2 or w in _UNGROUNDED_STOPWORDS
                or w in _DETAIL_FUNCTION_WORDS or w in out):
            continue
        if _grounded_loosely(w, heard):
            continue
        # A meaning word stands on its CLASS, not on itself. The caller who
        # said "don't" made the negation; the caller who said "as long as"
        # made the condition. Which word the model reached for afterwards is
        # not evidence of anything.
        cls = _meaning_class(w)
        if cls and _class_present(cls, heard):
            continue
        out.append(w)
    return out


# Words whose removal would change what the sentence CLAIMS rather than how it
# reads — grouped into CLASSES, and the grouping is the fix.
#
# These were a flat set, checked word by word against the transcript, and that
# made the guard fire hardest on exactly the answers the client most wants. A
# model paraphrasing the connective is the single most predictable thing it
# does:
#
#   caller "as long as they've got the right insurance"
#   model  "only if they have the right insurance"      -> EMPTIED
#   caller "they need a referral from their primary"
#   model  "only with a referral from their primary care doctor" -> EMPTIED
#
# and the same on the other side:
#
#   caller "we don't take new patients until January"
#   model  "not taking new patients until January"      -> EMPTIED
#
# In every one of those the caller DID negate, or DID make it conditional. The
# model reached for a different word for the same move. Asking whether the
# CALLER SAID "only" is the wrong question; the question is whether the caller
# expressed conditionality at all.
#
# So membership is checked per CLASS: a meaning word counts as grounded when
# ANY member of its class appears in what the caller asserted. An invented
# condition — "only if insured" on a call where nothing was conditional — still
# has no class-mate to stand on, and still drops the whole qualifier.
_MEANING_CLASSES: dict = {
    # Reversing the polarity of the claim.
    "negation": frozenset({
        "not", "never", "without", "cannot", "cant", "dont", "doesnt",
        "isnt", "arent", "wont", "wouldnt", "couldnt", "nor", "none",
        "neither", "no", "nope", "stopped", "closed", "refuse", "refused",
    }),
    # Making the claim conditional — the shape CAQH is after: "yes, but only if
    # you have insurance with this particular company". Necessity words belong
    # here too: "they need a referral" and "only with a referral" are the same
    # move, and a model will swap one for the other without hesitating.
    "condition": frozenset({
        "only", "unless", "except", "provided", "depends", "depending",
        "whether", "case", "long", "need", "needs", "needed", "require",
        "requires", "required", "must", "if", "when", "certain", "some",
    }),
}


# Auxiliaries, copulas and light verbs. Skipped outright in a QUALIFIER, the
# way _UNGROUNDED_STOPWORDS are skipped everywhere: their presence or absence
# says nothing about whether the model invented anything, and checking them
# produced "only if they the right insurance" — a sentence mangled by the
# removal of "have" because the caller had said "got".
_DETAIL_FUNCTION_WORDS = frozenset({
    "are", "is", "was", "were", "been", "being", "have", "has", "had",
    "having", "does", "did", "doing", "will", "would", "can", "could",
    "shall", "should", "get", "gets", "got", "with", "from", "their",
    "them", "they", "your", "you", "our", "its", "it", "that", "this",
    "these", "those", "there", "here", "and", "but", "for", "the", "any",
    "all", "one", "also", "just", "then", "than", "who", "which", "what",
    "about", "into", "onto", "over", "under", "been",
})


def _stem(word: str) -> str:
    """Crude stem, so an inflection is not mistaken for an invention.

    "we don't TAKE new patients" and "not TAKING new patients" are the same
    verb, and reporting "taking" as a word nobody said cost the qualifier on
    call-20260825. Suffix-stripped and de-silent-e'd, so take/takes/taking all
    reduce to "tak".

    Used ONLY for the free-text qualifier. Branch grounding keeps its exact
    comparison: a stem match is a loosening, and the branch field is where a
    loosening costs a wrong address in the directory.
    """
    w = word.lower().strip("'")
    for suf in ("ings", "ing", "ers", "er", "ies", "ied", "es", "ed", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    return w[:-1] if len(w) > 3 and w.endswith("e") else w


def _grounded_loosely(word: str, heard: str) -> bool:
    """Grounded allowing for inflection. Qualifier fields only.

    Two steps, because suffix-stripping only reaches inflections of one lemma.
    "cardiology" and "cardiologists" are the same fact in different parts of
    speech, and no amount of trimming -s and -ing turns one into the other —
    on call-20260825-1226 the caller said "cardiologists", the model wrote
    "cardiology", and the identity qualifier was emptied over the difference.
    A shared long prefix catches that family without reaching anything else:
    the words have to agree for six characters, which ordinary English pairs
    almost never do by accident.

    Qualifier fields ONLY. A prefix rule is a real loosening, and the branch
    field is where a loosening costs a wrong address.
    """
    if _grounded_in(word, heard):
        return True
    target = _stem(word)
    spoken = re.findall(r"[a-z']+", heard.lower())
    if any(_stem(w) == target for w in spoken):
        return True
    return len(word) >= 6 and any(
        len(w) >= 6 and w[:6] == word[:6] for w in spoken)


def _meaning_class(word: str) -> str:
    """Which meaning class this word belongs to, or "" for ordinary content."""
    w = word.replace("'", "").lower()
    for name, members in _MEANING_CLASSES.items():
        if w in members:
            return name
    return ""


def _class_present(name: str, heard: str) -> bool:
    """Did the caller make this KIND of move, in any words at all?

    Whole-word membership, not the substring test the content words use: "if"
    inside "different" is not a condition, and a class check that fired on it
    would ground every qualifier ever written.
    """
    spoken = {w.replace("'", "") for w in re.findall(r"[a-z']+", heard.lower())}
    return bool(spoken & _MEANING_CLASSES[name])


def _strip_ungrounded_detail(args: dict, sess: "RealtimeSession",
                             key: str) -> tuple:
    """Drop the words nobody said, KEEP the rest. Returns (dropped, reason).

    Discarding the whole string was the first rule and it traded badly. On
    call-20260825-0922 the caller said "you will be the number 21", the model
    wrote "you would be number 21", and one mismatched verb tense — will against
    would — emptied the field and took "number 21" with it. On a waitlist call
    the queue position is the most valuable thing in the record; losing it to a
    tense is not a defensible price for tidiness.

    So the ungrounded words go and the remainder stays. Three things bound that:

      * A NEGATOR IS NEVER STRIPPED. See _MEANING_WORDS. Removing one rewrites
        the claim instead of trimming it, so an ungrounded negator drops the
        whole qualifier — the one case where discarding is still right.
      * DIGITS ARE NEVER STRIPPED, because they are never checked: the token
        pattern is alphabetic, so "21" is not a candidate and cannot be dropped.
        That is what carries a queue position through.
      * A REMAINDER WITH NOTHING IN IT IS NOT KEPT. If stripping leaves no digit
        and no content word, the field is emptied — but RECORDED as emptied,
        never silently, because a quietly blank field reads exactly like a
        caller who volunteered nothing.
    """
    value = str(args.get(key) or "").strip()
    if not value:
        return (), ""
    dropped = _ungrounded_detail(args, sess, key)
    if not dropped:
        return (), ""

    # A meaning word only reaches `dropped` when its ENTIRE CLASS was absent
    # from the transcript — see _ungrounded_detail. So this is no longer "the
    # model used a word they did not", it is "the model made a move they never
    # made": invented a negation, or invented a condition. That still rewrites
    # the claim rather than trimming it, and still drops the whole qualifier.
    risky = [w for w in dropped if _meaning_class(w)]
    if risky:
        args[key] = ""
        return tuple(dropped), (
            "dropped whole - " + ", ".join(repr(w) for w in risky)
            + " changes what it claims, and trimming a negator rewrites the "
              "sentence")

    # Keep each whitespace word unless its alphabetic core was ungrounded, so
    # punctuation and digits ride along with the words that survive.
    kept = []
    for word in value.split():
        core = "".join(re.findall(r"[a-z']+", word.lower())).strip("'")
        if core and core in dropped:
            continue
        kept.append(word)
    remainder = " ".join(kept).strip()
    # Trim the danglers a deletion leaves behind. Removing "desk" from "call
    # the front desk" leaves "call the", which is not wrong so much as visibly
    # broken, and a reviewer who sees that stops trusting the field. Only
    # trailing function words go, and only from the end — nothing in the middle
    # is touched, so this cannot change what the remainder says.
    while True:
        _stripped = remainder.rstrip(" .,;:-")
        _tail = _stripped.rsplit(" ", 1)[-1].lower() if " " in _stripped else ""
        if _tail in {"the", "a", "an", "or", "and", "of", "to", "for", "with",
                     "from", "at", "in", "on", "by", "your", "their"}:
            remainder = _stripped[: _stripped.rfind(" ")].rstrip(" .,;:-")
            continue
        remainder = _stripped
        break

    informative = bool(re.search(r"\d", remainder)) or any(
        w for w in re.findall(r"[a-z']+", remainder.lower())
        if len(w) > 2 and w not in _UNGROUNDED_STOPWORDS)
    if not informative:
        args[key] = ""
        return tuple(dropped), "dropped whole - nothing informative survived"

    args[key] = remainder
    return tuple(dropped), "trimmed to " + repr(remainder)


def _collapse(text: str) -> str:
    """Letters and digits only — word boundaries removed, SEQUENCE preserved.

    call-20260824-2113: the caller said "east side clinic" and the model saved
    "Eastside Clinic". `clinic` is a grounding stopword, so `eastside` was the
    only content word left, and it is not a substring of "east side" — the
    space breaks it. Rejected four times, twice while the caller was repeating
    themselves verbatim, and the call recorded "could not obtain the location"
    about a cooperative person who answered immediately and confirmed it.

    THIS IS NOT FUZZY MATCHING, and the difference is the whole reason it is
    allowed where a similarity threshold was not. There is no score and no
    tolerance: every letter must still appear in the same order. It cannot
    rescue "Riverside" from "resides at" — measured, along with every other
    fabrication on record — because those differ in their letters, not in where
    the spaces fall. The (a)/(b) ambiguity that made fuzzy matching unusable
    needs two readings of the same string, and an exact character sequence
    admits only one.

    The model normalising "east side" to "Eastside" is a reasonable thing to do
    with a place name, and the same collapse covers north side/Northside, mid
    town/Midtown, Saint Mary's/St Mary's — a large fraction of real US branch
    names. Measured across all 36 resolved calls: zero rows change.

    DIGITS ARE NOT ROUTED THROUGH THIS. The digit rule keeps its own exact
    comparison of digit RUNS, so "1855" still cannot ground on "1825"; that
    guard exists because a house number nobody said reached the directory, and
    nothing here loosens it.
    """
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _grounded_in(term: str, heard: str) -> bool:
    """Did the caller say this term, allowing for where the spaces fell?"""
    return term in heard or _collapse(term) in _collapse(heard)


def _rode_along(args: dict, sess: "RealtimeSession") -> list:
    """Content words in the saved value that the caller was never heard to say.

    NOT A BLOCK — `_ungrounded_terms` has already decided whether to accept the
    value, and this changes none of that. It answers the narrower question the
    accept does not: WHICH PARTS were actually corroborated.

    The two are different because grounding deliberately accepts on ONE content
    word. "Mission Bay Clinic" grounds if "bay" was said; the digit rule was
    added when that tolerance put a house number nobody said into the directory,
    and this is the alphabetic half of the same hole. On call-20260824-2014 the
    transcriber rendered "Riverside campus" as "She resides at campus", the
    model retried with "Riverside Campus, 1825 4th Street", and it was accepted
    on the street number — correctly, the caller really had said that — while
    "Riverside" itself was never corroborated by anything. The row went to the
    directory stamped "verified against caller transcript", which was true of
    the address and false of the name.

    Blocking that would be wrong: the value was RIGHT, and refusing it costs a
    real answer, which is this project's expensive direction. Recording it costs
    nothing and makes the row's provenance answerable — a reviewer can ask which
    rows contain a token nobody was heard to say, which is exactly the question
    you want to be able to ask after a transcription failure.
    """
    heard = _asserted_caller_text(sess)
    if not heard.strip():
        return []
    out: list = []
    for field in ("branch", "city"):
        value = (args.get(field) or "").strip()
        if not value:
            continue
        # Same tokenizer as _ungrounded_detail, for the same reason: a
        # whitespace split leaves "Mid-Town" as one non-alpha token and skips it.
        for w in re.findall(r"[a-z']+", value.lower()):
            w = w.strip("'")
            if (w and w not in _UNGROUNDED_STOPWORDS
                    and len(w) > 2 and not _grounded_in(w, heard)
                    and w not in out):
                out.append(w)
    return out


def _asserted_caller_text(sess: "RealtimeSession") -> str:
    """Everything the caller ASSERTED, lowercased, as one blob.

    Extracted so the grounding check and the rode-along report read the same
    evidence. Two copies of "what did the caller actually tell us" is two
    answers to that question the first time one of them is edited.
    """
    return " ".join(
        t.text.lower() for t in sess.turns
        if t.role == "caller" and t.text.strip() != "[...]"
        and _turn_asserts(t.text, sess))


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
    # ASSERTIVE TURNS ONLY. A value the caller ASKED us about is not
    # evidence that they told us it — see _turn_asserts. _usable is left
    # whole because the hint-echo check below asks a different question
    # ("was this term only ever a bare echo on dead air"), which a
    # question-shaped turn answers just as well as a statement.
    heard = _asserted_caller_text(sess)
    if not heard.strip():
        # Nothing transcribed, or nothing ASSERTED — cannot judge either
        # way, so do not block. Same conservative direction as before.
        return ""

    missing = []
    for field in ("branch", "city"):
        value = (args.get(field) or "").strip()
        if not value:
            continue
        terms = [w.strip(".,!?-—'\"") for w in value.lower().split()]

        # ── DIGITS MUST MATCH EXACTLY ───────────────────────────────────────
        # The word rule below is deliberately lenient: one content word
        # matching is enough, because transcription is imperfect and a real
        # answer is worth more than a blocked one. That tolerance is right for
        # words and exactly wrong for numbers.
        #
        # call-20260819-1716: the caller said "1825 4th Street". The agent
        # saved "Mission Bay Clinic, 1855 Fourth Street" and grounding PASSED
        # it — because "bay" appeared, and one word was enough. A four-digit
        # house number nobody said went into the client directory.
        #
        # That is the worst failure this whole system exists to prevent. Not an
        # empty row and not an obviously wrong one, but a PLAUSIBLE one: no
        # reviewer spots it, and someone sent to 1855 Fourth Street finds the
        # wrong building. A misheard street name is recoverable; a misheard
        # street number is a wrong address that looks right.
        #
        # So numbers get no tolerance at all. Every digit run in the value must
        # appear verbatim in what the caller actually said.
        # A NUMBER SAID AS A WORD IS STILL THAT NUMBER. This is normalisation,
        # not tolerance: the rule's zero-tolerance is about numbers the caller
        # never GAVE, not about which notation they arrived in.
        #
        # call-20260825-1226: the caller said "Riverside Campus Seventh Street"
        # twice, the model wrote "7th", and the digit rule reported "number 7
        # not in what the caller said" — three refusals, and the branch that
        # finally saved was a bare "Riverside" with the campus and the street
        # both lost. The map was already there and already knew seventh -> 7;
        # only the caller's side of the comparison was not consulting it. The
        # reverse direction — value spelled out, caller said digits — has been
        # handled since the spelled-number bypass was closed, so this is the
        # missing half of a check that was always meant to be symmetric.
        _said_nums = set(re.findall(r"\d+", heard))
        _said_nums |= {str(_NUMBER_WORD_VALUE[w])
                       for w in re.findall(r"[a-z]+", heard)
                       if w in _NUMBER_WORD_VALUE}
        _value_nums = set(re.findall(r"\d+", value))
        _invented = sorted(_value_nums - _said_nums)
        if _invented:
            missing.append(
                f"{field}={value!r} (number{'s' if len(_invented) > 1 else ''} "
                f"{', '.join(_invented)} not in what the caller said)")
            continue

        # ── AND THE SAME TOLERANCE FOR NUMBERS SPELLED OUT ──────────────────
        # The rule above only inspects digit runs, so a value carrying NO
        # digits skips it entirely — _value_nums is empty, nothing is compared,
        # and the check passes vacuously. Spelling the number in words is
        # therefore a complete bypass of the strictest guard in this file.
        #
        # call-20260820-1321 walked straight into it, and the guard drove it
        # there. The caller said "It's Mission Bay Clinic, 1844th Street."
        #   1st try  'Mission Bay Clinic, 18 4th Street'    -> REJECTED, rightly
        #   2nd try  'Mission Bay Clinic, 18 4th Street'    -> REJECTED, rightly
        #   3rd try  'mission bay clinic, eighteen forty fourth street' -> SAVED
        # and it reached doctors.json as "partially_verified" with grounding
        # "verified against caller transcript". Nothing verified it: there were
        # no digits to check. Two rejections reading "NEED: wording the caller
        # used out loud" taught the model that digits were the problem, so it
        # wrote them as words and the guard waved it through — a guard that
        # trains the model to evade it is worse than no guard, because the
        # result carries a verification stamp.
        #
        # Same zero tolerance, then, but applied per word so that RENDERING
        # still passes. A number-word is grounded if the caller said that word
        # ("Seven Hills Clinic"), or said the digit it stands for
        # ("4th Street" -> "Fourth Street"). Neither, and it was substituted.
        _heard_words = set(re.findall(r"[a-z]+", heard))
        _value_numwords = {p for t in terms for p in t.split("-")
                           if p in _NUMBER_WORD_VALUE}
        _spelled = sorted(
            w for w in _value_numwords
            if w not in _heard_words
            and str(_NUMBER_WORD_VALUE[w]) not in _said_nums)
        if _spelled:
            missing.append(
                f"{field}={value!r} (numbers as words: "
                f"{', '.join(_spelled)} | caller did not say them)")
            continue

        content = [w for w in terms if w and w not in _UNGROUNDED_STOPWORDS]
        if not content:
            continue
        # One content word appearing is enough — transcription is imperfect and
        # we would rather let a real answer through than block it. See the digit
        # rule above for where this tolerance had to stop.
        if not any(_grounded_in(w, heard) for w in content):
            missing.append(f"{field}={value!r}")
            continue
        # It appears. Check it did not appear ONLY as a bare echo on dead air.
        _support = [t for t in _usable
                    if any(_grounded_in(w, t.text.lower()) for w in content)]
        _level = _caller_speech_level(sess)
        if _support and all(_is_hint_echo(t, content, _level) for t in _support):
            missing.append(
                f"{field}={value!r} (only heard as a bare term on silent audio)")
    return " and ".join(missing)


def _ungrounded_choice(args: dict, sess: "RealtimeSession", *,
                       arg: str, probe, classifier, states,
                       label: str) -> str:
    """Grounding for a closed-set field. Empty string means it checks out.

    PARAMETRISED OVER THE VOCABULARY, not copied per field. `probe` is the
    pattern that recognises the ask this answer belongs to — it is what anchors
    the search, and it is the same Field.probe the objective and the ask budget
    already use, so a template cannot end up with three different opinions
    about what counts as asking the question.

    WHY THIS IS NOT `classify_choice(heard)` OVER THE CALLER BLOB, which is the
    obvious reading of "do for CHOICE what _ungrounded_terms does for PLACE".
    That would be strictly weaker than the location check, not equivalent, for
    three reasons that all come from the same place: a location is a
    high-entropy proper noun and a status is two bits.

    1. THE BLOB IS THE WRONG SCOPE. "Northgate" appearing anywhere in a call is
       evidence somebody said Northgate. "Yes" appearing anywhere is evidence of
       nothing — callers say it constantly for other reasons. "Yes, speaking."
       at pickup would ground a YES for a call where the accepting question was
       never answered at all. So the evidence must come from the turns AFTER we
       asked, not from the call.
    2. THE BARE-TERM CONJUNCT COLLAPSES. _is_hint_echo requires two signals to
       fail together: the turn is nothing but the term, AND the audio carried no
       signal. For a location the first is a real discriminator, because a
       genuine answer usually arrives with surrounding words. A genuine status
       answer IS "Yes." — bare is the normal shape — so that conjunct is
       satisfied by every true answer and the audio measurement is doing all of
       the work alone. Which means it needs the rms check MORE than the location
       check does, not less: it is the only signal left.
    3. THE TRANSCRIBER FABRICATES EXACTLY THIS. 0.7s of near-silence produced a
       whole receptionist greeting on call-20260820-1732, and the hint that did
       it has since been cut — but the old hint's own "Likely phrases: yes,
       speaking" came back four times on call-20260813-1409. A phantom "Yes." on
       dead air is squarely inside what this transcriber does, and unlike a
       phantom place name there is no distinctiveness left to catch it.

    So: anchored to the ask, asserted rather than asked back, classified to the
    state being claimed, and rejected when its only support is a bare token on
    silent audio. When there was no ask to anchor to, the turn must additionally
    be ABOUT new patients — see the check below, which is where reason 1 would
    otherwise creep back in.
    """
    status = str(args.get(arg) or "").strip().lower()
    if status not in states:
        return ""      # tools.py rejects this on its own terms; not our call.

    # ANCHORED. Only turns after the most recent ASK about THIS field count.
    #
    # `_is_ask_for`, not a bare probe match, and the difference is a false
    # accept. The probe recognises the TOPIC; an agent turn can be about the
    # topic while asking nothing — a read-back, an acknowledgement, a closing
    # line. On call-20260824-2014 the agent's own "I heard you say she's taking
    # the new patients" matched the probe, advanced the anchor past every
    # caller turn that had answered, and left the window empty; the guard then
    # took its own "no evidence since the ask" branch and ACCEPTED a status it
    # had just refused three times. The model cannot be allowed to move the
    # goalposts by talking, which is the same principle as _ungrounded_terms
    # excluding the agent's words from `heard`.
    since = 0
    asked = False
    for i, t in enumerate(sess.turns):
        if t.role == "agent" and _is_ask_for(t.text or "", probe):
            since = i + 1
            asked = True
    usable = [t for t in sess.turns[since:]
              if t.role == "caller" and t.text.strip() != "[...]"]
    if not usable:
        # NOTHING TRANSCRIBED SINCE WE ASKED — stand down, do not block. Same
        # conservative direction as every other guard here: absence of
        # transcript is not evidence of fabrication, and a transcript can lag
        # the tool call that follows it.
        return ""

    # ASSERTED, not asked back. "Is she accepting new patients?" repeated by a
    # receptionist checking what we want is not them telling us. Same predicate
    # the location check uses, for the same reason.
    asserted = [t for t in usable if _turn_asserts(t.text, sess)]
    if not asserted:
        # THEY SPOKE AND NONE OF IT WAS AN ANSWER. This is NOT the same as the
        # silence above and must not share its verdict.
        #
        # The location guard can afford to stand down here, because a saved
        # branch still has to survive the blob check — its words must appear in
        # what the caller said, so there is a second gate underneath. A status
        # has no second gate: it is two bits, tools.py accepts any of the four
        # by definition, and this function is the only thing standing between a
        # model's guess and the directory. Standing down when the caller has
        # demonstrably only asked questions back would mean any status could be
        # saved at that moment, which is precisely the fabrication case.
        said = "; ".join(t.text.strip()[:60] for t in usable[-2:])
        return (f"{label}={status!r} — since you asked, they have only asked "
                f"back, not answered | THEY SAID: {said!r}")

    matching = []
    for t in asserted:
        heard_state = classifier(t.text)
        if heard_state is None or heard_state.value != status:
            continue
        # NEVER ASKED -> THE TURN MUST BE ABOUT NEW PATIENTS.
        #
        # Without an ask there is no anchor, so `since` is 0 and every caller
        # turn from pickup onwards is in scope — which reopens reason 1 above
        # in the one case it bites hardest. "Yes, speaking." is the single most
        # common opening utterance in this corpus (it is the phrase the retired
        # transcription hint echoed four times on call-20260813-1409), it
        # classifies as YES on a bare affirmative, and with no ask to anchor
        # against it would ground a new-patient status nobody was ever asked
        # for. The permissive branch was contradicting this function's own
        # docstring.
        #
        # Volunteering the answer is still honoured, which is what the anchor
        # was relaxed for in the first place: a receptionist who says "we're
        # not taking new patients right now" while you are still on the branch
        # question has told you, and that turn is ABOUT the thing. A bare "yes"
        # is not. The cost of being wrong here is one turn — the agent asks,
        # they repeat, it grounds normally — against a wrong directory row that
        # nobody can spot afterwards.
        # NEVER ASKED -> THE TURN MUST STAND ON ITS OWN.
        #
        # This tested whether the turn contained the ASK's vocabulary, and that
        # was the wrong question. On call-20260825-0915 the caller said "we are
        # full right now, but I can put you on the list. You would be number
        # 21." — a textbook waitlist answer — while the agent was still asking
        # about the BRANCH, so nothing had matched ACCEPTING_ASK and the
        # never-asked path applied. That sentence contains none of "accepting",
        # "taking new" or "new patients", so the vocabulary test threw it out.
        # Refused twice, and the queue position the client most wanted never
        # reached the record.
        #
        # What the rule was actually defending against is a BARE AFFIRMATIVE
        # with no anchor: "Yes, speaking." at pickup classifies YES on its
        # opening token alone and asserts nothing about new patients. So test
        # for that directly — strip the leading yes and see whether what remains
        # still says the same thing. A turn that states the condition in its own
        # words survives; one that was only ever a "yes" does not.
        if not asked and not states_in_its_own_right(
                t.text, status, classifier):
            continue
        matching.append(t)
    if not matching:
        said = "; ".join(t.text.strip()[:60] for t in asserted[-3:])
        return (f"{label}={status!r} — nothing the caller said since you asked "
                f"reads as that answer | THEY SAID: {said!r}")

    # The only support is a bare token on audio that carried nothing. For this
    # field that is the whole test, not a tiebreak — see 2. above.
    level = _caller_speech_level(sess)
    tokens = [w for t in matching
              for w in re.findall(r"[a-z']+", t.text.lower())]
    if all(_is_hint_echo(t, tokens, level) for t in matching):
        return (f"{label}={status!r} — only heard as a bare word on silent "
                f"audio, which is what a transcription artefact looks like")

    # ── SELECTION, NOT VALIDATION ───────────────────────────────────────────
    # `heard` is supposed to be what the caller said. It arrives model-authored
    # and, until now, entirely unchecked — while all three tool schemas told the
    # model it was "checked against the call transcript". On
    # call-20260824-2116 the model inserted clauses nobody uttered:
    #
    #   caller : "Yeah, definitely, you can reach out to them."
    #   heard  : "Yeah, definitely, they're taking new patients also. You can
    #             reach out to them."
    #   caller : "Yeah, you need to book through online or call. Please do that."
    #   heard  : "...call FROM THE FRONT DESK. Please do that."
    #
    # A fabricated quote is worse than a wrong status, because it reads as
    # verbatim to whoever audits the row and there is nothing in the record to
    # say it is not.
    #
    # So the model's string is not checked — it is DISCARDED. This function has
    # already identified the caller turn that corroborated the status; that
    # turn's real text is what gets stored, and the model's version becomes
    # irrelevant rather than merely suspect. Checking would leave the failure
    # mode in place with a detector in front of it; selection removes it.
    #
    # WHICH matching turn, when there are several.
    #
    # Last-wins was the first rule and it is wrong. On call-20260825-0915 the
    # caller circled back and the VAD split their final answer, so the last
    # turn classifying as WAITLIST was the fragment "The status waitlist is" —
    # a mid-sentence scrap that went into the record as the quotation
    # justifying the state, while "we are full right now, but I can put you on
    # the list. You would be number 21." sat further up.
    #
    # FIRST-WINS IS NOT THE ANSWER EITHER, and the reason is worth stating
    # because it is the tempting flip: a fragment can just as easily arrive
    # first, and on a call where the caller corrects themselves the earliest
    # match would be the superseded answer.
    #
    # LONGEST WINS, and the justification is that there is no correctness
    # dimension left to trade. `matching` is already filtered to turns whose
    # classification EQUALS the state being saved — every candidate asserts the
    # same thing — so recency buys nothing, and the only remaining question is
    # which of several agreeing sentences is the fullest statement of it. A
    # truncated fragment is short by construction; a complete answer carries its
    # qualifiers with it, which is also how "number 21" survives into the record
    # rather than being summarised away.
    #
    # Ties go to the later turn: same length, same claim, prefer the one they
    # most recently stood behind.
    args["heard"] = max(
        enumerate(matching),
        key=lambda pair: (len(pair[1].text.strip()), pair[0]),
    )[1].text.strip()
    return ""


# Every closed-set save tool, with the argument carrying its value, the guard
# that grounds it, the NEED fragment for a rejection, and where the verdict is
# recorded. A TABLE rather than three near-identical elif branches: the three
# differ only in vocabulary, and the branch that handles one is the branch that
# must handle the next — which is exactly the drift that let the ask budget be
# generalised in the counters but not in the gate feeding them.
_CHOICE_SAVE_TOOLS: dict = {}


# The three closed-set fields, each bound to the ask that anchors it. Wrappers
# rather than call-site keyword soup: the pairing of a field with its probe and
# its vocabulary is a fact about the field, and stating it once here is what
# stops a caller passing ACCEPTING_ASK while classifying with the referral
# vocabulary and getting a guard that can never fire.
def _ungrounded_status(args: dict, sess: "RealtimeSession") -> str:
    """Grounding for the new-patient status."""
    from agents.voice.objectives import CHOICE_STATES, classify_choice
    return _ungrounded_choice(args, sess, arg="status", probe=ACCEPTING_ASK,
                              classifier=classify_choice, states=CHOICE_STATES,
                              label="status")


# A doctor named in speech: "Dr. Kapoor", "Doctor Smith", "Dr Okafor's".
_NAMED_DOCTOR = re.compile(r"\b(?:dr\.?|doctor)\s+([a-z][a-z'-]{2,})", re.I)


def _surnames_named(text: str) -> list:
    """Every surname the caller attached to a doctor title, lowercased."""
    return [m.group(1).lower().rstrip("'s").rstrip("'")
            for m in _NAMED_DOCTOR.finditer(_norm_quotes(text or ""))]


def _wrong_doctor_named(text: str, sess: "RealtimeSession") -> str:
    """A surname in this turn that is NOT the doctor on record. "" if fine.

    THE CHECK THE IDENTITY FIELD EXISTED FOR AND DID NOT HAVE. On
    call-20260825-1226 the record said Dr. Okafor, the caller said "that's
    right, Dr. Kapoor is one of our cardiologists", and identity saved
    CONFIRMED — because the guard classified the affirmative and never looked
    at the name. Okafor and Kapoor are not the same person. The field exists to
    answer "are we talking about the right doctor", and it answered yes about a
    different one.

    STRICT, AND DELIBERATELY THE OPPOSITE OF THE BRANCH RULE. Branch grounding
    is lenient because refusing a real answer costs a lost row; here leniency
    costs a row CONFIRMED against the wrong person, attached to a real
    practice, which is the worst output this system has. So a named surname
    must match the one in CALL CONTEXT — collapsed for spacing and hyphens,
    which preserves the letter sequence, and nothing fuzzier. If the
    transcriber mangled our doctor's name we refuse a correct confirmation:
    that is the right way round, because "the ASR mangled it" and "they named
    someone else" produce the same string and only one of them is safe to
    assume.

    Silence about the name is not a mismatch. A caller who says "yes, speaking"
    names nobody, and this returns "" — the classification stands on its own.
    """
    named = _surnames_named(text)
    if not named:
        return ""
    from agents.voice.templates import clean_doctor_name
    _full = clean_doctor_name(getattr(sess.doctor, "doctor_name", "") or "")
    ours = (_full.split()[-1] if _full.split() else _full).lower()
    if not ours:
        return ""
    if any(_collapse(n) == _collapse(ours) for n in named):
        return ""
    return ", ".join(sorted(set(named)))


def _ungrounded_identity(args: dict, sess: "RealtimeSession") -> str:
    """Grounding for whether we reached the right doctor. Its own vocabulary.

    Plus the name check, which the vocabulary alone cannot do: an affirmative
    is an affirmative whoever it is about, so confirming REQUIRES that no other
    doctor was named in the same breath.
    """
    from agents.voice.objectives import IDENTITY_STATES, classify_identity

    claimed = str(args.get("identity") or "").strip().lower()
    if claimed == "confirmed":
        # Only the turns that could be the evidence — after the ask, asserted.
        for t in reversed(sess.turns):
            if t.role != "caller" or t.text.strip() == "[...]":
                continue
            other = _wrong_doctor_named(t.text, sess)
            if other:
                sess.memory.update(wrong_doctor_named=other)
                return (f"identity='confirmed' — they named {other!r}, and the "
                        f"doctor on this call is different | THEY SAID: "
                        f"{t.text.strip()[:70]!r}")
            if _surnames_named(t.text):
                break      # our doctor was named, and matched
    return _ungrounded_choice(args, sess, arg="identity", probe=IDENTITY_ASK,
                              classifier=classify_identity,
                              states=IDENTITY_STATES, label="identity")


def _ungrounded_scheduling(args: dict, sess: "RealtimeSession") -> str:
    """Grounding for whether a new patient can actually be booked in."""
    from agents.voice.objectives import CHOICE_STATES, classify_choice
    return _ungrounded_choice(args, sess, arg="status", probe=SCHEDULING_ASK,
                              classifier=classify_choice, states=CHOICE_STATES,
                              label="scheduling")


def _ungrounded_referral(args: dict, sess: "RealtimeSession") -> str:
    """Grounding for the referral requirement. Its own vocabulary."""
    from agents.voice.objectives import REFERRAL_STATES, classify_referral
    return _ungrounded_choice(args, sess, arg="requirement", probe=REFERRAL_ASK,
                              classifier=classify_referral,
                              states=REFERRAL_STATES, label="referral")


_CHOICE_SAVE_TOOLS.update({
    "save_doctor_identity": (
        "identity", _ungrounded_identity,
        "their own words on whether this is the right doctor, or 'unsure'",
        "identity_grounding"),
    "save_new_patient_status": (
        "status", _ungrounded_status,
        "their own words on new patients, or 'unsure'", "status_grounding"),
    "save_scheduling_status": (
        "status", _ungrounded_scheduling,
        "their own words on booking a new patient, or 'unsure'",
        "scheduling_grounding"),
    "save_referral_requirement": (
        "requirement", _ungrounded_referral,
        "their own words on referrals: always, depends on what, or 'unsure'",
        "referral_grounding"),
})


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


# ── Hint regurgitation: our own prompt coming back as "speech" ───────────────
#
# The transcription hint is sent to the transcriber as `prompt`. It is not a
# vocabulary filter — it is text prepended to that model's context, so anything
# in it can come back out as transcript. Proven beyond argument on
# call-20260819-1324, where the ENTIRE hint arrived as a caller turn, verbatim:
#
#   "We are having only one branch, that is the downtown branch in Los
#    Angeles. Phone call with a hospital or medical office receptionist.
#    Health systems: Mercy, Ascension, CommonSpirit, ..."
#
# and on call-20260819-1323, where "Mercy Hospital" — the first health system
# in the hint — arrived at audio_rms 0.011 on a call where the callee never
# spoke at all, and the agent answered it.
#
# THE ARCHITECTURAL POINT. Every guard in this file reads `sess.turns` as
# ground truth. _is_hint_echo was only ever consulted inside save_branch
# grounding, so a fabricated turn that did not trigger a save entered the
# transcript unexamined — steering the conversation, and on call-20260819-1324
# feeding _discarded_location a 'Northwell' the caller never said, which
# blocked a legitimate escalation and left the agent unable to end the call.
#
# So the check belongs at INGESTION, not at one consumer. Quarantine here and
# every downstream guard is correct by construction.

# A verbatim run this long from the hint cannot be coincidence. Six words of
# ordinary speech overlapping the hint is possible; six CONSECUTIVE ones in the
# hint's own order is the prompt being read back.
_HINT_RUN_WORDS = 6


# Section headings in the hint, capitalised but carrying no identity.
_HINT_HEADINGS = frozenset({"phone", "health", "location", "call", "systems", "words"})


# ── Fabrication vocabulary — DECOUPLED FROM THE TRANSCRIPTION HINT ──────────
# These names used to be read out of _US_TRANSCRIBE_HINT by capitalisation, so
# the detector's reach was whatever we happened to be sending the transcriber.
# On 2026-08-20 the hint lost its health-system list, because a controlled A/B
# on identical audio showed the list was the SOURCE of the fabrications: 0.7s
# of near-silence returned "Hello, this is the Methodist Hospital. How may I
# assist you?" with the list present, and single non-English tokens without it.
# The hint is now location vocabulary only.
#
# Shrinking the hint disarmed the detector along with it — _hint_proper_nouns
# returned an empty set and every observed fabrication ("Mercy Hospital",
# "...at the Mayo", "the Northwell campus") stopped being recognised. So the
# two jobs are separated. They were never the same job:
#
#   the HINT   is what we SEND the transcriber    -> must not prime
#   this VOCAB is what we RECOGNISE as fabricated -> must stay broad
#
# Removing the source is a mitigation, not a cure. The transcriber still
# fabricates on thin audio — it simply fabricates location words now instead of
# hospital names — and it keeps its own priors whatever we send it. A detector
# that could only see our own prompt was never the right shape.
#
# ROT RISK, named because the original coupling existed to avoid it: a
# duplicated list goes stale when the original changes. This one derives from
# nothing — it is US health systems, not config — so there is no original to
# drift from. Do NOT re-derive it from the hint, and do NOT put these names
# back into the hint in order to feed it.
# Held as the RETIRED HINT TEXT rather than a bare word list, because the two
# detectors need different shapes from it and a second constant would be the
# duplication the rot warning is about:
#
#   _reads_as_hint_vocabulary  needs MEMBERSHIP  -> _FABRICATION_VOCAB below
#   _strip_hint_run            needs ORDER       -> 6-grams of this text
#
# A recitation arrives in the hint's own word order ("...Mercy, Ascension,
# CommonSpirit, Providence..."), so run detection cannot work from a set. This
# is verbatim what was deleted from _US_TRANSCRIBE_HINT on 2026-08-20 and it is
# never sent to anyone — it exists so the transcriber reciting the hint we USED
# to send is still recognised.
_RETIRED_HINT_TEXT = (
    "Phone call with a hospital or medical office receptionist. "
    "Health systems: Mercy, Ascension, CommonSpirit, Providence, Sutter, "
    "Kaiser Permanente, HCA, Tenet, Baptist, Methodist, Presbyterian, Mount "
    "Sinai, Cleveland Clinic, Mayo Clinic, Johns Hopkins, Banner, Advocate, "
    "Trinity Health, Northwell, NewYork-Presbyterian, Cedars-Sinai."
)

_FABRICATION_VOCAB = frozenset(
    {w.lower() for w in re.findall(r"\b[A-Z][A-Za-z]{2,}\b", _RETIRED_HINT_TEXT)}
    - _UNGROUNDED_STOPWORDS - _HINT_HEADINGS)


def _hint_proper_nouns(hint: str) -> frozenset:
    """The named health systems in the hint — the words it can put in a mouth.

    Derived from CAPITALISATION rather than a hardcoded list, because the hint
    is written that way: the health systems are proper nouns ("Mercy",
    "Kaiser", "Mayo", "Northwell") while the location words are deliberately
    lowercase ("campus", "clinic", "medical center"). So the capitalised set is
    exactly the part a caller would not volunteer by accident, and it tracks
    the hint automatically if the hint is ever edited.
    """
    caps = {w.lower() for w in re.findall(r"\b[A-Z][A-Za-z]{2,}\b", hint or "")}
    caps -= _UNGROUNDED_STOPWORDS | _HINT_HEADINGS
    # UNION, not replacement. The static vocabulary is the floor and survives
    # the hint being minimised; anything capitalised in whatever hint is
    # actually in force is added on top, so a hint that regains proper nouns
    # stays covered without editing _FABRICATION_VOCAB.
    return frozenset(caps | _FABRICATION_VOCAB)


def _reads_as_hint_vocabulary(text: str, hint: str) -> bool:
    """Does `text` name a health system straight out of our own hint?

    The SECOND signal the quarantine requires, and it has to be the narrow one.
    Requiring every content word to come from the hint was too strict: the
    fabrication on call-20260819-2006 was "Hello, I need to schedule an
    appointment at the Mayo", where "schedule" and "appointment" are ordinary
    English and only "Mayo" came from us.

    A named system appearing on silent audio is the transcriber reading its
    own prompt. A caller genuinely saying "Mercy" is audible when they do —
    which is why this is paired with the audio test and never used alone.
    """
    # `hint` may now carry no proper nouns at all — the detector no longer
    # depends on it, so only the text is required.
    if not text:
        return False
    said = {w for w in re.findall(r"[a-z]+", text.lower())}
    return bool(said & _hint_proper_nouns(hint))


def _hint_vocabulary(hint: str) -> frozenset:
    """Every word the transcriber was primed with, lowercased.

    Derived from the live hint, never listed. The hint is the only thing that
    decides what the transcriber CAN echo, so it has to be the only thing that
    decides what we refuse to believe. A hardcoded list goes stale the moment
    the hint is edited — which is exactly what happened to _hint_proper_nouns
    when the health-system names came out of it, and why _RETIRED_HINT_TEXT
    had to be pinned separately to keep that detector alive.
    """
    return frozenset(re.findall(r"[a-z]+", (hint or "").lower()))


def _is_bare_hint_word(value: str, hint: str) -> bool:
    """Is this candidate location one word straight out of our own prompt?

    call-20260821-1705: the caller said "hmm". The transcriber had no lexical
    content to decode, sampled its own conditioning prompt instead, and
    returned "Suite." Re-decoding that same 0.55s four times returned
    'campus', 'Suite,', the entire hint verbatim, and Urdu script — outputs
    that disagree with each other on identical bytes, which is the proof that
    nothing was being recovered from the audio.

    Grounding cannot see this and never could: the fabricated word IS in the
    transcript, so _ungrounded_terms checking the value against the transcript
    is circular. Both gates that might have caught it were false for sound
    reasons — the audio was real (rms 0.038, peak 0.134), and the hint's
    location words are lowercase so a capitalisation-derived proper-noun set
    cannot contain them. This is the one check that asks where the word came
    from rather than whether it was said.

    ONE bare word only, and that restraint is the whole safety of it.
    "Downtown East", "Riverside Clinic", "Baptist Medical Center" and "1420
    Beacon Street" every one contain a hint word and every one names a real
    place; refusing a location for merely containing hint vocabulary would
    reject most of the true ones. An echo arrives alone because the
    transcriber sampled a single token, not a phrase.
    """
    words = re.findall(r"[A-Za-z0-9]+", value or "")
    if len(words) != 1:
        return False
    return words[0].lower() in _hint_vocabulary(hint)


def _strip_hint_run(text: str, hint: str) -> str:
    """Truncate `text` at the first verbatim run of >= _HINT_RUN_WORDS hint words.

    Truncate rather than excise: once the transcriber starts reciting the
    prompt it does not come back to the caller mid-sentence, so everything from
    the first run onward is prompt. Cutting a window out of the middle left
    the rest of the recited list in place on call-20260819-1324.

    Truncate rather than drop the turn, because the two get mixed: on that call
    the caller genuinely said "We are having only one branch, that is the
    downtown branch in Los Angeles" and the transcriber appended the whole hint
    to it. Dropping the turn would have discarded a real answer.
    """
    if not text:
        return text
    # The CURRENT hint plus the RETIRED one. A recitation reproduces whatever
    # the transcriber was primed with, and the health-system list was in that
    # prompt for weeks — on call-20260819-1324 it came back in full. Dropping
    # the list from _US_TRANSCRIBE_HINT must not also drop our ability to
    # recognise it coming back, so both are searched. See _RETIRED_HINT_TEXT.
    hw = [w for w in re.findall(r"[a-z]+",
                                ((hint or "") + " " + _RETIRED_HINT_TEXT).lower())]
    if len(hw) < _HINT_RUN_WORDS:
        return text
    runs = {tuple(hw[i:i + _HINT_RUN_WORDS])
            for i in range(len(hw) - _HINT_RUN_WORDS + 1)}
    words = re.findall(r"\S+", text)
    keys = [re.sub(r"[^a-z]", "", w.lower()) for w in words]
    for i in range(len(words) - _HINT_RUN_WORDS + 1):
        window = tuple(k for k in keys[i:i + _HINT_RUN_WORDS] if k)
        if len(window) == _HINT_RUN_WORDS and window in runs:
            return " ".join(words[:i]).strip()
    return text


def _audio_carried_nothing(rms: Optional[float],
                           speech_level: Optional[float]) -> bool:
    """Did the audio under this transcript carry any signal at all?

    If not, the words did not come from the caller — there was nothing there to
    transcribe. This is the rule that catches a fabrication whose wording is
    ordinary enough to pass a vocabulary test: on call-20260819-1324, "Sure,
    our clinic is located on 123 Main Street, across from the Northwell campus"
    arrived at audio_rms 0.000259, which is digital silence, and fed a
    'Northwell' to _discarded_location that blocked a legitimate escalation.

    Calibrated in the same place as the hint-echo threshold: across 30 calls
    with dual-channel recordings, no genuine caller turn measured below
    median * _QUIET_FRACTION. An unmeasured turn (None) is given the benefit of
    the doubt, as everywhere else in this file.
    """
    if rms is None:
        return False
    quiet_below = _LOW_AUDIO_RMS
    if speech_level is not None:
        quiet_below = max(quiet_below, speech_level * _QUIET_FRACTION)
    return rms < quiet_below


# SILENT is not QUIET, and the distinction is what makes an audio-only
# judgement safe. _LOW_AUDIO_RMS (0.015) and _QUIET_FRACTION answer "was this
# turn faint for this caller" — a question real speech can fail, which is why
# discounting on it alone was correctly refused.
#
# This answers a different and much cruder question: was there ANY signal? The
# three confirmed fabrications all sit at the mu-law digital-silence floor,
# adjudicated from the Twilio caller channel rather than from our own numbers:
#
#   call-20260819-2006  "...schedule an appointment at the Mayo"    silence
#   call-20260820-1154  "...appointment for my annual check-up"     0.0003, 13s of it
#   call-20260820-1230  "Hello,"                                    0.0003, 72-80s
#
# Against that, the quietest GENUINE turn ever measured on this rig is 0.030
# (Twilio channel, across 48 recordings; the recorded band on call-20260820-1230
# is 0.097-0.188). 0.002 sits roughly 8x above the digital floor and 15x below
# the quietest real speech — a gap no calibration drift is going to close.
#
# Do not conflate this with _LOW_AUDIO_RMS by "simplifying" them into one
# constant later. They are deliberately answering different questions, and the
# whole reason this one may act alone is that its question has no ambiguous
# middle.
_SILENT_AUDIO_RMS = 0.002


def _audio_was_silent(rms: Optional[float]) -> bool:
    """True only when the audio under a transcript carried no signal at all.

    Deliberately NOT level-relative. A fraction of the caller's own median is
    the right shape for "faint"; it is the wrong shape for "nothing there",
    because nothing-there is an absolute fact about the line and the median it
    would be compared against is itself computed from turns this predicate
    exists to exclude.

    None means unmeasured, which is not evidence of anything — the same
    benefit of the doubt every other check in this file gives.
    """
    if rms is None:
        return False
    return rms < _SILENT_AUDIO_RMS


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
        # Caller PCM: continuous stream from Twilio — already timeline-aligned.
        # EVERY inbound frame, so save() can lay a gapless caller channel
        # against the agent blocks. Not safe to measure utterances against:
        # see _caller_oai_pcm.
        self._caller_pcm: list[bytes] = []
        # The frames OpenAI actually received, and only those. Separate from
        # _caller_pcm because the two answer different questions and the
        # answers diverge: the recording wants every frame, the measurement
        # wants OpenAI's own timeline. Merging them cost call-20260821-1856 —
        # 173 frames withheld from OpenAI but kept for the recording put our
        # index 3.46s ahead of OpenAI's ms clock, and the caller's real answers
        # were deleted as fabrications. See the append site in the media loop.
        self._caller_oai_pcm: list[bytes] = []
        # How far into _caller_oai_pcm OpenAI's input buffer begins. Zero by
        # construction now — nothing is forwarded before listen_enabled is set,
        # and that buffer only receives forwarded frames — but kept, computed
        # and asserted rather than assumed, because this is the third distinct
        # cause of an audio-clock offset on this codebase.
        self._listen_start_bytes: int = 0
        # When the Twilio stream started (set on "start" event)
        self._stream_start_time: Optional[datetime] = None
        # Set when response.create for the greeting is sent; cleared once the
        # first audio delta arrives, so we measure the callee's dead air.
        self._greeting_requested_at: Optional[float] = None
        # time.monotonic() when Twilio's /answer webhook fired — the pickup.
        # Set by handle_realtime; None when the caller could not supply it.
        self._answered_at: Optional[float] = None
        # Seconds from pickup to the first sound the callee heard. THE number
        # the question "why does it take so long to say hello" is about, and
        # the one nothing measured: the only greeting figure this project had
        # started its clock at our own response.create, well after the pickup.
        self.pickup_to_greeting_s: Optional[float] = None
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
        # What the call is trying to collect, and what counts as done. Declared
        # by the template; defaulted here so a session built without one (every
        # unit check in the test suite) still has an objective to reason about.
        self.objective: CallObjective = default_objective()
        # ── The two ask counters ────────────────────────────────────────────
        # CONSECUTIVE asks the caller did not answer. This is the budget: it is
        # what ends a call, and it resets to zero the moment they say something,
        # so four answered asks per doctor across several doctors never touch
        # it. Replaces _location_asks, which counted asks that HAD been
        # answered and ended call-20260821-1931 with the answer in hand.
        self._unanswered_asks: int = 0
        # Asks since anything was last COLLECTED. The liveness bound that the
        # old counter was providing by accident — a caller who answers every
        # ask and supplies nothing would otherwise never end the call, and
        # there is no duration cap anywhere in this path.
        self._asks_without_progress: int = 0
        # Exchanges where the caller questioned the agent back instead of
        # answering. Bounded by _MAX_VETTING_REASKS.
        self._vetting_reasks: int = 0
        # Normalised wordings the location has already been asked in, so the
        # identical clause going out a second time is detectable.
        self._ask_phrasings: set[str] = set()
        self._verbatim_ask_nudged: bool = False
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
        # The transcription hint that was sent for this call. Held so an
        # arriving transcript can be compared against the prompt that may have
        # produced it — see _strip_hint_run.
        self.transcribe_hint: str = ""
        # Turns suppressed as hint regurgitation, recorded in the artifact so a
        # silent drop is never invisible.
        self.suppressed_echoes: list = []
        # Transcripts produced over a line that carried no signal. Separate
        # from suppressed_echoes, which is a mixed bag of everything the
        # transcriber got wrong: this one counts a specific, nameable defect
        # so "did the silence guard fire, and how often" is answerable from
        # the artifact without parsing a heterogeneous list.
        self.fabricated_turns: list = []
        # Told the caller the branch was saved when the tool then rejected it.
        # Times the caller was told the location was saved while the save
        # had in fact been rejected. A count, not a flag: the flag was a
        # one-shot gate, and once the retry loop is bounded there is no
        # reason to leave the second false statement standing.
        self._false_save_claims: int = 0
        # save_branch calls that came back rejected. Every correction at
        # that site is one-shot, so without a count a model that cannot
        # produce an acceptable value retries forever, saying goodbye each
        # time. See _MAX_SAVE_REJECTIONS.
        self._save_rejections: int = 0
        # When the agent said the job was done while memory was still
        # empty. Checked by the watchdog once any tool call has landed.
        self._claimed_done_at: float = 0.0
        self._claimed_done_nudged: bool = False
        # Rejected one save for omitting a street address the caller gave.
        self._address_nudged: bool = False
        # Told to answer the "is this about a patient?" half explicitly.
        self._patient_nudged: bool = False
        # OpenAI's audio_start_ms for the utterance in progress. Its own index
        # into the buffer we feed it, which is the only reliable way to cut an
        # utterance out — see _utterance_slice.
        self._speech_start_ms: Optional[int] = None
        # When the audio already handed to Twilio finishes playing, in
        # time.monotonic() terms. 0.0 means nothing is queued.
        self._playback_ends_at: float = 0.0
        # Replies that began sending while the previous one was still playing
        # out to the caller. Counted, not just fixed: this is the failure the
        # callee experiences as being unable to get a word in, and a count is
        # the only way to tell whether the gap actually stopped it happening.
        self._stacked_replies: int = 0
        # Assistant item ids whose audio was withheld because they were a
        # SECOND spoken item inside one response. Held on the session rather
        # than in the loop so the transcript handler — a separate function —
        # knows not to print or record a turn the caller never heard.
        self._muted_items: set[str] = set()
        # When the caller stopped speaking (monotonic), cleared by the first
        # audio delta of the reply. See note_reply_latency.
        self._caller_stopped_at: Optional[float] = None
        # How long the DETECTOR took to tell us the caller had stopped,
        # measured from audio_end_ms rather than assumed per detector. This
        # is the term that separates server_vad from semantic_vad, and it was
        # a hardcoded guess in both directions before it was measured.
        self._last_stop_lag_s: float = 0.0
        self.detector_lags: list[float] = []
        # Every measured gap between a caller finishing and the agent's first
        # sound, in seconds. One number per turn beats one impression per call.
        self.reply_latencies: list[float] = []
        # What those dropped items would have said, for the artifact. A guard
        # that fires invisibly cannot be reviewed after the call.
        self.dropped_second_items: list[str] = []
        # Text that was muted mid-response and carried something the spoken
        # half did not. Owed to the caller, and said on the next turn rather
        # than lost. See _drop_lost_substance.
        self._owed_substance: str = ""
        self._owed_recovered: int = 0
        # The recovery directive is injected once and the response may be
        # refused (playback gate) and retried. Without this the retry would
        # re-inject the directive every tick.
        self._owed_directive_sent: bool = False
        # Has the CURRENT response put any audio on the wire yet? Distinct
        # from _response_active ("a response exists") and from
        # agent_speaking ("audio is playing out"). This one answers the
        # only question that matters when a transcript is rejected: is
        # there still time to stop the reply, or has the caller already
        # heard it. False at response.created, True at the first delta.
        self._response_audio_started: bool = False
        self._response_created_at: float = 0.0
        # Set when a response was cancelled because the transcript that
        # caused it was rejected. Read by _handle_agent_transcript, which
        # must skip that response's transcript exactly as it skips a
        # barge-in cancel — the caller never heard a word of it.
        self._suppressed_response: bool = False
        # One entry per rejected transcript: was the reply stopped in
        # time, and by how much. The margin is the number that decides
        # whether cancelling is enough or response creation has to be
        # taken off OpenAI's VAD entirely.
        self.rejection_cancels: list = []
        # Why responses failed. Seven failed on call-20260819-2216 and the
        # reason was in every event, unread — so the dead air they caused was
        # diagnosed by guesswork twice before anyone read the field.
        self.response_failures: list[dict] = []
        # Backchannels. When the caller's current utterance began (None if they
        # are not speaking), whether we already made a noise during it, and the
        # last clip used so the same one is not repeated.
        # While the caller is away checking, the silence watchdog must not
        # prompt. Set when they ask for a moment, cleared when they come back
        # with something substantive. See _HOLD_GRACE_S.
        self._hold_until: float = 0.0
        self._caller_speaking_since: Optional[float] = None
        self._backchannel_done_this_utterance: bool = False
        self._last_backchannel_at: float = 0.0
        self._last_backchannel_clip: Optional[str] = None
        self._backchannels_sent: int = 0
        # Wall clock until which our own backchannel may still be audible on
        # the callee's line. See _BACKCHANNEL_ECHO_MARGIN_S.
        self._backchannel_mute_until: float = 0.0
        # Frames withheld inside that window because they were too quiet to be
        # the caller. Non-zero means the speakerphone echo is real and was
        # caught; zero across a call means it never happened.
        self._backchannel_echo_frames: int = 0
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
        # Which ceiling ran out: "unanswered" (they stopped replying) or
        # "no_progress" (they replied and never supplied). It selects the
        # directive AND the escalate reason, so a call that ends this way
        # records why in words that are true of it.
        self._give_up_trigger: str = "no_progress"

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

    # Optional, because the body has always handled None and the callers have
    # always been able to pass it: an unmeasurable segment is None, not 0.0,
    # and collapsing the two is what made a silent slice look like a real
    # measurement. The annotation said `float` and was simply wrong.
    def note_utterance_rms(self, rms: Optional[float]) -> None:
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

    def note_reply_latency(self, seconds: float) -> None:
        """One measured caller-stops → agent-speaks gap.

        Bounded because a stray measurement would poison the median that goes
        in the artifact: anything past 30s is not a reply latency, it is a turn
        that never came, and the silence watchdog owns that failure.

        The CEILING is the point. The floor used to exclude 0.0 too, which
        cost nothing while server_vad added a fixed 0.7s to every sample and
        silently dropped the fastest replies once semantic_vad set that term
        to zero. The call site has already established the measurement is
        real — it only runs when _caller_stopped_at was set — so a gap that
        rounds to zero is a fast reply, not a stray.
        """
        if 0.0 <= seconds < 30.0:
            self.reply_latencies.append(seconds)

    def collected_fields(self) -> dict:
        """Every field the objective declares, with what this call learned.

        THE VALUES WERE BEING DROPPED ON THE FLOOR. Until 2026-08-24 the call
        artifact recorded `collected: ["branch","accepting","scheduling",
        "referral"]` and `outcome: complete` and NOT ONE of the three status
        values — they were written to CallMemory, which is a per-call scratchpad
        with a one-hour TTL, and never copied into the record. A call that got
        everything it came for wrote down THAT it had succeeded and not WHAT it
        learned, which is a more complete defeat of a four-field script than any
        fabricated quote.

        Derived from the objective rather than from a hand-written list of keys,
        so a template that adds a fifth field does not also have to remember to
        add it here — that omission is exactly how the first three went missing.
        """
        out: dict = {}
        for f in _objective_of(self).fields:
            value = self.memory.get(f.memory_key)
            if value is None:
                continue
            entry = {"value": value}
            # Their own words, and the qualifier, where the tool records them.
            for suffix in ("heard", "detail", "depends_on"):
                extra = self.memory.get(f"{f.memory_key}_{suffix}")
                if extra:
                    entry[suffix] = extra
            out[f.name] = entry
        return out

    def reset_ask_budget(self, why: str) -> None:
        """The caller engaged, or something was collected: start the budget over.

        THREE SITES USED TO DO THIS BY HAND and they disagreed about which
        counters to clear — the escalation-block site reset the vetting count,
        the hold site and the named-a-place site did not, for no stated reason.
        Each of those resets fires precisely when the caller has just proved
        they are engaging, which is the condition for clearing all of them.

        Prints, because a guard that silently undoes another guard's work is how
        the give-up directive came to fire on a call that had already been
        answered twice.
        """
        if self._give_up_sent or self._unanswered_asks or self._asks_without_progress:
            print(f"[Realtime] Ask budget reset — {why} "
                  f"(was unanswered={self._unanswered_asks} "
                  f"no-progress={self._asks_without_progress} "
                  f"give_up={self._give_up_sent})", flush=True)
        self._give_up_sent = False
        self._give_up_at_turn = None
        self._unanswered_asks = 0
        self._asks_without_progress = 0
        self._vetting_reasks = 0

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

    def _enrich_doctor(self, branch: Optional[str], outcome: Outcome) -> dict:
        """Apply what this call learned to self.doctor, and describe the result.

        Mirrors the email agent's node_parse_done, which is the only other
        place a Doctor is enriched — same fields, same intent, so a record
        touched by voice and one touched by email stay comparable.

        TAKES THE OUTCOME, NOT A BOOLEAN, and the difference is the point of the
        2026-08-24 change. The write used to be gated on `resolved and branch`,
        so a call-level verdict decided whether a FIELD got written — and since
        save_branch was the only thing that could set that verdict, a call which
        collected something else and no branch wrote nothing at all. A field
        that passed every guard is worth recording whatever the call as a whole
        managed; the outcome decides the STATUS, not whether the data lands.

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
        if branch:
            doc.branch = branch
            city = self.memory.get("city")
            if city:
                doc.city = city
            # The first assignment of Source.VOICE anywhere in the programme.
            doc.source = Source.VOICE
            # VERIFIED only when the call met its whole objective AND the record
            # is otherwise usable. A partial call that got the branch still
            # writes the branch — it just does not claim the record is verified.
            doc.status = (DoctorStatus.VERIFIED
                          if outcome is Outcome.COMPLETE and doc.is_complete()
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
            # The non-branch fields, on the row the client actually reads. The
            # Doctor model has no column for them and inventing one here would
            # put a second schema in the directory, so they travel as a nested
            # dict beside the columns that do exist — visible, and clearly not
            # pretending to be validated Doctor fields.
            "collected_fields": self.collected_fields(),
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
        # RE-DERIVED HERE, not read as a fact somebody else asserted. The
        # objective is the authority on what this call was for; memory holds a
        # copy written after each tool call for the readers that only know
        # about `resolved`, and recomputing means a call that ended without a
        # final tool call still reports truthfully.
        objective = _objective_of(self)
        outcome  = objective.outcome(self.memory)
        resolved = objective.is_success(self.memory)
        collected = list(objective.collected(self.memory))
        missing   = list(objective.missing(self.memory))
        # Write what the call learned back onto the record it came from. Until
        # now nothing did: a resolved call wrote a CallRecord and the Doctor
        # that started it was never touched, so Source.VOICE was assigned to
        # nothing anywhere in the repo. The programme's purpose is enriching a
        # client directory, and the enrichment was ending at the call log.
        doctor_record = self._enrich_doctor(branch, outcome)
        # clean_doctor_name strips "Dr." so we don't get "Dr. Dr. John"
        from agents.voice.templates import clean_doctor_name
        doctor_display = clean_doctor_name(self.doctor.doctor_name)
        # Reports the BRANCH on its own terms. It used to say "Branch could not
        # be confirmed" whenever `resolved` was false, which for a partial call
        # would be a false sentence sitting next to the branch it denies.
        summary = (
            f"Called {self.doctor.hospital_name} to verify Dr. {doctor_display}'s branch. "
            + (f"Branch confirmed: {branch}." if branch
               else "Branch could not be confirmed.")
            + (f" Not collected: {', '.join(missing)}." if missing else "")
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

                # A verbatim repeat is the DEFECT, not noise — keep both.
                #
                # call-20260819-2044: the agent said "Sure, no rush." twice in
                # one breath, the live detector printed 🔁 REPEATED SENTENCE,
                # and the saved artifact showed one turn and
                # `repeated sentences 0`. The fragment merge below fired first
                # ("Sure, no rush." is 3 words, under the ≤4 fragment
                # threshold) and replaced the pair with a single turn, and the
                # duplicate-drop further down would have removed it anyway.
                #
                # Both rules were written for a LOGGING artifact — the same
                # turn recorded twice by a barge-in double-fire. They cannot
                # tell that from the model genuinely saying it twice, so they
                # deleted the evidence for the one case where it mattered. An
                # instrument that reads zero on the fault it exists to count is
                # worse than no instrument, because it is believed.
                if (prev.role == "agent" == turn.role
                        and _norm_clause(prev.text) == _norm_clause(turn.text)):
                    merged.append(turn)
                    continue

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

                # (The duplicate-agent-turn drop that used to live here is gone
                #  — see the keep-both rule above. It is unreachable now
                #  regardless, since identical adjacent agent turns are settled
                #  before either merge runs.)

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
            # THREE-VALUED, next to the boolean rather than instead of it. Every
            # existing reader wants `resolved`; none of them can express "got
            # the branch, never got the accepting status", which is the shape
            # the next script produces on a good call that ran out of patience.
            "outcome":        outcome.label,
            "collected":      collected,
            "missing":        missing,
            # WHAT it learned, not merely THAT it learned something. `collected`
            # is a list of field NAMES; without this the values never left
            # CallMemory and the artifact could not answer "is this practice
            # taking new patients", which is the question the call was placed to
            # answer. Carries each field's value, the caller's own words as
            # selected from the transcript, and any qualifier that survived
            # grounding.
            "fields":         self.collected_fields(),
            "success_at":     objective.success_at.label,
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
            # Turns the quarantine discarded. Recorded because a SILENT DROP
            # is invisible: on call-20260819-2006 two turns were dropped and
            # the artifact said nothing, so the only evidence was a terminal
            # someone happened to still have open. A guard that removes a
            # caller's words has to leave a trace of what it removed.
            "suppressed_echoes": self.suppressed_echoes or None,
            # Turns the transcriber produced over a silent line. Non-null means
            # the model was told words the caller never said.
            "fabricated_turns": self.fabricated_turns or None,
            # One entry per rejected transcript: whether the reply it had
            # already provoked was stopped in time, and the margin. This
            # is the measurement that decides whether cancelling is
            # enough or response creation has to come off OpenAI's VAD.
            "rejection_cancels": self.rejection_cancels or None,
            # Second-spoken-item audio withheld before it reached the caller.
            # Non-null here means the model tried to talk over itself.
            "dropped_second_items": self.dropped_second_items or None,
            # Backchannels played, and inbound frames withheld while one was
            # still audible. The second number is the whole reason the first
            # can be trusted: our clips are "mm-hm"/"okay"/"right"/"sure", and
            # a caller genuinely saying "Okay." transcribes identically, so
            # echo could never be found in the transcript afterwards. Non-zero
            # means speakerphone echo is real on this line and was stopped.
            "backchannels_sent": self._backchannels_sent or None,
            "backchannel_echo_frames": self._backchannel_echo_frames or None,
            # Of those, how many carried the substance of the turn and were
            # said on the next one instead of lost.
            "owed_substance_recovered": self._owed_recovered or None,
            # Replies that began playing on top of the previous one's queue.
            # Each is a stretch where the callee had no gap to speak into.
            "stacked_replies": self._stacked_replies or None,
            # Times the caller heard "saved" for a save that was rejected.
            "false_save_claims": self._false_save_claims or None,
            # Why any response failed. Non-null means dead air with a cause.
            "response_failures": self.response_failures or None,
            # Pickup to first sound, in seconds. The figure the callee
            # experiences; None when /answer could not be timed.
            "pickup_to_greeting_s": self.pickup_to_greeting_s,
            # Measured caller-stops → agent-speaks gaps, in seconds. The median
            # is the number to compare across calls; the max is the one the
            # callee remembers.
            "reply_latency": {
                "turns":  len(self.reply_latencies),
                "median": round(median(self.reply_latencies), 2)
                          if self.reply_latencies else None,
                "worst":  round(max(self.reply_latencies), 2)
                          if self.reply_latencies else None,
                # Measured, not assumed: median time the detector took to
                # report the stop. vad_hold_s used to be silence_ms echoed
                # back, which told you the setting, never the behaviour.
                "detector_lag_s": (round(median(self.detector_lags), 2)
                                   if self.detector_lags else None),
            } if self.reply_latencies else None,
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
                self._give_up_sent, bool(self.memory.get("escalated")),
                self._give_up_trigger),
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
                "outcome":          outcome.label,
                "missing":          missing,
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
        log.info("Realtime call saved: %s (%s resolved=%s branch=%s)",
                 self.call_id, _describe_objective(objective, self.memory),
                 resolved, branch)

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
        elif outcome is Outcome.PARTIAL:
            # PRINTED AS PARTIAL, not as a failure. A call that collected some
            # of what it came for and is filed as NOT RESOLVED is the defect
            # this outcome exists to stop, and the console is where it would be
            # believed first.
            print(f"  Result   : ◐ PARTIAL — got {', '.join(collected)}; "
                  f"missing {', '.join(missing)}", flush=True)
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
        print(f"    said twice in a row  {m['back_to_back_repeats']}"
              f"{'   <- word for word, back to back' if m['back_to_back_repeats'] else ''}",
              flush=True)
        if self.dropped_second_items:
            print(f"    2nd items muted      {len(self.dropped_second_items)}"
                  f"   (would have talked over itself)", flush=True)
        if self.pickup_to_greeting_s is not None:
            print(f"    pickup -> greeting   {self.pickup_to_greeting_s:.2f}s"
                  f"{'   <- dead air before the agent says anything' if self.pickup_to_greeting_s > 3 else ''}",
                  flush=True)
        if self.response_failures:
            print(f"    responses failed     {len(self.response_failures)}"
                  f"   <- each one is dead air on the line", flush=True)
            _seen_why: dict = {}
            for _f in self.response_failures:
                _seen_why[_f["reason"]] = _seen_why.get(_f["reason"], 0) + 1
            for _why, _n in sorted(_seen_why.items(), key=lambda kv: -kv[1]):
                print(f"        {_n}x  {_why[:88]}", flush=True)
        if self.reply_latencies:
            _vad = (median(self.detector_lags) if self.detector_lags
                    else 0.0)
            print(f"    reply gap            "
                  f"median {median(self.reply_latencies):.2f}s, "
                  f"worst {max(self.reply_latencies):.2f}s "
                  f"({len(self.reply_latencies)} turns)", flush=True)
            print(f"      of which detector  {_vad:.2f}s — measured, then "
                  f"inference plus the round trip", flush=True)
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

# ── Pre-warming ──────────────────────────────────────────────────────────────
# Measured on call-20260819-1915: 6.4 SECONDS between the callee pressing
# answer and hearing a word. All of it before the media stream opened —
#
#     /answer webhook  -> ngrok -> India -> TwiML back      ~1.0s
#     media WebSocket  -> ngrok -> India                    ~1.0s
#     OpenAI handshake FROM India                           ~1.7s
#     session.update round trip                             ~0.5s
#     response.create -> first audio                        ~1.1s
#
# The middle two are ours, and they only start once someone has already picked
# up — the phone rings for seconds while we do nothing with the time.
#
# The connect and the session configuration need NOTHING call-specific: the
# instructions are the template's static text, and the audio block comes from
# settings. Only the context item (doctor, hospital, greeting) is per-call, and
# that is sent after the stream opens. So the whole handshake can happen while
# the phone is still ringing.
#
# Failure is free: if pre-warming does not finish, or the callee never answers,
# handle_realtime connects the old way. Nothing depends on it succeeding.
_PREWARMED: dict[str, tuple] = {}

# A session held longer than this was almost certainly for a call nobody
# answered. Closed rather than handed to a later call, which would give that
# call a socket that has been idle for minutes.
_PREWARM_TTL_S = 150.0


async def _open_realtime_session(template) -> tuple:
    """Connect to OpenAI and apply session.update. Returns (conn, ws).

    Extracted so the pre-warm and the connect-on-answer path cannot drift —
    a second copy of this would be one more place for the audio config or the
    cached-prefix rule to be got subtly wrong.
    """
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
    model   = settings.realtime_model
    ws_obj = None
    conn = None
    for _attempt in (1, 2):
        try:
            conn = websockets.connect(REALTIME_URL.format(model=model),
                                      additional_headers=headers,
                                      open_timeout=_OAI_CONNECT_TIMEOUT_S)
            ws_obj = await conn.__aenter__()
            break
        except Exception as e:
            log.warning("[Realtime] OpenAI handshake attempt %d/2 failed: %s: %s",
                        _attempt, type(e).__name__, e)
            if _attempt == 2:
                raise
    if ws_obj is None or conn is None:
        raise RuntimeError("realtime handshake returned no socket")
    try:
        raw = await asyncio.wait_for(ws_obj.recv(), timeout=10.0)
        first = json.loads(raw)
        if first.get("type") == "error":
            err = first.get("error", {})
            raise RuntimeError(f"{model} rejected the connection: {err.get('message')}")
        # ONE session.update, everything in it. Splitting it churned the cached
        # prefix. `instructions` is the template's STATIC text — no doctor, no
        # hospital, no time of day; those go in the per-call conversation item.
        await ws_obj.send(json.dumps({
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
        for _ in range(10):
            sc = json.loads(await asyncio.wait_for(ws_obj.recv(), timeout=10.0))
            ev = sc.get("type", "")
            if ev == "error":
                err = sc.get("error", {})
                raise RuntimeError(
                    f"session.update rejected: {err.get('code')} {err.get('message')}")
            if ev == "session.updated":
                break
        return conn, ws_obj
    except Exception:
        # Close the socket we opened — the old code leaked it on the timeout path.
        await conn.__aexit__(None, None, None)
        raise


async def prewarm_realtime(call_sid: str) -> None:
    """Open and configure a session while the phone rings. Never raises."""
    try:
        _sweep_prewarmed()
        conn, ws = await _open_realtime_session(get_template(settings.call_template))
        _PREWARMED[call_sid] = (conn, ws, time.time())
        print(f"[Realtime] Pre-warmed a session while the phone rings — the "
              f"greeting will not wait on a handshake", flush=True)
    except Exception as e:
        # Deliberately swallowed. The call still works; it just pays the
        # handshake on answer, exactly as it did before.
        log.warning("[Realtime] pre-warm failed (%s: %s) — connecting on answer "
                    "instead", type(e).__name__, e)


def _sweep_prewarmed() -> None:
    """Close sessions whose call was never answered."""
    now = time.time()
    for sid in [s for s, (_, _, t) in _PREWARMED.items() if now - t > _PREWARM_TTL_S]:
        conn, _, _ = _PREWARMED.pop(sid)
        asyncio.create_task(_close_quietly(conn))
        log.info("[Realtime] discarded a stale pre-warmed session for %s", sid)


async def _close_quietly(conn) -> None:
    try:
        await conn.__aexit__(None, None, None)
    except Exception:
        pass


def take_prewarmed(call_sid: str) -> Optional[tuple]:
    """Claim the pre-warmed session for this call, if one is ready and fresh."""
    entry = _PREWARMED.pop(call_sid, None)
    if entry is None:
        return None
    conn, ws, made_at = entry
    if time.time() - made_at > _PREWARM_TTL_S:
        asyncio.create_task(_close_quietly(conn))
        return None
    return conn, ws


async def handle_realtime(twilio_ws: WebSocket, call_sid: str, doctor: Doctor,
                          answered_at: Optional[float] = None) -> None:
    """Bridge Twilio WebSocket ↔ OpenAI Realtime API for a single call.

    ``answered_at`` is time.monotonic() at the /answer webhook — the moment
    Twilio says the callee picked up. Optional because the caller may not have
    it (the test harness does not), and a missing value simply means the
    pickup-to-greeting figure is not reported rather than a wrong one being
    reported.

    IT IS A FLOOR, NOT THE FIGURE. The clock starts when the webhook ARRIVES
    here, and the callee has been holding a silent line since before Twilio
    sent it. Twilio's own event log for call-20260820-1154: pickup 06:23:56,
    /answer fetched 06:23:57, round trip 623ms — so ~1.3s of the callee's
    silence had already elapsed before this value starts counting. Reported
    6.73s; actual ~7.0s. To get the true figure, read `start_time` off the
    Twilio call record afterwards; there is no way to know it during the call.
    """
    sess     = RealtimeSession(call_sid, doctor)
    sess._answered_at = answered_at
    template = get_template(settings.call_template)

    # Never let configured settings be silently ignored — someone set them for a
    # reason, and a call going out under the wrong org name or in the wrong
    # language is not recoverable once the callee has heard it.
    # ORG_NAME is deliberately NOT passed: it is a per-call value now, reaching
    # the model through build_context(), so there is nothing about it to warn
    # on. It used to be passed into a parameter that ignored it, which read
    # from here like a check that was not happening.
    for warning in template.config_warnings(agent_language=settings.agent_language):
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
    sess.transcribe_hint = template.transcribe_hint
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

    # Claim the session pre-warmed while the phone was ringing, or connect
    # now. See prewarm_realtime: the handshake and session.update need nothing
    # call-specific, so they can happen before anyone answers — and on
    # call-20260819-1915 they were 2.2s of the 6.4s the callee spent listening
    # to silence.
    _pre = take_prewarmed(call_sid)
    if _pre is not None:
        conn, ws_obj = _pre
        print(f"[Realtime] Connected: {settings.realtime_model} (pre-warmed)", flush=True)
    else:
        conn, ws_obj = await _open_realtime_session(template)
        print(f"[Realtime] Connected: {settings.realtime_model}", flush=True)
    print(f"[Realtime] Session configured — template={template.name} "
          f"voice={settings.realtime_voice}", flush=True)

    oai_ws_ctx = conn

    try:
        oai_ws = ws_obj

        # ── 3. Wait for Twilio stream to start ────────────────────────
        print("[Realtime] Waiting for Twilio stream start...", flush=True)
        _socket_open_at = time.monotonic()
        async for raw_msg in twilio_ws.iter_text():
            msg = json.loads(raw_msg)
            if msg.get("event") == "start":
                sess.stream_sid = msg["start"]["streamSid"]
                sess._stream_start_time = datetime.now()
                # The last unmeasured leg of the dead air. Together with the
                # "/answer -> socket open" line in twilio_worker, this splits
                # the single "Twilio setup" figure into the two halves that
                # have different fixes: the socket-open half is tunnel and
                # transport, this half is Twilio's own handshake and is not
                # something a different tunnel would change.
                print(f"[Realtime] Twilio stream started: {sess.stream_sid} "
                      f"({time.monotonic() - _socket_open_at:.2f}s after the "
                      f"socket opened — Twilio's own handshake)", flush=True)
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
            asyncio.create_task(_silence_watchdog(oai_ws, sess, done_event, twilio_ws),
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
                        _oai_bytes = raw_bytes
                        oai_payload = payload
                    else:
                        raw_bytes = base64.b64decode(payload)
                        pcm_24k = (resample(_mulaw_decode(raw_bytes), _TWILIO_SR, _OAI_SR) * 32767).astype(np.int16)
                        _oai_bytes = pcm_24k.tobytes()
                        oai_payload = base64.b64encode(_oai_bytes).decode()
                    # THE RECORDING. Every inbound frame, unconditionally, so
                    # save() gets a gapless timeline. Deliberately NOT the
                    # buffer OpenAI's timestamps index into — see the mirror
                    # append at the send below.
                    sess._caller_pcm.append(_oai_bytes)
                    if not sess.listen_enabled.is_set():
                        continue
                    # Our own backchannel, coming back off a speakerphone.
                    # Energy, not a hard mute: the caller is BY DEFINITION
                    # mid-utterance here (a clip only fires 2.8s into their
                    # turn), so muting outright would cut real speech. Their
                    # measured level on the Twilio channel is 0.079-0.240
                    # against a threshold of 0.020 — real speech passes with
                    # a wide margin, an attenuated "mm-hm" does not.
                    if _is_own_backchannel_echo(sess, raw_bytes):
                        sess._backchannel_echo_frames += 1
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
                    # THE MIRROR. Appended here, at the send, and nowhere else
                    # — so a frame is in this buffer if and only if OpenAI
                    # received it. Every `continue` above is a frame OpenAI
                    # never saw, and any of them appending would put our index
                    # ahead of OpenAI's ms clock by exactly that much.
                    #
                    # That is not hypothetical. On call-20260821-1856 the
                    # backchannel echo guard withheld 173 frames while
                    # _caller_pcm still took them: 3.46s of drift, after which
                    # _utterance_slice read past the end of every utterance
                    # into mu-law silence and reported rms=0.000244 — the same
                    # fingerprint _utterance_slice's own docstring names. The
                    # quarantine then deleted the caller's real answers
                    # ("She's in San Francisco.", "clinic") as fabrications and
                    # the call ended unresolved with the answer in hand.
                    sess._caller_oai_pcm.append(_oai_bytes)
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

# How long the silence watchdog stands down after the caller asks for a moment.
# On call-20260819-1619 the caller said "give me a minute I just need to check",
# the agent correctly answered "No rush." — and the watchdog then fired 7s later
# and made it ask again, twice in one call, while the caller was still looking.
# The prompt already says "THE HOLD LASTS UNTIL THEY COME BACK WITH AN ANSWER.
# Not one turn — the whole time." The model obeyed it; the watchdog, which had
# no idea a hold was in progress, overrode it.
#
# Long enough to actually look something up. Bounded so a caller who never
# returns still eventually gets a "still there?" instead of silence forever.
# How many rejected save_branch attempts before the agent is handed the
# caller's verbatim words and told to stop rephrasing. Three, because two
# is a normal correction cycle — the first attempt is often genuinely wrong
# and the second fixes it — while the third is the point at which the model
# is demonstrably guessing rather than reading the transcript.
_MAX_SAVE_REJECTIONS = 3

_HOLD_GRACE_S = 45.0
_MAX_SILENCE_PROMPTS = 2


async def _silence_watchdog(oai_ws, sess: "RealtimeSession",
                            done_event: asyncio.Event,
                            twilio_ws=None) -> None:
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

        # ── Backchannel ────────────────────────────────────────────────────
        # Owned here because it is the only thing that runs BETWEEN events: no
        # OpenAI event arrives while the caller is mid-utterance, so the event
        # loop cannot notice that they have been talking for three seconds.
        #
        # Injected straight into the Twilio stream — no response.create, so it
        # cannot collide with turn detection, cannot be cancelled by the
        # caller's own speech, and costs nothing. It is a noise, not a turn:
        # nothing downstream records it.
        _spk = sess._caller_speaking_since
        if (settings.realtime_backchannels
                and _spk is not None and not sess.done
                and not sess.agent_speaking
                and not sess._backchannel_done_this_utterance
                and sess.listen_enabled.is_set()
                and sess.stream_sid and twilio_ws is not None
                and time.time() - _spk >= _BACKCHANNEL_AFTER_S
                and time.time() - sess._last_backchannel_at >= _BACKCHANNEL_COOLDOWN_S):
            _payload = backchannel.pick(settings.realtime_voice,
                                        exclude=sess._last_backchannel_clip)
            # None means no clips are installed for this voice; the feature is
            # simply off, which is the behaviour that already existed.
            if _payload:
                sess._backchannel_done_this_utterance = True
                sess._last_backchannel_at = time.time()
                sess._last_backchannel_clip = _payload
                sess._backchannels_sent += 1
                # Shut the echo window BEFORE the clip goes out, and size it
                # from the clip's own length — 8000 bytes/s of 8kHz mu-law.
                sess._backchannel_mute_until = (
                    time.time() + len(base64.b64decode(_payload)) / 8000.0
                    + _BACKCHANNEL_ECHO_MARGIN_S)
                try:
                    await twilio_ws.send_text(json.dumps({
                        "event": "media", "streamSid": sess.stream_sid,
                        "media": {"payload": _payload},
                    }))
                    print(f"[Realtime] 👂 backchannel while they talk "
                          f"({time.time() - _spk:.1f}s in)", flush=True)
                except Exception as e:
                    log.warning("[Realtime] backchannel send failed: %s", e)

        # Deferred completion-claim check. Waits for any tool call belonging
        # to that response to land, so this only fires when save_branch was
        # never called at all — not when it was called and rejected, which has
        # its own correction at the tool site.
        if (sess._claimed_done_at and not sess._claimed_done_nudged
                and not sess.done
                and time.time() - sess._claimed_done_at >= 1.5):
            sess._claimed_done_at = 0.0
            if not sess.memory.get("branch") and not sess.memory.get("escalated"):
                sess._claimed_done_nudged = True
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"⚠️  CLAIMED DONE, NOTHING SAVED — telling the agent to "
                      f"actually record it", flush=True)
                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text", "text": (
                                 "(system: you told them you were finished, but "
                                 "nothing has been recorded — save_branch was "
                                 "never called. If they gave you a location, "
                                 "call save_branch NOW with their exact words. "
                                 "If they did not, do not imply the call is "
                                 "over: ask for it.)")}]},
                }))

        # Deferred substance recovery. Owned by the watchdog for the same
        # reason the goodbye retry is: the drop is detected inside the OpenAI
        # event pump while a response is still settling, and creating one from
        # there collides with it. Here, _create_response's own policy applies —
        # including the playback gate, so the recovery cannot itself stack.
        #
        # call-20260820-1421: the caller asked "can you repeat that question
        # please?", the answer was muted behind "Sure, I'll repeat it clearly.",
        # and nothing said it. Seven seconds later the silence watchdog asked
        # "Are you still with me?" — which is the wrong sentence, because the
        # line was not the problem. They asked twice more and hung up at 88s.
        if (sess._owed_substance and not sess.done
                and not sess._response_active
                and sess.listen_enabled.is_set()):
            # STATE CHANGES AFTER THE OPERATION THAT DECIDES SUCCESS, not
            # before. The first cut of this cleared _owed_substance, counted a
            # recovery and printed "saying it now" — all ahead of a
            # _create_response that can refuse.
            #
            # It refused on the very next call. call-20260820-1440: the owed
            # text was detected at t=45.0s while the previous reply's audio ran
            # to t=45.86s, so the playback gate declined, no response was ever
            # created, and the owed half was dropped. Ten agent blocks, ten
            # spoken turns, no recovery among them — and the log said it had
            # been said. Exactly the false-save shape: a success message
            # emitted before the thing that determines success.
            #
            # Left in place on refusal, so the next tick retries once the queue
            # has drained — but the DIRECTIVE is sent only once. Retrying the
            # whole block would inject it again on every tick, and the model
            # would be told the same thing several times over.
            _owed = sess._owed_substance
            if not sess._owed_directive_sent:
                sess._owed_directive_sent = True
                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text", "text": (
                                 "(system: only the first half of your last "
                                 "turn reached them. They did not hear this: "
                                 f"\"{_owed}\". Say just that, now, in one "
                                 "short sentence. Do not repeat the half they "
                                 "did hear and do not apologise for it.)")}]},
                }))
            if await _create_response(oai_ws, sess, why="owed substance"):
                sess._owed_substance = ""
                sess._owed_directive_sent = False
                sess._owed_recovered += 1
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"💬 OWED SUBSTANCE — the half they never heard, "
                      f"saying it now: {_owed[:60]!r}", flush=True)
            continue

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
        # They asked for a moment. Silence is them doing what they said they
        # would do, not a dropped line — prompting here is the badgering the
        # prompt's hold rules exist to prevent, and the watchdog was the one
        # doing it. See _HOLD_GRACE_S.
        if time.time() < sess._hold_until:
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

    # What the call had collected BEFORE this tool ran, so the no-progress
    # ceiling can be reset by progress rather than by a guess about which tool
    # constitutes progress. save_branch is not the only way a field arrives —
    # a template may point a field at a note_* key — and hard-coding the tool
    # name here is how the success condition ended up inside save_branch in the
    # first place.
    _collected_before = set(_objective_of(sess).collected(sess.memory))

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
        # THE CALLER ANSWERED. Whatever happens to the value below, the model
        # only reaches here because it believed it heard a place — so the ask
        # budget, which exists to stop the agent pestering someone who will not
        # engage, has no business counting this call against them.
        #
        # It did, and it cost call-20260821-1931. The caller said "Mission Bay
        # Clinic, 1825 4th Street"; the live transcript mangled it to "Ford
        # Street"; grounding rejected the model's correct reading of it; the
        # agent asked again, that re-ask hit the 4-ask limit, and the give-up
        # directive fired. The caller then repeated the address cleanly — and
        # _ungrounded_terms passes on that transcript, verified — but the agent
        # had already been told to stop, so it said goodbye instead of
        # retrying. The recovery path existed and the budget closed it.
        #
        # Safe because the two budgets measure different things and only this
        # one is being reset: a model that keeps offering bad values is still
        # bounded by _MAX_SAVE_REJECTIONS, which counts up while this counts
        # down. Charging a rejected save to both is double jeopardy, and the
        # person paying it is the caller who answered.
        if str(args.get("branch") or "").strip():
            sess.reset_ask_budget("caller named a place")
        heard_any = any(t.role == "caller" and t.text.strip() != "[...]"
                        for t in sess.turns)
        # QUALIFIED, not just asserted. Grounding accepts on one content word,
        # so "verified against caller transcript" was being stamped on values
        # whose distinctive part nobody was heard to say — see _rode_along.
        _rode = _rode_along(args, sess)
        sess.memory.update(
            grounding=("verified against caller transcript"
                       + (f" EXCEPT {', '.join(repr(w) for w in _rode)}, "
                          f"which the caller was never transcribed saying"
                          if _rode else "")) if heard_any
            else "SKIPPED — no caller speech was transcribed on this "
                 "call, so the saved location could not be checked "
                 "against anything the caller actually said"
        )
        if _rode:
            sess.memory.update(rode_along=_rode)
            print(f"[Realtime] ⚠️  grounded on other words — "
                  f"{', '.join(repr(w) for w in _rode)} never appeared in the "
                  f"caller transcript", flush=True)
        ungrounded = _ungrounded_terms(args, sess)
        # FIRST, because it is the only gate that can see this one. A bare
        # hint word passes grounding (it is in the transcript), passes the
        # address check, and passes the organisation check — all three were
        # measured false on call-20260821-1705 while "Suite" sat in args.
        _val = str(args.get("branch") or "")
        if _is_bare_hint_word(_val, getattr(sess, "transcribe_hint", "") or ""):
            sess.memory.update(untrusted_location=_val)
            result = {
                "ok": False,
                "error": (
                    f"NOT SAVED — {_val!r} is one generic location word, not "
                    f"the name of a place | LIKELY a transcription artifact "
                    f"on a turn that carried no speech "
                    f"| NEED: the site name in full, as they said it"
                ),
            }
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                  f"🎣 HINT ECHO BLOCKED: {_val!r} came from our own "
                  f"transcription prompt", flush=True)
        elif ungrounded:
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
            # SAY WHICH PART IS WRONG. _ungrounded_terms has always computed a
            # specific reason — which field, which value, which number — and
            # this site discarded it and sent a generic line instead. Same
            # shape as 5aed263, where the failure reason was in every event and
            # was thrown away: the diagnosis existed and never reached anyone.
            #
            # It cost a real call. On call-20260820-1321 two rejections in a
            # row said only "NEED: wording the caller used out loud", so the
            # model could not tell that its NUMBER was the problem — and "out
            # loud" reads as "as spoken", which is an active nudge toward
            # spelling digits into words. It did exactly that on the third try
            # and bypassed the digit guard entirely.
            #
            # The reason text is built terse and non-speakable for the same
            # reason the rest of these are: it is machinery, and on
            # call-20260818-1112 the agent read one of these out to a caller.
            result = {
                "ok": False,
                "error": (
                    f"REJECTED — {ungrounded} "
                    f"| RE-READ: caller turns, verbatim; a valid "
                    f"location is often already among them "
                    f"| NEED: their own words, any number in digits"
                ),
            }
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                  f"🚫 HALLUCINATED BRANCH BLOCKED: {args}", flush=True)
        elif (_dropped := _address_dropped(args, sess)) and not sess._address_nudged:
            # ONE-SHOT. The value being saved is CORRECT, only less complete
            # than what they said, so this must never be able to stop the call
            # finishing — a true-but-thin record beats no record at all.
            #
            # The rejection points at the transcript rather than at the caller:
            # they already supplied it, and a wording that sends the agent back
            # to ask again is how call-20260818-1112 lost an answer that was
            # already on the call.
            sess._address_nudged = True
            sess.memory.update(address_offered=_dropped)
            result = {"ok": False, "error": (
                f"NOT SAVED — a street address was given and this value omits "
                f"it | THEY SAID: {_dropped!r} | RETRY: save_branch with both, "
                f"comma-separated | ALREADY SUPPLIED, nothing further needed "
                f"from them")}
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                  f"📍 ADDRESS DROPPED — they gave {_dropped!r}; asking for it "
                  f"to be saved too", flush=True)
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
            result = run_tool(name, sess.memory, args, sess.objective)
    elif name in _CHOICE_SAVE_TOOLS:
        # THE CALLER ANSWERED, whatever becomes of the value — same reasoning as
        # the save_branch reset above. The model only reaches here because it
        # believed it heard a state, so the budget that exists to stop the agent
        # pestering someone who will not engage has no business counting it.
        _arg, _guard, _need, _gkey = _CHOICE_SAVE_TOOLS[name]
        if str(args.get(_arg) or "").strip():
            sess.reset_ask_budget(f"caller answered: {name}")
        ungrounded_choice = _guard(args, sess)
        if ungrounded_choice:
            sess.memory.update(**{_gkey: f"BLOCKED — {ungrounded_choice}"})
            # Terse fragments, no fluent imperative. A rejection the model can
            # paraphrase into a grammatical sentence is a rejection it will read
            # out loud — see _reject's docstring in tools.py, which exists
            # because a live call relayed one to a receptionist verbatim.
            result = {"ok": False, "error": (
                f"NOT SAVED — {ungrounded_choice} | NEED: {_need}")}
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                  f"🚫 UNGROUNDED ANSWER BLOCKED: {name}({args})", flush=True)
        else:
            sess.memory.update(**{_gkey: "verified against caller transcript"})
            # THE QUALIFIER IS DROPPED, NOT THE SAVE. The status is grounded and
            # worth recording; a summary carrying a word nobody said is not, and
            # refusing the whole call over it would throw away a verified answer
            # to protect a footnote. Recorded either way — a field silently
            # emptied is the same invisibility the fabricated version had.
            for _dkey in ("detail", "depends_on"):
                _was = args.get(_dkey)
                _bad, _what = _strip_ungrounded_detail(args, sess, _dkey)
                if _bad:
                    sess.memory.update(**{
                        f"{_gkey}_{_dkey}_as_written": _was,
                        f"{_gkey}_dropped_words": list(_bad)})
                    print(f"[Realtime] ⚠️  {_dkey}: "
                          f"{', '.join(repr(w) for w in _bad)} never "
                          f"appeared in the caller transcript — {_what}",
                          flush=True)
            result = run_tool(name, sess.memory, args, sess.objective)
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
        # Two shapes of "not a refusal", blocked the same way. A hold request
        # is "wait, I'm getting it"; an invitation is "what do you want?" —
        # and on call-20260819-2121 the agent answered the second by hanging
        # up. The caller had asked three screening questions, the budget
        # counted all three, the give-up directive went out, and then they
        # said "How can I help you?" — the most willing thing anyone said on
        # that call — and the agent closed on it.
        _blocked = ""
        if not sess.memory.get("branch"):
            if is_hold_request(last_caller):
                _blocked = "hold"
            elif _invites_continuation(last_caller):
                _blocked = "invitation"
        if _blocked:
            if _blocked == "hold":
                result = {"ok": False, "error": (
                    "NOT ESCALATED — caller is mid-lookup, not refusing "
                    "| NEED: a two-word hold acknowledgement, then "
                    "silence until they return")}
                _line = "⏳ ESCALATION BLOCKED — caller is checking"
                _say = ("(system: disregard the earlier instruction to "
                        "stop and escalate. They are looking the branch up "
                        "right now. Wait for them.)")
            else:
                result = {"ok": False, "error": (
                    "NOT ESCALATED — caller just asked what you need "
                    "| NEED: tell them plainly, in one sentence, which "
                    "doctor and that you want the branch")}
                _line = "🚪 ESCALATION BLOCKED — caller asked what you need"
                _say = ("(system: disregard the earlier instruction to "
                        "stop and escalate. They have just asked what you "
                        "want, which means they are willing to help and "
                        "have not refused anything. Answer them: name the "
                        "doctor and say you are trying to find out which "
                        "branch they work out of. One sentence, then "
                        "wait.)")
            # The budget put us here, and it was wrong: they were engaging
            # the whole time. Reset it or the very next ask escalates again.
            sess.reset_ask_budget("escalation blocked — caller is engaging")
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                  f"{_line}: {last_caller[:60]!r}", flush=True)
            await oai_ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": _say}]},
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
            result = run_tool(name, sess.memory, args, sess.objective)
    else:
        result = run_tool(name, sess.memory, args, sess.objective)

    # Something new was collected: the no-progress ceiling starts over, whether
    # the call is finished or has another field (or another doctor) to go. This
    # is the reset that makes one ceiling work for a multi-field, multi-doctor
    # call without the counter having to know either number.
    _gained = set(_objective_of(sess).collected(sess.memory)) - _collected_before
    if _gained:
        sess.reset_ask_budget("collected " + ", ".join(sorted(_gained)))
        print(f"[Realtime] 🎯 {_describe_objective(_objective_of(sess), sess.memory)}",
              flush=True)

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
            # ── NOTHING BOUNDED THIS ────────────────────────────────────────
            # Every correction here is one-shot, and there was no counter at
            # all, so a model that cannot produce an acceptable value simply
            # keeps trying. call-20260820-1321: three attempts, each with a
            # closing line attached — "I'll note that and wrap up", "I'll note
            # it and let you go", "take care" — twenty seconds of a caller
            # being thanked for a branch that had not been recorded. The second
            # rejection got no correction at all, because _false_save_nudged
            # was already spent on the first.
            #
            # That call ended only because the third attempt slipped through
            # the spelled-number bypass. Closing that bypass removes the
            # accidental exit and leaves the loop unbounded, so the bound has
            # to be explicit — a fix that makes a guard stricter has to carry
            # the liveness that the leak was accidentally providing.
            #
            # Guessing is not the way out of this. The caller's own words are
            # already on the transcript and _candidate_location can quote them,
            # so at the limit the model is handed the answer verbatim rather
            # than asked to try again. If it still cannot save, escalating with
            # a true reason beats a call that never ends.
            sess._save_rejections += 1
            if sess._save_rejections >= _MAX_SAVE_REJECTIONS and not sess.done:
                _cand = _candidate_location(sess)
                print(f"[{ts}] 🧱 {sess._save_rejections} save attempts "
                      f"rejected — handing the agent the caller's own words",
                      flush=True)
                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text", "text": (
                                 # Opens with plain lowercase words, like every
                                 # other directive here: the suite finds them by
                                 # reading the source, and an f-string starting
                                 # with a placeholder is invisible to it.
                                 f"(system: nothing has been recorded and "
                                 f"{sess._save_rejections} save attempts have "
                                 f"been rejected. Stop rephrasing it. "
                                 + (f"The caller's own words were: {_cand}. "
                                    f"Call save_branch with exactly that "
                                    f"wording, copying any number digit for "
                                    f"digit. " if _cand else "")
                                 + "If that is rejected too, call escalate "
                                   "with reason 'could not obtain the "
                                   "location'. Do not tell them it is saved "
                                   "and do not say goodbye again until one of "
                                   "those succeeds.)")}]},
                }))
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
            # FIRES ON EVERY FALSE CLAIM, not once per call. It was
            # one-shot because nothing bounded the retry loop and a guard
            # that can nag forever is its own failure. _MAX_SAVE_REJECTIONS
            # bounds it now, so this can cost at most that many nudges —
            # and each one answers a separate thing the caller was actually
            # told. On call-20260820-1321 the second claim, "Thanks for
            # that branch name — I'll note it and let you go", got no
            # correction at all: the flag was spent on the first, so the
            # caller was left believing a branch had been recorded that
            # had not. Leaving a false statement standing to avoid
            # repeating yourself is the wrong trade.
            if _claims_saved(_said) and not sess.done:
                sess._false_save_claims += 1
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
    elif name in _CHOICE_SAVE_TOOLS:
        _short = name.replace("save_", "").replace("_status", "")
        if ok:
            print(f"\n[{ts}] ✅ {_short.upper():<14}: {args}", flush=True)
        else:
            print(f"\n[{ts}] ⛔ {_short.upper():<14}: REJECTED {args}", flush=True)
            print(f"          reason: {result.get('error', '')}", flush=True)
    else:
        print(f"[{ts}] 🔧 TOOL           : {name}({args}) → {result}", flush=True)

    # WHEN THE CALL IS OVER, asked of the objective rather than of the tool.
    #
    # This was `name in ("save_branch", "escalate")`, which made a successful
    # save_branch the end of the call by definition — correct only for as long
    # as the branch was the only thing any call collected. On a template that
    # also collects the new-patient status it would hang up the moment the
    # branch landed, before the second question was ever asked, and the artifact
    # would record a PARTIAL call with no sign that we cut it short ourselves.
    #
    # COMPLETE, deliberately, not `is_success`. `success_at` says what counts as
    # a reportable success when the call is over; it must not decide when to
    # stop asking. A template that accepts a partial as success still wants the
    # rest of what it came for.
    if name == "escalate" and result.get("ok"):
        sess.done = True
    elif (result.get("ok")
            and (name == "save_branch" or name in _CHOICE_SAVE_TOOLS)
            and _objective_of(sess).outcome(sess.memory) is Outcome.COMPLETE):
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


async def _suppress_reply_to(sess: "RealtimeSession", oai_ws, text: str) -> str:
    """Stop the agent answering a transcript we have just rejected.

    REJECTING A TRANSCRIPT AND PREVENTING A REPLY TO IT ARE DIFFERENT THINGS,
    and until 2026-08-20 this file only did the first. `create_response` is not
    set in build_audio_config, so it runs on the API default of true and
    OpenAI's server VAD creates the response at speech_stopped — strictly
    before transcription exists. By the time a guard sees the words, the reply
    is already being generated:

        speech_stopped -> [VAD creates response] -> response.created
            -> input_audio_transcription.completed -> our guard rejects
            -> response.output_audio.delta  ... the agent answers it anyway

    call-20260820-1611: "Hi, I'm looking to schedule an appointment at Mercy
    Hospital" was dropped as unevidenced, and the agent replied "Okay, I'll
    hold." to it 1.61s later. The drop line printed BEFORE the first audio
    delta, so the reply was suppressible and nothing tried.

    THE STATE DISTINCTION IS THE WHOLE POINT. Cancelling only helps while no
    audio has reached the caller. Once it has, the words are out and pretending
    otherwise would be its own lie — so that case is reported, not swallowed.
    Returns the outcome for logging and for the artifact.
    """
    _since_stop = (time.monotonic() - sess._caller_stopped_at
                   if sess._caller_stopped_at else None)
    if not sess._response_active:
        outcome = "no reply in flight"
    elif sess._response_audio_started:
        # Too late by design, not by accident. Recorded so the margin can be
        # measured across calls rather than guessed at.
        outcome = "TOO LATE — audio already reaching the caller"
    else:
        await oai_ws.send(json.dumps({"type": "response.cancel"}))
        sess._response_active = False
        sess._suppressed_response = True
        outcome = "cancelled before any audio"
        # No Twilio `clear` here on purpose: this branch is only reached when
        # nothing from THIS response was ever forwarded, and a clear would
        # flush audio still legitimately playing from the previous one.
    sess.rejection_cancels.append({
        "text": text[:60],
        "outcome": outcome,
        "since_speech_stopped_s": (round(_since_stop, 3)
                                   if _since_stop is not None else None),
        "since_response_created_s": (
            round(time.monotonic() - sess._response_created_at, 3)
            if sess._response_created_at else None),
    })
    return outcome


async def _handle_caller_transcript(msg: dict, sess: "RealtimeSession", oai_ws) -> None:
    """One completed caller transcript: log it, and run the turn-level guards.

    Extracted from _oai_to_twilio to bring that function back under pyright's
    analysis ceiling — see _handle_tool_call for why that matters. This one
    shares NO mutable state with the event loop: everything it changes lives on
    `sess`, and its only `break` is a local for-loop break. That is what made it
    the safest of the large handlers to move.
    """
    text = msg.get("transcript", "").strip()

    # ── Quarantine hint regurgitation BEFORE it becomes a turn ──────────────
    # Everything downstream reads sess.turns as ground truth, so a fabricated
    # turn does not just mislead the model — it feeds the grounding guards.
    # On call-20260819-1324 a silent turn containing "Northwell campus" (all
    # hint vocabulary, audio_rms 0.000259) made _discarded_location block a
    # legitimate escalation, and the agent could not end the call.
    _hint = getattr(sess, "transcribe_hint", "") or ""
    if text and _hint:
        _cleaned = _strip_hint_run(text, _hint)
        if _cleaned != text:
            sess.suppressed_echoes.append(
                {"kind": "verbatim hint run", "raw": text, "kept": _cleaned})
            print(f"[Realtime] 🚱 HINT ECHO stripped from caller turn — the "
                  f"transcriber returned the prompt as speech", flush=True)
            text = _cleaned
    # Words on silence did not come from the caller. Rather than drop them
    # quietly — which leaves the agent apparently ignoring someone — treat it
    # as what it actually is: we did not hear them. The existing faint-line
    # nudge already says the right thing.
    # TWO SIGNALS, NEVER AUDIO ALONE.
    #
    # This used to drop on the audio measurement by itself, and that
    # measurement has now been observed wrong twice — 0.000244 (mu-law digital
    # silence) recorded for turns where the Twilio caller channel measures
    # 0.24. _utterance_slice fixes the cause, but a guard that DISCARDS a
    # caller's words must not rest on a single number that has been wrong
    # before: the cost of being wrong is throwing away a real answer, which is
    # the expensive direction for a directory.
    #
    # So the words must ALSO look like the transcription hint coming back —
    # every distinctive word drawn from the vocabulary we handed the
    # transcriber. Both fabrications on call-20260819-2006 clear that bar
    # ("Mayo", "appointment"); a quietly-spoken real branch name does not.
    #
    # A hallucinated "Yes." on true silence now survives. That is the trade:
    # it enters the transcript but corrupts nothing, because every location
    # still has to clear grounding. Dropping a real answer corrupts the result.
    # ── ...AND ONE SIGNAL IS ENOUGH WHEN IT IS SILENCE, NOT QUIET ───────────
    # The two-signal rule above has a hole the vocabulary test cannot cover: a
    # fabrication in ordinary English. Three are now confirmed against the
    # Twilio caller channel, and _reads_as_hint_vocabulary returns False for
    # every one of them, because none quotes the hint:
    #
    #   "Hi, I need to schedule an appointment for my annual check-up."
    #   "Hello,"
    #
    # The paragraph above says such a turn "corrupts nothing, because every
    # location still has to clear grounding". That was measured against the
    # SAVE and it is true of the save. It is not true of the CALL. On
    # call-20260820-1230 the phantom "Hello," drew a reply out of OpenAI's VAD,
    # that reply was queued on top of audio still playing, and the callee got
    # 7.35 unbroken seconds during which they said "Hello?", "campus",
    # "Hello," into a line that never paused. A fabricated turn does not need
    # to reach the directory to cost the call.
    #
    # Why this may act on audio alone when the rule above may not: it is a
    # different threshold answering a different question. _audio_carried_nothing
    # asks "faint for this caller", and real speech can be faint.
    # _audio_was_silent asks "was there any signal", and real speech cannot be
    # digital silence. See _SILENT_AUDIO_RMS for the 15x margin.
    #
    # This is only trustworthy because the measurement was fixed and CHECKED:
    # _listen_start_bytes landed 2026-08-20, and on the next live call every
    # caller turn measured 0.097-0.188 against a Twilio channel of 0.079-0.240,
    # with none at the floor. Before that fix this branch would have fired on
    # real speech constantly, and its own input would have been the fabrication.
    _rms_now = sess._pending_utterance_rms
    _silent = text and _audio_was_silent(_rms_now)
    if (text and not _silent
            and _audio_carried_nothing(_rms_now, _caller_speech_level(sess))
            and _reads_as_hint_vocabulary(text, _hint)):
        sess.suppressed_echoes.append(
            {"kind": "hint vocabulary on silent audio", "raw": text,
             "audio_rms": _rms_now})
        print(f"[Realtime] 🚱 UNEVIDENCED TURN dropped: {text[:52]!r} "
              f"— audio carried nothing (rms={_rms_now})", flush=True)
        _sup = await _suppress_reply_to(sess, oai_ws, text)
        print(f"[Realtime]   ^ reply to it: {_sup}", flush=True)
        sess.take_utterance_rms()
        if not sess._low_audio_warned:
            sess._low_audio_warned = True
            await oai_ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": (
                             "(system: nothing audible came through just then. "
                             "Do not respond to anything you think you heard. "
                             "Ask them to say it again.)")}]},
            }))
        return

    if _silent:
        sess.suppressed_echoes.append(
            {"kind": "transcript on digital silence", "raw": text,
             "audio_rms": _rms_now})
        sess.fabricated_turns.append(text)
        print(f"[Realtime] 🚱 TRANSCRIPT ON SILENCE dropped: {text[:52]!r} "
              f"— the line carried no signal at all (rms={_rms_now:.6f}, "
              f"floor {_SILENT_AUDIO_RMS})", flush=True)
        _sup = await _suppress_reply_to(sess, oai_ws, text)
        print(f"[Realtime]   ^ reply to it: {_sup}", flush=True)
        sess.take_utterance_rms()
        # Always nudge, not once per call like the faint-line warning. That one
        # is advice about the LINE and repeating it is nagging; this one exists
        # to stop the model answering a specific phantom, and there can be more
        # than one phantom on a call. Suppressing the second would leave exactly
        # the failure this branch was written for.
        await oai_ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": (
                         "(system: the line was silent just then — whatever "
                         "you think you just heard was not said. Do not answer "
                         "it and do not treat it as a reply. Stay quiet and "
                         "wait for them.)")}]},
        }))
        return

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
        # Same shape as the identity nudge below, and for the same reason: a
        # predictable, high-frequency question that the prompt's general rule
        # ("Answer EVERY one of them") failed to cover on two calls running.
        # At a medical office "is this about a patient" decides whether they
        # pull a record or route to clinical staff — it cannot be left to be
        # inferred from "listing check".
        if (_asks_about_patient(text) and not sess.done
                and not sess._patient_nudged):
            sess._patient_nudged = True
            print(f"[Realtime] Caller asked if this concerns a patient — "
                  f"telling the agent to say NO explicitly", flush=True)
            await oai_ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": (
                             # The word this directive exists to suppress used
                             # to be IN it: "answering only the 'urgent' half"
                             # arrived in context immediately before the model
                             # spoke, priming the exact reflex it was written
                             # to correct. On call-20260821-1952 the caller
                             # asked only about a patient and got both answers
                             # stapled together — "No, nothing urgent — it's
                             # just about the listing. No, no patient is
                             # involved here." Two nos, two answers, one
                             # question. Says what to answer now, and nothing
                             # about what not to.
                             "(system: they asked whether this is about a "
                             "patient. Say plainly that it is NOT — no patient "
                             "is involved — before anything else. At a medical "
                             "office that question decides how they handle the "
                             "call.)")}]},
            }))

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
            # Stand the watchdog down for the whole hold, not just this turn.
            sess._hold_until = time.time() + _HOLD_GRACE_S
            sess.reset_ask_budget("caller is going to check")
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
    if _barge_in_pending or sess._suppressed_response:
        # This transcript was cancelled — never fully heard, skip it.
        # _suppressed_response is the same situation reached from the
        # other side: we cancelled because the transcript that PROVOKED
        # this response was rejected. No audio was sent either way, so
        # letting it become a turn would put words in the transcript the
        # caller never heard and hand them to the guards as evidence.
        _barge_in_pending = False
        sess._suppressed_response = False
        _agent_text_buf = ""
        return _agent_text_buf, _barge_in_pending
    # A second spoken item in the same response, whose audio was withheld in
    # the delta handler. The caller never heard it, so it is not a turn: the
    # guards must not react to it, the metrics must not count it, and the
    # transcript must not claim it was said. Kept out of the way in
    # dropped_second_items so the artifact still shows what was suppressed.
    _item = msg.get("item_id") or ""
    if _item and _item in sess._muted_items:
        _dropped = (msg.get("transcript") or _agent_text_buf).strip()
        if _dropped:
            sess.dropped_second_items.append(_dropped)
            print(f"[Realtime]   ^ it would have said: {_dropped!r}", flush=True)
            # Muting was forced; losing the content was not. If the half
            # that reached the caller did not carry this, it is owed.
            _spoken = next((t.text for t in reversed(sess.turns)
                            if t.role == "agent" and t.text), "")
            if not sess.done and _drop_lost_substance(_spoken, _dropped):
                sess._owed_substance = _dropped
                print(f"[Realtime]   ^ that was the substance of the turn — "
                      f"owed to the caller, will be said next", flush=True)
        return "", _barge_in_pending
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
        #
        # AND EXEMPT AN ANSWER TO A DIRECT WHO QUESTION. The docstring argued
        # this guard could not key off "did they ask who I am", because the
        # case it was built for had a MIS-TRANSCRIPTION ("Hi, Ms. Mage") that
        # the model read as an identity question. That reasoning held while
        # _IDENTITY_ASK could not see the commonest phrasing; it does not hold
        # now that it can.
        #
        # call-20260820-1440: the caller asked "Sorry, who's calling again?"
        # and the agent answered "Oh, sorry Varun — I'm David, calling on
        # behalf of Definitive Healthcare." That is the correct answer, and the
        # prompt's own EXCEPTION requires it — identity facts get repeated
        # every time they are asked. Flagging it told the model to stop doing
        # the one thing it had just done right.
        #
        # Only the turn IMMEDIATELY BEFORE counts. An identity question four
        # turns back does not license re-delivering the greeting now, which is
        # the failure this guard exists for.
        _prev_caller = next((t.text for t in reversed(sess.turns)
                             if t.role == "caller" and t.text
                             and t.text != "[...]"), "")
        _answered_who = bool(_IDENTITY_ASK.search(_prev_caller))
        _agent_turns = sum(1 for t in sess.turns if t.role == "agent")
        if (_agent_turns >= 1 and not sess._reintro_nudged
                and not sess.done and not _answered_who
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

        # ── Claimed it was done, and saved nothing ──────────────────────────
        # call-20260819-1619: the caller said "It's actually at 100 Main
        # Street" — a street address, a perfectly valid location — and the
        # agent replied "Thanks for the address, that's all I needed" and
        # stopped. save_branch was never called. resolved=False, branch=None.
        # A resolvable call, answered, and thrown away.
        #
        # DEFERRED to the watchdog rather than decided here. The tool call for
        # this same response has not arrived yet, so "it was never called" is
        # not knowable at this point — firing now also fires on the rejected
        # save, which is a different failure with its own correction.
        if (_claims_saved(text) and not sess.memory.get("branch")
                and not sess.memory.get("escalated") and not sess.done):
            sess._claimed_done_at = time.time()

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
        if _is_objective_ask(text, sess) and not sess.done:
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
            # WHICH ASKS SPEND THE BUDGET, INVERTED 2026-08-24.
            #
            # This used to increment on every ask the caller HAD answered, and
            # give up at four. Four answered asks is a call going well: it is
            # the new script's happy path exactly — branch, accepting new
            # patients, referral requirement, what it depends on — and that is
            # per doctor, with several doctors per call now in scope. The
            # mechanism had already ended a live call where the caller gave a
            # complete correct answer twice (call-20260821-1931).
            #
            # So the budget counts the asks they did NOT answer, consecutively,
            # and resets on any reply. Nothing about it needs to know how many
            # doctors or fields a call covers, because progress is what clears
            # it — see reset_ask_budget and _asks_without_progress.
            #
            # COUPLED TO _is_filler_reply. "Did they answer" is now judged
            # against what was ASKED: a bare "Yes." answers "are you accepting
            # new patients?" and does not answer "which branch?". Before that
            # change this inversion would have been strictly worse than what it
            # replaced — every "Yes." would have read as silence, and the
            # counter that ends the call is the one reading it.
            _first_ask = sess._last_ask_turn_idx < 0
            _answered = _first_ask or _caller_answered_since(
                sess, sess._last_ask_turn_idx)
            # They replied, but with a question rather than an answer. That is
            # a front desk deciding whether to engage, not a caller refusing,
            # and it must not spend either counter — see _caller_is_vetting.
            # Bounded, or a caller who only ever asks questions would keep the
            # call alive indefinitely.
            _vetted = (not _first_ask
                       and sess._vetting_reasks < _MAX_VETTING_REASKS
                       and _caller_vetted_since(sess, sess._last_ask_turn_idx))
            if _vetted:
                sess._vetting_reasks += 1
                sess._unanswered_asks = 0
                print(f"[Realtime] They asked a question back rather than "
                      f"answering ({sess._vetting_reasks}/"
                      f"{_MAX_VETTING_REASKS}) — spending nothing "
                      f"(unanswered={sess._unanswered_asks}/"
                      f"{settings.realtime_max_unanswered_asks}, "
                      f"no-progress={sess._asks_without_progress}/"
                      f"{settings.realtime_max_asks_without_progress})",
                      flush=True)
            elif _answered:
                # An answered ask costs the budget nothing. It still counts
                # toward the no-progress ceiling: engaging is not the same as
                # supplying, and that ceiling is the only thing left that ends
                # a call where they talk and never tell.
                sess._unanswered_asks = 0
                sess._asks_without_progress += 1
            else:
                sess._unanswered_asks += 1
                sess._asks_without_progress += 1
                print(f"[Realtime] Ask into silence "
                      f"({sess._unanswered_asks}/"
                      f"{settings.realtime_max_unanswered_asks} unanswered, "
                      f"{sess._asks_without_progress}/"
                      f"{settings.realtime_max_asks_without_progress} "
                      f"without progress)", flush=True)
            sess._last_ask_turn_idx = len(sess.turns)
            # The SAME WORDS, again.
            #
            # call-20260819-2121 ended every one of its four turns with "which
            # branch Dr. Okafor works out of" — greeting, then stapled onto the
            # answer to each of three screening questions. Nothing caught it:
            # _MIN_REASK_GAP_S measures speed and the gaps were eleven seconds,
            # the ask budget counts asks and not their wording, and
            # repeated_sentences is computed after the call is over.
            #
            # Re-asking is sometimes right. Re-asking in the identical clause
            # is never right — it is the single clearest tell that nobody is
            # listening on this end, because a person who has to ask twice
            # rephrases without thinking about it.
            _ask_clauses = {_norm_clause(c) for s in _sentences(text)
                            for c in _clauses(s) if _is_location_ask(c)}
            _repeat_phrasing = _ask_clauses & sess._ask_phrasings
            sess._ask_phrasings |= _ask_clauses
            if (_repeat_phrasing and not sess._verbatim_ask_nudged
                    and not sess._give_up_sent):
                sess._verbatim_ask_nudged = True
                print(f"[Realtime] 🗣  Asked in the SAME WORDS again: "
                      f"{sorted(_repeat_phrasing)[0][:60]!r} — telling the "
                      f"agent to stop stapling it on", flush=True)
                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text", "text": (
                                 "(system: you have now asked for the branch "
                                 "in those exact words more than once, and "
                                 "you are attaching it to the end of every "
                                 "reply. They heard it the first time. When "
                                 "they ask you something, answer it and STOP "
                                 "— no question on the end. Ask again only "
                                 "once they have answered you and gone quiet, "
                                 "and when you do, use different words.)")}]},
                }))
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
            # TWO TRIGGERS, TWO TRUE REASONS. The old one had a single
            # condition and a single escalate reason — "caller engaged but
            # never provided a location" — which was already the wrong
            # sentence for a caller who had said nothing at all, and
            # _discarded_location exists because a false reason in the record
            # is indistinguishable from a true one to whoever reads it.
            _out_of_budget = (sess._unanswered_asks
                              >= settings.realtime_max_unanswered_asks)
            _no_progress = (sess._asks_without_progress
                            >= settings.realtime_max_asks_without_progress)
            if (_out_of_budget or _no_progress) and not sess._give_up_sent:
                sess._give_up_sent = True
                sess._give_up_at_turn = len(sess.turns)
                sess._give_up_trigger = ("unanswered" if _out_of_budget
                                         else "no_progress")
                if _out_of_budget:
                    print(f"[Realtime] {sess._unanswered_asks} asks with no "
                          f"reply from the caller — telling the agent to stop "
                          f"and escalate", flush=True)
                else:
                    print(f"[Realtime] {sess._asks_without_progress} asks with "
                          f"nothing collected — telling the agent to stop and "
                          f"escalate", flush=True)
                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{
                            "type": "input_text",
                            "text": give_up_directive(
                                sess, sess._give_up_trigger),
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
    # The FIRST assistant item in this response that produced audio. A phone
    # turn is one spoken item; a second one is the agent talking to itself.
    # Distinct from _current_item_id, which follows what is playing and is what
    # a barge-in truncates.
    _spoken_item_id: Optional[str] = None
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
                sess._caller_speaking_since = time.time()
                sess._backchannel_done_this_utterance = False
                _speech_start_pcm_pos = len(sess._caller_oai_pcm)  # fallback only
                # OpenAI's own offset into the buffer we feed it. The chunk
                # position above is where the EVENT ARRIVED, which is up to a
                # second late from India — see _utterance_slice.
                sess._speech_start_ms = msg.get("audio_start_ms")
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
                    sess._caller_speaking_since = None
                    # Start the clock on the reply — at the moment the CALLER
                    # STOPPED TALKING, not the moment we heard about it.
                    #
                    # This event arrives after the detector has made up its
                    # mind, and how long that takes is the difference between
                    # the detectors, not a constant. Anchoring here and adding
                    # a per-detector guess for the rest was wrong twice: 0.7s
                    # charged to semantic_vad (which sends no silence timer)
                    # inflated every gap on call-20260821-1856, and 0.0s
                    # charged to it on call-20260821-1931 reported 0.81s while
                    # the Twilio recording measures 3.67s — the instrument
                    # moved the opposite way to the thing it measures.
                    #
                    # audio_end_ms says when the caller actually stopped,
                    # indexed into the buffer we control. The mirror makes that
                    # index exact, so the lag can be computed instead of
                    # assumed: bytes buffered since that point, over the wire
                    # rate. No detector-specific term survives.
                    _lag_s = 0.0
                    _end_ms = msg.get("audio_end_ms")
                    if isinstance(_end_ms, (int, float)):
                        _bps = _wire_bytes_per_ms() * 1000.0
                        _end_byte = sess._listen_start_bytes + _end_ms * _wire_bytes_per_ms()
                        _have = sum(len(c) for c in sess._caller_oai_pcm)
                        if _bps > 0:
                            # Clamped: a negative lag would mean the caller
                            # stopped after audio we have not buffered yet, and
                            # a huge one means the buffers disagree — neither is
                            # a latency, and both must fall back to "now".
                            _lag_s = max(0.0, min((_have - _end_byte) / _bps, 10.0))
                    sess._caller_stopped_at = time.monotonic() - _lag_s
                    sess._last_stop_lag_s = _lag_s
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
                    utterance = _utterance_slice(
                        sess, sess._speech_start_ms, msg.get("audio_end_ms"),
                        _speech_start_pcm_pos)
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
                sess._response_audio_started = False
                sess._response_created_at = time.monotonic()

            # ── Audio → Twilio ─────────────────────────────────────────────
            # gpt-realtime-2 uses response.output_audio.delta (not response.audio.delta)
            elif event_type == "response.output_audio.delta":
                delta = msg.get("delta", "")

                # ── One spoken item per response ───────────────────────────
                # call-20260819-2044: ONE response, ONE response.done, 2.85s of
                # audio, and two `response.output_audio_transcript.done` events
                # both reading "Sure, no rush." The callee heard it twice in a
                # single breath. Same shape as the "of course, of course" turn
                # the day before.
                #
                # Nothing in this codebase asked for it twice — the hold branch
                # only stands the watchdog down, and a second response.create
                # would have produced a second response.done. The model emitted
                # two assistant items in one response and both were spoken.
                #
                # Every guard downstream of this is powerless here: the
                # transcript arrives after the audio, so 🔁 REPEATED SENTENCE
                # can only narrate what the callee already heard. The audio
                # deltas are where it can still be stopped, and they are also
                # where the two items are distinguishable — item_id changes.
                #
                # Dropping rather than cancelling is deliberate. response.cancel
                # is protocol state that races with response.done, and the
                # remaining tokens are a fraction of a second either way. Not
                # forwarding costs nothing and cannot desynchronise anything.
                #
                # This does not fire on a tool call followed by speech: a
                # function_call item emits no audio deltas, so the first item
                # seen here is the spoken one.
                _delta_item = msg.get("item_id") or ""
                if delta and _delta_item:
                    if _spoken_item_id is None:
                        _spoken_item_id = _delta_item
                    elif _delta_item != _spoken_item_id:
                        if _delta_item not in sess._muted_items:
                            sess._muted_items.add(_delta_item)
                            print(f"[Realtime] 🔇 second spoken item in one "
                                  f"response — dropping it before it reaches "
                                  f"the caller", flush=True)
                        delta = ""

                if delta:
                    sess.agent_speaking  = True
                    sess._response_active = True
                    sess._response_audio_started = True
                    _response_had_audio  = True
                if delta and sess.stream_sid:
                    try:
                        raw_pcm = base64.b64decode(delta)
                        _samples_this_response += _wire_samples(raw_pcm)
                        if _first_delta_sent_at is None:
                            # STACKED REPLY. _create_response already refuses to
                            # start one while audio is still playing out — but
                            # it only sees the responses WE create, and the
                            # ordinary ones are created by OpenAI's VAD, which
                            # never passes through it. So the gate has been in
                            # place and unreachable for the common case.
                            #
                            # call-20260820-1230, blocks 5 and 6: audio sent at
                            # 71.90s ran to 76.95s, the next reply began sending
                            # at 76.30s. Twilio does not mix, it queues — so the
                            # callee heard 7.35 unbroken seconds with nowhere to
                            # speak, and said "Hello?", "campus", "Hello," while
                            # it ran. Same shape again at blocks 7/8.
                            #
                            # Give them the gap a person would leave. Closing is
                            # exempt for the reason _create_response exempts it:
                            # a goodbye that waits for the queue is a goodbye
                            # that arrives after the line is being torn down.
                            _still_playing = sess._playback_ends_at - time.monotonic()
                            if _still_playing > 0 and not sess.done:
                                sess._stacked_replies += 1
                                print(f"[Realtime] 🫁 stacked reply — "
                                      f"{_still_playing:.2f}s still playing out; "
                                      f"inserting a {_STACK_BREATH_S:.1f}s gap so "
                                      f"they can speak", flush=True)
                                await _send_breath(twilio_ws, sess, _STACK_BREATH_S)
                                # The real audio now begins after the queue AND
                                # the gap, so _playback_ends_at (derived from
                                # this) stays honest. Using time.monotonic()
                                # here would under-report the queue by exactly
                                # the amount that caused the overlap.
                                _first_delta_sent_at = (sess._playback_ends_at
                                                        + _STACK_BREATH_S)
                            else:
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
                            # The figure the callee actually experiences. The
                            # line above starts its clock at OUR request, which
                            # is after /answer, after the media WebSocket, and
                            # after Twilio's stream-start handshake — so it can
                            # read 1.08s on a call that felt like ten seconds
                            # of nothing, and did.
                            if sess._answered_at is not None:
                                _pu = time.monotonic() - sess._answered_at
                                sess.pickup_to_greeting_s = round(_pu, 2)
                                _setup = max(0.0, _pu - gap)
                                print(f"[Realtime] 📞 Greeting {_pu:.2f}s after "
                                      f"they picked up ({_setup:.2f}s Twilio "
                                      f"setup + {gap:.2f}s to first audio)",
                                      flush=True)
                            if gap > 2.0:
                                print(f"[Realtime]   ^ that is dead air on the "
                                      f"callee's end before the greeting starts",
                                      flush=True)
                        # Every other turn. The wait the caller actually feels
                        # starts when they stop talking, not when OpenAI's VAD
                        # notices: the silence window elapses first and is part
                        # of the gap, so it is added back rather than hidden.
                        # Splitting it out is the point — the window is a knob
                        # we own (realtime_silence_ms) and the rest is inference
                        # plus the round trip to a US datacentre, which is not.
                        elif sess._caller_stopped_at is not None:
                            # _caller_stopped_at is backdated to when the
                            # caller actually stopped, so this IS the felt
                            # gap. Nothing is added to it.
                            _felt = time.monotonic() - sess._caller_stopped_at
                            sess._caller_stopped_at = None
                            _vad = sess._last_stop_lag_s
                            sess.detector_lags.append(_vad)
                            _after_vad = max(0.0, _felt - _vad)
                            sess.note_reply_latency(_felt)
                            print(f"[Realtime] Reply {_felt:.2f}s after the "
                                  f"caller stopped ({_vad:.2f}s detector + "
                                  f"{_after_vad:.2f}s think/round-trip)",
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
                # WHY it failed, which was being thrown away.
                #
                # call-20260819-2216 had SEVEN `[failed]` responses with
                # in_text=0, and four stretches of 8-11 seconds where nobody on
                # the call made a sound — the failures and the dead air line up
                # one for one. Twilio's own recording showed every agent block
                # reaching the line within 0.4s of generation, so the transport
                # was never the problem, and two rounds of diagnosis went into
                # guessing at a reason the event carried all along.
                #
                # `status_details` holds {type, reason} and, for failures, an
                # {error: {type, code, message}}. Printed, not logged, so it
                # lands in the call log next to the response it explains.
                _sd = ((msg.get("response") or {}).get("status_details") or {})
                if _resp_status in ("failed", "incomplete") and _sd:
                    _sd_err = _sd.get("error") or {}
                    _why_failed = (_sd_err.get("message")
                                   or _sd_err.get("code")
                                   or _sd.get("reason") or "no reason given")
                    print(f"[Realtime] ⚠️  response {_resp_status}: "
                          f"{_why_failed}", flush=True)
                    sess.response_failures.append(
                        {"status": _resp_status,
                         "reason": str(_why_failed)[:200]})
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
                    # Kept on the session so _create_response can see it. We
                    # hand Twilio audio as fast as OpenAI produces it, and
                    # OpenAI produces far faster than realtime — a 6.25s reply
                    # arrives in about a second. Everything after that sits in
                    # Twilio's queue. Creating another response before the
                    # queue drains does not talk OVER the caller; it appends,
                    # so they hear one unbroken monologue with no gap to speak
                    # into. On call-20260819-2006 that came out as three
                    # identical questions in a single 50-word turn, and she
                    # hung up.
                    sess._playback_ends_at = _playback_ends_at
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
                _spoken_item_id = None
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
                    # out_text is printed alongside out_audio because the token
                    # CAP counts both, and only out_audio was ever shown. When
                    # call-20260820-1230 came back "incomplete:
                    # max_output_tokens" the line read out_audio=151 against a
                    # cap of 400, which looks like it had plenty of room and
                    # made the truncation unexplainable from the log alone.
                    # The missing half was the text.
                    _ot_audio = details_out.get("audio_tokens", 0)
                    _ot_text  = details_out.get("text_tokens", 0)
                    print(f"[Realtime] usage: in_text={details_in.get('text_tokens', 0)} "
                          f"(cached {c_text})  in_audio={details_in.get('audio_tokens', 0)} "
                          f"(cached {c_audio})  out_audio={_ot_audio}  out_text={_ot_text}"
                          f"  (cap {settings.realtime_max_response_tokens})"
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
                    # Everything buffered up to here was never sent to OpenAI,
                    # so its ms timestamps count from THIS point, not from
                    # stream start. Record where that is before any caller turn
                    # can exist — every utterance slice is measured from it.
                    sess._listen_start_bytes = sum(len(c) for c in sess._caller_oai_pcm)
                    _lead_s = sess._listen_start_bytes / max(_wire_bytes_per_ms(), 1e-9) / 1000
                    print(f"[Realtime] Greeting done — now listening to caller "
                          f"(OpenAI's audio clock starts {_lead_s:.2f}s into ours)",
                          flush=True)
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
                elif (code == "conversation_already_has_active_response"
                      and sess.done):
                    # THE SAME CONDITION _create_response ALREADY HANDLES, seen
                    # from the server instead of from our own flag.
                    #
                    # `_response_active` is a lagging indicator by construction:
                    # OpenAI's server VAD can create a response the instant the
                    # caller speaks, and we do not know until that
                    # `response.created` is read off the socket. The goodbye
                    # retry is a separate task precisely so the pump keeps
                    # reading (see _goodbye_retry_at), which closed the version
                    # of this race caused by sleeping inside a handler — but it
                    # cannot close the gap between the server deciding and us
                    # hearing, and no client-side flag can.
                    #
                    # On call-20260824-2113 the caller said "Yes, I'm there"
                    # in that gap. The retry lost the race, the server refused
                    # it, and the call closed correctly anyway a second later
                    # ("Take care.") — because a response being in flight is
                    # exactly the case where the retry was unnecessary.
                    #
                    # So it is reported as what it is. Printing API ERROR for a
                    # benign, expected, already-handled race is how a log
                    # teaches people to ignore it.
                    print("[Realtime] Goodbye retry raced OpenAI's own "
                          "response and lost — the line is not silent, "
                          "nothing to do", flush=True)
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
