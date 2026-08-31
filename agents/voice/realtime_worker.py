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
from __future__ import annotations as annotations

import asyncio
import base64
import json
import logging
import time
import traceback
from datetime import datetime
from typing import NamedTuple as NamedTuple, Optional

import numpy as np
import websockets
import websockets.exceptions
from fastapi import WebSocket

from core.config import settings, persona_for_voice
from core.models import Doctor, DoctorStatus as DoctorStatus, Source as Source, TranscriptTurn as TranscriptTurn
from core.memory import CallMemory as CallMemory
from agents.voice.objectives import (
    ACCEPTING_ASK,
    IDENTITY_ASK,
    REFERRAL_ASK,
    SCHEDULING_ASK,
    Outcome as Outcome,
    expected_answers as expected_answers,
    sentences as _sentences,
)
from agents.voice.templates import get_template
from agents.voice.tools import run_tool as run_tool, TOOL_SCHEMAS
from core.audio_utils import resample, _mulaw_decode, _mulaw_encode as _mulaw_encode
# The evidence guards, in their own module now. See evidence.py for why the
# line falls where it does. Split in two on purpose:
#
#   the first group this file CALLS;
#   the second it only passes through, so that existing callers — and every
#   `rw._thing` in the suite — keep working. Those use the redundant
#   `X as X` form, which is how a checker is told a re-export is deliberate.
#   Written plainly they were reported as 25 unused imports, and a standing
#   hint storm is how a real one gets scrolled past.
# AT MODULE LEVEL, and it matters: outbound_audio binds scipy on import, and a
# deferred import would bind it inside whatever mock.patch.dict("sys.modules")
# context happened to be active — after which scipy is evicted and its C
# extensions refuse to reload for the life of the process. See that module.
from agents.voice.outbound_audio import (
    DISABLED_REASON as OUTBOUND_UNAVAILABLE,
)
# Re-exported for the suite, which addresses both as rw._stage_row and
# rw._fmt_stages. Plain form, not `as X`: the worker calls both itself.
from agents.voice.latency import _stage_row as _stage_row, _fmt_stages as _fmt_stages

# Re-exported: the suite and the rest of this module address these as
# rw.<name>. See agents/voice/audio.py for why they moved.
from agents.voice.lifecycle import (
    _ResponseDone as _ResponseDone,
    _end_speaking_gate as _end_speaking_gate,
    _handle_response_done as _handle_response_done,
)
from agents.voice.audio import (
    _TWILIO_SR,
    _OAI_SR,
    _passthrough_enabled,
    _outbound_conditioned as _outbound_conditioned,
    _effective_output_format,
    _AGENT_WIRE_SR as _AGENT_WIRE_SR,
    _agent_wire_sample_rate,
    _agent_wire_samples as _agent_wire_samples,
    _wire_to_pcm16,
    _wire_samples as _wire_samples,
    _wire_bytes_per_ms,
    _utterance_slice,
    _STACK_BREATH_S as _STACK_BREATH_S,
    _TWILIO_SILENCE_FRAME as _TWILIO_SILENCE_FRAME,
    _send_breath as _send_breath,
    _loudest_window_rms,
    _audio_carried_nothing as _audio_carried_nothing,
    _SILENT_AUDIO_RMS as _SILENT_AUDIO_RMS,
    _audio_was_silent as _audio_was_silent,
    _AudioDelta,
    _drop_held_items as _drop_held_items,
    _flush_held_item as _flush_held_item,
    _MAX_HELD_ITEM_CHUNKS as _MAX_HELD_ITEM_CHUNKS,
    _handle_audio_delta,
)
# Re-exported: the suite and the rest of this module address these as
# rw.<name>. See agents/voice/grounding.py for why the set is a closure.
from agents.voice.grounding import (
    _IDENTITY_ASK as _IDENTITY_ASK,
    _CLAIMS_SAVED as _CLAIMS_SAVED,
    _claims_saved as _claims_saved,
    _spoken_farewell as _spoken_farewell,
    _HOLD_REQUEST as _HOLD_REQUEST,
    _CALLER_WILL_ACT as _CALLER_WILL_ACT,
    is_hold_request as is_hold_request,
    _FACTUAL_ESCALATIONS as _FACTUAL_ESCALATIONS,
    _ungrounded_escalation as _ungrounded_escalation,
    _candidate_location as _candidate_location,
    _CALL_SHAPE_EXITS as _CALL_SHAPE_EXITS,
    _discarded_location as _discarded_location,
    _create_response,
    _STREET_ADDRESS as _STREET_ADDRESS,
    _address_offered as _address_offered,
    _address_dropped as _address_dropped,
    _SELF_ID as _SELF_ID,
    _SELF_ID_WEAK as _SELF_ID_WEAK,
    _ORG_WORD as _ORG_WORD,
    hospital_mismatch as hospital_mismatch,
    _strip_ungrounded_detail as _strip_ungrounded_detail,
    _CHOICE_SAVE_TOOLS,
    _RETIRED_VOCAB_TEXT as _RETIRED_VOCAB_TEXT,
    _hint_vocabulary as _hint_vocabulary,
    _is_bare_hint_word as _is_bare_hint_word,
    _MAX_SAVE_REJECTIONS as _MAX_SAVE_REJECTIONS,
    _ToolOutcome as _ToolOutcome,
    _handle_tool_call,
    _resolve_deferred_save as _resolve_deferred_save,
    _STREET_SUFFIX as _STREET_SUFFIX,
)
# Re-exported: the suite and the rest of this module address these as
# rw.<name>. See agents/voice/turns.py for what the closure covers.
from agents.voice.turns import (
    _ACK_WORDS as _ACK_WORDS,
    _ACK_REPLY as _ACK_REPLY,
    _AFFIRM_REPLY as _AFFIRM_REPLY,
    _HAS_AFFIRM as _HAS_AFFIRM,
    _is_filler_reply as _is_filler_reply,
    _pending_expectation as _pending_expectation,
    _caller_answered_since as _caller_answered_since,
    _MAX_VETTING_REASKS as _MAX_VETTING_REASKS,
    _caller_vetted_since as _caller_vetted_since,
    _REPAIR_WINDOW_S as _REPAIR_WINDOW_S,
    _CUT_SHORT_MS as _CUT_SHORT_MS,
    _BACKCHANNEL_AFTER_S as _BACKCHANNEL_AFTER_S,
    _BACKCHANNEL_COOLDOWN_S as _BACKCHANNEL_COOLDOWN_S,
    _BACKCHANNEL_ECHO_MARGIN_S as _BACKCHANNEL_ECHO_MARGIN_S,
    _MIN_REASK_GAP_S as _MIN_REASK_GAP_S,
    _is_reintroduction as _is_reintroduction,
    _claims_employment as _claims_employment,
    _is_objective_ask as _is_objective_ask,
    GIVE_UP_REASONS as GIVE_UP_REASONS,
    _field_vocabulary as _field_vocabulary,
    _field_already_answered as _field_already_answered,
    _volunteered_fields as _volunteered_fields,
    give_up_directive as give_up_directive,
    _PATIENT_ASK as _PATIENT_ASK,
    _asks_about_patient as _asks_about_patient,
    _content_words as _content_words,
    caller_repeated_answer as caller_repeated_answer,
    _HINT_RUN_WORDS as _HINT_RUN_WORDS,
    _HINT_HEADINGS as _HINT_HEADINGS,
    _RETIRED_HINT_TEXT as _RETIRED_HINT_TEXT,
    _FABRICATION_VOCAB as _FABRICATION_VOCAB,
    _hint_proper_nouns as _hint_proper_nouns,
    _reads_as_hint_vocabulary as _reads_as_hint_vocabulary,
    _strip_hint_run as _strip_hint_run,
    _SILENCE_PROMPT_FIRST as _SILENCE_PROMPT_FIRST,
    _SILENCE_PROMPT_AFTER as _SILENCE_PROMPT_AFTER,
    _HOLD_GRACE_S as _HOLD_GRACE_S,
    _MAX_SILENCE_PROMPTS as _MAX_SILENCE_PROMPTS,
    _silence_watchdog,
    _suppress_reply_to as _suppress_reply_to,
    _handle_caller_transcript,
    _handle_agent_transcript,
)
# Re-exported: the suite addresses these as rw.<name>.
from agents.voice.metrics import (
    _double_ask as _double_ask,
    conversation_metrics as conversation_metrics,
)
# Re-exported: the suite addresses these as rw.<name>.
from agents.voice.session import (
    _PROJECT_ROOT as _PROJECT_ROOT,
    audio_dir as audio_dir,
    json_dir as json_dir,
    _MASTER_LOCK as _MASTER_LOCK,
    _DOCTORS_LOCK as _DOCTORS_LOCK,
    _ask_budget_outcome as _ask_budget_outcome,
    RealtimeSession,
)
from agents.voice.evidence import (
    _LOCATION_ANCHORS as _LOCATION_ANCHORS,
    _LOW_AUDIO_RMS,
    _NON_PLACE as _NON_PLACE,
    _ORG_STOPWORDS as _ORG_STOPWORDS,
    _QUIET_FRACTION as _QUIET_FRACTION,
    _UNGROUNDED_STOPWORDS as _UNGROUNDED_STOPWORDS,
    _transcript_pending as _transcript_pending,
    _caller_ends_call as _caller_ends_call,
    _caller_is_vetting as _caller_is_vetting,
    _caller_speech_level as _caller_speech_level,
    _distinctive as _distinctive,
    _drop_lost_substance as _drop_lost_substance,
    _grounding_verdict as _grounding_verdict,
    _invites_continuation as _invites_continuation,
    _is_ask_for as _is_ask_for,
    _is_hint_echo as _is_hint_echo,
    _is_location_ask as _is_location_ask,
    _meaning_class as _meaning_class,
    _name_mismatch,
    _owed_key as _owed_key,
    _owed_refusal as _owed_refusal,
    _revisit_grounding as _revisit_grounding,
    _rode_along as _rode_along,
    _spell_out as _spell_out,
    _spelled_out as _spelled_out,
    _surnames_named,
    _ungrounded_choice,
    _ungrounded_detail as _ungrounded_detail,
    _ungrounded_terms as _ungrounded_terms,
    _wrong_doctor_named,
)
from agents.voice.evidence import (  # re-exported for callers
    _ACK_TAKES_VALUE as _ACK_TAKES_VALUE,
    _CONFIRMS_VALUE as _CONFIRMS_VALUE,
    _DETAIL_FUNCTION_WORDS as _DETAIL_FUNCTION_WORDS,
    _INVITATION as _INVITATION,
    _MAX_OWED_PER_CALL as _MAX_OWED_PER_CALL,
    _MAX_OWED_PER_TEXT as _MAX_OWED_PER_TEXT,
    _MEANING_CLASSES as _MEANING_CLASSES,
    _MIN_TURNS_FOR_ADAPTIVE as _MIN_TURNS_FOR_ADAPTIVE,
    _NAMED_DOCTOR as _NAMED_DOCTOR,
    _NOT_AN_ASK as _NOT_AN_ASK,
    _NUMBER_WORD_VALUE as _NUMBER_WORD_VALUE,
    _POSSESSIVE as _POSSESSIVE,
    _REPORTS_FAILURE as _REPORTS_FAILURE,
    _VETTING_OPENER as _VETTING_OPENER,
    _asserted_caller_text as _asserted_caller_text,
    _class_present as _class_present,
    _collapse as _collapse,
    _ever_transcribed as _ever_transcribed,
    _grounded_in as _grounded_in,
    _grounded_loosely as _grounded_loosely,
    _stem as _stem,
    _turn_asserts as _turn_asserts,
)

log = logging.getLogger(__name__)

REALTIME_URL = "wss://api.openai.com/v1/realtime?model={model}"
# Per-attempt ceiling on the OpenAI handshake. Deliberately below the
# websockets default of 10s: the callee is already on the line and every
# second here is silence they hear, so a stall must be caught early enough to
# retry inside their patience rather than after it. Measured healthy: 1.7s.
_OAI_CONNECT_TIMEOUT_S = 6.0


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
    # The watchdog's own exit, added when it stopped falling silent after
    # spending _MAX_SILENCE_PROMPTS. A trigger with no marker is a trigger no
    # absence assertion can vouch for — that is what the paired length check
    # in the suite is there to force.
    "no_response": "the line has gone quiet",
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
                       language: str = "en",
                       audio_format: str, noise_reduction: str,
                       turn_detection: str, eagerness: str,
                       voice: str, silence_ms: int = 500,
                       interrupt_response: bool = True,
                       output_format: str = "") -> dict:
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
    def _fmt(kind: str) -> dict:
        return ({"type": "audio/pcmu"} if kind == "pcmu"
                else {"type": "audio/pcm", "rate": _OAI_SR})

    fmt = _fmt(audio_format)
    # THE TWO LEGS ARE NEGOTIATED SEPARATELY. Inbound stays μ-law — the model
    # is the consumer and nothing we insert helps it — while outbound asks for
    # PCM16 so there is something left to condition. Defaults to the inbound
    # value so a caller that does not care gets the old single-format
    # behaviour, which is what check_realtime.py's probes want.
    out_fmt = _fmt(output_format or audio_format)

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

    _transcription: dict = {
        "model": transcribe_model,
    }
    # OMITTED WHEN EMPTY, like the prompt below it. Passing "" would assert a
    # language named "", and the point of an empty value is to make no claim
    # at all and let the transcriber decide. Default is "en", unchanged.
    if language:
        _transcription["language"] = language
    # OMITTED, NOT SENT EMPTY. A retired hint should leave no prompt on the
    # request at all — sending "" invites the question of whether an empty
    # prompt is still a prompt, and the whole point of retiring it is that the
    # transcriber has nothing of ours to recite.
    if transcribe_hint:
        _transcription["prompt"] = transcribe_hint

    audio_in: dict = {
        "format": fmt,
        "transcription": _transcription,
        "turn_detection": td,
    }
    if noise_reduction and noise_reduction != "off":
        audio_in["noise_reduction"] = {"type": noise_reduction}

    return {"input": audio_in, "output": {"format": out_fmt, "voice": voice}}


# ── Grounding: a saved location must be one the caller actually said ─────────

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

def _ungrounded_identity(args: dict, sess: "RealtimeSession") -> str:
    """Grounding for whether we reached the right doctor. Its own vocabulary.

    Plus the name check, which the vocabulary alone cannot do: an affirmative
    is an affirmative whoever it is about, so confirming REQUIRES that no other
    doctor was named in the same breath.
    """
    from agents.voice.objectives import IDENTITY_STATES, classify_identity

    claimed = str(args.get("identity") or "").strip().lower()
    if claimed == "confirmed":
        # THE SCAN STARTS AFTER THE SPELLING, and that is the repair actually
        # taking effect rather than merely being requested. Scanning the whole
        # call means the mangled "Dr. Riaz" from before we spelled the letters
        # is still sitting there to be found, so a caller who then hears
        # "R-E-Y-E-S" and says "yes, that's him" is refused on evidence the
        # spelling was performed to supersede — which is the loop 1437 was in.
        # Turns before the letters were about a name nobody could hear; turns
        # after them are about ours.
        #
        # Only the turns that could be the evidence — after the ask, asserted.
        for t in reversed(sess.turns[sess._name_spelled_at:]):
            if t.role != "caller" or t.text.strip() == "[...]":
                continue
            other = _wrong_doctor_named(t.text, sess)
            if other:
                return _name_mismatch(sess, other, t.text)
            if _surnames_named(t.text):
                break      # our doctor was named, and matched
    return _ungrounded_choice(args, sess, arg="identity", probe=IDENTITY_ASK,
                              classifier=classify_identity,
                              states=IDENTITY_STATES, label="identity",
                              since_at_least=sess._name_spelled_at,
                              floor_reason=("you spelled the name out to them "
                                            "and they have not answered yet"))


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
                    language=settings.realtime_transcribe_language,
                    transcribe_hint=template.transcribe_hint,
                    audio_format=settings.realtime_audio_format,
                    noise_reduction=settings.realtime_noise_reduction,
                    turn_detection=settings.realtime_turn_detection,
                    eagerness=settings.realtime_vad_eagerness,
                    voice=settings.realtime_voice,
                    silence_ms=settings.realtime_silence_ms,
                    output_format=_effective_output_format(),
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
    # ONLY EVER ASSIGNED INSIDE response.done until that block was extracted on
    # 2026-08-27, so it never needed an initial value: nothing read it before
    # the first response finished. Building the handler's state tuple reads all
    # twelve on every response.done, INCLUDING the first, and an unbound local
    # there raised UnboundLocalError straight into the loop's crash handler —
    # which caught it, printed "EVENT LOOP CRASHED", and ended the call. The
    # whole suite went red in the same shape: everything after the first
    # response.done simply never ran.
    _echo_cooldown        = 0.3     # replaced by the real figure on every response
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
                    # Anything still held is a turn they have just talked over.
                    # Playing it after the fact is worse than muting it ever
                    # was — see _drop_held_items.
                    _drop_held_items(sess, "the caller interrupted")
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
                            generated_ms = int(_samples_this_response
                                               / _agent_wire_sample_rate() * 1000)
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
                    # t0. Opened here and NOT at response.created, because the
                    # stage that turned out to carry the variance is the one
                    # BEFORE the response exists. A record started later cannot
                    # measure it. Overwritten if the caller stops again before
                    # replying — a split turn is one turn, and the last stop is
                    # the one the reply answers, matching _caller_stopped_at.
                    sess._stage = {"t0": sess._caller_stopped_at,
                                   "detector_s": round(_lag_s, 3)}
                    # OpenAI will open a response for this turn on its own.
                    sess._vad_response_due_until = time.monotonic() + 2.0
                    # Placeholder — filled in by the session's own inline
                    # transcription when conversation.item.input_audio_
                    # transcription.completed arrives. Nothing else fills it:
                    # the out-of-band whisper-1 HTTP fallback was removed to
                    # keep this path pure speech-to-speech. Placeholders that
                    # never resolve are dropped from the transcript in save().
                    sess.add_turn("caller", "[...]")
                    # WHEN the placeholder was made, so a wait can tell an
                    # utterance still being transcribed from one whose result
                    # already came back and was thrown away. See
                    # _transcript_pending.
                    sess._placeholder_at = time.monotonic()

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
                # t1. First response of this turn only: a tool turn opens a
                # SECOND response after the deferral, and stamping that one too
                # would overwrite the mark inference 1 is measured from.
                sess._vad_response_due_until = 0.0
                if sess._stage is not None and "t1" not in sess._stage:
                    sess._stage["t1"] = time.monotonic()
                sess._response_audio_started = False
                sess._response_created_at = time.monotonic()

            # ── Audio → Twilio ─────────────────────────────────────────────
            # gpt-realtime-2 uses response.output_audio.delta (not response.audio.delta)
            elif event_type == "response.output_audio.delta":
                _ad = await _handle_audio_delta(
                    msg, sess, twilio_ws, _current_response_pcm,
                    _AudioDelta(_samples_this_response, _first_delta_sent_at,
                                _current_response_start, _spoken_item_id,
                                _response_had_audio, _current_item_id))
                _samples_this_response = _ad.samples_this_response
                _first_delta_sent_at = _ad.first_delta_sent_at
                _current_response_start = _ad.current_response_start
                _spoken_item_id = _ad.spoken_item_id
                _response_had_audio = _ad.response_had_audio
                _current_item_id = _ad.current_item_id

            # ── Agent transcript ───────────────────────────────────────────
            elif event_type == "response.output_audio_transcript.delta":
                _agent_text_buf += msg.get("delta", "")

            elif event_type == "response.output_audio_transcript.done":
                _agent_text_buf, _barge_in_pending = await _handle_agent_transcript(
                    msg, sess, oai_ws, _agent_text_buf, _barge_in_pending)
                # A HELD SECOND ITEM THE VERDICT RELEASED. The decision is
                # taken in the handler, which has the two transcripts to
                # compare; the playing is done here, which is where twilio_ws
                # and the response's sample count are. Same split as the
                # deferred close, for the same reason.
                #
                # The samples are added to this response's own count because
                # _playback_ends_at is computed from it at response.done —
                # audio the callee is hearing that nothing counted is a hang-up
                # that cuts them off mid-sentence, and an echo cooldown that
                # ends while the agent is still talking.
                if sess._release_item:
                    _rel = sess._release_item
                    sess._release_item = ""
                    _samples_this_response += await _flush_held_item(
                        sess, twilio_ws, _rel, _current_response_pcm)

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
                # 397 lines lived here until 2026-08-27. See lifecycle.py, and
                # pyrightconfig.json for why the remedy is a split and not a
                # complexity ceiling.
                _rd = await _handle_response_done(
                    msg, sess, oai_ws, twilio_ws, done_event,
                    _current_response_pcm, _counted_responses,
                    _ResponseDone(
                        _samples_this_response, _first_delta_sent_at,
                        _current_response_start, _spoken_item_id,
                        _response_had_audio, _current_item_id, _closing_sent,
                        _closing_retries, _empty_responses,
                        _pending_response_create, _barge_in_pending,
                        _echo_cooldown))
                _samples_this_response = _rd.samples_this_response
                _first_delta_sent_at = _rd.first_delta_sent_at
                _current_response_start = _rd.current_response_start
                _spoken_item_id = _rd.spoken_item_id
                _response_had_audio = _rd.response_had_audio
                _current_item_id = _rd.current_item_id
                _closing_sent = _rd.closing_sent
                _closing_retries = _rd.closing_retries
                _empty_responses = _rd.empty_responses
                _pending_response_create = _rd.pending_response_create
                _barge_in_pending = _rd.barge_in_pending
                _echo_cooldown = _rd.echo_cooldown
                # THE TWO EXITS. `continue` and `break` cannot cross a function
                # boundary, so the handler names the loop's next move and this
                # is the only place that acts on it. Dropping either one leaves
                # the goodbye unretried, or the line open after the hang-up.
                if _rd.flow == "continue":
                    continue
                if _rd.flow == "break":
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
        # This used to log at INFO with no traceback, so a bug in the event loop silently ended the call and looked, from the caller's side, exactly
        # like a dropped line. A NameError here cost 12 test failures that all
        # pointed somewhere else.
        log.exception("[Realtime] OAI→Twilio loop CRASHED: %s", e)
        print(f"\n[Realtime] ❌ EVENT LOOP CRASHED — the call was cut short.\n"
              f"           {type(e).__name__}: {e}\n"
              f"{traceback.format_exc()}", flush=True)


# The re-export surface, declared. This module is now an orchestration layer
# over six others and most of what it imports it does not call itself - the
# suite reaches those through rw.<name>. An unaliased import says so with the
# redundant `X as X` form; an ALIASED one has no such form, so the whole
# surface is named here instead. Same purpose as the list in evidence.py.
__all__ = [    "ACCEPTING_ASK",
    "CallMemory",
    "Doctor",
    "DoctorStatus",
    "GIVE_UP_MARKERS",
    "GIVE_UP_REASONS",
    "IDENTITY_ASK",
    "NamedTuple",
    "OUTBOUND_UNAVAILABLE",
    "Optional",
    "Outcome",
    "REALTIME_URL",
    "REFERRAL_ASK",
    "RealtimeSession",
    "SCHEDULING_ASK",
    "Source",
    "TOOL_SCHEMAS",
    "TranscriptTurn",
    "WebSocket",
    "_ACK_REPLY",
    "_ACK_TAKES_VALUE",
    "_ACK_WORDS",
    "_AFFIRM_REPLY",
    "_AGENT_WIRE_SR",
    "_AudioDelta",
    "_MAX_HELD_ITEM_CHUNKS",
    "_drop_held_items",
    "_flush_held_item",
    "_BACKCHANNEL_AFTER_S",
    "_BACKCHANNEL_COOLDOWN_S",
    "_BACKCHANNEL_ECHO_MARGIN_S",
    "_CALLER_WILL_ACT",
    "_CALL_SHAPE_EXITS",
    "_CHOICE_SAVE_TOOLS",
    "_CLAIMS_SAVED",
    "_CONFIRMS_VALUE",
    "_CUT_SHORT_MS",
    "_DETAIL_FUNCTION_WORDS",
    "_DOCTORS_LOCK",
    "_FABRICATION_VOCAB",
    "_FACTUAL_ESCALATIONS",
    "_HAS_AFFIRM",
    "_HINT_HEADINGS",
    "_HINT_RUN_WORDS",
    "_HOLD_GRACE_S",
    "_HOLD_REQUEST",
    "_IDENTITY_ASK",
    "_INVITATION",
    "_LOCATION_ANCHORS",
    "_LOW_AUDIO_RMS",
    "_MASTER_LOCK",
    "_MAX_OWED_PER_CALL",
    "_MAX_OWED_PER_TEXT",
    "_MAX_SAVE_REJECTIONS",
    "_MAX_SILENCE_PROMPTS",
    "_MAX_VETTING_REASKS",
    "_MEANING_CLASSES",
    "_MIN_REASK_GAP_S",
    "_MIN_TURNS_FOR_ADAPTIVE",
    "_NAMED_DOCTOR",
    "_NON_PLACE",
    "_NOT_AN_ASK",
    "_NUMBER_WORD_VALUE",
    "_OAI_CONNECT_TIMEOUT_S",
    "_OAI_SR",
    "_ORG_STOPWORDS",
    "_ORG_WORD",
    "_PATIENT_ASK",
    "_POSSESSIVE",
    "_PREWARMED",
    "_PREWARM_TTL_S",
    "_PROJECT_ROOT",
    "_QUIET_FRACTION",
    "_REPAIR_WINDOW_S",
    "_REPORTS_FAILURE",
    "_RETIRED_HINT_TEXT",
    "_RETIRED_VOCAB_TEXT",
    "_SELF_ID",
    "_SELF_ID_WEAK",
    "_SILENCE_PROMPT_AFTER",
    "_SILENCE_PROMPT_FIRST",
    "_SILENT_AUDIO_RMS",
    "_STACK_BREATH_S",
    "_STREET_ADDRESS",
    "_STREET_SUFFIX",
    "_TWILIO_SILENCE_FRAME",
    "_TWILIO_SR",
    "_ToolOutcome",
    "_UNGROUNDED_STOPWORDS",
    "_VETTING_OPENER",
    "_above_echo_floor",
    "_address_dropped",
    "_address_offered",
    "_agent_wire_sample_rate",
    "_agent_wire_samples",
    "_ask_budget_outcome",
    "_asks_about_patient",
    "_asserted_caller_text",
    "_audio_carried_nothing",
    "_audio_was_silent",
    "_caller_answered_since",
    "_caller_ends_call",
    "_caller_is_vetting",
    "_caller_speech_level",
    "_caller_vetted_since",
    "_candidate_location",
    "_claims_employment",
    "_claims_saved",
    "_spoken_farewell",
    "_class_present",
    "_close_quietly",
    "_collapse",
    "_content_words",
    "_create_response",
    "_discarded_location",
    "_distinctive",
    "_double_ask",
    "_drop_lost_substance",
    "_echo_gate_allows",
    "_effective_output_format",
    "_end_speaking_gate",
    "_ResponseDone",
    "_handle_response_done",
    "_ever_transcribed",
    "_field_already_answered",
    "_volunteered_fields",
    "_field_vocabulary",
    "_fmt_stages",
    "_grounded_in",
    "_grounded_loosely",
    "_grounding_verdict",
    "_handle_agent_transcript",
    "_handle_audio_delta",
    "_handle_caller_transcript",
    "_handle_tool_call",
    "_hint_proper_nouns",
    "_hint_vocabulary",
    "_invites_continuation",
    "_is_ask_for",
    "_is_bare_hint_word",
    "_is_filler_reply",
    "_is_hint_echo",
    "_is_location_ask",
    "_is_objective_ask",
    "_is_own_backchannel_echo",
    "_is_reintroduction",
    "_loudest_window_rms",
    "_meaning_class",
    "_mulaw_decode",
    "_mulaw_encode",
    "_name_mismatch",
    "_oai_to_twilio",
    "_open_realtime_session",
    "_outbound_conditioned",
    "_owed_key",
    "_owed_refusal",
    "_passthrough_enabled",
    "_pending_expectation",
    "_reads_as_hint_vocabulary",
    "_realtime_tools",
    "_resolve_deferred_save",
    "_revisit_grounding",
    "_rode_along",
    "_send_breath",
    "_sentences",
    "_silence_watchdog",
    "_spell_out",
    "_spelled_out",
    "_stage_row",
    "_stem",
    "_strip_hint_run",
    "_strip_ungrounded_detail",
    "_suppress_reply_to",
    "_surnames_named",
    "_sweep_prewarmed",
    "_transcript_pending",
    "_turn_asserts",
    "_twilio_to_oai",
    "_ungrounded_choice",
    "_ungrounded_detail",
    "_ungrounded_escalation",
    "_ungrounded_identity",
    "_ungrounded_referral",
    "_ungrounded_scheduling",
    "_ungrounded_status",
    "_ungrounded_terms",
    "_utterance_slice",
    "_wire_bytes_per_ms",
    "_wire_samples",
    "_wire_to_pcm16",
    "_wrong_doctor_named",
    "asyncio",
    "audio_dir",
    "base64",
    "build_audio_config",
    "caller_repeated_answer",
    "conversation_metrics",
    "datetime",
    "expected_answers",
    "get_template",
    "give_up_directive",
    "handle_realtime",
    "hospital_mismatch",
    "is_hold_request",
    "json",
    "json_dir",
    "logging",
    "np",
    "persona_for_voice",
    "prewarm_realtime",
    "resample",
    "run_tool",
    "settings",
    "take_prewarmed",
    "time",
    "traceback",
    "websockets",
]
