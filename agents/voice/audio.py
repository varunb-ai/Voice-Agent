"""Wire formats, buffers, and the outbound delta path.

Split from realtime_worker 2026-08-26, verbatim.

- Not a second audio_utils. Mu-law and resampling come from
  agents/experiment/audio_utils; here is only what means something inside a
  CALL - which format each leg negotiated, how many samples a buffer holds,
  where one utterance sits in the caller recording.
- The wire constants came too. Each has exactly one reader, and splitting a
  constant from its reader is how a sample rate ends up defined twice.
- One way: realtime_worker imports this, never the reverse. RealtimeSession is
  bound under TYPE_CHECKING only, so no runtime import exists.
"""
from __future__ import annotations

import base64
import json
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, NamedTuple, Optional

import numpy as np

from core.config import settings
from agents.experiment.audio_utils import resample, _mulaw_decode
from agents.voice.outbound_audio import DISABLED_REASON as OUTBOUND_UNAVAILABLE
from agents.voice.latency import _stage_row, _fmt_stages
from agents.voice.evidence import _LOW_AUDIO_RMS, _QUIET_FRACTION

if TYPE_CHECKING:                    # pragma: no cover - typing only
    # One way: realtime_worker imports audio, never the reverse.
    from agents.voice.realtime_worker import RealtimeSession

log = logging.getLogger(__name__)



_TWILIO_SR = 8_000
_OAI_SR    = 24_000


# ── Audio format conversion ───────────────────────────────────────────────────
#
# _convert_oai_to_twilio IS GONE, and it is worth saying why rather than
# leaving a smaller version of it here. It did the 24k -> 8k conversion one
# delta at a time, which meant the anti-alias filter restarted from zero at
# every 400ms boundary:
#
#     error RMS   0.00108     SNR 48.6 dB
#     error peak  0.09014     <- a -15 dB transient, 2.5x a second
#     share of that error within 5ms of a chunk boundary:  100%
#
# The aggregate SNR reads as harmless and the distribution does not. It never
# fired in production only because pcmu passthrough routed around the function
# entirely — so it sat here, correct-looking, waiting for the day somebody
# turned the pcm path on. That day is now, which is why the replacement is a
# stateful object (agents/voice/outbound_audio.OutboundConditioner) and not a
# function: filter memory, decimation phase and compressor envelope all have to
# survive the gap between one delta and the next.
#
# Verified: chunked output is now byte-identical to single-shot over the same
# audio, error exactly 0.0.


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
    """Do the RECORDING BUFFERS hold 8kHz μ-law?

    This governs the buffers, not the wire, and the two stopped being the same
    question when the outbound leg moved to PCM. `_outbound_conditioned` is the
    wire question for the agent's audio.

    It stays true under conditioning because the agent's recording now stores
    the μ-law THAT WAS SENT rather than the PCM that arrived — see the delta
    handler. That keeps both channels in one format, which save() requires, and
    it makes the recording the audio the caller actually heard instead of the
    audio we started from. A recording of the unconditioned signal would be a
    recording of something nobody was played.
    """
    return settings.realtime_audio_format == "pcmu"


def _outbound_conditioned() -> bool:
    """Is OpenAI sending us PCM16 24kHz for us to condition and convert?

    FALSE WHEN THE CONDITIONER CANNOT RUN, not only when it is switched off,
    and that second clause is load-bearing. Asking OpenAI for PCM we have no
    filter for would leave the 24k->8k decimation with nothing in front of it,
    and decimating unfiltered audio ALIASES — every component between 4 and
    8 kHz folds back down into speech. That is materially worse than the dull
    line conditioning was added to fix, and it is the exact trap
    audio_utils.resample already contains (it drops to np.interp, silently,
    with no anti-alias filter, when scipy is missing).

    So the two decisions are made together: no scipy means we negotiate μ-law
    and pass it through untouched, exactly as before any of this existed.
    """
    if settings.realtime_output_format != "pcm":
        return False
    return not OUTBOUND_UNAVAILABLE


def _effective_output_format() -> str:
    """What we actually ask OpenAI to send on the output leg.

    NOT settings.realtime_output_format read directly, because "pcm" is only a
    coherent request when there is a conditioner to receive it. Without scipy,
    asking for PCM and then forwarding it to Twilio unchanged would put linear
    PCM bytes on a wire that reads them as μ-law — noise, on every call.

    One function so the negotiation and the handling cannot disagree: whatever
    this returns is what session.update asks for AND what the delta handler
    expects to arrive.
    """
    return "pcm" if _outbound_conditioned() else "pcmu"


# THE AGENT RECORDING BUFFER IS ALWAYS 8kHz μ-LAW. Not conditionally — always,
# and stating it as an invariant rather than a lookup is the fix for a bug this
# module had for exactly one afternoon.
#
# It holds what was SENT TO TWILIO, and Twilio's wire is μ-law 8k on every call
# whatever OpenAI was asked for: either OpenAI sent μ-law and it was forwarded,
# or it sent PCM and the conditioner converted it. There is no third path.
#
# The version this replaces derived the answer from the format settings and got
# one of the four combinations wrong — inbound pcm with outbound pcmu reported a
# one-second block as 0.167s, six times out, which is a garbled agent channel in
# the WAV and a barge-in truncation against the wrong clock. Deriving a constant
# is how that happens. The caller side still needs _passthrough_enabled,
# because its buffer really does change format with the inbound setting.
_AGENT_WIRE_SR = _TWILIO_SR


def _agent_wire_sample_rate() -> int:
    return _AGENT_WIRE_SR


def _agent_wire_samples(raw: bytes) -> int:
    return len(raw)          # μ-law is one byte per sample


def _agent_wire_to_pcm16(raw: bytes) -> np.ndarray:
    return _mulaw_decode(raw)


def _agent_to_caller_rate(arr: np.ndarray, caller_sr: int) -> np.ndarray:
    """Put an agent block on the caller channel's timebase before mixing.

    save() lays both channels into ONE array at one sample rate. While the two
    buffers shared a format this was free; conditioning can make the agent 8kHz
    while the caller is 24kHz, and summing those without a resample plays the
    agent back at a third of speed for a third of the duration.
    """
    agent_sr = _agent_wire_sample_rate()
    if agent_sr == caller_sr or arr.size == 0:
        return arr
    return resample(arr, agent_sr, caller_sr)


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

class _AudioDelta(NamedTuple):
    """What one audio delta changed in _oai_to_twilio's locals.

    THE SAME SHAPE AS _ToolOutcome, and for the same reason. pyright refused to
    analyse _oai_to_twilio — "Code is too complex to analyze; reduce complexity
    by refactoring into subroutines" — which is not a cosmetic warning: once it
    gives up it can no longer prove any local in the function is read, so the
    editor greys out dozens of live names and stops seeing the calls the
    function makes. pyrightconfig.json records that every recurring bug of that
    week lived in exactly that unanalysed region, and says to answer this by
    splitting the function rather than by raising maxCodeComplexity, which does
    not work anyway.

    The audio-delta block is the largest self-contained piece left, and it grew
    again when outbound conditioning landed. Its coupling to the loop is these
    five locals and nothing else.
    """
    samples_this_response: int
    first_delta_sent_at: Optional[float]
    current_response_start: Optional[float]
    spoken_item_id: Optional[str]
    response_had_audio: bool
    # THE ONE THAT WAS MISSED, and pyright is why it was found. The extraction
    # shipped without it and _current_item_id was read on line 1 of the block
    # while unbound — UnboundLocalError on the FIRST audio delta of every call,
    # which is to say no call would have produced sound at all.
    #
    # It is also the exact argument pyrightconfig.json makes: the complexity
    # bail is not a cosmetic warning. While the analyser was giving up on this
    # function, this error was invisible; the split that cleared the bail
    # surfaced it in the same run.
    #
    # It has to round-trip because the two halves both use it: this block
    # follows the item being spoken, and the barge-in handler truncates that
    # item to what the caller actually heard.
    current_item_id: Optional[str]


# Roughly 40s of deltas at gpt-realtime-2's chunk size. Nothing legitimate
# comes near it; it exists so a held item cannot grow without bound.
_MAX_HELD_ITEM_CHUNKS = 400


def _drop_held_items(sess: "RealtimeSession", why: str) -> None:
    """Throw away every held second item. Called when nothing may be played.

    A barge-in and a cancelled response BOTH land here, and they have to: the
    held audio is a turn the caller has just talked over, and playing it after
    the fact is the one thing worse than muting it. `_muted_items` is left
    alone — the verdict site still needs to know the item was withheld so it
    does not become a turn nobody heard.
    """
    if sess._held_item_pcm:
        print(f"[Realtime] 🔇 held second item discarded — {why}", flush=True)
    sess._held_item_pcm.clear()
    sess._release_item = ""


async def _flush_held_item(sess: "RealtimeSession", twilio_ws, item_id: str,
                           current_response_pcm: list) -> int:
    """Play a held second item the verdict judged to carry real substance.

    Returns the wire samples played, which the caller adds to the response's
    own count — `_playback_ends_at` is computed from it at response.done, and
    audio the callee is hearing that nothing counted is a hang-up that cuts
    them off mid-sentence.

    CONDITIONED HERE, not at the delta, and in arrival order: the conditioner
    is stateful per call, and this is the first moment we know the audio will
    actually be played. Same bytes to the wire and to the recording, which is
    the invariant the delta path exists to keep.
    """
    chunks = sess._held_item_pcm.pop(item_id, None)
    if not chunks or not sess.stream_sid:
        return 0
    _samples = 0
    for _delta in chunks:
        try:
            raw_pcm = base64.b64decode(_delta)
            if _outbound_conditioned():
                raw_pcm = sess.outbound.process(raw_pcm)
            _samples += _agent_wire_samples(raw_pcm)
            current_response_pcm.append(raw_pcm)
            await twilio_ws.send_text(json.dumps({
                "event":     "media",
                "streamSid": sess.stream_sid,
                "media":     {"payload": (base64.b64encode(raw_pcm).decode()
                                          if _outbound_conditioned() else _delta)},
            }))
        except Exception as e:
            log.error("[Realtime] held-item send error: %s", e)
            break
    return _samples


async def _handle_audio_delta(
    msg: dict,
    sess: "RealtimeSession",
    twilio_ws,
    current_response_pcm: list,
    state: _AudioDelta,
) -> _AudioDelta:
    """Forward one chunk of the agent's speech to Twilio.

    `current_response_pcm` is mutated in place — it is the recording buffer for
    the response being spoken, and appending is the whole of its interaction
    with this function, so it does not need to travel through the return value.
    """
    samples_this_response = state.samples_this_response
    _first_delta_sent_at = state.first_delta_sent_at
    _current_response_start = state.current_response_start
    _spoken_item_id = state.spoken_item_id
    _response_had_audio = state.response_had_audio
    _current_item_id = state.current_item_id
    _current_response_pcm = current_response_pcm


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
    #
    # HELD, NOT DROPPED, since call-20260827-1130. Muting is still forced here
    # — the first item is already on the wire and item_id is the only thing
    # this early that distinguishes them — but DELETING was a decision taken
    # before the evidence for it existed. On that call the model split every
    # one of four recovery attempts into a filler intro and the actual
    # question; the question was muted each time, the owed-substance recovery
    # re-asked and was split again, and the caller heard "Sorry, quick check on
    # that.", "Let me put the referral question clearly.", "Let me ask the
    # referral part directly.", "Let me say the referral question clearly." and
    # never the question. The recovery directive already said "say just that,
    # in one short sentence, do not apologise" — a prose rule the model ignored
    # four times running, which is why the fix is here and not in the prompt.
    #
    # So the audio waits for its own transcript, and the verdict site in
    # _handle_agent_transcript decides: a repeat is discarded exactly as
    # before, substance the spoken half does not carry is played.
    _delta_item = msg.get("item_id") or ""
    if delta and _delta_item:
        if _spoken_item_id is None:
            _spoken_item_id = _delta_item
        elif _delta_item != _spoken_item_id:
            if _delta_item not in sess._muted_items:
                sess._muted_items.add(_delta_item)
                print(f"[Realtime] 🔇 second spoken item in one "
                      f"response — holding it back until its "
                      f"transcript says what it is", flush=True)
            _held = sess._held_item_pcm.setdefault(_delta_item, [])
            # A CEILING, because this buffers a stream. Past it the item
            # degrades to the old behaviour — dropped — rather than growing
            # without bound on a model that will not stop talking.
            if len(_held) < _MAX_HELD_ITEM_CHUNKS:
                _held.append(delta)
            delta = ""

    if delta:
        sess.agent_speaking  = True
        sess._response_active = True
        sess._response_audio_started = True
        _response_had_audio  = True
    if delta and sess.stream_sid:
        try:
            # CONDITIONED HERE, ONCE, AT THE TOP OF THE PATH — so
            # everything downstream (the sample count that bounds a
            # truncation, the recording buffer, the payload) is the
            # SAME bytes the caller is played. An earlier shape
            # converted at the send and recorded the unconditioned
            # PCM, which would have made the recording a document of
            # audio nobody heard.
            #
            # The conditioner is stateful and per-call: filter
            # memory, decimation phase and compressor envelope all
            # carry across deltas, which is what stops the seam
            # artefact the old per-chunk resample produced.
            raw_pcm = base64.b64decode(delta)
            if _outbound_conditioned():
                raw_pcm = sess.outbound.process(raw_pcm)
            samples_this_response += _agent_wire_samples(raw_pcm)
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
                # t5 — the agent made a sound. Close the record.
                _st = sess._stage
                sess._stage = None
                if _st is not None and "t0" in _st:
                    _st["t5"] = time.monotonic()
                    _row = _stage_row(_st, _felt)
                    sess.turn_stages.append(_row)
                    # Printed as well as recorded: the whole reason this exists
                    # is that a slow turn was unattributable while the call was
                    # still in front of someone.
                    print(f"[Realtime]   ^ stages: {_fmt_stages(_row)}",
                          flush=True)
            twilio_payload = (base64.b64encode(raw_pcm).decode()
                              if _outbound_conditioned() else delta)
            await twilio_ws.send_text(json.dumps({
                "event":    "media",
                "streamSid": sess.stream_sid,
                "media":    {"payload": twilio_payload},
            }))
        except Exception as e:
            log.error("[Realtime] audio send error: %s", e)
    elif delta and not sess.stream_sid:
        log.warning("[Realtime] Audio delta received but stream_sid empty — dropped")

    return _AudioDelta(samples_this_response, _first_delta_sent_at,
                       _current_response_start, _spoken_item_id,
                       _response_had_audio, _current_item_id)


# The re-exported surface, declared. These are called from realtime_worker and
# from audio.py, never from inside this module, so without this the checker
# reports the module's whole reason for existing as unused. Same purpose as the
# list in evidence.py: it says what the module is FOR, and it keeps a hint storm
# from burying a real warning.
__all__ = [
    "_AGENT_WIRE_SR",
    "_AudioDelta",
    "_OAI_SR",
    "_SILENT_AUDIO_RMS",
    "_STACK_BREATH_S",
    "_TWILIO_SILENCE_FRAME",
    "_TWILIO_SR",
    "_agent_to_caller_rate",
    "_agent_wire_sample_rate",
    "_MAX_HELD_ITEM_CHUNKS",
    "_agent_wire_samples",
    "_drop_held_items",
    "_flush_held_item",
    "_agent_wire_to_pcm16",
    "_audio_carried_nothing",
    "_audio_was_silent",
    "_effective_output_format",
    "_handle_audio_delta",
    "_loudest_window_rms",
    "_outbound_conditioned",
    "_passthrough_enabled",
    "_send_breath",
    "_utterance_slice",
    "_wire_bytes_per_ms",
    "_wire_sample_rate",
    "_wire_samples",
    "_wire_to_pcm16",
]
