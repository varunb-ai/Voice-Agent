"""The call's state, and the record it leaves behind.

Split from realtime_worker 2026-08-26. RealtimeSession moved whole and verbatim
- 108 fields, fourteen methods, not one line rewritten.

- The CLASS moved, not save() alone. Extracting save() as a free function
  taking `sess` is the smaller move and the riskier one: 628 lines reading
  three dozen private fields would reach across a module boundary for each.
  Moving the class puts the boundary where the coupling already was.
- This had to go last. RealtimeSession reaches _objective_of and
  hospital_mismatch (grounding), the _agent_wire_* family (audio),
  _norm_clause (turns) and conversation_metrics (metrics). While those were in
  the worker, moving the class meant importing the worker back.
- The guards still import RealtimeSession from realtime_worker under
  TYPE_CHECKING. That resolves because the worker re-exports it, and stays
  TYPE_CHECKING-only, so no runtime import exists in either direction.
"""
from __future__ import annotations

import logging
from core.memory import CallMemory
from agents.voice.audio import _outbound_conditioned, _effective_output_format, _agent_wire_sample_rate, _agent_wire_samples, _agent_wire_to_pcm16, _agent_to_caller_rate, _wire_sample_rate, _wire_to_pcm16
from agents.voice.evidence import _is_location_ask, _revisit_grounding
from agents.voice.grounding import _objective_of, hospital_mismatch
from agents.voice.metrics import conversation_metrics
from agents.voice.objectives import CallObjective, Outcome, default_objective, describe as _describe_objective
from agents.voice.outbound_audio import DISABLED_REASON as OUTBOUND_UNAVAILABLE, OutboundConditioner
from agents.voice.turns import _norm_clause
from core.config import settings
from core.models import Doctor, DoctorStatus, Source, TranscriptTurn
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Optional
import asyncio
import json
import numpy as np
import threading

log = logging.getLogger(__name__)


# Where call artefacts land. Indirected through functions so tests can point
# them at a temp directory — the protocol suite used to write real WAVs and
# JSON into data/ on every run, polluting the actual call records.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def audio_dir() -> Path:
    return _PROJECT_ROOT / "data" / "3 cases voice"

def json_dir() -> Path:
    return _PROJECT_ROOT / "data" / "3 cases jsons"

# Serialises the read-modify-write of master.json. See save().
_MASTER_LOCK = threading.Lock()

# Same, for the doctor directory. A separate lock rather than reusing
# _MASTER_LOCK: the two files are independent and never written nested, so
# sharing one would only couple them.
_DOCTORS_LOCK = threading.Lock()

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
        # Normalised wordings any objective field has already been asked in, so
        # the identical clause going out a second time is detectable.
        self._ask_phrasings: set[str] = set()
        self._verbatim_ask_nudged: bool = False
        self._give_up_sent: bool = False
        # Turn index at which each field was last asked, keyed by field name.
        # PER FIELD, because a call collecting five of them has five separate
        # conversations running and "when did we last ask" has five answers.
        # The single _last_ask_turn_idx the budget keeps is the last ask about
        # ANYTHING, which cannot tell "they answered the branch question" from
        # "they answered the identity question". See _field_already_answered.
        self._field_ask_at: dict[str, int] = {}
        self._answered_reask_nudged: bool = False
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
        # Every save a guard refused ON THE SPOT, with the caller turn that
        # caused it. The counter above bounds the retry loop; this is the
        # evidence, and it is what check_refusals.py reads to find the next
        # probe gap without a person having to read a console log.
        self.save_refusals: list[dict] = []
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
        # Audio for a second spoken item, WITHHELD at the delta and not yet
        # judged. {item_id: [raw base64 delta payloads]}.
        #
        # Held rather than dropped because the mute has to fire on the first
        # audio delta of the second item — item_id is the only handle that
        # early — and the transcript that says whether it is a repeat or the
        # answer arrives afterwards. Dropping decided the question before the
        # evidence existed; this defers it, exactly as the save gate does.
        #
        # RAW, NOT CONDITIONED. sess.outbound.process is stateful per call
        # (filter memory, decimation phase, compressor envelope), so running it
        # over audio that is then discarded would carry state across a seam
        # nobody heard. Conditioning happens at the flush, in arrival order.
        self._held_item_pcm: dict[str, list] = {}
        # The item whose held audio the verdict released. Read and cleared by
        # the event loop, which is where twilio_ws and the sample counter live.
        self._release_item: str = ""
        # Second items that were held, judged NEW SUBSTANCE, and played.
        self.released_second_items: list[dict] = []
        # When the caller stopped speaking (monotonic), cleared by the first
        # audio delta of the reply. See note_reply_latency.
        self._caller_stopped_at: Optional[float] = None
        # How long the DETECTOR took to tell us the caller had stopped,
        # measured from audio_end_ms rather than assumed per detector. This
        # is the term that separates server_vad from semantic_vad, and it was
        # a hardcoded guess in both directions before it was measured.
        self._last_stop_lag_s: float = 0.0
        self.detector_lags: list[float] = []
        # ── PER-TURN STAGE CLOCKS ────────────────────────────────────────────
        # WHY THIS EXISTS. reply_latency reports one number per turn, and on
        # call-20260826-1134 that number ranged 1.69s to 6.64s across five
        # structurally identical tool turns with no way to say which stage
        # moved. The console log's whole-second timestamps could prove the
        # SPREAD sat before the tool call (4.99s, against 0.55s after it) but
        # not the absolute split, because the call-start second and the tool
        # print second collapse into one shared unknown offset.
        #
        # Six marks, all on events already handled here, so nothing new is
        # asked of the socket and no behaviour changes:
        #   t0 speech_stopped        the caller stopped (already backdated)
        #   t1 response.created      OpenAI opened the response
        #   t2 function_call_args    the tool call arrived   [t2-t1 = inference 1]
        #   t3 output submitted      we answered it          [t3-t2 = our work]
        #   t4 response.done         the tool response closed[t4-t3 = the DEFERRAL]
        #   t5 first audio delta     the agent spoke         [t5-t4 = inference 2]
        # MEASURE-ONLY. Nothing here is read by any guard or gate.
        self.turn_stages: list[dict] = []
        self._stage: Optional[dict] = None
        # ── A SAVE WAITING FOR ITS EVIDENCE ─────────────────────────────────
        # The model answers from audio in ~0.5s; transcription lands in ~2s.
        # When the model wins that race the guards are asked their question
        # before the evidence for it exists, and on call-20260826-1422 that
        # happened on SIX of six saves — every wait timed out, every save was
        # refused, and the caller was asked twice for answers already given.
        #
        # The refusal was designed to cost "one more turn". It did not: the
        # model read "nothing has been transcribed" as "they did not answer",
        # apologised, and re-asked. So the decision is deferred instead of
        # refused — parked here, re-judged by the SAME guard the moment the
        # transcript lands. Nothing is written on a deferral.
        # The caller's own words, when they are why the call ended. None on
        # every call that ended the ordinary way, so the artifact can tell a
        # hangup we chose from one we were asked for.
        self.ended_by_caller: Optional[str] = None
        self._deferred_save: Optional[dict] = None
        self.deferred_saves: list[dict] = []
        # ── OPENAI'S VAD CREATES RESPONSES WE DID NOT ASK FOR ───────────────
        # `create_response` is not set in build_audio_config, so it runs on the
        # API default of true and the server opens a response at every
        # speech_stopped. We learn of it only when `response.created` reaches
        # us — and until then `_response_active` is False, so our own guard
        # waves a second create through and the server refuses it.
        #
        # call-20260826-1422 at 14:24:56: a deferred tool-result create fired
        # just as two caller transcripts landed, and lost —
        # conversation_already_has_active_response, resp_EH3hIa4MQ3lMDIzyo1cO.
        # Nothing was silent that time because the VAD's own response covered
        # the turn, which is luck, not design.
        #
        # So the window is treated as occupied: from speech_stopped until the
        # response.created that answers it. Bounded, because a window that can
        # only be closed by an event that may never come is a call that can
        # never speak again.
        self._vad_response_due_until: float = 0.0
        # Every measured gap between a caller finishing and the agent's first
        # sound, in seconds. One number per turn beats one impression per call.
        self.reply_latencies: list[float] = []
        # What those dropped items would have said, for the artifact, each
        # carrying the VERDICT that says why it was dropped.
        #
        # A bare list of strings could not answer the only question a reviewer
        # has about this list. On call-20260825-1428 the model emitted the
        # greeting twice and the mute stopped the caller hearing it twice —
        # the guard working exactly as intended. call-20260825-1435 has an
        # entry of identical shape that was the caller's answer being deleted.
        # Same list, same kind of string, opposite meanings.
        #
        #   "duplicate"  the spoken half already carried it. Nothing was lost,
        #                and this entry is the guard earning its keep.
        #   "owed"       substance the caller did not hear. Recovery scheduled.
        #   "abandoned"  substance lost AND the recovery caps were reached.
        #                This one is a defect on the call and is meant to be
        #                greppable across a batch of artifacts.
        self.dropped_second_items: list[dict] = []
        # Text that was muted mid-response and carried something the spoken
        # half did not. Owed to the caller, and said on the next turn rather
        # than lost. See _drop_lost_substance.
        self._owed_substance: str = ""
        self._owed_recovered: int = 0
        # Recovery attempts, per owed sentence and per call. See
        # _MAX_OWED_PER_TEXT: without these the recovery scheduled itself
        # forever, because the response it creates can be muted the same way
        # the turn that created the debt was.
        self._owed_attempts: dict = {}
        self._owed_tried: int = 0
        # Owed text the caps refused to chase again, with why. A recovery that
        # gives up has to leave a trace, or it is indistinguishable in the
        # artifact from a turn that never owed anything.
        self.owed_abandoned: list[dict] = []
        # Every branch value the grounding guard refused, with the reason and
        # the clock time. Non-empty means the saved branch was contested: a
        # value was rejected and the wording that finally landed may be the
        # transcriber's rather than the caller's. See _grounding_verdict, which
        # reads this so the stamp cannot say "verified" without saying that.
        self.branch_rejections: list[dict] = []
        # "The objective finished on a DEFERRED save; close after the response
        # that is already in flight." Set by _resolve_deferred_save, acted on
        # at response.done — see the comment there for why it cannot be both.
        self._close_after_response: bool = False
        # "The objective finished while the caller was still owed an answer to
        # the question we had just asked them; close once they have given it."
        # Set by the tool handler, consumed in _handle_caller_transcript, which
        # hands it on to _close_after_response so the agent's REPLY to that
        # answer is the turn that closes. Two flags because they mark different
        # moments: one waits on a response we already asked for, this one waits
        # on a person. See call-20260831-1048.
        self._close_when_answered: bool = False
        # escalate refused once because the caller's last turn was still
        # transcribing. ONE-SHOT: the placeholder resolves within a turn either
        # way, and a guard that can refuse forever cannot end a call.
        self._escalation_held: bool = False
        # Fields the volunteered-info guard has already spoken up about. One
        # directive per field per call: a second copy of one the model ignored
        # is context spent for nothing, the same rule the other nudges use.
        self._volunteered_seen: set[str] = set()
        # The caller offered to help and was told to spend it. One-shot, like
        # the patient and identity nudges beside it.
        self._offer_nudged: bool = False
        # The agent said goodbye while nothing had ended the call. One-shot,
        # and recorded: a guard that fires invisibly cannot be checked after.
        self._farewell_nudged: bool = False
        self._refusal_nudged: bool = False
        self.hard_refusal: str = ""
        # Deferred, exactly like _claimed_done_at above it: a farewell is
        # judged 1.5s after it is spoken, so a tool call belonging to the
        # same response has landed and sess.done is knowable. Judging it at
        # transcript time fires on every correct close.
        # Last inbound frame loud enough to be a person rather than our own
        # echo, and how loud. Written only while agent audio is playing out;
        # read only by the drain barge-in, which will not fire without a
        # recent one.
        self._last_voiced_frame_at: float = 0.0
        self._last_voiced_frame_rms: float = 0.0
        self.drain_barge_ins: list[dict] = []
        self._farewell_at: float = 0.0
        self._farewell_said: str = ""
        self.farewell_without_close: list[str] = []
        # Answers the caller gave to questions nobody had asked, with the state
        # they read as. Non-empty means the guard caught a field the ordinary
        # path — which only ever looks at the question on the table — would
        # have gone on to ask for again.
        self.volunteered_answers: list[dict] = []
        # Closed-set saves accepted without their quote ever being checked,
        # because nothing on the call transcribed at all. `heard` on these is
        # the model's own string, and on call-20260825-1731 a model-authored
        # quote ("Okay.") stood as the provenance for a CONFIRMED identity.
        # Non-empty here means the row's quotes are unaudited.
        self.unverified_quotes: list[dict] = []
        # Monotonic stamps for "a caller turn is awaiting transcription": when
        # the placeholder was created, and when the transcriber last answered
        # for ANY utterance. A result that arrived and was discarded still
        # counts as answered — see _transcript_pending.
        self._placeholder_at: float = 0.0
        self._transcript_at: float = 0.0
        # Every surname heard that was not the doctor on record. THE FAILURE
        # MADE VISIBLE: calls 1433 and 1437 both ended with identity
        # unconfirmed because the transcriber rendered "Reyes" as "Riaz", "Yes"
        # and "Ayers", and no artifact said so — the only trace was a memory
        # key nothing read. A call that could not confirm the name and a call
        # that never asked must not produce the same record.
        self.name_mismatches: list[dict] = []
        # Index into `turns` just after the agent spelled our doctor's surname
        # letter by letter. 0 means it never did. Everything before it is
        # evidence about a name the line mangled; everything after it is
        # evidence about ours. See _name_mismatch.
        self._name_spelled_at: int = 0
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
        # Outbound conditioning, one per call because every stage of it carries
        # state between deltas. Built unconditionally — it is cheap, and a
        # session that flips to passthrough mid-call simply never calls it.
        self.outbound = OutboundConditioner()
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

    def notes(self) -> dict:
        """Everything note_info recorded, as {key: value}. {} when it wrote none.

        WRITE-ONLY UNTIL 2026-08-27, and that is the whole reason this exists.
        `note_info` did `memory.update(**{f"note_{key}": value})` and returned
        ok — and nothing anywhere read a `note_` key back out. Not the artifact,
        not doctors.json, not the summary. A tool that reports success and whose
        output no downstream reader can see is the "acts and leaves no trace"
        defect wearing a third hat.

        call-20260827-1516 is the case. The caller said it was a bad time and
        asked to be rung back; the agent got the window ("afternoon"), called
        note_info TWICE to record it, and the call filed
        `outcome: none, collected: []`. The one actionable thing the call
        learned survived only as prose inside the transcript array, where
        nothing can schedule anything from it.

        The `note_` prefix is stripped, because it is namespacing for the memory
        dict and noise to a reader. Fields may point at note_ keys (see
        objectives.py), so a key that is ALSO a collected field still appears
        here — the two views are of different things and neither is authoritative
        over the other.
        """
        # snapshot(), not .items(): CallMemory is a store with a backend, not a
        # dict, and reaching past its API is how a change of backend becomes a
        # silent breakage here.
        _snap = self.memory.snapshot() or {}
        return {k[len("note_"):]: v
                for k, v in sorted(_snap.items())
                if k.startswith("note_") and v not in (None, "")}

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
            # Same reasoning as collected_fields directly above: the Doctor
            # model has no column for a callback window or a referral URL, and
            # inventing one here would put a second schema in the directory. It
            # travels as a nested dict beside the columns that do exist. This is
            # the row a person acts on, so a call whose only product was "ring
            # back this afternoon" has to put it HERE, not only on the artifact.
            "notes":            self.notes(),
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
                arr   = _agent_to_caller_rate(_agent_wire_to_pcm16(pcm_bytes), _SR)
                end   = min(start + len(arr), n)
                if end > start:
                    agent[start:end] += arr[:end - start]
                print(f"  agent block: t={t_offset:.2f}s, "
                      f"dur={_agent_wire_samples(pcm_bytes)/_agent_wire_sample_rate():.2f}s, "
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
                e = min(s + int(_agent_wire_samples(pcm_b)
                               / _agent_wire_sample_rate() * _SR)
                        + int(0.30 * _SR), n)
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

        # The transcript is final now — nothing else will arrive to change
        # what the caller was heard to say. This is the first moment the
        # grounding question has a settled answer; it was decided during the
        # call against a transcript that was still filling in. See
        # _revisit_grounding.
        _revisit_grounding(self)

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
            # The branch values the guard refused on the way to the one above.
            # A rejection that only ever reached the console is a guard that
            # left no trace — the failure family this project keeps paying for.
            "branch_rejections": self.branch_rejections or None,
            # Everything note_info recorded. On call-20260827-1516 this was the
            # ONLY actionable thing the call learned — a callback window — and
            # it reached no structured field at all.
            "notes":          self.notes() or None,
            # Refusals, with the words that caused them. See check_refusals.py.
            "save_refusals":  self.save_refusals or None,
            # Sign-offs spoken while no tool had ended the call. Non-empty
            # means the agent tried to leave without writing a reason.
            "farewell_without_close": self.farewell_without_close or None,
            "hard_refusal": self.hard_refusal or None,
            "drain_barge_ins": self.drain_barge_ins or None,
            # What `grounding` said while the call was still running, present
            # only when the finished transcript changed the answer. A verdict
            # that improves silently after the fact cannot be audited — this
            # keeps "did the guard fire during the call" answerable.
            "grounding_at_save": self.memory.get("grounding_at_save"),
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
            # Non-null here means the model tried to talk over itself. Read the
            # `verdict` on each, not the count: "duplicate" is the guard
            # working, "abandoned" is a caller who asked something and was
            # never answered.
            "dropped_second_items": self.dropped_second_items or None,
            # Second items the caller DID hear, because the transcript showed
            # they carried substance the spoken half did not. Non-empty here is
            # the mute declining to delete an answer.
            "released_second_items": self.released_second_items or None,
            "volunteered_answers": self.volunteered_answers or None,
            # Substance the caller was owed and the recovery gave up on, with
            # the cap that stopped it. Non-null is always a defect on the call.
            "owed_abandoned": self.owed_abandoned or None,
            # What the save gate did. A value it HELD is one the caller
            # really gave and the record never received, because nothing yet
            # said who it was about — correct, and not something that may
            # happen quietly. `held_saves` non-empty at the end of a call is
            # an answer that was collected and lost; the other two are the
            # gate resolving, which is the path that is supposed to happen.
            "held_saves":     self.memory.get("deferred_saves") or None,
            "held_applied":   self.memory.get("deferred_applied") or None,
            "held_dropped":   self.memory.get("deferred_dropped") or None,
            # Closed-set values accepted with a model-authored quote, because
            # this call transcribed nothing to check it against. Non-null means
            # `heard` on those fields was never corroborated.
            "unverified_quotes": self.unverified_quotes or None,
            # Guards held for a caller turn that was still transcribing, and
            # whether it arrived. `landed: false` entries are the calls where
            # the wait bought nothing and the ceiling may be too low.
            # A save whose evidence never arrived is closed out here rather
            # than vanishing: it is an answer the model offered that the record
            # never got, and it must read as such in the artifact.
            "deferred_saves": ((self.deferred_saves + (
                [{"tool": self._deferred_save["name"],
                  "args": self._deferred_save["args"],
                  "held_because": self._deferred_save["why"],
                  "outcome": "never_arrived"}]
                if self._deferred_save else []))
                or None),
            "ended_by_caller": self.ended_by_caller,
            # Surnames heard that were not the doctor on record, each with
            # whether it was still happening AFTER we spelled the name out.
            # Non-null with after_spelling false is a transcription problem;
            # with it true, the practice really does have somebody else.
            "name_mismatches": self.name_mismatches or None,
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
                # Per-turn stage breakdown. reply_latency says how long the
                # caller waited; this says which stage spent it.
                "stages": self.turn_stages or None,
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
                "input_format":   settings.realtime_audio_format,
                # THE EFFECTIVE OUTPUT FORMAT, NOT THE CONFIGURED ONE. The whole
                # case for the pcm leg is that it is falsifiable by reverting one
                # value and comparing two calls — and that comparison reads this
                # field, so it has to say what the call DID, not what it asked
                # for. Those differ: without scipy, "pcm" silently negotiates
                # mu-law passthrough instead (_effective_output_format), and a
                # record that copied the setting would label an unconditioned
                # call "pcm" and quietly corrupt the A/B it exists to serve.
                "output_format":     _effective_output_format(),
                "output_conditioned": _outbound_conditioned(),
                # Present ONLY when the request was downgraded, and carrying the
                # cause. A conditioner that stands down invisibly is the same
                # defect as a guard that refuses invisibly: the call sounds dull,
                # the artifact says nothing, and the search starts at the prompt.
                **({"output_downgraded": OUTBOUND_UNAVAILABLE}
                   if settings.realtime_output_format == "pcm"
                   and not _outbound_conditioned() else {}),
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
            _by: dict = {}
            for _d in self.dropped_second_items:
                _v = _d.get("verdict", "?")
                _by[_v] = _by.get(_v, 0) + 1
            print(f"    2nd items muted      {len(self.dropped_second_items)}"
                  f"   ({', '.join(f'{n} {v}' for v, n in sorted(_by.items()))})",
                  flush=True)
            if _by.get("abandoned"):
                print(f"      ^ {_by['abandoned']} of them was substance the "
                      f"caller never heard and the recovery gave up on",
                      flush=True)
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
        if self.unverified_quotes:
            print(f"    quotes never checked  {len(self.unverified_quotes)}"
                  f"   <- nothing transcribed on this call, so `heard` is the "
                  f"model's own words", flush=True)
        _held = self.memory.get("deferred_saves") or []
        if _held:
            print(f"    held and never filed {len(_held)}"
                  f"   <- the caller answered these and the record never got "
                  f"them", flush=True)
            for _h in _held:
                print(f"        {_h.get('field')}  (waiting on "
                      f"{_h.get('gate')})", flush=True)
        if self.name_mismatches:
            _after = sum(1 for m in self.name_mismatches
                         if m.get("after_spelling"))
            print(f"    wrong name heard     {len(self.name_mismatches)}"
                  f"   ({', '.join(sorted({m['heard'] for m in self.name_mismatches}))}"
                  f" vs {self.name_mismatches[0]['ours']!r})", flush=True)
            _why = ("the practice named someone else even after we spelled "
                    "it out — not a transcription problem") if _after else (
                    "never spelled out to them, so this is the line mangling "
                    "the name and not the practice")
            print(f"      ^ {_why}", flush=True)
        if self._employment_claimed:
            print(f"    ⚠️  FALSE EMPLOYMENT CLAIM made on this call",
                  flush=True)

        self._print_cost(duration)


__all__ = [
    "RealtimeSession",
    "_DOCTORS_LOCK",
    "_MASTER_LOCK",
    "_PROJECT_ROOT",
    "_ask_budget_outcome",
    "audio_dir",
    "json_dir",
]
