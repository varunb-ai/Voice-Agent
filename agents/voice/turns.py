"""One caller turn, from speech_stopped to the reply it earns.

Split from realtime_worker 2026-08-26, verbatim.

- Holds the turn lifecycle: both transcript handlers, the silence watchdog, the
  reply suppressor, and the detectors they consult about the turn just heard.
- The hint quarantine came too, because it runs at INGESTION - a transcript is
  scrubbed of our own prompt before it becomes a turn.
- One way, and now a chain:
      realtime_worker -> session -> metrics -> turns -> grounding -> evidence
                                            -> audio  -> latency
  RealtimeSession is TYPE_CHECKING-only here. Without that binding every
  `sess: "RealtimeSession"` is an unresolved name that `from __future__ import
  annotations` hides from the interpreter and from the suite, but not pyright.
- _oai_to_twilio stays in the worker: it threads six mutable locals through
  these handlers by parameter and return, and extracting it would mean
  inventing shared state to replace them.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:                    # pragma: no cover - typing only
    from agents.voice.realtime_worker import RealtimeSession

from agents.voice import backchannel
from agents.voice.audio import _audio_carried_nothing, _SILENT_AUDIO_RMS, _audio_was_silent
from agents.voice.evidence import _UNGROUNDED_STOPWORDS, _invites_continuation, _caller_ends_call, _caller_is_vetting, _caller_speech_level, _drop_lost_substance, _is_ask_for, is_hard_refusal, _is_location_ask, _note_name_heard, _our_surname, _performed_name_repair, _owed_key, _owed_refusal, _spell_out, _spelled_out
from agents.voice.grounding import _objective_of, _IDENTITY_ASK, _claims_saved, _gave_name_and_dob, _gave_own_detail, _only_acknowledged, _said_not_a_patient, _stale_own_detail, _detail_left_bare, _spoken_farewell, _announced_an_ask, is_hold_request, _create_response, closing_directive, _RETIRED_VOCAB_TEXT, _resolve_deferred_save, _ack_opener, _reacted_to_news, _housekeeping_turn, _narrated_the_reply, workflow_narration, _leaked_the_instructions, _stapled_own_detail, cadence_directive
from agents.voice.objectives import AnswerKind, clauses as _clauses, expected_answers, norm_quotes as _norm_quotes, sentences as _sentences
from agents.voice.objectives import states_in_its_own_right
from core.audio_utils import _mulaw_decode
from core.config import settings
from core.models import TranscriptTurn
from datetime import datetime
from typing import Callable, Optional
import asyncio
import base64
import json
import re
import time

log = logging.getLogger(__name__)


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
# SPLIT AGAIN 2026-09-10, and this one is about SPEAKING, not about
# answering. The list above is judged as one thing because one question is
# being asked of it — did this turn carry an answer — and for that the three
# shapes below are interchangeable. For the question `_only_marked_time` asks,
# they are opposites: one of them means the caller is waiting on us.
#
# The caller marking that they heard, and nothing more. Nobody is owed a
# reply to these; the turn is still theirs.
_ACK_MARKERS = (r"ok|okay|sure|right|alright|"
                r"mm+|hm+|uh+|um+|er+|ah+|oh+|that'?s fine|i see|fine")

# ...and the rest of the old list, which is NOT the same thing. A greeting
# arriving mid-call is the caller checking the line is alive; "sorry",
# "pardon", "say again", "what" are requests to repeat; "go ahead" is an
# invitation to speak. Every one of them OWES the caller speech, and the only
# thing they share with the markers above is carrying no answer. Measured over
# 211 call artifacts: of the 51 caller turns that follow an objective ask
# carrying no answer, 11 are these — 9 greetings, one "hello sorry?", one
# "sure, sure. go ahead." Going quiet on any of them is the failure this split
# exists to prevent.
_ACK_REPAIRS = (r"hello|hullo|hi|hey|go ahead|"
                r"sorry|pardon|come again|say again|what|huh|"
                r"are you there|still there|can you hear me")

# THE UNION IS UNCHANGED, and the suite proves it rather than asserting it:
# every caller turn in the corpus is re-judged through both spellings and the
# verdicts must match. Alternation order does not change the language matched
# by an anchored `(?:...)+$`, but this repo has been bitten by a "behaviour
# preserving" refactor before, so it is checked, not reasoned about.
_ACK_WORDS = rf"{_ACK_MARKERS}|{_ACK_REPAIRS}"

# ACKNOWLEDGEMENT ONLY, and these stay filler even when a yes/no answer is
# what we asked for. "Mm-hm" to "are you accepting new patients?" is an
# affirmative in real speech, and it is ALSO the exact text of our own
# backchannel clips coming back up the line off a speakerphone.
# _BACKCHANNEL_ECHO_MARGIN_S says why that cannot be told apart after the fact:
# "There would be no way to tell our own echo from a real backchannel". Letting
# one of those strings satisfy a field would let the agent answer its own
# question. So the affirmative set below is the EXPLICIT one only.
#
# THE CLIP POOL IS NARROWER THAN THIS LIST AND THE LIST STAYS WIDE ANYWAY.
# This used to read "mm-hm, okay, right, sure", naming the pool as it stood;
# make_backchannels.TOKENS is now the two nasal tokens only (mm-hm, mm) because
# a lexical token can be heard as agreeing to something. The words stay in
# `_ACK_WORDS` regardless — they are what a REAL callee says, and this list is
# about judging their turns, not about what we play. Keep the two questions
# apart: narrowing the pool is a speaking decision, and narrowing this would be
# a listening one.
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

# The repair/greeting/invitation half of _ACK_WORDS, searched rather than
# matched: _only_marked_time is only ever asked about a turn `_is_filler_reply`
# has already judged to be nothing but acknowledgement tokens, so finding one
# of these anywhere in it means the turn IS one.
_ACK_REPAIR_REPLY = re.compile(rf"\b(?:{_ACK_REPAIRS})\b", re.I)


def _only_marked_time(text: str, agent_name: str = "",
                      expects: Optional[frozenset] = None) -> bool:
    """They acknowledged and said nothing else — so say nothing back.

    A STRICT SUBSET of `_is_filler_reply`, and the subset is the whole point.
    That predicate answers "did this turn carry an answer", which is the right
    question for the ask budget and the wrong one for deciding whether to
    SPEAK. It returns True for "Sorry?" and "Are you there?" — turns that carry
    no answer and are exactly the ones that must draw one.

    So this is composed ON it rather than replacing it: whatever it decides
    about substance stands, including the `expects` rule that makes a bare
    "Yeah." an ANSWER to a yes/no field and filler to a location ask. This adds
    one condition — that the acknowledgement is not also a request for speech.
    """
    if not _is_filler_reply(text, agent_name, expects):
        return False
    return not _ACK_REPAIR_REPLY.search(_norm_quotes(text or ""))


def _marking_time_on_an_open_ask(sess: "RealtimeSession", text: str) -> bool:
    """They acknowledged, and the question is still theirs to answer.

    THE TURN-TAKING DEFECT, not a retry defect. On call-20260910-1042 the
    caller said "Okay." at rec 43.96-44.43 — acknowledging a narration — and
    the agent's question began at 44.76, 0.33s LATER. They had not heard it
    when they spoke. OpenAI's server VAD opened a response on that "Okay."
    anyway (`create_response` is unset in build_audio_config, so it runs on the
    API default), the model had an unanswered ask in view, and it put the
    question again 1.00s after the first copy finished playing.

    Nothing re-asked. No threshold was crossed: `_MIN_REASK_GAP_S` is read by
    one detector and gates nothing, and the ask budget only counts. What was
    missing is any answer to "does this turn deserve the floor".

    The three conditions are all necessary. Without the open-ask test this
    would go quiet on an acknowledgement that closes an exchange; without
    `missing()` it would go quiet on a call with nothing left to ask; and
    without `_caller_answered_since` it would ignore the 2026-08-24 rule that
    a bare "Yeah." ANSWERS a closed-set question — which would undo the split
    that stopped every "Yes." reading as silence.
    """
    if sess._last_ask_turn_idx < 0:
        return False
    if not _objective_of(sess).missing(sess.memory):
        return False
    expects = _pending_expectation(sess, sess._last_ask_turn_idx)
    if not _only_marked_time(text, sess.agent_name, expects):
        return False
    return not _caller_answered_since(sess, sess._last_ask_turn_idx)


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
# than none.
#
# NECESSARY, NOT SUFFICIENT. This used to be the whole test, and the comment
# here said the pause "is not observable from the events we get" — true of the
# events and false of the audio, which this process already has. See
# _BACKCHANNEL_PAUSE_MIN_S.
#
# RAISED 2.8 -> 4.5 on 2026-09-03 after call-20260903-1807. Five clips fired on
# that call, all inside the pause window and all technically correct, and the
# callee still heard them as hiccups. THE LOG CANNOT ADJUDICATE IT — the
# transcript line is written when transcription completes, which lags the
# audio, so "the clip printed before their turn" is not evidence it landed at
# the end of it. What the log DOES show is the shape: every one fired after
# 2.9-4.0s of speech, i.e. on short answers, where the caller is delivering one
# fact and a listening noise has nothing to listen to.
#
# 4.5s keeps the feature for what it was built for — the caller explaining at
# length, where silence from us reads as an empty line — and gives up the
# marginal case rather than defending it. Set REALTIME_BACKCHANNELS=false to
# switch it off entirely; that is the zero-risk lever if they still intrude.
#
# 6.5, RAISED FROM 4.5 ON THE CLIENT'S READ. The log showed backchannels after
# 6-8s of speech on short answers ("yes, we do have a waitlist"), which is
# listening-theatre on a turn that did not need it. A backchannel earns its
# place when the receptionist is genuinely explaining — a long hold, a
# waitlist rundown — not on every competent sentence.
#
# ── THE BAR WAS RIGHT AND THE CLOCK WAS WRONG (2026-09-04) ──────────────────
# The sentence above contains its own contradiction and nobody caught it: a
# SHORT ANSWER cannot contain 6-8s of speech. It was never measuring speech.
# The comparison was against `time.time() - _caller_speaking_since`, a mark set
# at `speech_started`, and the server VAD does not close a segment until 400ms
# of silence — so the wall time since it counts every thinking pause inside the
# caller's own turn. "Yes, we do have a waitlist", delivered with two pauses in
# it, clears a 6.5s WALL bar while being exactly the short answer the client
# objected to. Raising the bar could not fix that, and it did something worse:
# it switched the feature off. `backchannels_sent` has been null on every call
# since 2026-09-04 10:13.
#
# Measured on the CALLEE leg of 50 dual-channel Twilio recordings — 858
# utterances, frames at 20ms against the same relative floor the live gate
# uses, segments split on 400ms of silence exactly as the server VAD does:
#
#     wall    p50 1.04  p90 3.94  p95 5.06  p99 6.54   max 8.12
#     voiced  p50 0.78  p90 3.08  p95 3.92  p99 4.86   max 6.44
#
#     >= 6.5s wall   :  10 of 858, 5 of them with a usable interior gap
#     >= 6.5s VOICED :   0 of 858
#
# So on the honest clock the old bar is not strict, it is UNSATISFIABLE — no
# receptionist in the corpus has ever spoken 6.5 seconds.
#
# 3.5s OF VOICED SPEECH sits just above p90: the top tenth of turns by how much
# was actually said, which is what "a medium or longer explanation" means once
# the distribution is consulted instead of guessed at.
#
# CHOSEN OFF A SWEEP, NOT BY FEEL. Replaying the callee leg of all 50
# recordings through these same constants — the real cooldown, the real pause
# window, one clip per utterance:
#
#     voiced bar   clips   per call   1 per     max on one call
#        2.5s       140      2.80      0.6 min        8
#        3.5s        61      1.22      1.4 min        4     <- here
#        4.5s        22      0.44      3.9 min        2
#        6.5s (old)   0         0        never        0
#
# 3.5 rather than 3.0 (which gives 1.84/call, max 5) because this feature has
# drawn a complaint from the client twice and the standing rule is that a clip
# which never fires is the safe failure. It is not set at 4.5, because the
# whole finding here is that the feature had stopped existing and the point is
# to give it back. Roughly one clip per 70 seconds of callee speech, silence
# everywhere else.
#
# THIS BAR IS ON A DIFFERENT CLOCK FROM THE ONE IT REPLACES and the numbers are
# not comparable: 3.0 voiced is a LONGER turn than 4.5 wall was, not a shorter
# one. Anyone tempted to "put it back to 4.5" should read the two rows above
# first. Set REALTIME_BACKCHANNELS=false to switch the feature off entirely;
# that is still the zero-risk lever if clips intrude.
_BACKCHANNEL_AFTER_S = 3.5

# At most one per caller utterance, and never twice inside this window — two in
# quick succession is a tic, not listening.
_BACKCHANNEL_COOLDOWN_S = 9.0


# ── The gap the clip has to land in ──────────────────────────────────────────
# THE SECOND HALF OF THE TEST, and the elapsed timer above is worthless without
# it. On call-20260903-1422 all three clips fired at 3.0-3.2s into the caller's
# turn — the timer expiring — with no question asked about whether she was
# between words or mid-syllable. Two of the three landed inside one. A "mm-hm"
# through the middle of a word is not listening, it is an interruption, and it
# is the thing a person never does.
#
# So: they must have been speaking long enough (above) AND be in a gap right
# now. A caller who runs on without drawing breath gets no clip at all until
# their first natural lull, which is the correct behaviour and not a
# degradation — the alternative is talking over them.
#
# LOWER BOUND — long enough to be a breath or a comma rather than the closure
# before a plosive. Stops and affricates ("t", "k", "p") carry 40-80ms of near
# silence inside a word; 150ms is clear of all of them.
#
# RAISED 0.15 -> 0.25 on 2026-09-03. 150ms clears a plosive closure, which is
# what it was chosen for, but it does NOT clear an ordinary gap between words —
# so a caller in full flow still presented gaps this wide and got a clip
# through the middle of their sentence. The callee's report was exactly that:
# "it should come when the speaker gives a pause, not when they are
# continuously speaking." 250ms is a pause someone is taking, not a gap between
# two words, and it narrows the window to [0.25, 0.32] against the VAD's 400ms
# — deliberately tight, because a clip that never fires is the safe failure and
# one that lands mid-word is not.
_BACKCHANNEL_PAUSE_MIN_S = 0.25

# UPPER BOUND — AND THIS ONE IS DERIVED, not chosen. OpenAI's server VAD ends
# the turn after `realtime_silence_ms` of silence. That was 400ms until
# 2026-09-07 and is now 700ms: measured off the caller channel of four calls,
# 34% of their WITHIN-TURN pauses ran longer than 400ms, so a third of the
# time a pause to think was read as the end of their turn and the agent spoke
# over them. At 700ms that is 20%. The `vad->resp` stage reads as this figure
# plus transport.
# A gap longer than that is not a pause inside a turn, it is the END of the
# turn: the caller has finished, a real response is already being created, and
# our "mm-hm" would play into the reply rather than under their speech.
#
# Held one notch below the VAD's own figure rather than at it. Our clock reads
# frames as they arrive over Twilio and OpenAI reads its own copy, so the two
# measurements of the same gap differ by the transport jitter between them;
# spending that margin here buys nothing and risks landing on the boundary.
_BACKCHANNEL_PAUSE_MAX_S = max(
    _BACKCHANNEL_PAUSE_MIN_S + 0.05,
    settings.realtime_silence_ms / 1000.0 - 0.08)


# How often the silence watchdog looks while a clip is pending.
#
# THE POLL PERIOD IS PART OF THE GATE. With the pause window at [0.25, 0.62]
# (0.07s wide, after the 0.15->0.25 raise), a half-second tick would miss every
# gap, and the old 100ms tick still only lands inside roughly two-thirds of
# them. 50ms puts a look inside every qualifying gap.
#
# NARROWED TO THE WINDOW WHERE IT CAN MATTER, deliberately, rather than
# speeding the loop up for the whole call. It applies only while the caller is
# mid-utterance and a clip is still owed for it — bounded by
# _backchannel_done_this_utterance and the cooldown — so nothing else in the
# watchdog runs materially more often than it does today. Every other check in
# that loop is a threshold on elapsed time or a one-shot flag, so a faster tick
# cannot change what any of them decides, only when it notices.
_BACKCHANNEL_TICK_S = 0.05

# ── How many times one cadence directive may go out ─────────────────────────
# ONE-SHOT WAS WRONG, AND call-20260904-1734 IS THE EVIDENCE. Six of that
# call's nine agent turns were the announcement cadence — "Got it, thanks for
# confirming, let me just ask a quick question about where he sees patients."
# Five of the six were DETECTED and only THREE directives went out, because
# each guard below was a single boolean that latched on its first firing. The
# rest were written to the artifact and never mentioned to the model, which is
# the [[detector is not an enforcement]] failure arriving from the other side:
# here the enforcement existed and disarmed itself while the defect continued.
#
# TWO CAPS, ANSWERING DIFFERENT QUESTIONS.
#
# Per kind — "has this correction stopped working". check_cadence_loop measures
# roughly two good corrections in three, so a second attempt is worth having;
# a third identical instruction is the model ignoring it, and repeating it a
# third time only spends context.
#
# Total — "how much of this call may be us talking to ourselves". Each
# directive enters the conversation as a turn the model reads, so on a nine
# turn call four is already a quarter of the transcript. ON THE WORST CALLS
# THIS BINDS FIRST AND THAT IS DELIBERATE: a call that trips five different
# cadence guards is not going to be rescued by a fifth note, and the artifact
# still records every occurrence whether or not a directive was sent.
_CADENCE_NUDGE_PER_KIND = 2
_CADENCE_NUDGE_TOTAL = 4

_WATCHDOG_TICK_S = 0.5


def _watchdog_tick_s(sess: "RealtimeSession") -> float:
    """How long the watchdog may sleep before it has to look again.

    A function rather than a constant because the answer depends on whether a
    backchannel is currently pending: see _BACKCHANNEL_TICK_S for why the poll
    period is part of the gate rather than an implementation detail of the
    loop.
    """
    if (settings.realtime_backchannels
            and sess._caller_speaking_since is not None
            and not sess._backchannel_done_this_utterance
            and not sess.done):
        return _BACKCHANNEL_TICK_S
    return _WATCHDOG_TICK_S


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
    # THE WATCHDOG'S OWN EXIT. It used to prompt _MAX_SILENCE_PROMPTS times and
    # then `continue` forever, leaving the call open on a line nobody was on —
    # the escalation was carried only as a prompt rule ("Silence -> ... If it
    # continues, escalate"), which is a rule the model has to remember from
    # 6,000 tokens back at the one moment it has stopped being spoken to.
    "no_response": "caller stopped responding and did not come back",
    # Not a budget outcome. The caller drew a line, and the record has to say
    # that rather than "engaged but never provided a location", which reads as
    # a call that might work next time.
    "refused": "caller refused to give the information",
}

def _field_vocabulary(field) -> Optional[Callable]:
    """The classifier that reads THIS field's answers, or None.

    KEYED ON THE FIELD'S OWN `states`, which is a fact the template already
    declares, rather than on a second table of field names to keep in step with
    it. A field whose states are not one of the three declared sets gets None
    and is simply not checked — silence, not a guess.

    Matching a field with another field's vocabulary produces a check that
    either can never fire or fires on the wrong answer. That is not
    hypothetical: objectives.states_in_its_own_right defaulted to
    classify_choice and read "No, not at the moment." — an answer to the
    SCHEDULING question — as a referral answer, because classify_choice
    recognises it and classify_referral does not.
    """
    from agents.voice.objectives import (CHOICE_STATES, IDENTITY_STATES,
                                         REFERRAL_STATES, classify_choice,
                                         classify_identity, classify_referral)
    states = getattr(field, "states", None) or frozenset()
    if states == IDENTITY_STATES:
        return classify_identity
    if states == REFERRAL_STATES:
        return classify_referral
    if states == CHOICE_STATES:
        return classify_choice
    return None

def _volunteered_fields(sess: "RealtimeSession", text: str) -> list:
    """Fields this caller turn answers that nobody has asked for yet.

    THE PROMPT RULE THIS REPLACES WAS NEVER WRITTEN, and that is deliberate.
    "If they give you an answer to a question you haven't asked, save it" is
    exactly the shape of rule this project has repeatedly watched the model
    ignore — the recovery directive on call-20260827-1130 said "say just that,
    in one short sentence, do not apologise" and was disobeyed four times
    inside one call. A guard that fires on the event costs no prompt tokens and
    does not depend on the model remembering anything.

    TWO CONDITIONS, AND THE SECOND ONE IS THE WHOLE SAFETY OF IT.

      1. the field's own vocabulary reads a state out of the turn, and
      2. the caller's words NAME THE TOPIC — `field.probe` matches.

    Condition 1 alone is a fabrication engine. Every CHOICE field shares
    `classify_choice`, so a bare "Yes." answering the branch question would be
    read as "they volunteered that the doctor is accepting new patients", and
    the directive below would then tell the model to record it. That is the
    same defect `_field_already_answered` had until 2026-08-27, arriving from
    the other end — and there it merely mis-fired a nudge, where here it would
    put a value in the record the caller never gave.

    So the turn has to be ABOUT the field, in the caller's own words. It is the
    same anchoring `_ungrounded_status` applies to its evidence window: "when
    there was no ask to anchor to, the turn must additionally be ABOUT new
    patients".

    ONLY FIELDS NOBODY HAS ASKED FOR. Once a field has been asked, the ordinary
    path owns it — the ask budget, the re-ask guard and the save gate all key
    off that ask, and a second opinion from here would double-count.
    """
    out = []
    if sess.done or not (text or "").strip() or text.strip() == "[...]":
        return out
    for f in _objective_of(sess).fields:
        if f.name in sess._volunteered_seen:
            continue                       # one directive per field per call
        if sess._field_ask_at.get(f.name) is not None:
            continue                       # asked; the ordinary path owns it
        try:
            if f.present(sess.memory):
                continue                   # already collected
        except Exception:
            pass
        classify = _field_vocabulary(f)
        probe = getattr(f, "probe", None)
        if classify is None or probe is None:
            continue
        if not probe.search(text):
            continue                       # not about this field — see above
        got = classify(text)
        if got is not None and got.value in (getattr(f, "states", None)
                                             or frozenset()):
            out.append((f, got.value))
    return out


def _uptake_only(text: str, sess: "RealtimeSession") -> bool:
    """Strip the bare affirmative — is everything left a question?

    THE DISCRIMINATOR FOR "Yes, before that, can you confirm your full name and
    your date of birth?". `classify_choice` takes the leading "Yes" and calls
    it an answer; every word after it is a REQUEST, and the field it was asked
    about is left exactly as unknown as before the caller spoke.

    `_caller_is_vetting` was the first thing tried here and it is too blunt:
    it is True for any turn ending in a question, and the corpus's single most
    common identity confirmation is

        "Yes, this is Dr. Brown's office. How can I help you?"

    — an answer with a courtesy question stapled on. Using it would have
    discarded ~40 of those. `states_in_its_own_right` does not rescue them
    either: the identity classifier returns None for "this is Dr. Brown's
    office" standing alone, so that test is structurally False for EVERY
    identity confirmation, however explicit.

    ASSERTION IS SENTENCE-LEVEL — the same conclusion _asserted_caller_text
    reached, for the same reason, one layer down. A turn is uptake only when NO
    sentence in it asserts anything once the affirmative is gone.

    A HOLD IS SENTENCE-LEVEL FOR THE SAME REASON. "Let me check" asserts, but
    what it asserts is that the answer is still coming, so it is dropped here
    with the questions. Testing the whole turn for a hold instead — which is
    what the first cut did — discarded

        "Yeah, she works out for Northgate Clinic. Let me pull up that file.
         What is your full name?"

    whole, identity confirmation and all, because a LATER sentence held. Per
    sentence, the first one survives and the turn is an answer.

    Returns False for a bare "Yeah.": nothing survives the strip, and that case
    belongs to the 2026-08-24 rule that made a bare affirmative in direct reply
    an answer. This predicate does not reopen it.

    NEEDS A SESSION `_caller_is_vetting` CAN READ — `sess.doctor`,
    `sess.org_name` and `sess.agent_name`, via _turn_asserts. That is a
    stricter requirement than the rest of
    _field_already_answered, which reads its namespaces through getattr, so a
    test double handed to that guard must now carry a doctor or the guard
    raises instead of judging.
    """
    from agents.voice.evidence.window import _turn_asserts
    from agents.voice.objectives import without_bare_affirmative
    stripped = without_bare_affirmative(text)
    if not stripped:
        return False
    return not any(_turn_asserts(s, sess) and not is_hold_request(s)
                   for s in _sentences(stripped))


def _field_already_answered(sess: "RealtimeSession", field,
                            since_idx: int) -> str:
    """A caller turn since `since_idx` that reads as an answer to `field`.

    Returns the turn's text, or "" if they have not answered it. The point is
    the question NOTHING ELSE IN THIS FILE ASKS: not "was that re-ask too fast"
    (_MIN_REASK_GAP_S) and not "have they stopped answering altogether" (the
    ask budget), but "did they already answer THIS".

    call-20260825-1847 is the whole case for it. The caller said "Yes?" to the
    identity question; grounding refused it on the transcriber's punctuation
    (see _turn_asserts, now fixed); the model was told to get their words and
    asked the identical question again. The gap was 19s so the speed guard
    stood down, the caller had answered so the budget stood down, and the
    identity clause was invisible to the phrasing guard. Three guards, none of
    them asking whether the answer was already on the call.

    DELIBERATELY INDEPENDENT OF WHETHER THE SAVE SUCCEEDED. Reading memory
    instead would miss exactly this case — identity was not in memory, because
    the guard had just rejected it. What matters is that the CALLER did their
    part, which is a fact about the transcript and not about our bookkeeping.

    Read with the FIELD'S OWN vocabulary — see _field_vocabulary.

    AND ONLY WHILE THIS FIELD STILL HOLDS THE FLOOR. The window opens at OUR
    ask and runs to now, and on a multi-field call other questions get asked
    inside it. Vocabulary alone cannot tell those answers apart: `classify_*`
    reads a bare "No." and every field with CHOICE_STATES claims it, so the
    answer to whichever question was actually on the table is attributed to all
    of them.

    call-20260827-1010 is the case. The window for `accepting` opened on a turn
    that asked nothing (see _ANNOUNCES_ASK, now fixed); inside it the agent
    asked twice for the street address and the caller said "No, I don't have
    it."; and when the new-patient question was finally put — for the FIRST
    time — this scan handed that sentence back as proof they had already
    answered it. The nudge that fires on the result does not merely note the
    re-ask: it tells the model `they said X, take that as their answer, record
    it`. A guard that instructs the model to file a value the caller never gave
    for that field is the fabrication every other guard in this file exists to
    stop, arriving from inside.

    So the scan tracks WHOSE QUESTION IS OUTSTANDING. Our ask opens the floor;
    a later agent ask for some other field takes it; a later ask that includes
    ours takes it back. A caller turn counts only while we hold it. This is the
    same anchoring `_ungrounded_status` uses on the evidence window, applied to
    the other side of the same question, and it is deliberately the
    false-NEGATIVE direction: a missed re-ask costs one clumsy turn, and a
    false one puts words in the caller's mouth.
    """
    classify = _field_vocabulary(field)
    if classify is None:
        return ""
    states = getattr(field, "states", None) or frozenset()
    others = [f for f in _objective_of(sess).fields
              if f.name != getattr(field, "name", None)]
    # getattr, for the same reason _objective_of uses it: these guards are
    # handed namespaces carrying only the attributes they read, and one that
    # raises on a test double is one that stops being tested.
    def _asks(txt: str, f) -> bool:
        p = getattr(f, "probe", None)
        return p is not None and _is_ask_for(txt, p)

    ours = True                    # turns[since_idx] is OUR ask; it opens it
    # DOES THIS TURN DIRECTLY ANSWER OUR ASK, or merely arrive while we still
    # hold the floor? `ours` cannot tell those apart, and on call-20260907-1602
    # the difference is the whole defect — see the bare-affirmative gate below.
    replying = True                # turns[since_idx] is our ask; the next
                                   # caller turn is a reply to it
    for _i, t in enumerate(sess.turns[max(0, since_idx):]):
        text = t.text or ""
        if t.role == "agent":
            # Ours first: one turn can ask for several fields at once, and a
            # turn that asks for ours keeps the floor whatever else it names.
            if _asks(text, field):
                ours = True
                replying = True
            elif any(_asks(text, f) for f in others):
                ours = False
                replying = False
            elif _i:
                # WE SPOKE AND ASKED NOTHING. The floor is still ours, but what
                # they say next is a reply to THAT, not to a question upstream.
                #
                # `_i` GUARDS IT because turns[since_idx] IS our ask by
                # construction — the caller reached this function through it.
                # Clearing on that turn made the answer spoken straight back at
                # us unreachable whenever `_asks` could not confirm the ask,
                # which is every field carrying no probe.
                replying = False
            continue
        if t.role != "caller" or not ours or text.strip() == "[...]":
            continue
        got = classify(text)
        if got is None or got.value not in states:
            continue
        # THE SAME EVIDENCE DECISION THE SAVE GUARD MAKES, and the reason is
        # call-20260907-1602. The agent re-asked the accepting question at
        # 16:02:54; this scan handed back the caller's bare "Yeah." — spoken to
        # acknowledge our NAME three turns earlier — and the nudge told the
        # model `they already answered it, they said 'Yeah.'`. The model did as
        # it was told and called save_new_patient_status(heard='Yeah.').
        # _ungrounded_choice refused it, correctly; and worse, that same
        # re-asking turn had just moved the evidence anchor PAST "Yeah.", so
        # the window it was judged in no longer contained it.
        #
        # The model was left holding two of our own statements that cannot both
        # be true: "they answered, it was 'Yeah.'" and "'Yeah.' is not an
        # answer". It retried the identical arguments three times across four
        # responses that emitted NO AUDIO AT ALL, while a direct question from
        # the receptionist went unanswered for 22 seconds and they said
        # "Hello, are you there?".
        #
        # states_in_its_own_right is the save guard's own per-turn test, so
        # this can no longer claim a turn that guard would reject on content.
        # Applied here unconditionally where the guard applies it only on the
        # never-asked path — deliberately STRICTER, in the direction this
        # docstring already argues for: a missed re-ask costs one clumsy turn,
        # a false one puts words in the caller's mouth.
        # ── UPTAKE IS NOT AN ANSWER, EVEN WHEN IT IS A DIRECT REPLY ─────
        # `not replying` was the whole condition, and call-20260910-2032 is
        # what it misses. The agent asked "are you taking new patients right
        # now?"; the receptionist replied
        #
        #     "Yes, before that, can you confirm your full name and your
        #      date of birth?"
        #
        # — a direct reply, so `replying` was True and the gate below was
        # skipped. classify_choice took the leading "Yes" and this handed the
        # counter-request back as the availability answer. Two things then
        # acted on it: the re-ask nudge told the model `they already answered
        # it, they said "Yes, before that, can you confirm your full name"`,
        # and _steer_went_stale withdrew the steer with `still: null`, removing
        # the only instruction pointing at the field nobody had answered.
        #
        # A REPLY THAT IS ONLY UPTAKE ANSWERS NOTHING — see _uptake_only for
        # why this is asserted per sentence and not `_caller_is_vetting`, which
        # would have taken ~40 real identity confirmations with it.
        #
        # STILL SUBORDINATE TO states_in_its_own_right, so a turn that states
        # the answer in its own words survives however it is wrapped. The bare
        # "Yeah." is untouched: nothing survives the affirmative strip, so
        # _uptake_only is False and the 2026-08-24 rule still owns that case.
        # A HOLD IS UPTAKE TOO — folded into _uptake_only, per sentence.
        # Task 1 taught _HOLD_REQUEST the present
        # continuous and vetoed "Yeah, I'm, I'm checking wait, yeah." at the
        # SAVE guard — but the save guard is one of three paths that ask
        # whether the caller answered, and this one had no hold test at all.
        # Left alone it would tell the model `they already answered it, they
        # said "Yeah, I'm, I'm checking"` about someone still mid-lookup, which
        # is how call-20260907-1602 began: a nudge and a guard contradicting
        # each other, three refused saves, four silent responses.
        #
        # Same subordination as above — "Let me check... yes, we are taking
        # them" states the answer on its own and is kept on content.
        if ((not replying or _uptake_only(text, sess))
                and not states_in_its_own_right(text, got.value, classify)):
            # A BARE AFFIRMATIVE THAT IS NOT A REPLY TO OUR ASK. On
            # call-20260907-1602 the agent asked the accepting question, the
            # receptionist asked for a name, the agent GAVE the name, and they
            # said "Yeah." — acknowledging the name. `ours` was still True,
            # because answering a question asks for nothing, so this scan
            # handed "Yeah." back and the nudge told the model `take that as
            # their answer, record it`. It did: save_new_patient_status(
            # heard='Yeah.'), refused three times across four responses that
            # emitted no audio, while the receptionist asked twice whether
            # anyone was there.
            #
            # states_in_its_own_right is the save guard's own per-turn test,
            # applied ONLY where position has stopped vouching for the turn.
            # Applying it unconditionally was the first cut and it is wrong:
            # it also refuses "Yes?" spoken straight back at the identity
            # question, which is call-20260825-1847 — a real answer, and the
            # case this whole function was written for.
            replying = False
            continue
        replying = False
        return text.strip()
    return ""

def _steer_went_stale(sess: "RealtimeSession") -> tuple:
    """The standing steer named a field the caller has since answered.

    THE DEFECT THIS EXISTS FOR, traced end to end on call-20260910-1534. At
    15:34:38 a save injected "what is still missing is which office they're
    at". At 15:34:40 the agent asked it. At 15:34:50 the caller said "He works
    at our Northgate location." At 15:34:51 the agent said:

        "Alright, let me think about what to ask next so I can be sure about
         availability."

    The steer was satisfied and still standing, and its replacement is
    manufactured by the save INSIDE the response being generated — so at the
    moment of generation the model held one imperative it must not obey and
    nothing else. An instruction to act can be performed or reported; when
    performing is unavailable, reporting is the only move left. Measured over
    664 non-greeting turns, 21.4% carry some form of that report.

    ASKED OF THE TRANSCRIPT, NOT OF MEMORY. The save that would put this in
    memory has not run yet — it runs inside the response this is trying to get
    ahead of. What matters is that the CALLER did their part.

    ANCHORED ON `_field_ask_at`, AND ANSWERED BY `_caller_answered_since`, and
    the first cut of this used `_field_already_answered` instead and could not
    see the case it was written for. That predicate reads a field with its OWN
    vocabulary, keyed on `field.states` — so a CHOICE field is visible to it
    and a PLACE field, which declares no states, returns "silence, not a
    guess". The branch is a PLACE field, and the branch is exactly what
    call-20260910-1534 answered. The pair used here is the ask budget's own,
    and it is kind-agnostic: it asks whether the caller said anything
    substantive against the expectation the ask set up, which is True for "He
    works at our Northgate location." and False for "Okay."

    NOT ASKED YET -> NOT STALE. A steer whose field nobody has put to them is
    still an instruction the model can obey, and withdrawing it would remove
    the only thing telling it what the turn is for.

    Returns (field_they_answered, what_is_still_unknown). The second is "" when
    nothing is left, and the caller site does not speak then: a finished
    objective belongs to the close, not to a state note.
    """
    _name = getattr(sess, "_steer_field", "")
    if not _name:
        return None, ""
    _obj = _objective_of(sess)
    _fld = next((f for f in _obj.fields if f.name == _name), None)
    if _fld is None:
        return None, ""
    _asked_at = sess._field_ask_at.get(_name)
    if _asked_at is None:
        return None, ""
    # ── FIELD-AWARE FIRST, AND THE BUDGET'S PREDICATE ONLY AS A FALLBACK ────
    # This used `_caller_answered_since` alone, and on call-20260910-2032 that
    # cost the field the only instruction pointing at it. That predicate is the
    # ASK BUDGET's, and the budget's question is "are they still engaging" —
    # deliberately loose, because a caller who answers with a question is
    # engaging. Borrowed here it answered a different question, "did they
    # answer THIS", and a counter-request came back as yes.
    #
    # So: where the field declares a vocabulary, ask the field-aware predicate,
    # which now rejects a vetting turn that states nothing on its own. Where it
    # does not — a PLACE field declares no states, which is why the first cut
    # of this reached for the budget at all — keep the budget's predicate but
    # apply the SAME uptake exclusion, per caller turn. `_caller_vetted_since`
    # was the obvious reach here and is the wrong tool for the same reason it
    # is wrong above: "Yes, this is Dr. Brown's office. How can I help you?"
    # vets, and it is an answer. One turn that asserts something is enough.
    if _field_vocabulary(_fld) is not None:
        if not _field_already_answered(sess, _fld, _asked_at):
            return None, ""
    elif not _caller_answered_since(sess, _asked_at) or not any(
            t.role == "caller" and (t.text or "").strip() != "[...]"
            and not _uptake_only(t.text or "", sess)
            for t in sess.turns[max(0, _asked_at):]):
        return None, ""
    _still = next((f.label for f in _obj.fields
                   if f.name != _name and f.is_required(_obj, sess.memory)
                   and not f.present(sess.memory)), "")
    return _fld, _still


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
    if trigger == "no_response":
        opening = ("(system: the line has gone quiet and they have not come "
                   "back after being checked on twice. ")
    elif trigger == "refused":
        opening = ("(system: they have just refused to give you the "
                   "information, or asked not to be called. That is final and "
                   "asking again in any wording is badgering them. ")
    elif trigger == "unanswered":
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

def _norm_clause(text: str) -> str:
    """Normalised form for equality: case, quotes, whitespace, edge punctuation.

    "...working out of?" and "...working out of." are the same thing said
    twice, and a detector that treats them as different is measuring
    punctuation rather than repetition.
    """
    t = _norm_quotes(text).lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t.strip(".,!?;:—–- ")

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
# "scheduling" joined these on 2026-08-26 for the same reason the others are
# here: it is the capitalised head of "Scheduling words:" in the retired text,
# and _FABRICATION_VOCAB is built from capitalisation. Without it, a caller
# saying "scheduling" once could be condemned as a fabrication marker — the
# widening this set exists to prevent, caught by its own test.
_HINT_HEADINGS = frozenset({"phone", "health", "location", "call", "systems",
                            "words", "scheduling"})

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
    "Trinity Health, Northwell, NewYork-Presbyterian, Cedars-Sinai. "
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
    hw = [w for w in re.findall(
        r"[a-z]+", ((hint or "") + " " + _RETIRED_HINT_TEXT
                    + " " + _RETIRED_VOCAB_TEXT).lower())]
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


_HOLD_GRACE_S = 45.0


# ── When the hold is over ────────────────────────────────────────────────────
# `_hold_until` was a DURATION, never a state. Its original reader is the
# silence watchdog, which only ever needed "do not badger for a while", and a
# fixed timer answers that. _next_move_item then borrowed the same field to ask
# a STATE question — is a hold in flight right now — and nothing had ever
# cleared it, because nothing had ever needed it cleared.
#
# call-20260908-1114: they asked for a minute at 11:15:45, CAME BACK AND
# ANSWERED at 11:16:10, and the save at 11:16:12 still read as mid-hold — 27s
# into a 45s grace. The steer was withheld under a premise that had stopped
# being true two seconds earlier, and the model, handed a turn with nothing to
# fill it with, filled it with "Okay, thanks for checking that."
#
# TWO SIGNALS, AND THE QUESTION MARK IS THE ONE THAT MATTERS. Word count alone
# put "Sorry, still waiting on the system." and "I'm just checking with my
# colleague." (4 content words each, and is_hold_request catches neither) on
# the same side as "Okay, and your date of birth?" — which is the receptionist
# BACK and working through intake, and appears three times in the corpus.
# Someone still reading a screen does not ask you a question, so a question is
# a return of attention whatever its length; anything else has to carry real
# content to count.
#
# FAILING SAFE MEANS KEEPING THE HOLD. A missed clear leaves exactly the
# behaviour that shipped; a wrong clear invents a new one — the badgering the
# grace exists to prevent. Measured over the corpus: of 41 caller turns
# following a hold request, this clears 31 and keeps 10, and the 10 are every
# pure filler in the set ("Mm.", "Oh.", "Ok.", "Got it. Thank you.").
#
# The 45s expiry stays as the fallback: a caller who never says anything
# substantive again still comes off hold on the timer.
_HOLD_RESUME_MIN_WORDS = 5


def _caller_resumed(text: str) -> bool:
    """Has the caller come back from a hold with something to say?

    NOT a judgement about whether they answered the question — only about
    whether they are talking to us again. is_hold_request is checked by the
    call site first, so a fresh "one moment" extends the hold rather than
    reaching this.
    """
    t = (text or "").strip()
    if not t:
        return False
    if "?" in t:
        return True
    _words = re.sub(r"[^a-z0-9 ]", " ", t.lower()).split()
    return len([w for w in _words
                if len(w) > 2 and w not in _UNGROUNDED_STOPWORDS]
               ) >= _HOLD_RESUME_MIN_WORDS


_MAX_SILENCE_PROMPTS = 2


# How long a close deferred as "spoken" waits for a turn that may never come.
#
# THE OTHER END OF THE HOLD. _decide_close stops the goodbye stacking on the
# agent's own turn by waiting for the caller; if the caller simply says nothing
# back, that wait has to end somewhere or the change trades one closing defect
# for a longer one.
#
# NOT _SILENCE_PROMPT_AFTER, and shorter than it deliberately, because the two
# silences are not the same silence. Seven seconds is thinking room for someone
# who has been ASKED something. Nobody has been asked anything here — the
# objective is complete, we have answered them, and the only thing left is to
# say goodbye. So this fires first, and what it fires is the close rather than
# "are you still with me?", which is the wrong sentence about a call that is
# already over. Same correction the padding chase made against this watchdog's
# other silence, one guard over.
#
# ONLY for the "spoken" deferral. "ours" and "theirs" both have a question on
# the table and somebody who owes a turn, which is exactly the silence the
# prompt budget below is right about.
_DEFERRED_CLOSE_WAIT_S = 4.5

# How long a promised question may go unasked before the line is chased.
#
# WELL UNDER _SILENCE_PROMPT_AFTER, and that gap is the point. The silence
# watchdog waits seven seconds because someone who has just been asked
# something is usually thinking. Nobody is thinking here: they were told a
# question was coming and given nothing to answer, so every one of those seven
# seconds is dead air with no one on either end of it. Measured from the moment
# the agent's audio finished PLAYING (see _agent_quiet_since), not from
# response.done, so this is 2.5s of real silence on the wire.
_PADDING_PROMPT_AFTER = 2.5

# ── AND THE FAST PATH, WHICH IS THE ORDINARY CASE ───────────────────────────
# 2.5s of confirmed silence is evidence we do not need. The TRANSCRIPT has
# already said the agent promised a question and asked none; waiting proves
# only that the caller has not yet filled the gap — and on call-20260908-1359
# they filled it twice, with "Okay." at 14:00:10 and again at 14:00:30, turning
# a two-turn repair into a three-turn exchange and billing 8.17s to one reply.
#
# WHAT THE WAIT IS ACTUALLY BUYING, and why it cannot go to zero:
#   * NOT protection from our own audio. `_agent_quiet_since` is set to
#     time.time() + _playback_remaining — a moment in the FUTURE — so the
#     comparison is already measured from the end of PLAYBACK and any
#     non-negative threshold clears it.
#   * NOT protection from a caller who is talking. `speech_started` sets
#     `_agent_quiet_since = None` and the loop skips the branch entirely.
#   * IT IS the lag before a caller who HAS started speaking is reported to
#     us. Measured over 684 turns of corpus: median 0.428s, p90 0.476s. One
#     detector-lag is the honest floor, and the watchdog ticks at 0.5s anyway.
#
# THE CASE THAT USED TO NEED THE LONG WAIT IS CLOSED UPSTREAM, and it is worth
# writing down because it is the reason to be careful here. On
# call-20260902-2207 `_announced_an_ask` read the taught close ("I'll sort out
# my schedule and think it over") as a promised question, and the only thing
# that stopped a chase demanding a question after the sign-off was the caller
# speaking at 1.3s — inside 2.5s. The suite still records that near-miss.
#
# It cannot recur, and NOT because of this threshold: `_announced_an_ask`
# returns False whenever `_SPOKEN_FAREWELL` matches, so the two are mutually
# exclusive by construction and a goodbye never arms the debt at all. A
# farewell branch here would be dead code — checked, not assumed. If that
# exemption is ever removed, this window is not what will save it; restore the
# check there.
_PADDING_SETTLE_S = 0.5

# Bounded for the reason every recovery here is bounded: the response this
# creates can be padded exactly the way the one that caused the debt was, and
# an unbounded chase for a question the model will not ask is a livelock. Two,
# then the artifact records that the caller never got it.
_MAX_PADDING_NUDGES = 2

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
        await asyncio.sleep(_watchdog_tick_s(sess))

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
                # VOICED seconds inside this utterance, not wall seconds since
                # it opened. See _BACKCHANNEL_AFTER_S: the wall clock counts
                # the caller's own thinking pauses, so a short answer told
                # slowly used to clear the bar and a real explanation delivered
                # briskly did not.
                and sess._utterance_voiced_s >= _BACKCHANNEL_AFTER_S
                and time.time() - sess._last_backchannel_at >= _BACKCHANNEL_COOLDOWN_S):
            # ── ARE THEY BETWEEN WORDS RIGHT NOW ────────────────────────────
            # Everything above says a clip is DUE. This says whether it may go
            # out yet, and it is checked here — immediately before the send —
            # rather than folded into the condition above so the two questions
            # stay separable in the log: held is not the same as not due, and
            # only one of them is a number worth counting.
            #
            # None means no clean inbound frame has been measured, which is not
            # a gap. See caller_lull_s: an unmeasured line must not read as a
            # quiet one.
            #
            # HELD, NOT SPENT, and NOT `continue`. Nothing is marked done and
            # nothing is charged to the cooldown, so the next tick tries again
            # — the clip stays owed for this utterance until they draw breath
            # or the utterance ends. Skipping the rest of the loop body would
            # be a different change entirely: this tick runs at 100ms while
            # they speak, and the claimed-done, sign-off, owed-substance and
            # goodbye-retry checks below all have to keep running through it.
            _lull = sess.caller_lull_s()
            _in_gap = (_lull is not None
                       and _BACKCHANNEL_PAUSE_MIN_S <= _lull
                       <= _BACKCHANNEL_PAUSE_MAX_S)
            if not _in_gap:
                sess._backchannels_held += 1
            _payload = (backchannel.pick(settings.realtime_voice,
                                         exclude=sess._last_backchannel_clip)
                        if _in_gap else None)
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
                    # THROUGH THE MIXER WHEN IT OWNS THE WIRE. The clip itself
                    # is unchanged — same bytes, same length, same echo window
                    # above — but Twilio queues rather than mixes, so a clip
                    # pushed around the pump would play with the bed cut out
                    # from under it for its whole duration. That is the
                    # ambience-ON/OFF/ON artefact, arriving on the one sound
                    # that is supposed to say "a person is still here".
                    if sess.ambience is not None:
                        sess.ambience.push_voice(
                            _mulaw_decode(base64.b64decode(_payload)))
                    else:
                        await twilio_ws.send_text(json.dumps({
                            "event": "media", "streamSid": sess.stream_sid,
                            "media": {"payload": _payload},
                        }))
                    print(f"[Realtime] 👂 backchannel in the gap after "
                          f"{time.time() - _spk:.1f}s of speech "
                          f"({(_lull or 0.0) * 1000:.0f}ms since their last word"
                          + (f", held {sess._backchannels_held}x"
                             if sess._backchannels_held else "") + ")",
                          flush=True)
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

        # Deferred sign-off check. Same 1.5s wait as the claim check above,
        # for the same reason: any tool call belonging to that response has
        # landed by now, so sess.done means what it says.
        #
        # A PENDING CLOSE COUNTS AS AN ENDING. _close_after_response and
        # _close_when_answered are the two ways the objective path says "this
        # call is ending, just not in this event" — the deferred save sets the
        # first and the unanswered-question branch sets the second. Neither sets
        # sess.done yet, and nudging into either would ask for an escalate on a
        # call that is closing correctly.
        if (sess._farewell_at and not sess._farewell_nudged
                and not sess.done
                and not sess._close_after_response
                and not sess._close_when_answered
                and time.time() - sess._farewell_at >= 1.5):
            _said = sess._farewell_said
            sess._farewell_at = 0.0
            sess._farewell_said = ""
            sess._farewell_nudged = True
            sess.farewell_without_close.append(_said[:160])
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"👋 SIGNED OFF WITH NOTHING RECORDED — no tool has ended "
                  f"this call; asking for escalate", flush=True)
            await oai_ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": (
                             "(system: you just signed off, but nothing "
                             "has ended this call — you have not called a "
                             "tool. Call escalate now with the true reason "
                             "this call is ending. Do not say goodbye "
                             "again and do not start a new topic.)")}]},
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
            # THE EXIT. Checked here as well as where the debt is taken on,
            # because a debt already standing when the cap is reached would
            # otherwise be retried on every tick for the rest of the call.
            _stop = _owed_refusal(sess, _owed)
            if _stop:
                sess.owed_abandoned.append({"text": _owed, "why": _stop})
                sess._owed_substance = ""
                sess._owed_directive_sent = False
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"⚠️  OWED SUBSTANCE ABANDONED ({_stop}) — the caller "
                      f"never heard: {_owed[:60]!r}", flush=True)
                continue
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
            if await _create_response(oai_ws, sess, why="owed substance",
                                      allow_when_vad_pending=True):
                # Counted on the ATTEMPT, not on the recovery landing — the
                # whole failure is that the attempt can be muted exactly like
                # the turn that created the debt, and an attempt that is never
                # heard is still an attempt spent.
                _k = _owed_key(_owed)
                sess._owed_attempts[_k] = sess._owed_attempts.get(_k, 0) + 1
                sess._owed_tried += 1
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

        # ── PROMISED A QUESTION, ASKED NOTHING ──────────────────────────────
        # Tool-call padding: "let me just ask one more thing", then a tool
        # call, then nothing. The caller is not thinking about an answer
        # because they were not given a question, so the seven-second silence
        # budget below is measuring the wrong kind of quiet — see
        # _PADDING_PROMPT_AFTER and _announced_an_ask.
        #
        # BEFORE the silence prompt, deliberately. Both fire on the same
        # `quiet_since`, and "are you still with me?" is the wrong sentence
        # here for the same reason it was wrong on call-20260820-1421: the line
        # is not the problem, the missing question is.
        # A PENDING CLOSE STANDS THIS DOWN, exactly as it stands down the
        # sign-off nudge beside it. _close_after_response and
        # _close_when_answered are the two ways the objective path says "this
        # call is ending, just not in this event"; neither has set sess.done
        # yet, so the check above cannot see them. On call-20260902-2207 the
        # close was deferred on "Would you like me to add you?" and the agent's
        # reply armed a padding debt - chasing it would have asked a question
        # into a call that was closing correctly, which is the same error the
        # sign-off nudge was given these two flags for.
        _padding = sess._padding_owed
        # READ WITH THE DEBT, NOT AT THE APPEND. Both branches below clear the
        # debt (and reset this to "announced") BEFORE they file their row, so
        # reading it at the append recorded "announced" every time -- the
        # metric tidying away the distinction it exists to keep.
        # getattr, FOR THE REASON _objective_of GIVES: the watchdog is driven
        # in the suite with a namespace carrying only the attributes it reads,
        # and a guard that raises on a test double is a guard that stops being
        # tested. It raised on the first check of the run.
        _armed_by = getattr(sess, "_padding_reason", "announced")
        if (_padding and not sess._response_active
                and not sess._close_after_response
                and not sess._close_when_answered
                and time.time() - quiet_since >= _PADDING_SETTLE_S):
            _silent_for = round(time.time() - quiet_since, 2)
            # THE EXIT, checked here as well as after each chase, because a
            # debt still standing when the cap is reached would otherwise be
            # retried on every tick for the rest of the call.
            if sess._padding_nudges >= _MAX_PADDING_NUDGES:
                sess._padding_owed = ""
                sess._padding_reason = "announced"
                sess._padding_directive_sent = False
                sess.tool_call_padding.append(
                    {"said": _padding[:160], "silent_s": _silent_for,
                     "armed_by": _armed_by,
                     "outcome": "abandoned"})
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"⚠️  ANNOUNCED QUESTION ABANDONED "
                      f"({sess._padding_nudges} chases spent) — the caller was "
                      f"promised a question and never got one", flush=True)
                continue
            if not sess._padding_directive_sent:
                sess._padding_directive_sent = True
                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text", "text": (
                                 "(system: you told them you were about to "
                                 "ask something and then said nothing. They "
                                 "are waiting on a question that never came. "
                                 "Ask it now, in one short sentence, and "
                                 "nothing else. Do not apologise, do not "
                                 "thank them again, do not re-introduce "
                                 "yourself.)")}]},
                }))
            # STATE CHANGES AFTER THE OPERATION THAT DECIDES SUCCESS. The
            # owed-substance recovery beside this one shipped the other way
            # round and printed "saying it now" for a response that was
            # refused and never created; the debt is cleared only once
            # something is actually on its way.
            if await _create_response(oai_ws, sess, why="announced ask, none made",
                                      allow_when_vad_pending=True):
                sess._padding_owed = ""
                sess._padding_reason = "announced"
                sess._padding_directive_sent = False
                sess._padding_nudges += 1
                sess.tool_call_padding.append(
                    {"said": _padding[:160], "silent_s": _silent_for,
                     "armed_by": _armed_by,
                     "outcome": "chased"})
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"❓ PROMISED A QUESTION, ASKED NONE ({_silent_for:.1f}s "
                      f"of silence) — asking for it now: {_padding[:60]!r}",
                      flush=True)
            continue

        # ── A CLOSE HELD FOR A TURN THAT NEVER CAME ─────────────────────────
        # _decide_close's "spoken" branch stops the goodbye landing on top of
        # the agent's own finished turn by waiting for the caller. Almost
        # always they say "okay, no problem" and the re-arm fires. When they do
        # not, this is the other end of that wait — without it the hold is
        # unbounded and the next thing the caller hears is the watchdog asking
        # whether they are still there, about a call that is already complete.
        #
        # BEFORE the silence prompt and on a shorter clock, so it is this that
        # fires rather than that. See _DEFERRED_CLOSE_WAIT_S for why the two
        # silences are not the same silence.
        #
        # ONLY the "spoken" deferral: the other two have a question on the
        # table and somebody who owes a turn.
        if (sess._close_when_answered
                and sess._close_deferred_reason == "spoken"
                and not sess._response_active
                and time.time() - quiet_since >= _DEFERRED_CLOSE_WAIT_S):
            _last_agent = next((t.text for t in reversed(sess.turns)
                                if t.role == "agent" and t.text), "")
            # STATE FIRST HERE, unlike the recoveries above, and for the
            # opposite reason: those must not claim success before the create
            # that decides it, while this must not re-enter on the next tick
            # if the create is refused. A close that fires twice says goodbye
            # twice, which is the defect this whole branch exists to prevent.
            sess._close_when_answered = False
            sess._close_deferred_reason = ""
            sess._agent_quiet_since = None
            sess.done = True
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🏁 CLOSE HELD "
                  f"{time.time() - quiet_since:.1f}s FOR A REPLY THAT DID NOT "
                  f"COME — closing now rather than asking whether they are "
                  f"still there", flush=True)
            try:
                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text",
                                          "text": closing_directive(
                                              _last_agent)}]},
                }))
                sess._close_requested = True
                await _create_response(oai_ws, sess, why="deferred close",
                                       allow_when_done=True,
                                       allow_when_vad_pending=True)
            except Exception:
                return
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
            # THE DANGLING STATE, closed. Both budgets spent used to mean this
            # loop span on doing nothing until something else ended the call —
            # and on a line where nobody is talking, nothing else does. The
            # exit was in the prompt and nowhere in the process.
            #
            # Through give_up_directive rather than a fresh string, so the
            # wording the test suite asserts on is the wording that goes out,
            # and _give_up_sent makes it one-shot exactly like the budget's.
            if not sess._give_up_sent and not sess.done:
                sess._give_up_sent = True
                sess._give_up_trigger = "no_response"
                print(f"[Realtime] 🤐 silence budget spent ({_used}/"
                      f"{_MAX_SILENCE_PROMPTS} {_phase}) — closing the call "
                      f"rather than holding a line nobody is on", flush=True)
                try:
                    await oai_ws.send(json.dumps({
                        "type": "conversation.item.create",
                        "item": {"type": "message", "role": "user",
                                 "content": [{"type": "input_text",
                                              "text": give_up_directive(
                                                  sess, "no_response")}]},
                    }))
                    await _create_response(oai_ws, sess,
                                           why="silence budget spent",
                                           allow_when_vad_pending=True)
                except Exception:
                    return
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
                         # NO PHRASE IS QUOTED HERE ANY MORE. This used to say
                         # "in a few words — 'still with me?' —" and the
                         # transcript then carried "Still with me?" verbatim
                         # six times, word for word, across separate calls: a
                         # directive that quotes a line is a directive that
                         # hands over a line. Same finding as the prompt's own
                         # housekeeping ban, reached from the code side.
                         "content": [{"type": "input_text", "text": (
                             "(system: they have not said anything since you "
                             "stopped speaking. Say ONE short thing in your "
                             "own words — check the line is still there, or "
                             "put your question again more simply. Never the "
                             "same words you used the last time you did "
                             "this.)")}]},
            }))
            await _create_response(oai_ws, sess, why="silence watchdog",
                                   allow_when_vad_pending=True)
        except Exception:
            return

async def _suppress_reply_to(sess: "RealtimeSession", oai_ws, text: str,
                             record_to: str = "rejection_cancels") -> str:
    """Stop the agent answering a turn that should draw no reply.

    `record_to` NAMES THE LIST, and it defaults to the one this was written
    for. The mechanism — cancel while no audio has gone out, report honestly
    when it has — turned out to serve two different findings, and they must not
    be filed together. A REJECTED transcript means the caller was not heard; a
    bare acknowledgement means they were heard perfectly and said nothing that
    needed answering, and the turn is kept. Both produce a cancel and both want
    the same race margin; only one of them is a fault on the line. Putting them
    in one list would tell whoever reads the artifact that a caller who was
    heard was not, which is the reporting error `_discarded_location` exists to
    prevent one layer over.

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
    # CAPTURED BEFORE THE BRANCHES, because the cancelling branch clears
    # _response_active three lines later and would leave every cancelled row —
    # the only rows anyone wants to join — recording no response at all.
    # Null when nothing is in flight: there is then no response for this to be
    # about, and a stale id would read as one.
    _resp = (sess._response_id[-8:]
             if sess._response_id and sess._response_active else None)
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
    getattr(sess, record_to).append({
        # MILLISECONDS, not the transcript's whole seconds. Three of these can
        # land inside one second on a fast turn, and "which of them" is the
        # question the field exists to answer.
        "t": datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "turn": len(sess.turns),
        "response": _resp,
        "text": text[:60],
        "outcome": outcome,
        "since_speech_stopped_s": (round(_since_stop, 3)
                                   if _since_stop is not None else None),
        "since_response_created_s": (
            round(time.monotonic() - sess._response_created_at, 3)
            if sess._response_created_at else None),
    })
    return outcome


def _rearm_close_if_answered(sess: "RealtimeSession", ts: str = "") -> bool:
    """The caller answered the question a pending close was waiting on.

    The objective can finish inside a response that has just asked them
    something — the save that completes it and the question are in the same
    turn. The tool handler holds the teardown rather than hanging up
    mid-question (see the _close_deferred branch in grounding); this is the
    event it was waiting for, and it is a person speaking rather than anything
    on a clock. Called from _handle_caller_transcript the moment their words
    are in sess.turns.

    HANDED TO _close_after_response, NOT STRAIGHT TO sess.done. Closing here
    would drop the line on the answer they just gave, with the agent never
    acknowledging it — the same discourtesy one turn later, and on
    call-20260831-1048 the discourtesy WAS the bug. The model's reply to this
    turn becomes the closing turn instead, and the block at response.done then
    makes the call it already knows how to make: does that reply stand as a
    goodbye, or does one have to be asked for.

    A separate function because it is a decision, and because a re-arm that
    only exists inline in a 300-line handler cannot be tested on its own — a
    deferral with no re-arm is a call that never ends, which is the failure
    mode this half has to be checked against.
    """
    if not sess._close_when_answered or sess.done:
        return False
    sess._close_when_answered = False
    sess._close_deferred_reason = ""
    sess._close_after_response = True
    print(f"[{ts}] ▶️  CLOSE RE-ARMED — they answered; closing after the "
          f"agent replies", flush=True)
    return True


async def _handle_caller_transcript(msg: dict, sess: "RealtimeSession", oai_ws) -> None:
    """One completed caller transcript: log it, and run the turn-level guards.

    Extracted from _oai_to_twilio to bring that function back under pyright's
    analysis ceiling — see _handle_tool_call for why that matters. This one
    shares NO mutable state with the event loop: everything it changes lives on
    `sess`, and its only `break` is a local for-loop break. That is what made it
    the safest of the large handlers to move.
    """
    # STAMPED BEFORE ANY FILTERING, and that is the whole point of putting it
    # here. This records that the TRANSCRIBER HAS ANSWERED for the current
    # utterance, not that the answer was any good. Everything below may discard
    # what arrived — a hint echo stripped to nothing, a fabricated turn on
    # silence, an empty string — and every one of those leaves the "[...]"
    # placeholder standing, which is indistinguishable from a transcript still
    # in flight unless something remembers that the reply already came.
    sess._transcript_at = time.monotonic()
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
    # ── AND NOT BEFORE THEY HAVE EVER SPOKEN ───────────────────────────────
    # FIFTH FALSE FIRE, and the first four are catalogued above the arming
    # site in realtime_worker. Every one is the same event: server VAD trips
    # on noise while WE are still talking, the slice contains no speech, no
    # transcript arrives, and the pair (quiet, nothing transcribed) is exactly
    # what this branch reads as "we cannot hear you".
    #
    # call-20260907-1814 and -1817 are the fifth and sixth. On -1817 the alarm
    # fired at 18:17:47, the directive said "ask them to repeat it", and the
    # agent asked its greeting a SECOND time at 18:17:50 — to a receptionist
    # who had not yet said a word. Their first syllable came at 18:17:52. The
    # transcript shows two agent turns back to back and nothing between them;
    # `asked twice running` and `spoke over itself` both recorded it.
    #
    # "measure here, decide there" was the fourth fix and it could not work:
    # a VAD trigger on noise and a caller too faint to transcribe produce the
    # SAME pair at transcript time. Position is the only thing that separates
    # them — a turn nobody has yet answered cannot be evidence about the line.
    #
    # MORE THAN ONE, not "any". The placeholder for the utterance being judged
    # right now is already in sess.turns: realtime_worker adds "[...]" at
    # speech_stopped, before this handler runs. So one caller turn means THIS
    # one and nothing before it.
    #
    # THE ALARM IS NOT DISABLED, only delayed by one utterance. A genuinely
    # faint caller is still faint on their second attempt, and it fires then —
    # which costs one exchange against a guard that has now spoken over the
    # opening of six calls and, on the record in this file, has never once
    # been right.
    # A GENUINE TURN, not a placeholder. Counting every caller entry counted
    # "[...]" — the marker realtime_worker writes at speech_stopped for an
    # utterance nobody has transcribed yet, including the VAD trip on our own
    # greeting that this whole gate exists to reject. Two noise trips would
    # have satisfied a bare count and fired the alarm anyway.
    #
    # This is the same predicate the arming site used to call heard_clearly,
    # moved to where the transcript is actually known. Requiring it HERE is
    # what makes the legitimate case reachable: they were heard, then a later
    # turn comes back quiet and empty, and that is a real faint line.
    _caller_spoke_before = any(
        _t.role == "caller" and _t.text.strip() not in ("", "[...]")
        for _t in sess.turns)
    # DID THE ALARM ASK FOR SPEECH ON *THIS* TURN. The empty-transcript gate at
    # the foot of this handler cancels the reply to a turn that carried nothing
    # — and this branch is the one case where a turn carrying nothing is owed
    # one anyway, because the directive it sends says "ask them to repeat it".
    # Cancelling on top of that would send the instruction and then remove the
    # turn that was supposed to act on it.
    _asked_to_repeat = False
    if (_low is not None and not text and not sess._low_audio_warned
            and _caller_spoke_before):
        sess._low_audio_warned = True
        _asked_to_repeat = True
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
                             "make out. Ask "
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
    # DID IT FIRE FOR *THIS* TURN. `sess._repair_nudged` is a one-shot for the
    # whole call and cannot answer that, and the acknowledgement gate at the
    # foot of this handler needs the per-turn answer: "Hello." from someone we
    # cut off is owed a restatement, and going quiet on it would rebuild the
    # exact failure the repair branch was written for.
    _repaired_now = False
    if (text and sess._truncated_at is not None
            and time.time() - sess._truncated_at <= _REPAIR_WINDOW_S
            and sess._truncated_heard_ms < _CUT_SHORT_MS
            and not sess._repair_nudged):
        sess._repair_nudged = True
        _repaired_now = True
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

    # ── VAD FIRED AND NOTHING CAME BACK ────────────────────────────────────
    # call-20260910-1618. Three responses, two caller turns: `reply_latency`
    # recorded a third reply with `detector=0.412s`, so speech_stopped genuinely
    # fired, but transcription returned nothing and no caller turn was created.
    # The response OpenAI's VAD had already opened went ahead regardless, and
    # the model — handed a turn with no new input — filled it with a reaction to
    # nothing, a narration, and the question it had asked four seconds earlier:
    #
    #   16:18:57  agent   "Which office does Dr. Browne see people at?"
    #   16:19:01  agent   "Got it, let me just note that and then I'll ask about
    #                      availability. Which site should I come to for Dr.
    #                      Browne?"
    #
    # The caller heard the same question twice in four seconds and hung up at
    # 38s. Measured across the corpus: 8 of 89 calls (9%) carry one more
    # response than they have caller turns, and half of those died inside 46
    # seconds with nothing collected.
    #
    # THE GUARDS ABOVE ALL REQUIRE TEXT. `_audio_was_silent` and
    # `_reads_as_hint_vocabulary` judge WORDS that arrived on a line that
    # carried none, and the repair branch judges a turn that was cut off. None
    # of them can see the case where nothing arrives at all, because every one
    # of them is behind `if (text and ...)`.
    #
    # SAME MECHANISM AS THE ACKNOWLEDGEMENT GATE, one step further out: that
    # one says a turn carrying only a receipt earns no reply, and this says a
    # turn carrying nothing earns none either. Cancel before any audio, record
    # it, and let the silence watchdog own what happens next — the cancelled
    # response still produces a response.done with no audio, so
    # `_agent_quiet_since` is set from a playback remainder of zero and the
    # existing budget takes over on its own clock.
    if (not text and not sess.done
            and not _asked_to_repeat
            and not sess._close_after_response
            and not sess._close_when_answered
            and not sess._give_up_sent
            and not sess._padding_owed
            # WE CUT OURSELVES OFF AND THEY MADE A SOUND. The repair branch
            # above cannot speak for this case — it is behind `if (text and
            # ...)`, so `_repaired_now` is dead on an empty turn and testing it
            # here would be a conjunct that can never fire. The condition it
            # was reaching for is the truncation window itself: if we truncated
            # our own audio a moment ago, the caller heard a fragment, and
            # going silent on top of that leaves them with half a sentence and
            # no one talking. Owed a restatement, not more silence.
            and not (sess._truncated_at is not None
                     and time.time() - sess._truncated_at <= _REPAIR_WINDOW_S)):
        _sup = await _suppress_reply_to(sess, oai_ws, "[nothing transcribed]",
                                        record_to="empty_turn_cancels")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🕳️  VAD FIRED AND "
              f"NOTHING CAME BACK — no reply to a turn that carried no words "
              f"({_sup})", flush=True)

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
        # ── THEY OFFERED. TAKE IT. ──────────────────────────────────────
        # `_invites_continuation` already existed and had exactly two callers,
        # both of them defensive: _caller_is_vetting, and the escalate blocker
        # that refuses to hang up on "how can I help?". Neither fires in the
        # ordinary case, where a front desk offers mid-call and the agent
        # returns the politeness instead of spending it. That was carried as
        # prompt prose — "This is the easiest ask you will get on the whole
        # call and it is routinely wasted" — which is a rule the model has to
        # recall 4,000 tokens later at the one moment it matters.
        #
        # Placed with the other caller-turn nudges and one-shot like them: a
        # second copy of a directive the model ignored is context spent for
        # nothing.
        if (_invites_continuation(text) and not sess.done
                and not sess._offer_nudged
                and _objective_of(sess).missing(sess.memory)):
            sess._offer_nudged = True
            print(f"[Realtime] 🎁 they offered — telling the agent to spend it "
                  f"now: {text[:52]!r}", flush=True)
            _want = (_objective_of(sess).missing_spoken(sess.memory)
                     or "what you called about")
            await oai_ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": (
                             f"(system: they just offered to help. Say the one "
                             f"thing you want, right now, plainly — you still "
                             f"need {_want}. Do not return the politeness and "
                             f"do not wait for a better moment.)")}]},
            }))

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
            # FIRST, so a second "one moment" mid-hold extends it rather than
            # being read as the caller coming back.
            sess._hold_until = time.time() + _HOLD_GRACE_S
            sess.reset_ask_budget("caller is going to check")
        elif sess._hold_until and _caller_resumed(text):
            # THEY ARE BACK. Only the grace is dropped — the ask budget keeps
            # whatever the hold reset bought it, because coming back with an
            # answer is not a reason to start spending asks again.
            _was_live = time.time() < sess._hold_until
            sess._hold_until = 0.0
            if _was_live:
                print(f"[Realtime] hold over — caller came back with "
                      f"something to say ({text[:48]!r})", flush=True)
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

        # THE EVIDENCE IS IN sess.turns AS OF THIS LINE, and not one line
        # earlier. Anything held for it is judged now, against the words
        # themselves — this is the event the 1.5s wait was standing in for.
        await _resolve_deferred_save(sess, oai_ws)

        # ── THEY NAMED A DOCTOR AND IT IS NOT OURS ──────────────────────────
        # RECORD ONLY. The active check lives inside save_doctor_identity's
        # grounding and therefore stops looking the moment identity is
        # CONFIRMED — so call-20260903-1126 filed name_mismatches: null on a
        # call where the receptionist said "Dr. Abel" twice against our "Dr.
        # Pediatric". Nothing is asked of the call here (see _note_name_heard);
        # this exists so the ARTIFACT shows the input was wrong.
        # "not challenged" WAS READ AS "the agent let it go", AND ON
        # call-20260904-1013 IT HAD NOT. This line fired at 10:14:34 and the
        # agent spelled the surname out at 10:14:42 — the very next turn, no
        # other agent turn between them — but the wording sent a reader looking
        # for a workflow defect that was not there. It describes what THIS call
        # site does, so say that: the recorder sends no directive, which is not
        # a claim about what the agent goes on to do.
        _heard_name = _note_name_heard(sess, text)
        if _heard_name:
            print(f"[{ts}] 📛 THEY SAID Dr. {_heard_name!r} — our record says "
                  f"{_our_surname(sess)!r}. Recorded for the artifact; no "
                  f"directive sent from here — the agent may still raise it "
                  f"on its own turn. Check the doctor field on this row.",
                  flush=True)

        # THEY ANSWERED THE QUESTION THE CLOSE WAS DEFERRED FOR.
        _rearm_close_if_answered(sess, ts)

        # ── THE STANDING STEER HAS BEEN OVERTAKEN ───────────────────────────
        # STATE, NOT A PLAN, and that is the whole of the design. The steer it
        # replaces names a move and a moment ("this is the turn it lands in");
        # this names only what is true and what is not. There is no verb for
        # the model to lift, no ordering to recite and no condition to evaluate
        # aloud — the three shapes every traced narration turned out to be.
        #
        # IT CANNOT RETRACT THE OLD ONE. An injected conversation item is
        # permanent, so the only lever is to stop the spent instruction being
        # the NEWEST thing in context when the next response is generated.
        # That is why this fires here, on their words, rather than waiting for
        # the save: the save runs inside the response this is getting ahead of.
        _spent, _still = _steer_went_stale(sess)
        if _spent is not None and not sess.done:
            sess._steer_field = ""
            sess.stale_steers.append(
                {"field": _spent.name, "still": _still or None, "at": ts})
            if _still:
                print(f"[{ts}] 🧭 THE STEER WAS OVERTAKEN — they answered "
                      f"{_spent.name!r}; replacing it with what is still "
                      f"unknown: {_still!r}", flush=True)
                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user",
                             # OPENS ON WORDS, NOT ON THE LABEL. The directive
                             # audit derives its population with
                             # r'"\(system: ([a-z][a-z ]{9,})' — a directive
                             # whose first character after the opening marker
                             # is an interpolation is counted as declared and
                             # never found, and the check that exists to catch
                             # a new directive the day it lands reports 41
                             # against 42 instead. It caught this one.
                             "content": [{"type": "input_text", "text": (
                                 f"(system: they have just told you "
                                 f"{_spent.label} — that is known now. "
                                 f"Still unknown: {_still}.)")}]},
                }))

        # ── THEY ANSWERED SOMETHING NOBODY ASKED ────────────────────────────
        # A front desk volunteers. "She's at Riverside, and she's not taking
        # new patients right now" answers two fields while we asked about one,
        # and the ordinary path only ever looks at the field on the table — so
        # the second answer sat in the transcript and the agent asked for it
        # again four turns later, which is the "robotic loop" complaint in its
        # most concrete form.
        #
        # AFTER _resolve_deferred_save, for the same reason the hang-up check
        # is: a field that just landed from a held save must count as
        # collected here, or this offers the model a value it already has.
        for _vf, _vstate in _volunteered_fields(sess, text):
            sess._volunteered_seen.add(_vf.name)
            sess.volunteered_answers.append(
                {"field": _vf.name, "state": _vstate, "heard": text[:160]})
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 💡 VOLUNTEERED "
                  f"— they answered {_vf.name!r} without being asked: "
                  f"{text[:60]!r}", flush=True)
            # Terse and non-speakable, like every directive in this file: on
            # call-20260818-1112 the agent read one of these out to a caller.
            await oai_ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": (
                             f"(system: they just answered "
                             f"{_vf.spoken or _vf.name} without being asked — "
                             f"they said {text[:80]!r}. Take that as their "
                             f"answer and do not ask them for it later.)"
                         )}]},
            }))

        # ── THE CALLER ASKED TO STOP ────────────────────────────────────────
        # Set only the flag. No goodbye is injected and no response is created
        # here, and that restraint is the whole design: the caller just spoke,
        # so OpenAI's server VAD has already opened a response for this turn.
        # That response plays the model's own farewell, its response.done finds
        # sess.done set with no closing pending, and the existing branch drains
        # the audio and hangs up. Injecting a second goodbye would collide with
        # the one already in flight — the collision this module has been bitten
        # by twice.
        #
        # AFTER _resolve_deferred_save, so a save whose evidence landed in this
        # very turn is still applied. Ending the call must not cost a field the
        # caller already gave.
        # ── THEY DREW A LINE ────────────────────────────────────────────────
        # A hard refusal is not a budget event and must not wait for one. The
        # budget exists to stop the agent pestering someone who will not engage,
        # and it measures that indirectly, over many turns; this is the caller
        # saying it outright, once, and it is the cheapest signal on the call.
        #
        # call-20260902-1716: "Can't share the information with you, thank you
        # but don't call me again." Nothing read it. The agent had asked six
        # times, the budget stood at 2 of 8, and the caller hung up six seconds
        # later. Measured across the corpus before being wired in -- 5 hits in
        # 762 caller turns, all real, none on a call that got a branch.
        #
        # THE DIRECTIVE, NOT A HANG-UP, for the reason the whole file gives:
        # escalate is what writes the reason down, and cutting the line here
        # would leave a call with no outcome. _give_up_sent is set too, so the
        # budget path cannot fire a second directive on top of this one.
        if (not sess.done and not sess._refusal_nudged
                and is_hard_refusal(text)):
            sess._refusal_nudged = True
            sess._give_up_sent = True
            sess._give_up_at_turn = len(sess.turns)
            sess._give_up_trigger = "refused"
            sess.hard_refusal = text[:160]
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                  f"⛔ THEY REFUSED - stopping the asks and escalating: "
                  f"{text[:70]!r}", flush=True)
            await oai_ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text",
                                      "text": give_up_directive(
                                          sess, "refused")}]},
            }))

        if not sess.done and _caller_ends_call(text):
            sess.done = True
            sess.ended_by_caller = text[:120]
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                  f"👋 CALLER ENDED THE CALL: {text[:70]!r}", flush=True)
            print("[Realtime]   ^ closing after the reply already in flight",
                  flush=True)

        # -- THEY ONLY ACKNOWLEDGED; THE QUESTION IS STILL THEIRS ------------
        # LAST IN THIS HANDLER, deliberately. Every flag this reads is written
        # by a branch above it - a deferred save can arm a close, a refusal
        # sets _give_up_sent - and a gate that read them before they are set
        # would cancel the goodbye those branches had just decided to send.
        # What the position costs is microseconds, against the 1.9s measured on
        # call-20260910-1042 between the transcript arriving and the first
        # audio delta of the reply it had already provoked.
        #
        # CANCEL, DO NOT REDIRECT, and there is no directive here on purpose.
        # Silence IS the correct turn: they acknowledged, the question is on
        # the table, and the next voice should be theirs. Injecting "wait for
        # them" would be a seventh prompt rule against a class where six have
        # already measured null.
        #
        # NOTHING NEW WAITS ON A CLOCK. The cancelled response still produces a
        # response.done carrying no audio, so `_agent_quiet_since` is set from
        # a playback remainder of zero and the silence watchdog takes it from
        # there on its existing budget - _SILENCE_PROMPT_AFTER of real quiet,
        # then "put your question again more simply, never the same words".
        # That was already the policy for a caller who says nothing; this is
        # what makes a caller who says only "Okay." reach it.
        if (not sess.done
                and not sess._close_after_response
                and not sess._close_when_answered
                and not sess._give_up_sent
                and not sess._padding_owed
                and not _repaired_now
                and time.time() >= sess._hold_until
                and _marking_time_on_an_open_ask(sess, text)):
            _sup = await _suppress_reply_to(sess, oai_ws, text,
                                            record_to="acks_left_alone")
            print(f"[{ts}] 🤫 THEY ONLY ACKNOWLEDGED - leaving them the "
                  f"floor rather than putting the question again: "
                  f"{text[:40]!r} ({_sup})", flush=True)


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
    # ── WHAT THE MODEL ACTUALLY PRODUCED, BEFORE THIS FUNCTION TOUCHES IT ──
    # OBSERVABILITY ONLY. Nothing here changes a verdict; it records the one
    # thing the artifact could never show. Asked to prove that narration was
    # not INTRODUCED by response post-processing, the honest answer was an
    # inference — 87.6% of narrated turns are single-item and no code rewrites
    # text — because the raw per-item output was never kept. Three of the four
    # branches below end without the item ever reaching `sess.turns`, so a turn
    # the caller heard and a turn nobody heard are indistinguishable afterwards.
    #
    # AT INGRESS, and that placement is the point: every branch below is
    # reachable from here, so no path can drop an item without it having been
    # written down first. The verdict is stamped where each path decides.
    _raw_item = msg.get("item_id") or ""
    _raw_text = (msg.get("transcript") or _agent_text_buf).strip()
    _was_muted = bool(_raw_item and _raw_item in sess._muted_items)
    _raw_row = None
    if _raw_text:
        # THE FOUR FIELDS THAT MAKE A ROW ADDRESSABLE. Without them a row says
        # what the model produced and nothing about when, in reply to what, or
        # as part of which response — so an item that was cancelled could not
        # be shown to BE the response a given caller turn provoked, which is
        # the one thing the PII question turns on.
        #
        #   t        wall clock to the millisecond, joining to the cancel rows
        #   turn     len(sess.turns) at ingress, so sess.turns[turn - 1] is the
        #            caller turn this response was generated for
        #   response the response this item belongs to; `response_id` off the
        #            event is authoritative for THIS item, and the session's
        #            in-flight value is the fallback for transports that omit it
        #   idx      output_index — which item of the response this is, the
        #            fact the whole second-item muting path turns on
        _raw_row = {"t": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    "turn": len(sess.turns),
                    "response": ((msg.get("response_id")
                                  or sess._response_id or "")[-8:] or None),
                    "item": _raw_item[-8:] or "?",
                    "idx": msg.get("output_index"),
                    "text": _raw_text[:200],
                    #   narration  MEASUREMENT ONLY. Nothing branches on this.
                    #            `_narrated_the_reply` keeps driving the
                    #            directive and `reply_narration`; this records
                    #            what the guard's narrow pattern cannot see —
                    #            it read 0 of 126 offline generations and 6 of
                    #            13 known live narrated turns. A metric that
                    #            misses half the population is why the
                    #            narration question needed three experiments
                    #            to ask. Recorded per item and BEFORE any
                    #            verdict, so a narrated item that was later
                    #            cancelled still counts as model output.
                    "narration": workflow_narration(_raw_text) or None,
                    "verdict": "spoken"}
        sess.raw_items.append(_raw_row)

    if _barge_in_pending or sess._suppressed_response:
        # This transcript was cancelled — never fully heard, skip it.
        # _suppressed_response is the same situation reached from the
        # other side: we cancelled because the transcript that PROVOKED
        # this response was rejected. No audio was sent either way, so
        # letting it become a turn would put words in the transcript the
        # caller never heard and hand them to the guards as evidence.
        if _raw_row is not None:
            _raw_row["verdict"] = "cancelled"
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
    # HOISTED so the turn walk further down can see it. A RELEASED second item
    # becomes a turn of its own — correctly, the caller heard it — and that
    # makes the agent turn directly behind it the model's OWN first item rather
    # than the last thing it said to them. Every guard down there that reasons
    # about "the previous agent turn" needs to know the difference.
    _released = False
    if _item and _item in sess._muted_items:
        _dropped = (msg.get("transcript") or _agent_text_buf).strip()
        if _dropped:
            print(f"[Realtime]   ^ it would have said: {_dropped!r}", flush=True)
            # Muting was forced; losing the content was not. If the half
            # that reached the caller did not carry this, it is owed.
            # EVERYTHING THEY HAVE HEARD SINCE THEY LAST SPOKE, not just the
            # last agent turn. A RELEASED item becomes a turn, so on a
            # three-item response item 3 was judged against item 2 alone and
            # item 1 had already dropped out of view:
            #
            #   item 1  "Do you know if there's a waiting list?"      (played)
            #   item 2  "I'm a bit disappointed to hear that."        (released)
            #   item 3  "Do you know if there's a waiting list, and how would
            #            I get on it?"                               (released)
            #
            # New substance against item 2, and a verbatim repeat of item 1 —
            # the caller heard the question twice in one turn, and the
            # artifact's own `repeated sentences` counter caught it. Two items
            # never showed this because there was nothing behind the spoken
            # half to forget.
            #
            # The walk stops at the caller, so this is scoped to the current
            # exchange and needs no new state to clear. Agent turns stacked
            # across two responses with nobody in between are INCLUDED
            # deliberately: they are equally something the caller has already
            # heard, and over-suppressing is the safe direction for a predicate
            # whose own docstring says repeating yourself is what makes people
            # hang up.
            _heard = []
            for _t in reversed(sess.turns):
                if _t.role != "agent":
                    break
                if _t.text:
                    _heard.append(_t.text)
            _spoken = " ".join(reversed(_heard))
            # THE VERDICT IS DECIDED HERE AND RECORDED WITH THE TEXT, because
            # here is the only place that still has the spoken half to compare
            # against. Deciding it later, from the artifact, is what nobody
            # could do — see dropped_second_items.
            _verdict = "duplicate"
            # ── THE AUDIO IS STILL IN HAND ─────────────────────────────────
            # Since call-20260827-1130 the delta handler HOLDS a second item
            # instead of deleting it, so the answer to "was that a repeat or
            # the substance?" now arrives while the audio can still be played.
            # If it carries something the spoken half did not, the caller hears
            # what the model actually said — which is strictly better than
            # asking the model to say it again, because on that call the model
            # split the retry the same way four times running and the caller
            # heard four filler intros and never the question.
            #
            # `sess.done` is excluded deliberately: a call that is closing must
            # not start playing extra audio into the hang-up.
            _released = False
            _held = sess._held_item_pcm.get(_item) or []
            # ── A STALE HALF-ANSWER IS NOT SUBSTANCE (call-20260909-1822) ──
            # The caller asked for two things in one breath; the agent gave the
            # name, the caller moved on ("Okay, Bennett. Okay, let me confirm
            # the doctor availability now."), and fourteen seconds later the
            # SECOND item of the next response was "November 3, 2000."
            #
            # _drop_lost_substance released it, correctly on its own terms: a
            # date of birth is not a repeat of "Okay, I'll wait." Its question
            # is whether the held item repeats OUR OWN speech, and it has no
            # opinion about whether anyone wants to hear it. Nothing else did
            # either — no runtime state tracks the parts of a multi-part
            # request, so the date was outstanding only in the model's reading
            # of the conversation.
            #
            # A HALF-ANSWERED REQUEST IS NOT A STANDING PERMISSION, and this is
            # the whole of the rule: a detail of ours is speakable while
            # somebody is asking for it and not otherwise. See
            # _stale_own_detail for why it is keyed on our own PII rather than
            # on substance in general — the bad-news reaction and its question
            # answer a caller turn that is a STATEMENT, and a broader test
            # would suppress exactly that.
            # ── THE LAST THING THEY SAID THAT CHANGED THE SUBJECT ────────
            # THIS WAS `next(... if t.role == "caller")` — the newest caller
            # turn, whatever it was — and on call-20260910-1726 that cost the
            # EHR guard its disclaimer. The caller asked "can I know what's
            # your full name and your date of birth?", the model correctly
            # produced "I'm not a patient here yet, just looking, and my name
            # is Devon Keswick." as its second item, and 0.8s later the caller
            # added a trailing "Yeah." The held item is judged at TRANSCRIPT
            # time, not at generation time, so by then "Yeah." was the newest
            # caller turn; it carries no request, the item was dropped, and
            # "October 6, 1974." then went out with no not-a-patient line in
            # front of it. `bare_pii_details` recorded exactly that. The
            # disclaimer was finally spoken ten seconds AFTER the date of
            # birth — the order the EHR rule exists to forbid.
            #
            # RECENCY IN A CONVERSATION IS NOT "THE LAST THING SAID". It is
            # the last thing said that CHANGED THE SUBJECT, and an
            # acknowledgement is definitionally the absence of one. So the walk
            # steps over acknowledgement-only turns to the substantive one
            # behind them. `_only_acknowledged` is already that predicate and
            # already returns True for "Yeah.", "Okay.", "Right.", "Sure." and
            # — usefully — for the "[...]" placeholder, which is an utterance
            # we have detected and cannot yet read. Treating "we do not know
            # what they said" as "they are not asking" was a third way to lose
            # the disclaimer.
            #
            # THE AGENT-TURN BOUND IS THE OTHER HALF, and without it this
            # over-corrects into the opposite defect. Once WE have spoken, the
            # exchange those words belonged to is over: a request answered two
            # turns ago must not keep a disclaimer alive into a new one, or the
            # caller hears it twice. So the walk skips only the agent items of
            # THIS response — the run at the end of the list — and stops at the
            # first agent turn it meets after crossing into caller territory.
            # This mirrors the `_heard` walk twelve lines above, whose own
            # comment says it is "scoped to the current exchange and needs no
            # new state to clear". That was true there and stays true here: no
            # session state is added.
            _asking = ""
            _left_agent_run = False
            for _t in reversed(sess.turns):
                if _t.role == "agent":
                    if _left_agent_run:
                        break          # a previous exchange, not this one
                    continue           # this response's own item(s)
                _left_agent_run = True
                if _only_acknowledged(_t.text or ""):
                    continue           # a continuation, not a subject change
                _asking = _t.text or ""
                break
            if _held and not sess.done and _stale_own_detail(_dropped,
                                                             _asking):
                _verdict = "stale_detail"
                sess.stale_detail_items.append(
                    {"text": _dropped, "after": (_asking or "")[:80]})
                print(f"[Realtime]   ^ 🔒 a detail nobody is asking for — held "
                      f"back; it answers a request they have moved on from",
                      flush=True)
            elif (_held and not sess.done
                    and _drop_lost_substance(_spoken, _dropped)):
                _released = True
                sess._muted_items.discard(_item)
                sess._release_item = _item
                sess.released_second_items.append({"text": _dropped})
                if _raw_row is not None:
                    _raw_row["verdict"] = "held:released"
                print(f"[Realtime]   ^ ✅ that is new substance, not a repeat "
                      f"— releasing the audio we held, so they DO hear it",
                      flush=True)
                # Falls through to the ordinary path below: the caller hears
                # it, so it is a real turn and the turn guards must see it.
            elif not sess.done and _drop_lost_substance(_spoken, _dropped):
                # NOTHING LEFT TO PLAY — the hold was capped out, or a barge-in
                # threw it away. The owed-substance recovery is what is left,
                # and it is still worth having for exactly this case.
                sess._held_item_pcm.pop(_item, None)
                _stop = _owed_refusal(sess, _dropped)
                if _stop:
                    # The recovery for this substance has already been muted
                    # its allowance of times. Owing it again is the livelock.
                    _verdict = "abandoned"
                    sess.owed_abandoned.append({"text": _dropped, "why": _stop})
                    print(f"[Realtime]   ^ ⚠️  that was the substance of the "
                          f"turn and the recovery is GIVING UP ({_stop}) — "
                          f"the caller never hears it", flush=True)
                else:
                    _verdict = "owed"
                    sess._owed_substance = _dropped
                    print(f"[Realtime]   ^ that was the substance of the turn — "
                          f"owed to the caller, will be said next", flush=True)
            if not _released:
                if _verdict == "duplicate":
                    # A repeat: discarded exactly as it always was. This is
                    # the guard working — the greeting muted so nobody hears
                    # it twice. Nothing about that case changes.
                    sess._held_item_pcm.pop(_item, None)
                # A RELEASED ITEM IS NOT A DROPPED ONE. It goes in
                # released_second_items and becomes a turn; recording it here
                # as well would make the artifact say the caller never heard
                # the sentence they did hear.
                sess.dropped_second_items.append(
                    {"text": _dropped, "verdict": _verdict})
                if _raw_row is not None:
                    _raw_row["verdict"] = "held:" + _verdict
        if _item in sess._muted_items:
            sess._held_item_pcm.pop(_item, None)
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
        # THE PREVIOUS AGENT TURN, AND HOW FAR BACK IT IS. One walk for both,
        # because the second question only makes sense about the first: "did
        # the model just repeat itself" and "is the disclaimer it said last
        # time still standing" are the same walk with different verdicts.
        #
        # BEFORE add_turn, deliberately — `text` is not in sess.turns yet, so
        # `_prev_agent` is the turn before this one rather than this one.
        # Placeholders and blanks are skipped: "[...]" is a caller transcript
        # still in flight, and counting it as a turn of theirs would age the
        # disclaimer on evidence that has not arrived.
        _prev_agent = ""
        _callers_since = 0
        # ── A RELEASED SECOND ITEM IS NOT ITS OWN PREVIOUS TURN ─────────────
        # The model emitted two spoken items in one response; the second was
        # held, judged to carry real substance, and played. It is a turn — the
        # caller heard it — but the thing standing behind it in sess.turns is
        # the FIRST HALF OF THE SAME BREATH, not the last thing we said to
        # them. Walking one agent turn further back is what makes every guard
        # below reason about the conversation instead of about the split.
        #
        # call-20260904-0026 is the cost of not doing it. The agent said "Oh,
        # I'm not a patient here yet — I'm just looking. It's Ingrid Bennett.",
        # they asked for the date of birth, and the model answered in two items:
        # "Okay, let me think about how to answer that." then "Sure, it's
        # November 3, 2000 — ...". CALL CONTEXT's one-turn exception says a
        # disclaimer spoken in the turn you JUST spoke is still standing, and it
        # WAS — one caller turn back. But `_prev_agent` resolved to the model's
        # own filler and `_callers_since` to 0, so `_detail_left_bare` saw a
        # date of birth handed over with nothing in front of it, recorded
        # bare_pii_details, and injected "tell them now that you are not
        # registered with them yet". The model obeyed on the next turn — which
        # was a hold — and said "Okay, I'm not registered with you yet, and
        # I'll wait while you check." The model had followed the rule exactly
        # and the guard punished it for it.
        _skip_agent = 1 if _released else 0
        for _t in reversed(sess.turns):
            _txt = (_t.text or "").strip()
            if not _txt or _txt == "[...]":
                continue
            if _t.role == "agent":
                if _skip_agent:
                    _skip_agent -= 1
                    continue
                _prev_agent = _t.text
                break
            _callers_since += 1
        # Exactly one turn of theirs in between is them asking the follow-up:
        # "name?" -> we answer -> "and date of birth?". Zero means we spoke
        # twice running, which is also the same breath. Two or more, or none
        # at all with no previous agent turn, and whatever we said last time
        # has been overtaken.
        _same_exchange = bool(_prev_agent) and _callers_since <= 1
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

        # ── THE ENQUIRY-BOT CADENCE ──────────────────────────────────────────
        # Acknowledgement -> required-field question, over and over. 70% of
        # non-greeting agent turns across 57 patient calls opened on an
        # acknowledgement and 177 ran back to back, and NOT ONE existing
        # counter could see it: call-20260904-1306 scored piled_turns 0,
        # stapled_questions 0, repeated_sentences 0, shared_opening_clauses 0
        # while five of its eleven turns were "Okay, thanks for confirming
        # that. Let me ask about availability next." See
        # grounding/vocabulary._ack_opener for the full measurement.
        #
        # THE PAIR IS THE DEFECT, NOT THE OPENER. One "Got it —" is a person
        # talking. Two running is the form being worked through, and it is the
        # only version of this a detector can be sure about — which is why the
        # predicate returns the opener and this decides.
        #
        # THE GREETING IS EXCLUDED because it is a fixed line, and a first turn
        # has nothing before it anyway; _prev_ack_opener starts "" and is only
        # ever set from a turn that reached this point.
        _ack = _ack_opener(text)
        # AN EARNED OPENER IS NOT THE TIC. "Oh, that's a shame - is there a
        # waitlist?" opens on 'oh', and 'oh' is 4th on the content-free list
        # with 47 corpus hits -- so the one turn shape this call is short of
        # was being booked as the cadence and answered with a directive that
        # says "no opener in front of it" for the rest of the call. The turn
        # is dropped from the run in BOTH positions, not merely spared the
        # directive: a reaction is not an instance of the form-filling
        # cadence, so it must not arm the next turn into one either.
        #
        # RECORDED, NOT JUST SUPPRESSED. An exemption that leaves no trace is
        # a counter that can shrink without anyone being able to see why it
        # shrank; the artifact carries what was exempted and what it said.
        if _ack and _reacted_to_news(text):
            sess.ack_reaction_turns.append(
                {"opener": _ack, "said": text.strip()[:160], "at": ts})
            print(f"[{ts}] 💬 REACTION, NOT A TIC - opened on {_ack!r} and "
                  f"said something about what they told us", flush=True)
            _ack = ""
        if _ack and sess._prev_ack_opener and not sess.done:
            sess.ack_opener_runs.append(
                {"first": sess._prev_ack_opener, "then": _ack,
                 "said": text.strip()[:120], "at": ts})
            print(f"[{ts}] 🪞 ACK OPENER TWICE RUNNING — "
                  f"{sess._prev_ack_opener!r} then {_ack!r}", flush=True)
            if sess.may_nudge_cadence("ack_run"):
                # THE WORDING LIVES IN cadence_directive, one definition for
                # this call site and for scripts/check_cadence_loop.py, which
                # replays these against the real model. Its docstring carries
                # the measurement that decided the wording — naming the
                # outstanding field rather than naming the failure.
                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text",
                                          "text": cadence_directive(
                                              "ack_run",
                                              _objective_of(sess)
                                              .next_spoken(sess.memory))}]},
                }))
        sess._prev_ack_opener = _ack

        # ── A TURN SPENT ON HOUSEKEEPING ─────────────────────────────────────
        # Thanking them for confirming, checking or waiting, with no question
        # in the same breath. The prompt banned this in capitals and QUOTED
        # THE FIVE PHRASES; the model then said those exact phrases in 67 of
        # 470 turns, which is what a ban written as a phrase library buys. The
        # quotes are gone from templates.py and this is what replaces them.
        if (_housekeeping_turn(text, asks=_is_objective_ask(text, sess))
                and not sess.done):
            sess.housekeeping_turns.append(
                {"said": text.strip()[:160], "at": ts})
            print(f"[{ts}] 🧹 HOUSEKEEPING TURN — thanked them for taking "
                  f"part and asked nothing", flush=True)
            _hk_nudged = False
            if sess.may_nudge_cadence("housekeeping"):
                _hk_nudged = True
                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text",
                                          "text": cadence_directive(
                                              "housekeeping",
                                              _objective_of(sess)
                                              .next_spoken(sess.memory))}]},
                }))
            # ── AND THE TURN IS STILL OWED A QUESTION ───────────────────────
            # call-20260910-1618: "Okay, thanks for confirming that." at
            # 16:18:50, the caller waited, said "Okay." at 16:18:56, and the
            # office question came at 16:18:57 — a whole exchange spent on a
            # turn with nothing in it, on a call that died at 38 seconds.
            #
            # THE DIRECTIVE ABOVE ALREADY FIXED THE NEXT TURN; what nothing
            # did was bring the next turn forward. That is exactly what the
            # padding debt is for — the caller has been given nothing to answer
            # — so it is armed here rather than a second recovery being
            # invented. The watchdog chases at _PADDING_SETTLE_S instead of the
            # caller filling the gap six seconds later.
            #
            # _padding_directive_sent IS SET FROM `_hk_nudged`, and that is the
            # load-bearing line. The chase's own directive opens "you told them
            # you were about to ask something" — false of a turn that announced
            # nothing, and a directive with a wrong premise is one this project
            # has watched get narrated. When the housekeeping directive went
            # out, this suppresses the chase's; when the cadence budget was
            # spent and none went, the chase sends its own rather than creating
            # a response with no steering at all.
            if (not sess._padding_owed
                    and _objective_of(sess).missing(sess.memory)
                    and not sess._close_after_response
                    and not sess._close_when_answered
                    and not sess._give_up_sent):
                sess._padding_owed = text
                sess._padding_directive_sent = _hk_nudged
                sess._padding_reason = "housekeeping"
                print(f"[{ts}] ⏳ NOTHING IN THAT TURN TO ANSWER — chasing "
                      f"the question rather than waiting for them", flush=True)

        # ── NARRATING THE REPLY INSTEAD OF GIVING IT ─────────────────────────
        # "Okay, let me answer that." to a caller who asked for a date of
        # birth. FOUND BY RENDERING RATHER THAN BY READING: the prompt rebuild
        # deleted every quoted phrase the model had been lifting, the A/B
        # renders confirmed the verbatim reproductions were gone — and this
        # came back in their place, generated fresh, in four takes out of
        # fourteen. The prompt has banned narration in capitals with an
        # operational test attached since it was written, which is exactly the
        # evidence that another prompt line was not the answer.
        #
        # DISTINCT FROM tool_call_padding, which is a promised QUESTION. Here
        # they are waiting on the fact they asked for, and the padding
        # detector is silent because it reads ask/check verbs.
        if _narrated_the_reply(text) and not sess.done:
            sess.reply_narration.append(
                {"said": text.strip()[:160], "at": ts})
            print(f"[{ts}] 💭 NARRATED THE REPLY — said it was coming "
                  f"instead of saying it", flush=True)
            if sess.may_nudge_cadence("reply_narration"):
                # NO OBJECTIVE FIELD PASSED, unlike the two above. They have
                # just asked for something specific and are waiting on it;
                # steering to the next missing field would answer a question
                # nobody asked.
                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text",
                                          "text": cadence_directive(
                                              "reply_narration")}]},
                }))

        # ── THE INSTRUCTIONS ARRIVING IN THE AUDIO ───────────────────────────
        # call-20260904-1651, first live call after the rebuild. The caller
        # said "Ah." and the agent said out loud:
        #
        #   "Let me listen carefully to what they said and then respond
        #    clearly."          <- a paraphrase of a RULE in templates.py
        #   "Take your time."   <- the correct turn, arriving second
        #
        # Detected by the pronoun rather than by a phrase list: the prompt and
        # every directive are written ABOUT the call, so they say "they" about
        # the caller, and narrating that person's words in the third person
        # while speaking to them is something no patient does. See
        # _leaked_the_instructions for the 481-turn precision check.
        if _leaked_the_instructions(text) and not sess.done:
            sess.leaked_instructions.append(
                {"said": text.strip()[:160], "at": ts})
            print(f"[{ts}] 📖 INSTRUCTIONS SPOKEN ALOUD — the turn narrated "
                  f"the exchange in the third person", flush=True)
            if sess.may_nudge_cadence("leaked_instructions"):
                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text",
                                          "text": cadence_directive(
                                              "leaked_instructions")}]},
                }))

        # ── A PLAIN FACT WITH A QUESTION STAPLED TO IT ───────────────────────
        # "April 21, 1984. Any chance you could tell me if Dr. Browne is
        # taking new patients right now?" — and the receptionist answered the
        # staple with "Okay.", so the question was asked again on the next
        # turn. The prompt carries "A PLAIN FACT IS A PLAIN ANSWER" in capitals
        # and this happened anyway, which is the standing evidence about prose
        # rules on this project.
        if _stapled_own_detail(text) and not sess.done:
            sess.stapled_details.append(
                {"said": text.strip()[:160], "at": ts})
            print(f"[{ts}] 📎 DETAIL WITH A QUESTION STAPLED ON — the fact "
                  f"goes past them and gets asked again", flush=True)
            if sess.may_nudge_cadence("stapled_detail"):
                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text",
                                          "text": cadence_directive(
                                              "stapled_detail")}]},
                }))
        sess.add_turn("agent", text)

        # ── Name AND date of birth, in one breath ────────────────────────────
        # THE EHR GUARD, ENFORCED. The prompt states this three separate ways
        # ("asked your name, you say your name and stop"; "a name and a date of
        # birth arriving together is the single event this whole section exists
        # to prevent") and it held on NONE of the nine calls where a
        # receptionist asked for intake details — every one of them a compound
        # ask, "can I get your first and last name and your date of birth?",
        # which the prompt never gave a rule for.
        #
        # THIS CANNOT UNSAY THE TURN, and the honesty about that is the same
        # one _back_to_back_asks carries: the audio is on the line by the time
        # the transcript arrives. What it can do is stop the run continuing and
        # make the failure visible — before this, nine calls carried it and no
        # artifact field recorded any of them.
        #
        # EVERY occurrence is recorded; the directive goes ONCE, like the
        # self-repeat nudge. A guard that re-injects on every turn spends the
        # call arguing with itself.
        if _gave_name_and_dob(text):
            sess.compound_pii_dumps.append(
                {"said": text.strip()[:160], "at": ts})
            print(f"\n[{ts}] 🩺 NAME + DOB IN ONE TURN — the pair is enough "
                  f"to start a record for a person who does not exist",
                  flush=True)
            if not sess._pii_pair_nudged and not sess.done:
                sess._pii_pair_nudged = True
                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text", "text": (
                                 "(system: you just gave your name and your "
                                 "date of birth in the same turn. They have a "
                                 "patient record open and the pair is all it "
                                 "takes to start one. For the rest of this "
                                 "call: when they ask for several details at "
                                 "once, answer ONLY the first one, with its "
                                 "disclaimer, and stop. Do not repeat what you "
                                 "have already given.)")}]},
                }))

        # ── The disclaimer, said twice running ───────────────────────────────
        # THE OTHER HALF OF THE DOUBLING, and the compound-dump guard above is
        # silent on it by design. call-20260903-1422:
        #
        #   14:23:32  "I haven't registered with you yet, but my name is
        #              Ingrid Bennett."
        #   14:23:41  caller: "Okay, and your date of birth is?"
        #   14:23:42  "I'm not in your system yet, but it's November 3, 2000."
        #
        # Two details, two TURNS, the caller between them — everything the
        # compound guard asks for, correctly answered, and the audio still
        # sounds like a recording. The same scripted construction twice inside
        # ten seconds is the tell, not the pair.
        #
        # RECORDED AND NUDGED, NOT REJECTED, and this cannot unsay the turn —
        # the same honesty _gave_name_and_dob carries. templates.py now teaches
        # the one-turn exception, so this measures whether that rule holds; a
        # prompt rule with no number against it is a rule nobody can tell has
        # stopped working, which is how the intake disclaimer reached nine
        # calls unnoticed in the first place.
        if (_said_not_a_patient(text) and _gave_own_detail(text)
                and _same_exchange and _said_not_a_patient(_prev_agent)):
            sess.repeated_disclaimers.append(
                {"said": text.strip()[:160], "after": _prev_agent.strip()[:160],
                 "at": ts})
            print(f"[{ts}] 🗣️  DISCLAIMER SAID TWICE RUNNING — the "
                  f"not-a-patient line was already standing from the turn "
                  f"before", flush=True)
            if not sess._disclaimer_repeat_nudged and not sess.done:
                sess._disclaimer_repeat_nudged = True
                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text", "text": (
                                 "(system: you told them you were not "
                                 "registered with them, and then said it "
                                 "again in your very next turn. It was still "
                                 "standing — they heard it seconds ago — and "
                                 "the same construction twice in a row sounds "
                                 "like a recording. For the rest of this "
                                 "call: say that line the FIRST time a detail "
                                 "is asked for, and if they come straight "
                                 "back for another one, give it bare and "
                                 "short. If an exchange about something else "
                                 "comes in between, say it again.)")}]},
                }))

        # ── A detail with no disclaimer standing at all ──────────────────────
        # THE FAILURE THE RELAXATION ABOVE COULD OTHERWISE OPEN, and the one
        # nothing in this repo has ever looked for. The rule "never hand the
        # detail over bare" has always been prose, stated three separate ways,
        # and prose is what failed on all nine calls where a receptionist ran
        # intake. Narrowing it by one turn without adding this check would be
        # trading a rule nobody enforced for a smaller rule nobody enforced.
        #
        # This is the one of the two that is a SAFETY event rather than an
        # audio one: the person on the other end has a record open, and a name
        # or a date of birth arriving with nothing in front of it is what they
        # type into it.
        if _detail_left_bare(text, _prev_agent, _same_exchange):
            sess.bare_pii_details.append(
                {"said": text.strip()[:160], "at": ts,
                 "prev_agent": _prev_agent.strip()[:160],
                 "callers_since": _callers_since})
            print(f"\n[{ts}] 🩺 DETAIL GIVEN BARE — no not-a-patient line was "
                  f"standing; they have a record open in front of them",
                  flush=True)
            if not sess._bare_detail_nudged and not sess.done:
                sess._bare_detail_nudged = True
                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text", "text": (
                                 "(system: you just gave one of your own "
                                 "details with nothing in front of it. They "
                                 "have a patient record open and that is what "
                                 "they type into it. Tell them now, in one "
                                 "short sentence, that you are not registered "
                                 "with them yet — and for the rest of this "
                                 "call that line goes in front of any detail "
                                 "unless you said it in the turn immediately "
                                 "before.)")}]},
                }))

        # ── The name, spelled out ────────────────────────────────────────────
        # Recorded when it HAPPENS, not when it is asked for. `_name_spelled_at`
        # moves the identity scan past every turn that came before it, so a
        # mangled surname from earlier in the call stops refusing a
        # confirmation the caller has now given against our actual letters. If
        # this were set where the rejection is written, the scan would advance
        # on a request the model ignored and any later "yes" would confirm.
        _ours = _our_surname(sess)
        # GATED, AND EITHER PROOF COUNTS. Spelling still advances the
        # barrier; so does a natural confirmation of our surname. Both
        # now require an ACTIVE mismatch to exist first, so a recited
        # example line cannot move a safety barrier on a call where
        # nothing was ever mismatched.
        if _ours and not sess._name_spelled_at and _performed_name_repair(
                text, sess):
            sess._name_spelled_at = len(sess.turns)
            print(f"[{ts}] 🔤 SPELLED THE NAME — {_spell_out(_ours)}; what "
                  f"they say next is evidence about our doctor", flush=True)

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

        # ── Signed off ──────────────────────────────────────────────────────
        # RECORDED ON EVERY AGENT TURN, which is the fix. This check used to
        # live at the tool site, inside the save_branch-REJECTED branch — so it
        # could only see a goodbye that happened to share a turn with a refused
        # branch save. It saw neither close on 2026-09-02: 1544 signed off with
        # save_branch long since succeeded, 1511 signed off carrying no tool at
        # all. farewell_without_close was null on both, which read as "the agent
        # closed properly" on two calls the CALLER had to end.
        #
        # Deferred for the same reason as the claim check above, and it matters
        # more here: the close path sets sess.done from the tool handler, and a
        # model that says its goodbye in the same response as the tool that
        # completes the objective is behaving correctly. Judged now, that is
        # indistinguishable from signing off with nothing recorded.
        if _spoken_farewell(text) and not sess.done:
            sess._farewell_at = time.time()
            sess._farewell_said = text

        # ── Promised a question and asked none ──────────────────────────────
        # DISARM FIRST, ARM SECOND, and the order is the whole guard. Reaching
        # this line means the agent has just spoken, so whatever it announced
        # last time has been followed by speech and is no longer owed — even
        # when this turn is itself another announcement, which is the case that
        # would otherwise make the debt look older than it is and fire the
        # recovery a beat too early.
        #
        # Nothing else here decides anything. Whether the announcement was kept
        # is a question about the silence after it, and the watchdog is the
        # only thing that can see silence.
        sess._padding_owed = ""
        sess._padding_reason = "announced"
        if (not sess.done and _announced_an_ask(text)
                and not _is_objective_ask(text, sess)):
            # NOT AN ASK BY CONTENT, checked with the same predicate the ask
            # budget uses. "I need to check which office she's at" announces
            # and asks in one breath — a statement-form request is a request,
            # and treating it as padding would chase the agent for a question
            # it had already put to them.
            sess._padding_owed = text
            print(f"[{ts}] ⏳ ANNOUNCED A QUESTION — nothing asked yet; "
                  f"watching the line", flush=True)

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
            # EVERY OBJECTIVE FIELD, not just the branch. `_is_location_ask`
            # here made this guard blind to three of the four things the call
            # collects, and that is what let call-20260825-1847 through: the
            # agent asked "is this Dr. Carol, Neurosurgery, at New York
            # Presbyterian?", the caller said "Yes?", and nine seconds later
            # the agent asked the IDENTICAL clause again. The identity clause
            # never entered `_ask_phrasings`, so the second copy matched
            # nothing.
            #
            # The outer gate is already `_is_objective_ask` and the ask budget
            # has counted every field since 2026-08-24 — this line was the last
            # place inside that block still assuming one field. Same reasoning
            # as _is_objective_ask's own docstring: the counters never knew
            # about branches, only the way in did.
            _ask_clauses = {_norm_clause(c) for s in _sentences(text)
                            for c in _clauses(s) if _is_objective_ask(c, sess)}
            _repeat_phrasing = _ask_clauses & sess._ask_phrasings
            sess._ask_phrasings |= _ask_clauses

            # ── ASKED AGAIN FOR SOMETHING THEY ALREADY ANSWERED ─────────────
            # The guard the other three leave a hole for. See
            # _field_already_answered for why speed, budget and phrasing all
            # stood down on call-20260825-1847.
            #
            # Fires AFTER the agent has spoken, like every nudge in this block
            # — the transcript is how we learn what it said. It cannot unsay
            # the re-ask; it stops the run continuing, which is the difference
            # between one clumsy turn and the caller saying "I already told you
            # that".
            for _f in _objective_of(sess).fields:
                if not _is_ask_for(text, _f.probe):
                    continue
                _prev_idx = sess._field_ask_at.get(_f.name)
                sess._field_ask_at[_f.name] = len(sess.turns)
                if _prev_idx is None:
                    continue        # first time we have asked this one
                _answer = _field_already_answered(sess, _f, _prev_idx)
                if (not _answer or sess._answered_reask_nudged
                        or sess._give_up_sent):
                    continue
                sess._answered_reask_nudged = True
                print(f"[{ts}] 🔁 RE-ASKED AN ANSWERED QUESTION — "
                      f"{_f.label!r}; they already said {_answer[:48]!r}",
                      flush=True)
                await oai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text", "text": (
                                 f"(system: you just asked about "
                                 f"{_f.label} again. They already answered "
                                 f"it — they said {_answer[:80]!r}. Do not "
                                 f"ask it a third time. Take that as their "
                                 f"answer and move on to what is still "
                                 f"missing.)")}]},
                }))
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
                                 "(system: you have now asked for the same "
                                 "thing in those exact words more than once, "
                                 "and you are attaching it to the end of "
                                 "every reply. They heard it the first time. "
                                 "When they ask you something, answer it and "
                                 "STOP — no question on the end. Ask again "
                                 "only once they have answered you and gone "
                                 "quiet, and when you do, use different "
                                 "words.)")}]},
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
                print(f"[Realtime] Re-asked "
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
                                "(system: you just asked the same kind "
                                "of question twice within a few seconds. "
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


__all__ = [
    "GIVE_UP_REASONS",
    "_ACK_MARKERS",
    "_ACK_REPAIRS",
    "_ACK_REPAIR_REPLY",
    "_ACK_REPLY",
    "_ACK_WORDS",
    "_AFFIRM_REPLY",
    "_BACKCHANNEL_AFTER_S",
    "_BACKCHANNEL_COOLDOWN_S",
    "_BACKCHANNEL_ECHO_MARGIN_S",
    "_BACKCHANNEL_PAUSE_MAX_S",
    "_BACKCHANNEL_PAUSE_MIN_S",
    "_BACKCHANNEL_TICK_S",
    "_CADENCE_NUDGE_PER_KIND",
    "_CADENCE_NUDGE_TOTAL",
    "_WATCHDOG_TICK_S",
    "_watchdog_tick_s",
    "_CUT_SHORT_MS",
    "_DEFERRED_CLOSE_WAIT_S",
    "_FABRICATION_VOCAB",
    "_HAS_AFFIRM",
    "_HINT_HEADINGS",
    "_HINT_RUN_WORDS",
    "_HOLD_GRACE_S",
    "_MAX_PADDING_NUDGES",
    "_MAX_SILENCE_PROMPTS",
    "_MAX_VETTING_REASKS",
    "_MIN_REASK_GAP_S",
    "_PATIENT_ASK",
    "_REPAIR_WINDOW_S",
    "_RETIRED_HINT_TEXT",
    "_PADDING_PROMPT_AFTER",
    "_PADDING_SETTLE_S",
    "_SILENCE_PROMPT_AFTER",
    "_SILENCE_PROMPT_FIRST",
    "_asks_about_patient",
    "_caller_answered_since",
    "_caller_vetted_since",
    "_claims_employment",
    "_content_words",
    "_field_already_answered",
    "_volunteered_fields",
    "_field_vocabulary",
    "_handle_agent_transcript",
    "_handle_caller_transcript",
    "_hint_proper_nouns",
    "_is_filler_reply",
    "_marking_time_on_an_open_ask",
    "_only_marked_time",
    "_is_objective_ask",
    "_is_reintroduction",
    "_norm_clause",
    "_pending_expectation",
    "_rearm_close_if_answered",
    "_reads_as_hint_vocabulary",
    "_silence_watchdog",
    "_strip_hint_run",
    "_suppress_reply_to",
    "caller_repeated_answer",
    "give_up_directive",
]
