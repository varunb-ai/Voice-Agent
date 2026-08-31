"""What a tool call may write, and what the model is told when it may not.

Split from realtime_worker 2026-08-26, verbatim. _handle_tool_call alone is 625
lines and was the largest function in the package.

- The set is a transitive CLOSURE, not a tidy list. is_hold_request,
  _objective_of and _hint_vocabulary are not "grounding" by any reading; they
  are here because the rejection path calls them, and leaving them behind would
  have made this module import the worker back.
- _create_response lives here for the same reason: its six call sites are split
  between this module and the worker, and it had to sit where both could import
  it. It is not a grounding concern. A turns module imports it rather than
  declaring a second one - the per-site policy in its docstring is load-bearing.
- RealtimeSession stays in the worker. Moving a class both sides mutate is how
  a cycle gets built.
- Checks that answer "did the caller say this?" stay in evidence.py and are
  imported. Nothing here re-implements one.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:                    # pragma: no cover - typing only
    # Every `sess: "RealtimeSession"` below needs a binding or Pylance
    # reports eleven errors while the suite runs green — `from __future__
    # import annotations` makes them lazy strings, so nothing fails at
    # runtime and nothing tells you either. Exactly the trap evidence.py
    # documents at its own guarded import.
    #
    # TYPE_CHECKING is False when Python runs, so no import happens and no
    # cycle exists. The runtime dependency stays one way: realtime_worker
    # imports grounding, never the reverse.
    from agents.voice.realtime_worker import RealtimeSession
from agents.voice.evidence import _LOCATION_ANCHORS, _NON_PLACE, _ORG_STOPWORDS, _UNGROUNDED_STOPWORDS, _transcript_pending, _caller_speech_level, _distinctive, _grounding_verdict, _invites_continuation, _is_hint_echo, _meaning_class, _rode_along, _ungrounded_detail, _ungrounded_terms
from agents.voice.objectives import CallObjective, Outcome, default_objective, describe as _describe_objective, norm_quotes as _norm_quotes
from agents.voice.tools import run_tool
from datetime import datetime
from typing import NamedTuple, Optional
import json
import re
import time

log = logging.getLogger(__name__)


def _collected_pairs(sess: "RealtimeSession") -> set:
    """Every collected field as (name, value), not just its name.

    A SET OF NAMES CANNOT SEE A STATE FLIP, and on call-20260831-1048 that hid
    the exact moment the call was decided. `identity` was already collected
    when it was overwritten `confirmed` -> `unsure`, so the difference of the
    two name-sets was empty, the 🎯 line did not print, and the objective went
    PARTIAL -> COMPLETE with nothing whatsoever in the log to say so. The one
    line that would have told an operator why the call ended three seconds
    later was suppressed by the very write that ended it — the same shape as
    the metrics that tidied away the repeat before it could be counted.

    A FLIP IS ALSO PROGRESS, which is the other half of what `_gained` feeds.
    confirmed -> not_here is the caller putting us right, and a no-progress
    counter that cannot see it would keep ticking through the most informative
    turn of the call. The regression to `unsure` that motivated this is refused
    in _save_state now and never reaches here.

    Values are read through the field's own memory_key and lowercased, so this
    compares the same strings `present()` accepted rather than a second opinion
    about them.
    """
    obj = _objective_of(sess)
    return {(f.name, str(sess.memory.get(f.memory_key) or "").strip().lower())
            for f in obj.fields if f.present(sess.memory)}


def _objective_of(sess: "RealtimeSession") -> CallObjective:
    """The objective this call is working to.

    getattr, because the guards in this module are routinely handed a namespace
    carrying only the four attributes they read — see `double()` in the test
    suite — and a guard that raises on a test double is a guard that stops being
    tested.
    """
    obj = getattr(sess, "objective", None)
    return obj if isinstance(obj, CallObjective) else default_objective()

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

# The agent telling the caller the location is recorded, or that the call is
# finished. Both are false the moment save_branch returns a rejection, and the
# second is worse: it invites them to hang up.
#
# Two families, because they fail differently. "I'll save that" is a claim
# about the tool; "we'll be all set" is a claim about the call. The model
# produced BOTH in one sentence on call-20260818-1613.
# WIDENED 2026-08-26, after call-20260826-1650. The agent said
#
#     "Got it, thanks for clarifying - I'll go with Eastside Clinic."
#
# and the artifact records grounding as "verified against caller transcript
# EXCEPT 'eastside', which the caller was never transcribed saying". branch is
# null: the guard did its job on the WRITE. But guards gate the tool call,
# never the speech, so the caller was told a site name they never gave.
#
# The old pattern could not see it. It required a PRONOUN object - "note THAT",
# "record IT" - so every claim that named the field escaped: "I have noted the
# branch as X", "let me just note the location", "I'll go with X". Same shape
# as the _is_location_ask and rstrip bugs: fluent in one phrasing, blind to its
# neighbour.
#
# WIDENING IS SAFE, and worth stating rather than assuming. Both consumers
# already require that nothing was saved - the grounding site sits inside a
# save_branch REJECTION, the turns site checks `not sess.memory.get("branch")`.
# A false positive can only fire where the branch genuinely is not recorded.
_CLAIMS_SAVED = re.compile(
    r"\b(i'?ll (save|note|record|log|put|get) (that|it|this|them)"
    r"|i'?ve (saved|noted|recorded|logged|got) (that|it|this)"
    r"|got (that|it) (saved|noted|recorded|down)"
    r"|that'?s (saved|noted|recorded|logged|in)"
    r"|i('| a)m saving (that|it)"
    # the same claim with the FIELD NAMED instead of a pronoun
    r"|(i'?ll|i'?ve|i have|let me( just)?|i am|i'?m) ?"
    r"(save|saved|note|noted|record|recorded|log|logged|capture|captured"
    r"|put|putting)( down)? (the|that|this|your) "
    r"(branch|location|address|site|clinic|campus|name|detail|details"
    r"|referral|status|info|information)"
    # choosing a value out loud is a claim about the record too
    r"|(i'?ll|we'?ll|let'?s) (go with|use|put you down as)"
    r"|mark(ed)? (it|that) as"
    r"|we'?(ll be|re) all set|we'?re (done|all done|set|good)"
    r"|that'?s (everything|us|it) (done|sorted)?"
    r"|that'?s all i (need|needed)|that'?s (what|all) i needed"
    r"|i have (everything|what) i need"
    r"|all (set|sorted|done))\b", re.I)

def _claims_saved(text: str) -> bool:
    """Did this agent turn tell the caller the location is recorded, or done?"""
    return bool(_CLAIMS_SAVED.search(_norm_quotes(text or "")))


# A SIGN-OFF. Deliberately kept apart from _CLAIMS_SAVED, which is about a
# false claim of RECORDING; this is about ending the conversation.
#
# NOTHING IN THIS CODEBASE WATCHED FOR ONE. `sess.done` moves on exactly four
# events — escalate succeeding, a save completing the objective, the deferred
# close, and the caller ending the call — and every one of them is a TOOL or
# the caller. The agent saying goodbye was invisible.
#
# call-20260827-1516: the caller said it was a bad time and asked to be rung
# back. At 15:17:00 the agent said "No problem — take care." and called no
# tool, so nothing had ended anything: OpenAI's VAD opened a response on the
# caller's "Okay.", the model filled it, and the call ran another twenty
# seconds of politeness before the CALLER had to end it. escalate arrived two
# seconds after that, on the reply already in flight.
_SPOKEN_FAREWELL = re.compile(
    # `take care OF` is not a sign-off — "I'll take care of that" is a promise
    # to act, and reading it as goodbye would inject the escalate directive in
    # the middle of a call that is going fine.
    r"\b(take care(?!\s+of\b)|good ?bye|bye now|"
    r"have a (good|great|nice) (day|one|afternoon|evening|weekend)|"
    r"thanks? (you )?for your time)\b", re.I)


def _spoken_farewell(text: str) -> bool:
    """Did this agent turn sign off?

    Used ONLY together with `not sess.done` — a farewell is correct once
    something has ended the call, and the whole point of the guard is the case
    where nothing has.
    """
    return bool(_SPOKEN_FAREWELL.search(_norm_quotes(text or "")))

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
                           allow_when_active: bool = False,
                           allow_when_vad_pending: bool = False) -> bool:
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
      allow_when_vad_pending=True
                               the three RECOVERY sites — silence watchdog,
                               owed substance, empty-response re-request. They
                               exist because the expected response did NOT
                               arrive, so refusing them on the grounds that one
                               is expected is the exact inversion of their job.
                               Adding this parameter was not optional: without
                               it the watchdog went silent in the suite, which
                               is the dead air it was written to end.

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
    # THE CALLER HAS STOPPED AND OPENAI IS ABOUT TO ANSWER THEM. Our create
    # would be the second one in that conversation and the server refuses the
    # loser. Skipping is right on the merits as well as the mechanics: the
    # response we would be asking for is the one the VAD is already opening.
    _vad_due = getattr(sess, "_vad_response_due_until", 0.0)
    if time.monotonic() < _vad_due and not (allow_when_active
                                            or allow_when_vad_pending):
        log.info("[Realtime] response.create skipped (%s): OpenAI's VAD is "
                 "already opening one for the turn that just ended", why)
        return False
    await oai_ws.send(json.dumps({"type": "response.create"}))
    # OPTIMISTIC, AND THAT IS THE POINT. _response_active was set only when
    # `response.created` came back, leaving a whole round trip in which a
    # second call site could pass this same guard. Marking it here closes the
    # window against ourselves; the VAD window above closes it against OpenAI.
    sess._response_active = True
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

# Every closed-set save tool, with the argument carrying its value, the guard
# that grounds it, the NEED fragment for a rejection, and where the verdict is
# recorded. A TABLE rather than three near-identical elif branches: the three
# differ only in vocabulary, and the branch that handles one is the branch that
# must handle the next — which is exactly the drift that let the ask budget be
# generalised in the counters but not in the gate feeding them.
_CHOICE_SAVE_TOOLS: dict = {}

# RETIRED 2026-08-26, verbatim from _US_TRANSCRIBE_HINT and
# _PROVIDER_VERIFICATION_HINT. HELD SEPARATELY FROM THE HEALTH SYSTEMS ABOVE,
# and the split is the whole point rather than tidiness.
#
# Both halves feed _strip_hint_run, which needs six consecutive words and
# cannot realistically false-positive. Only THIS half feeds _hint_vocabulary,
# which condemns a location on ONE word.
#
# The first attempt at this fix fed the guard both halves and it then refused a
# caller who answers "which branch?" with "Baptist", "Methodist", "Providence"
# or "Mercy" — while the call that prompted the retirement was to New York
# Baptist Hospital. A health system IS a plausible one-word answer; "suite" and
# "campus" identify nothing and never were one.
#
# Feeding this half back reproduces the guard's reach EXACTLY as it stood while
# the hint was live — these are the words that were in it — so the retirement
# costs no protection. The health systems came out of the live hint on
# 2026-08-20 and were never covered by this guard; putting them in now would be
# a widening disguised as a repair.
_RETIRED_VOCAB_TEXT = (
    "Location words: campus, clinic, medical center, satellite office, "
    "north, south, east, west, downtown, midtown, uptown, suite, "
    "boulevard, avenue, parkway, drive, street. "
    "Scheduling words: waitlist, waiting list, referral, new patients, "
    "accepting, scheduling, insurance."
)

def _hint_vocabulary(hint: str) -> frozenset:
    """Every word the transcriber was primed with, lowercased.

    Derived from the live hint, never listed. The hint is the only thing that
    decides what the transcriber CAN echo, so it has to be the only thing that
    decides what we refuse to believe. A hardcoded list goes stale the moment
    the hint is edited — which is exactly what happened to _hint_proper_nouns
    when the health-system names came out of it, and why _RETIRED_HINT_TEXT
    had to be pinned separately to keep that detector alive.
    """
    # LIVE HINT PLUS _RETIRED_VOCAB_TEXT — and pointedly NOT _RETIRED_HINT_TEXT.
    #
    # Retiring the hint on 2026-08-26 would otherwise have disarmed this guard
    # silently, which is the failure the docstring above predicts for
    # _hint_proper_nouns. Feeding it the retired VOCABULARY restores exactly the
    # reach it had while that vocabulary was live, so nothing is lost.
    #
    # Feeding it the retired HEALTH SYSTEMS as well was tried and reverted: this
    # test condemns on one word, and it then refuses a caller who answers "which
    # branch?" with "Baptist" — on a call to New York Baptist Hospital. Those
    # names left the live hint on 2026-08-20 and this guard never covered them.
    return frozenset(re.findall(
        r"[a-z]+", ((hint or "") + " " + _RETIRED_VOCAB_TEXT).lower()))

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

    # t2 — the tool call is here.
    if sess._stage is not None and "t2" not in sess._stage:
        sess._stage["t2"] = time.monotonic()
        sess._stage["tool"] = name

    # THERE WAS A BLOCKING WAIT HERE, and it is gone. Every guard below asks
    # what the caller said, and the model reached this line from audio the
    # transcript has not necessarily caught up with — so this used to hold the
    # whole handler up to 1.5s for the words. Measured over 119 artifacts it
    # never once returned early (14 waits, 12 timeouts, 0 landed) and cost
    # 1.5s a time; the deferral below does the same job on the transcript event
    # itself, which is where the evidence actually appears. See the comment
    # above _transcript_pending for the distribution that settled it.

    # What the call had collected BEFORE this tool ran, so the no-progress
    # ceiling can be reset by progress rather than by a guess about which tool
    # constitutes progress. save_branch is not the only way a field arrives —
    # a template may point a field at a note_* key — and hard-coding the tool
    # name here is how the success condition ended up inside save_branch in the
    # first place.
    #
    # (name, VALUE) pairs, not names — see _collected_pairs. A field that is
    # overwritten with a different state is progress this set has to be able to
    # see, and until 2026-08-31 it could not.
    _collected_before = _collected_pairs(sess)

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
        sess.memory.update(grounding=_grounding_verdict(
            _rode, heard_any, getattr(sess, "branch_rejections", ())))
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
        elif ungrounded and _transcript_pending(sess):
            # ── THE WORDS ARE STILL IN FLIGHT ───────────────────────────
            # The same hold the choice fields have had since 2026-08-26,
            # which this path was missed out of. call-20260827-0942 is what
            # that omission costs, twice on one call:
            #
            #   waited 1.50s for the transcript and it never came
            #   BLOCKED {"branch": "Riverside campus"}
            #   CALLER : He works out at Riverside Campus.   <- one line later
            #
            #   waited 1.50s ... never came
            #   BLOCKED {"branch": "Riverside campus, 1477 10th Street"}
            #           (numbers 10, 1477 not in what the caller said)
            #   CALLER : I think it's 1477 10th Street.      <- one line later
            #
            # The caller heard the question a third time and said "Oh, no,
            # that's the specific address. I already told that." Then
            # escalate was refused - correctly - because the discard guard
            # could see they HAD given a location. Every guard was right on
            # its own terms and the call still ended with branch = null.
            #
            # Nothing is written here. _resolve_deferred_save re-runs
            # _ungrounded_terms against the real words the moment they land.
            sess._deferred_save = {
                "name": name, "args": dict(args),
                "why": ungrounded, "at": time.monotonic(),
                "asked_turns": len(sess.turns),
            }
            result = {"ok": True, "pending": True, "note": (
                "Held — their words are still being transcribed and this will "
                "be checked against them. Do not ask again and do not apologise; "
                "carry on.")}
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                  f"⏸️  BRANCH HELD FOR EVIDENCE: {args}", flush=True)
            print(f"          the transcript for that turn is still in flight "
                  f"— judging it when it lands, not now", flush=True)
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
            # DURABLE, not just printed. This project's recurring defect is
            # a guard that acts and leaves no trace, and this was the last
            # branch guard still failing that way: on call-20260827-1010 the
            # block appears nowhere in the artifact — "HALLUCINAT", "REJECTED"
            # and "blocked" all score 0 against the JSON — while doctors.json
            # carries status="verified". Whoever reads that row later cannot
            # tell it from a clean one, which is the whole point of the row.
            sess.branch_rejections.append({
                "value": str(args.get("branch") or ""),
                "why": ungrounded,
                "at": datetime.now().strftime("%H:%M:%S"),
            })
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
        # ── THE EVIDENCE HAS NOT ARRIVED YET IS NOT THE SAME AS THE EVIDENCE
        #    CONTRADICTS YOU, and until now both ended in the same refusal.
        #
        # call-20260826-1422: six saves, six waits, six timeouts, zero landed.
        # Every rejection was followed within the same second by the caller
        # transcript containing the answer. "nothing has been transcribed since
        # you asked" was true when asked and false a heartbeat later.
        #
        # The old comment on this path argued the cost of refusing was "one
        # more turn: the model saves again when the transcript lands". That is
        # not what the model did. It apologised — "I'm just making sure I heard
        # you clearly, since phone audio can clip a bit" — and re-asked a
        # question already answered, twice, on a 151s happy path.
        #
        # So when the guard objects and the words are STILL IN FLIGHT, hold the
        # decision instead of taking it. Nothing is saved here. The same guard
        # runs again the instant the transcript lands, against the real words.
        # If they never land, nothing is ever written — which is the behaviour
        # this branch has always had, reached by waiting rather than by
        # guessing.
        if ungrounded_choice and _transcript_pending(sess):
            sess._deferred_save = {
                "name": name, "args": dict(args),
                "why": ungrounded_choice, "at": time.monotonic(),
                "asked_turns": len(sess.turns),
            }
            # ok=True with nothing written is deliberate and it is the whole
            # point. ok=False is what produced the apology and the re-ask; the
            # model needs to hear "this is in hand, carry on", and the promise
            # is kept by _resolve_deferred_save, which will inject a correction
            # if the words do not bear it out.
            result = {"ok": True, "pending": True, "note": (
                "Held — their words are still being transcribed and this will "
                "be checked against them. Do not ask again and do not apologise; "
                "carry on.")}
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                  f"⏸️  HELD FOR EVIDENCE: {name}({args})", flush=True)
            print(f"          the transcript for that turn is still in flight "
                  f"— judging it when it lands, not now", flush=True)
        elif ungrounded_choice:
            sess.memory.update(**{_gkey: f"BLOCKED — {ungrounded_choice}"})
            # Terse fragments, no fluent imperative. A rejection the model can
            # paraphrase into a grammatical sentence is a rejection it will read
            # out loud — see _reject's docstring in tools.py, which exists
            # because a live call relayed one to a receptionist verbatim.
            # RE-READ REACHES ALL FIVE FIELDS NOW, not just the branch.
            # save_branch has carried this fragment since call-20260820-1321,
            # where two bare rejections left the model unable to tell WHAT was
            # wrong and it rephrased into a worse answer. The four choice
            # fields never had it — they got the reason and nothing about where
            # to look — and the prompt was carrying the difference in prose
            # ("RE-READ WHAT THEY ACTUALLY SAID BEFORE YOU ASK AGAIN") for all
            # five. This is a tool result, not the cached prefix, so unifying
            # it costs nothing against the prompt ceiling and lets that prose
            # go: the instruction now arrives at the moment it applies, on the
            # field it applies to.
            result = {"ok": False, "error": (
                f"NOT SAVED — {ungrounded_choice} "
                f"| RE-READ: their turns, verbatim; the answer is often "
                f"already among them "
                f"| NEED: {_need}")}
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
            # THE ONE TOOL THE DELETED WAIT COVERED THAT CANNOT DEFER.
            # _TRANSCRIPT_READING_TOOLS held six tools; five of them are saves
            # whose guard hands an objection to _resolve_deferred_save when the
            # words are still in flight. escalate has no such path — it ends
            # the call — so removing the wait would leave _discarded_location
            # reading a transcript that has not caught up, and the answer the
            # caller just gave would be invisible to the guard whose entire job
            # is noticing it. One turn of grace instead, and ONE-SHOT like every
            # other injected directive here: the placeholder resolves either
            # way within a turn, and a guard that can refuse forever is a call
            # that cannot be ended.
            if _transcript_pending(sess) and not sess._escalation_held:
                _blocked = "in flight"
            elif is_hold_request(last_caller):
                _blocked = "hold"
            elif _invites_continuation(last_caller):
                _blocked = "invitation"
        if _blocked:
            if _blocked == "in flight":
                sess._escalation_held = True
                result = {"ok": False, "error": (
                    "NOT ESCALATED — their last turn is still transcribing "
                    "| NEED: wait one turn; the answer may be in it")}
                _line = ("🚪 ESCALATION HELD — their last words are still "
                         "transcribing")
                _say = ("(system: hold on before ending the call. They have "
                        "just said something that has not reached you yet. "
                        "Wait for it — say nothing new and do not end the "
                        "call.)")
            elif _blocked == "hold":
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
            #
            # NOT FOR "in flight". That one says nothing about whether they
            # were engaging — the budget may have run out for perfectly good
            # reasons and we are only asking it to wait for words already on
            # the way. Resetting there would buy a stall a fresh budget.
            if _blocked != "in flight":
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
    _gained = _collected_pairs(sess) - _collected_before
    if _gained:
        # Named with the value it landed on, because a flip and a first
        # collection are now both in here and "collected identity" alone no
        # longer says which happened.
        _what = ", ".join(f"{n}={v}" for n, v in sorted(_gained))
        sess.reset_ask_budget("collected " + _what)
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
    # ── EVERY REFUSAL, ONE RECORD, WITH THE WORDS THAT CAUSED IT ───────────
    # The counter below this only ever covered save_branch, and a choice-field
    # refusal reached the artifact through nothing at all: on
    # call-20260827-1428 the identity save was refused at 14:29:35 and the
    # finished JSON contains no "BLOCKED", no "REJECTED", no "NOT SAVED".
    # Seven probe gaps have been found on this project by a person reading a
    # console log, and this is why — the evidence was never written down.
    #
    # `heard` IS THE POINT. A refusal without the caller turn that caused it
    # says a guard fired; with it, it says which phrasing the probe could not
    # read, which is the one thing that turns an audit into a fix. The deferred
    # path is not double-counted: a hold returns ok=True and records itself in
    # deferred_saves, and its own refusal lands there as "contradicted".
    if not ok and name.startswith("save_"):
        sess.save_refusals.append({
            "tool": name,
            "args": args,
            "why": str(result.get("error", ""))[:200],
            "heard": next((t.text for t in reversed(sess.turns)
                           if t.role == "caller" and t.text
                           and t.text.strip() != "[...]"), ""),
            "at": ts,
        })
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
            # SIGNED OFF WITHOUT ENDING ANYTHING. The correction is to make
            # the TOOL fire, never to hang up here: escalate is what writes the
            # reason, and on 1516 it was the only record of why the call
            # produced nothing. Cutting the line at the farewell would have
            # traded twenty seconds of politeness for a call with no outcome —
            # the trace-less failure this file exists to stop.
            #
            # One-shot, unlike the false-save claim beside it. That one answers
            # a separate false statement each time; this one asks for a single
            # tool call, and a second copy of a directive the model ignored is
            # context spent for nothing.
            if (_spoken_farewell(_said) and not sess.done
                    and not sess._farewell_nudged):
                sess._farewell_nudged = True
                sess.farewell_without_close.append(_said[:160])
                print(f"[{ts}] 👋 SIGNED OFF WITH NOTHING RECORDED — no tool "
                      f"has ended this call; asking for escalate", flush=True)
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
    #
    # ESCALATE IS NOT DEFERRABLE and the branch below does not touch it. It is
    # the model saying it has given up; holding that open for an answer is how
    # a call that has already failed stays on the line. Only the objective path
    # can be deferred, because only it can finish WITHOUT anyone deciding to.
    _close_deferred = False
    if name == "escalate" and result.get("ok"):
        sess.done = True
    elif (result.get("ok")
            and (name == "save_branch" or name in _CHOICE_SAVE_TOOLS)
            and _objective_of(sess).outcome(sess.memory) is Outcome.COMPLETE):
        # THE OBJECTIVE FINISHED ON A QUESTION WE HAVE NOT HEARD BACK ON.
        #
        # call-20260831-1048, and it is the second half of the same defect the
        # `sounded_like_a_goodbye` test below was written for. That test asks
        # the right question — an utterance ending in "?" is not a farewell —
        # and then has nowhere to put the answer: its only two branches are
        # "let the model's line stand as the goodbye" and "ask for a goodbye
        # anyway". Neither is "do not hang up yet". So the agent asked "would
        # scheduling be the best group to ask about where she sees patients?",
        # the objective flipped COMPLETE inside the same response, a goodbye
        # was requested with allow_when_done (which bypasses the playback
        # guard), and its audio began 1.43s BEFORE the question had finished
        # playing out. The caller was talked over and hung up on, mid-question.
        #
        # The third branch, then. The objective really is complete and nothing
        # here disputes that — the close is deferred, not cancelled, and it
        # re-arms the moment they answer (see _close_when_answered, consumed in
        # _handle_caller_transcript). If they never answer, the silence
        # watchdog still ends the call on its own budget, so this cannot hold a
        # line open indefinitely.
        # UNANSWERED, WHICH IS NOT THE SAME AS "the last agent turn ends in ?".
        # The happy path ends every call on a question the caller then answered
        # — "which location is Dr. Okafor practising at?", "She's at the
        # Northgate campus." — and the save that completes the objective is
        # grounded in that very answer. Reading only the last AGENT turn would
        # defer the close on every well-run call in the suite. So walk back
        # from the end: a real caller turn in between means the question was
        # answered and nothing is owed. A "[...]" placeholder does not — that
        # is their answer still in flight, which is a reason to wait, not to
        # hang up.
        _unanswered = ""
        for _t in reversed(sess.turns):
            if _t.role == "caller":
                if (_t.text or "").strip() != "[...]":
                    break
                continue
            if _t.role == "agent":
                if (_t.text or "").rstrip().endswith("?"):
                    _unanswered = _t.text.rstrip()
                break
        if _unanswered:
            sess._close_when_answered = True
            _close_deferred = True
            print(f"\n[{ts}] ⏸️  CLOSE DEFERRED  : objective complete, but "
                  f"the turn just spoken is a question they have not "
                  f"answered — waiting for them\n"
                  f"          asked: {_unanswered[-70:]!r}", flush=True)
        else:
            sess.done = True

    await oai_ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {
            "type":    "function_call_output",
            "call_id": call_id,
            "output":  json.dumps(result),
        },
    }))
    # t3 — answered. Everything after this point is OpenAI's, and on the
    # deferred path (sess.done False) nothing is even ASKED of it until
    # response.done arrives. That gap is t4-t3 and it is the cost of the
    # deferral, isolated.
    if sess._stage is not None and "t3" not in sess._stage:
        sess._stage["t3"] = time.monotonic()

    # EVERY TOOL IN THE TURN, WITH ITS VERDICT - not just the first.
    #
    # t2/t3 above deliberately mark only the FIRST tool, because that is what
    # the inference_1 interval measures. But a response may carry several tool
    # calls, and recording only the first made call-20260826-1656 unreadable:
    # identity is saved `confirmed`, the only save_doctor_identity in the stage
    # data sits on a turn whose transcript the guard REJECTS, and the stored
    # quote appears in two different caller turns. Which turn grounded identity
    # could not be determined from the artifact at all.
    #
    # Appended here because `result` is final at this line - every accept,
    # reject and hold path has converged by the time the output goes out.
    #
    # Written with sess._stage[...] rather than .setdefault so it stays inside
    # the shapes the measure-only test allows: this list is written and never
    # read by anything that decides behaviour.
    if sess._stage is not None:
        if "tools" not in sess._stage:
            sess._stage["tools"] = []
        sess._stage["tools"].append(
            {"tool": name, "ok": bool(result.get("ok"))})

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
    elif _close_deferred:
        # NO CONVERSATION ITEM AND NO response.create. The response carrying
        # this tool call has already put a question to them; the only correct
        # next sound on this line is theirs.
        #
        # Which is why this is its own branch and not a fall-through to the
        # `else`. `_pending_response_create` fires a create at the next
        # response.done — that is right after an ordinary tool call, where the
        # model has a result to speak to, and wrong here, where speaking again
        # is the whole thing being avoided. It would talk over the question by
        # the same 1.4s the injected goodbye did, minus the goodbye.
        #
        # Left as None in the outcome below rather than set False, so the event
        # loop keeps whatever it already had — see _ToolOutcome on why None is
        # not False.
        pass
    else:
        _pending_response_create = True
    return _ToolOutcome(_agent_text_buf, _closing_sent,
                        _pending_response_create, False)

async def _resolve_deferred_save(sess: "RealtimeSession", oai_ws) -> None:
    """Judge a held save now that the caller's words have actually arrived.

    THE SAME GUARD, ON THE SAME ARGUMENTS, AGAINST REAL EVIDENCE. Nothing here
    is more permissive than the path that deferred: `_guard` is re-read from
    `_CHOICE_SAVE_TOOLS` rather than carried along, so a save can only pass by
    satisfying exactly the check that objected. The deferral buys the guard its
    evidence; it does not lower the bar, and the model's `heard` string is
    never consulted — selection runs over the transcript as it always did.

    Three outcomes, all recorded:
      applied      the words bear it out; the tool runs for real, late
      contradicted the words arrived and refuse it; the model is told to ask
                   again — the re-ask this whole mechanism avoids is CORRECT
                   here, because now there is evidence for it
      (unresolved) the transcript never came; nothing is written and the row
                   is closed out in save() as dropped

    WHY THE CORRECTION IS INJECTED RATHER THAN RETURNED. The tool call this
    belongs to was answered a turn ago, with ok=True. There is no result left
    to fail. A conversation item is the only channel that still reaches the
    model, and it is the same one the false-save and silence directives use.
    """
    held = sess._deferred_save
    if held is None:
        return
    sess._deferred_save = None
    name = held["name"]
    # TWO FAMILIES, ONE RESOLVER. The choice fields carry their guard in
    # _CHOICE_SAVE_TOOLS; save_branch has its own guard and no BLOCKED
    # memory key, so _gkey is None for it and the contradiction path below
    # skips the memory write rather than inventing a key.
    if name == "save_branch":
        _guard = _ungrounded_terms
        _need = "their own words, any number in digits"
        _gkey = None
    else:
        spec = _CHOICE_SAVE_TOOLS.get(name)
        if spec is None:
            return
        # The value argument is not needed here — the guard reads it off
        # `args` itself. Named out rather than indexed so this stays in step
        # with _CHOICE_SAVE_TOOLS if its shape changes.
        _, _guard, _need, _gkey = spec
    args = held["args"]
    waited = round(time.monotonic() - held["at"], 2)
    still = _guard(args, sess)
    ts = datetime.now().strftime("%H:%M:%S")

    if not still:
        result = run_tool(name, sess.memory, args, sess.objective)
        ok = bool(result.get("ok"))
        sess.deferred_saves.append(
            {"tool": name, "args": args, "waited_s": waited,
             "outcome": "applied" if ok else "refused_by_tool",
             "held_because": held["why"],
             "error": None if ok else str(result.get("error"))[:160]})
        if ok:
            sess.reset_ask_budget(f"deferred save landed: {name}")
            print(f"\n[{ts}] ✅ {name.upper()} (held {waited:.2f}s for the "
                  f"transcript): {args}", flush=True)
            print(f"[Realtime] 🎯 "
                  f"{_describe_objective(_objective_of(sess), sess.memory)}",
                  flush=True)
            # THE DEFERRED PATH COULD NOT END A CALL. The synchronous tool
            # handler sets sess.done when a successful save completes the
            # objective — see "WHEN THE CALL IS OVER" above. This path runs the
            # same tool, for real, a turn later, and had no such check. On
            # call-20260827-1010 the new-patient status landed here,
            # `outcome=complete` printed, and the call ran another 24 seconds
            # and four agent turns, ending on a "Take care." the model had
            # already said once.
            #
            # A FLAG, NOT THE GOODBYE, and both halves of that are forced:
            #  - `_closing_sent` is a local of the event loop. We are inside
            #    _handle_caller_transcript, which returns None and "shares NO
            #    mutable state with the event loop" by design. Injecting the
            #    closing here would leave `_closing_sent` False, so the
            #    in-flight response's own response.done would read "done,
            #    nothing pending" and hang up ON the goodbye we just asked for.
            #  - the sounded_like_a_goodbye test cannot run yet. The response
            #    in flight has not produced its transcript, so there is no last
            #    agent turn to inspect — and hanging up on a question is the
            #    live defect that test exists for.
            # Both are answerable one event later, which is where it is done.
            if (not sess.done
                    and _objective_of(sess).outcome(sess.memory)
                        is Outcome.COMPLETE):
                sess._close_after_response = True
                print(f"[Realtime] 🏁 objective complete on the deferred save "
                      f"— closing after the response already in flight",
                      flush=True)
        else:
            print(f"\n[{ts}] ⛔ {name.upper()} REJECTED after the wait: "
                  f"{result.get('error', '')}", flush=True)
        return

    # THE WORDS ARRIVED AND THEY DO NOT SUPPORT IT. Refuse exactly as the
    # undeferred path would have, and say so out loud — a held save that
    # quietly evaporates is the invisible-guard failure this project keeps
    # paying for.
    # save_branch has no BLOCKED key of its own; only the choice fields do.
    if _gkey:
        sess.memory.update(**{_gkey: f"BLOCKED — {still}"})
    sess.deferred_saves.append(
        {"tool": name, "args": args, "waited_s": waited,
         "outcome": "contradicted", "held_because": held["why"], "why": still})
    print(f"\n[{ts}] 🚫 HELD ANSWER REFUSED — the transcript arrived and does "
          f"not bear it out: {name}({args})", flush=True)
    print(f"          {still}", flush=True)
    await oai_ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": (
                     f"(system: the answer you recorded for {name} was not "
                     f"borne out by what they actually said. It has NOT been "
                     f"saved. Ask them again, plainly, and wait for their "
                     f"reply. NEED: {_need})")}]},
    }))


# EVERY NAME realtime_worker RE-EXPORTS, DECLARED. Two of these — the two
# entry points — are called only from the worker, so without this the
# checker reports the module's own reason for existing as dead code. Same
# purpose as evidence.py's list: it says what the module is FOR, and it
# keeps a hint storm from burying a real one.
__all__ = [
    "_CALLER_WILL_ACT",
    "_CALL_SHAPE_EXITS",
    "_CHOICE_SAVE_TOOLS",
    "_CLAIMS_SAVED",
    "_FACTUAL_ESCALATIONS",
    "_HOLD_REQUEST",
    "_IDENTITY_ASK",
    "_MAX_SAVE_REJECTIONS",
    "_ORG_WORD",
    "_RETIRED_VOCAB_TEXT",
    "_SELF_ID",
    "_SELF_ID_WEAK",
    "_STREET_ADDRESS",
    "_STREET_SUFFIX",
    "_ToolOutcome",
    "_address_dropped",
    "_address_offered",
    "_candidate_location",
    "_claims_saved",
    "_spoken_farewell",
    "_create_response",
    "_discarded_location",
    "_handle_tool_call",
    "_hint_vocabulary",
    "_is_bare_hint_word",
    "_collected_pairs",
    "_objective_of",
    "_resolve_deferred_save",
    "_strip_ungrounded_detail",
    "_ungrounded_escalation",
    "hospital_mismatch",
    "is_hold_request",
]
