"""The vocabulary of a tool call: patterns, thresholds, and the tool registry.

PURE. Nothing here takes a session and nothing here reaches up the package.
It is the bottom of the layering, so a module needing one phrase list can
take it without pulling a guard, a socket or a lifecycle in behind it.

_CHOICE_SAVE_TOOLS LIVES HERE, which reads oddly for a table of guards
until you ask who consults it: the handlers, the reporting, the close
decision and the deferred resolver, which sit in four different layers
above. A registry read by everything belongs under everything. It is also
MUTABLE ON PURPOSE - realtime_worker fills in the four field guards with
.update() once they exist, because each one needs a probe and a vocabulary
this layer must not know about.
"""
from __future__ import annotations

import re
import time

from agents.voice.evidence import (
    _ungrounded_terms,
)
from agents.voice.objectives import (
    norm_quotes as _norm_quotes,
)

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
    r"thanks? (you )?for your time"
    # THE TEMPLATE'S OWN GOODBYE WAS NOT IN HERE. patient_discovery
    # teaches a close that names no farewell word at all - templates.py:
    # "Let me just figure out my schedule, and I'll call back. Thanks!" -
    # and this pattern returned False on that sentence verbatim, on both
    # closes the model actually spoke on 2026-09-02 (1511 "Let me think
    # about it and I might call back", 1544 "Let me sort out my schedule,
    # and I'll call back"), and so on every well-behaved close the persona
    # makes. A guard that cannot recognise the goodbye its own prompt asks
    # for is measuring nothing.
    #
    # FIRST PERSON AND FUTURE, deliberately. "I'll call back" is a
    # sign-off; "should I call back later?" and a relay of the caller's
    # "we will call you back" are not, and neither matches - the subject
    # and the contraction are the whole test. Applied to AGENT turns only.
    r"|(?:i'?ll|i will|i might|i may) call (?:you )?back"
    # "let me think about it" IS NOT A SIGN-OFF ON ITS OWN, and it stood here
    # for one round before a live call proved it. It was drawn from 1511's
    # "Let me think about it and I might call back", where the farewell is the
    # CALL-BACK clause -- which the alternative above already matches. Alone it
    # is a conversational hold: on call-20260902-1842 the caller asked "Would
    # you like me to add you there?" and the agent said "let me think about it
    # for a moment", which this read as a goodbye. Every close the template
    # teaches carries a call-back or a have-a-good-day, so nothing is lost.
    r"|let me (?:just )?(?:figure|sort) out my schedule"
    r"|(?:i'?ll|i will) get back to you)\b", re.I)



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
    "_SPOKEN_FAREWELL",
    "_STREET_ADDRESS",
    "_STREET_SUFFIX",
    "_claims_saved",
    "_hint_vocabulary",
    "_is_bare_hint_word",
    "_spoken_farewell",
    "is_hold_request",
]
