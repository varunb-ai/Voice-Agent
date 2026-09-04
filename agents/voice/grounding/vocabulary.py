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
    #
    # AND IN THE FIRST PERSON FUTURE, which is the same correction arriving a
    # second time on the same alternative. This read `let me (figure|sort) out
    # my schedule` and call-20260902-2207 closed with "I appreciate you
    # explaining that - I'LL sort out my schedule and think it over. Thanks for
    # letting me know." That is the taught close, spoken well, and this pattern
    # returned False on it - so `farewell_without_close` was null again, on a
    # goodbye, which is exactly the blindness the guard exists to end. It cost
    # nothing there only because the objective was already COMPLETE and the
    # close deferred; on a call where nothing had ended it, nothing would have
    # noticed. Same shape as the "I'll call back" alternative above, and the
    # same reasoning: the subject and the contraction are the whole test.
    r"|(?:let me|i'?ll|i will) (?:just )?(?:figure|sort|work) out my schedule"
    r"|(?:i'?ll|i will) get back to you)\b", re.I)



def _spoken_farewell(text: str) -> bool:
    """Did this agent turn sign off?

    Used ONLY together with `not sess.done` — a farewell is correct once
    something has ended the call, and the whole point of the guard is the case
    where nothing has.
    """
    return bool(_SPOKEN_FAREWELL.search(_norm_quotes(text or "")))


# Already thanked them. Narrow on purpose: this decides whether to TELL the
# model it has thanked them, so a false positive is an instruction that is
# not true of the call.
_ALREADY_THANKED = re.compile(r"\b(thanks|thank you|appreciate|grateful)\b",
                              re.I)


def closing_directive(last_agent: str = "") -> str:
    """The item asked for when a goodbye has to be requested.

    ONE DEFINITION, TWO CALL SITES. This string was written out twice —
    teardown.py's tool-call close and lifecycle.py's deferred close — and the
    twin next to it, `sounded_like_a_goodbye`, was fixed in one place and left
    wrong in the other for a day and two live calls. The same shape does not
    get to happen to the sentence those two branches ask for.

    THE THANKS CLAUSE IS CONDITIONAL ON THE FACT. call-20260903-1422 closed
    with "Thanks, that helps — I'm just going to think about it for now."
    followed by "Okay, thanks for explaining that." — the second thank-you in
    three seconds, and the tell everybody heard. But a model told it has
    already thanked them when it has not is being lied to about its own call,
    which is how a directive stops being trusted; so the clause is added only
    when the last turn actually carries one.

    A DIRECTIVE, NOT A SCRIPT. It says what not to do and leaves the wording
    alone — a fixed farewell string is the identical-hold-acknowledgement tell
    the prompt already warns about, one turn later.
    """
    _base = "(say a brief warm goodbye now, then stop. ONE short sentence."
    if _ALREADY_THANKED.search(_norm_quotes(last_agent or "")):
        _base += (" You have ALREADY thanked them in the turn you just spoke"
                  " — do not thank them again.")
    return _base + (" Do not repeat or rephrase anything you just said, and"
                    " do not raise anything new.)")


# ── The turn that promises a question and does not ask one ───────────────────
#
# TOOL-CALL PADDING. The model opens its mouth, says it is about to ask
# something, fires a tool instead, and the response ends. Nothing is on the
# wire after that: the caller has been told a question is coming and given
# nothing to answer, so they wait, and the silence watchdog will not speak for
# seven seconds because as far as it knows the ball is with them.
#
# call-20260902-2002, twice inside forty-six seconds:
#   20:01:01  "Okay, thanks for that. Let me just ask one more thing."  -> save_branch
#   20:01:47  "Okay, thanks for checking - let me ask one quick thing
#              about that."                                            -> note_info
# and again on -2005 at 20:07:04. Both times the next sound on the line was
# the caller giving up on waiting.
#
# A PROMPT RULE WAS THE OTHER OPTION AND IT COSTS MORE THAN IT BUYS. The
# patient_discovery template is already 5,285 tokens against its own 5,400
# ceiling, so this rule could only arrive by evicting one that is carrying its
# weight - and the standing evidence on this project is ~14 code guards holding
# on live calls against ~6 prompt rules that did not. The shape of a turn is
# exactly what a guard can read.
_ANNOUNCED_ASK = re.compile(
    r"\b(?:"
    # First person, about to ask. The slack in the middle is deliberate: the
    # prompt tells the model to vary its wording, and five separate probe
    # defects on this project have been a regex that took it at its word.
    r"(?:let me|lemme|i'?ll|i will|i need to|i just need to|i want to|"
    r"i'?d like to|i have to|i'?m going to|i'?m gonna|gonna)\s+"
    r"(?:just\s+|quickly\s+|then\s+|also\s+|first\s+){0,2}"
    # THE VERB LIST IS THE OBSERVED ONE. `sort out`, `run through` and
    # `go over` stood here for one call and came out again: the template
    # teaches its close as "let me sort out my schedule, and I'll call back",
    # so `sort out` made that goodbye read as a promised question on
    # call-20260902-2207. Every observed padding turn used `ask` or `clear up`;
    # the rest were added on no evidence, and one of them collided with the one
    # sentence the prompt asks the model to say at the end of every call.
    r"(?:ask|check|confirm|clarify|clear up|double[- ]check)\b"
    # The question named and not asked. THE ADJECTIVE IS REQUIRED: a bare "a
    # question" matches "that's a good question", which is an answer to them
    # rather than a promise to them.
    r"|(?:one|a|another)\s+(?:more|last|final|quick|other|small)\s+"
    r"(?:thing|question|detail|point)\b"
    r"|quick question\b"
    # The wrap-up preamble, which is the same failure with a closing flavour -
    # and the more expensive one, because a caller who hears it starts saying
    # goodbye.
    r"|before\s+(?:we|i)\s+(?:wrap|finish|close|go|let you go|move on|end|"
    r"hang up)\b"
    r")", re.I)

# Anything in the turn that is itself the substance. A turn carrying a house
# number, or a name spelled out letter by letter, has DONE something: the
# spell-and-confirm repair is exactly that shape ("let me check the spelling -
# O-K-A-F-O-R") and it is correct for the line to go quiet afterwards, because
# the caller is being asked to confirm it.
_PADDING_SUBSTANCE = re.compile(r"\d|\b[A-Za-z](?:[-\s][A-Za-z]){2,}\b")

# Padding is short by nature - the three observed turns are 9, 12 and 15 words.
# Generous, because the cap is a bound and not the test: what actually decides
# this is the silence that follows, and a long turn that genuinely announced
# nothing simply does not reach the pattern above.
_PADDING_MAX_WORDS = 30


def _announced_an_ask(text: str) -> bool:
    """Did this agent turn promise a question without asking one?

    JUDGED ON THE TURN ALONE, so it says nothing about whether the question
    ever arrived. That is the caller's business and it is answered by the
    clock - see the padding recovery in _silence_watchdog, which fires only if
    the line then stays silent. A model that announces a question and asks it
    in the next breath trips this predicate and nothing happens, which is the
    correct outcome and the reason this can afford to be generous.

    A QUESTION MARK ANSWERS IT OUTRIGHT. "Let me just ask one more thing -
    which office is she at?" is a well-formed turn; the announcement is the
    preamble to an ask that is right there.
    """
    t = _norm_quotes(text or "").strip()
    if not t or "?" in t:
        return False
    if len(t.split()) > _PADDING_MAX_WORDS:
        return False
    if _PADDING_SUBSTANCE.search(t):
        return False
    # A SIGN-OFF IS NOT A PROMISE OF A QUESTION, and this is the general form
    # of the `sort out` collision above rather than a second patch on it. Every
    # close this persona is taught announces something the AGENT will go and do
    # - think it over, sort out a schedule, call back - which is the same
    # grammar as announcing a question and means the opposite of it. Whatever
    # _SPOKEN_FAREWELL can recognise, this must not chase.
    if _SPOKEN_FAREWELL.search(t):
        return False
    return bool(_ANNOUNCED_ASK.search(t))


# The agent announcing it will go and do something INTERNAL — think, note,
# check, work it out. Shape, not a phrase list, for the same reason
# _HOLD_REQUEST is: "ways to say you are about to go away and do something"
# is an open set.
#
# THE SIBLING OF _ANNOUNCED_ASK, not a duplicate of it. That one asks "did it
# promise a QUESTION and not ask one"; this asks "did it say anything that
# advances the line at all". "Let me just think about how I want to handle the
# waitlist" promises no question, so _announced_an_ask is correctly False on
# it — and it is still not an answer to anything.
_AGENT_STALL = re.compile(
    r"\b(?:let me|lemme|i'?ll|i will|i'?m going to|i'?m gonna|give me)\s+"
    r"(?:just\s+|quickly\s+|only\s+|a\s+)*"
    r"(?:think|thought|note|record|log|jot|write|mull|consider|figure|"
    r"work(?:\s+out)?|sort|check|look|see|review|process|handle|decide)\b",
    re.I)


# A NAME AND A DATE OF BIRTH IN ONE BREATH. The pair is the whole hazard: the
# person on the other end has a patient record open, and name + DOB is all it
# takes to start one for somebody who has never been seen.
#
# KEYED ON THE YEAR, NOT ON THE PERSONA'S NAME, for the reason the
# unsolicited_pii_dumps detector gives: the synthetic name varies per doctor
# and is not available to a pure predicate, while the year range is
# synthetic_identity's own (1960 + seed % 45). A four-digit year inside an
# agent turn that is also introducing itself is that date being read aloud.
_GAVE_NAME = re.compile(r"\b(?:my name(?:'s| is)|name's|i'?m called)\b", re.I)
_GAVE_DOB = re.compile(r"\b(?:19\d{2}|200[0-4])\b")


def _gave_name_and_dob(text: str) -> bool:
    """Did this ONE agent turn hand over both the name and the date of birth?

    call-20260903-1126 and eight before it. The prompt's rule — "asked your
    name, you say your name and stop", because "a name and a date of birth
    arriving together is the single event this whole section exists to
    prevent" — held on NONE of the nine calls where a receptionist asked for
    intake details. It is stated three separate ways in the prompt and was
    ignored every time, which is this repo's standing result for a prose rule
    against a guard.

    The robotic doubling everyone heard ("I haven't registered with you yet,
    but my name is X. I'm not in your system yet, but it's <date>.") is the
    same event: two scripted disclaimers concatenate precisely because two
    details left together. Fixing the safety failure removes the bad audio;
    smoothing the audio into one sentence would have removed the safeguard.

    NOT the same question as unsolicited_pii_dumps, which asks whether PII
    arrived with no prior ask and is silent here because the caller DID ask.
    """
    t = _norm_quotes(text or "")
    return bool(_GAVE_NAME.search(t) and _GAVE_DOB.search(t))


# ── The not-a-patient line ───────────────────────────────────────────────────
# THE SENTENCE THAT STOPS THEM TYPING. The person on the other end has a
# patient record open; the disclaimer in front of a detail is what keeps the
# detail out of it. Two questions are asked of it and they pull in opposite
# directions, which is why one predicate answers both:
#
#   said twice running  -> a recording, and the tell everyone heard on
#                          call-20260903-1422.
#   said not at all     -> the safety failure, and until now nothing in this
#                          repo looked for it. Nine calls of prose failing is
#                          the standing evidence that nothing was.
#
# MATCHED BY SHAPE, not by the two sentences CALL CONTEXT happens to build.
# Those are per-call strings and a predicate that pinned them would go quiet
# the moment the wording moved — the false-negative shape this suite has been
# bitten by repeatedly. A negation, then a not-yet-a-patient word within the
# same clause, with slack between them: the prompt asks the model to vary its
# wording and five separate probe defects on this project have been a regex
# that took it at its word.
#
# FIRST PERSON THROUGHOUT, and a bare "not" is deliberately not a cue. Some of
# the status words below are ordinary things to say about somebody else — "I'm
# not sure she's a patient of his" — so a negation with no subject attached
# would read that as the persona disclaiming. The subject is the whole test,
# the same discriminator _SPOKEN_FAREWELL's "I'll call back" alternative is
# built on.
_NOT_YET = (r"(?:haven'?t|have\s+not|hadn'?t|had\s+not|i'?m\s+not|"
            r"i\s+am\s+not|we'?re\s+not|we\s+are\s+not|"
            r"(?:i|we|they|you)'?(?:ve|d)\s+(?:not|never)|"
            r"(?:i|we)\s+(?:have\s+)?never)")

_NOT_A_PATIENT = re.compile(
    r"(?:"
    # The not-yet-on-your-books words. Slack between the two halves because
    # the prompt asks the model to vary its wording and five separate probe
    # defects on this project have been a regex that took it at its word —
    # but bounded by "no sentence end" as well as by length, so "I'm not sure.
    # My name is Ingrid." cannot match across the full stop.
    + _NOT_YET + r"[^.?!]{0,44}?"
    r"\b(?:register(?:ed|ing)?|in\s+(?:your|the)\s+system|on\s+file|"
    r"on\s+your\s+books|signed\s+up|set\s+up|enrolled|intake|"
    r"been\s+seen|come\s+in\s+before|seen\s+(?:him|her|them)\s+before)\b"
    r"|"
    # "a patient" gets NO slack, and that is the difference between "I'm not a
    # patient there yet" and "I'm not sure she's a patient of his". Forty
    # characters cannot tell those apart; adjacency can.
    r"\b(?:i'?m|i\s+am|we'?re|we\s+are)\s+not\s+(?:yet\s+)?"
    r"(?:a|an|your|their)\s+patient\b"
    r"|"
    r"\b(?:haven'?t|have\s+not|i'?ve\s+(?:not|never))\s+(?:ever\s+)?"
    r"been\s+(?:a\s+)?patient\b"
    r")",
    re.I)


def _said_not_a_patient(text: str) -> bool:
    """Did this agent turn carry the not-a-patient disclaimer?"""
    return bool(_NOT_A_PATIENT.search(_norm_quotes(text or "")))


def _gave_own_detail(text: str) -> bool:
    """Did this agent turn hand over one of the persona's own PII details?

    NAME AND DATE OF BIRTH ONLY, and the omission of the address is deliberate
    twice over. The pair is what starts a record — that is the whole hazard
    _gave_name_and_dob names — and an address predicate would have to tell the
    persona's street from the PRACTICE's, which the agent reads back all the
    time ("so that's the Northgate clinic on Main Street"). A detector that
    fires on the caller's own address is a detector nobody will trust.

    Keyed the same way `_gave_name_and_dob` is keyed, and for the reason its
    docstring gives: the synthetic name varies per doctor and is not available
    to a pure predicate, while the year range is synthetic_identity's own.
    """
    t = _norm_quotes(text or "")
    return bool(_GAVE_NAME.search(t) or _GAVE_DOB.search(t))


def _detail_left_bare(text: str, prev_agent: str,
                      same_exchange: bool) -> bool:
    """A detail handed over with no disclaimer standing in front of it.

    THE GUARD THAT PAYS FOR THE RELAXATION. templates.py used to demand the
    disclaimer on EVERY answer, however many times they asked, and the model
    duly produced two identical constructions ten seconds apart. That rule is
    now one turn narrower — a line said in the turn you just spoke is still
    standing — and this is what makes narrowing it an improvement rather than
    a hole: before, nothing anywhere checked that a detail carried a
    disclaimer at all. The rule was prose, stated three ways, and prose failed
    on all nine calls where a receptionist ran intake.

    `same_exchange` is the caller-turn count between this turn and the
    previous agent turn, collapsed to a bool by the caller: exactly one caller
    turn in between is them asking the follow-up. Two or more, or an exchange
    about something else, and the earlier disclaimer is stale — a receptionist
    who has been talking about the waiting list for a minute is not still
    holding "she isn't registered" in mind.
    """
    if not _gave_own_detail(text):
        return False
    if _said_not_a_patient(text):
        return False
    return not (same_exchange and _said_not_a_patient(prev_agent))


def _agent_stalled(text: str) -> bool:
    """Did this agent turn speak without answering or asking anything?

    WHY IT EXISTS: call-20260903-1126. The caller asked "Would you like me to
    add you to the list?", the agent said "Okay, let me just think about how I
    want to handle the waitlist", and the close walk in _decide_close read
    "there is an agent turn after their question" as "we answered them" and
    hung up mid-exchange. SPEAKING IS NOT ANSWERING, and that walk had no way
    to tell them apart.

    THE TWO EXCLUSIONS ARE THE WHOLE PREDICATE, and both are load-bearing:

      a question mark  — "Let me check: which office is that?" is an ask. The
                         turn puts the ball back in their court, which is
                         exactly what a stall does not do.
      a farewell       — every close this persona is taught has stall grammar
                         ("let me sort out my schedule, and I'll call back"),
                         so without this the taught goodbye reads as a stall
                         and the call could never end. Same collision
                         _announced_an_ask documents, and the same fix.

    Safe in one direction only, and it is the right one: a false positive
    keeps a question open for one more turn, while a false negative hangs up
    on someone mid-sentence.
    """
    t = _norm_quotes(text or "").strip()
    if not t or "?" in t:
        return False
    if _SPOKEN_FAREWELL.search(t):
        return False
    return bool(_AGENT_STALL.search(t))


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
    "_ALREADY_THANKED",
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
    "_NOT_A_PATIENT",
    "_agent_stalled",
    "_detail_left_bare",
    "_gave_name_and_dob",
    "_gave_own_detail",
    "_said_not_a_patient",
    "_announced_an_ask",
    "_spoken_farewell",
    "closing_directive",
    "is_hold_request",
]
