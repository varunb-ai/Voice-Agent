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


def closing_directive(last_agent: str = "", *,
                      react_to_news: bool = False) -> str:
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
    # WHEN THE SAVE THAT COMPLETED THE CALL IS THE NEWS ITSELF.
    # call-20260907-1755 closed on "Okay, thanks for that-let me just note what
    # you said and then I'll wrap up." followed by "Alright, take care." The
    # caller had just told them the doctor IS taking new patients -- the whole
    # reason for the call -- and the agent narrated its own bookkeeping at it.
    # The default below asks only for a goodbye, so a goodbye is all it got.
    #
    # POSITIVE ONLY, AND NO WORDING. It names what to react TO, never how; the
    # corpus records the model reproducing quoted phrases from directives 67
    # times in 470 turns, so this hands over none. The no-invention clause is
    # the one constraint that has to be stated: a patient who has just heard
    # good news is one turn away from thanking them for an appointment nobody
    # offered.
    if react_to_news:
        _base = ("(react briefly to what they just told you and what it means"
                 " for you, in your own words, then say goodbye. ONE or two"
                 " short sentences. Claim no appointment, visit or next step"
                 " they did not offer.")
    else:
        _base = "(say a brief warm goodbye now, then stop. ONE short sentence."
    if _ALREADY_THANKED.search(_norm_quotes(last_agent or "")):
        _base += (" You have ALREADY thanked them in the turn you just spoke"
                  " — do not thank them again.")
    return _base + (" Do not repeat or rephrase anything you just said, and"
                    " do not raise anything new.)")


# ── The enquiry-bot cadence, made observable ─────────────────────────────────
#
# WHY THIS IS CODE AND NOT A PROMPT RULE. Measured over the 57
# patient_discovery calls: 290 of 413 non-greeting agent turns (70%) opened
# with an acknowledgement and 177 of those ran back to back —
#
#   "Okay, thanks for that—let me ask one quick thing about where he sees
#    patients."      "Okay, I just need to check something simple first."
#   "Got it, thanks for confirming that. Let me check one more thing."
#   "Okay, thanks for confirming that. Let me ask about availability next."
#
# — five of the eleven agent turns on call-20260904-1306, and the receptionist
# learned nothing from any of them. THE ARTIFACT SCORED THAT CALL CLEAN:
# piled_turns 0, stapled_questions 0, repeated_sentences 0,
# shared_opening_clauses 0, tool_call_padding null. Every existing counter
# measures the STRUCTURE of a turn — sentence counts, repeats, staples — and
# this defect is in its CONTENT, so nothing saw it. shared_opening_clauses
# reads 0 here because "Okay, thanks for confirming that" and "Got it, thanks
# for confirming that" are two different clauses.
#
# The prompt has asked for the right thing throughout ("NOT owed every turn",
# "twice running is a tic", "Vary how turns open") and the standing evidence on
# this project is ~14 code guards holding on live calls against ~6 prompt rules
# that did not. So this is the guard.

# Content-free openers, in the order the corpus ranks them: thanks 84, got it
# 61, okay 58, oh 47, sure 23. YES AND NO ARE DELIBERATELY ABSENT — "Yes, this
# is an automated call" is an ANSWER, and the one thing this must never
# discourage is answering a question straight. "yeah" is here because in this
# corpus it only ever appeared as an acknowledgement, and it is bounded by the
# comma-or-stop the pattern requires after it.
_ACK_OPENER = re.compile(
    r"^\s*(?:so\s+)?"
    r"(oh|okay|ok|alright|all right|right|got it|gotcha|"
    r"thanks|thank you|sure|great|perfect|understood|i see|"
    r"well|mm-?hm+|mm|yeah|yep)"
    r"\b[\s,.!?—–-]*", re.I)


def _ack_opener(text: str) -> str:
    """The acknowledgement this turn opens with, or "" — never a judgement.

    One of these on its own is ordinary human speech and this says nothing
    about it. The tic is two in a row; that is the call site's question, not
    this function's.
    """
    m = _ACK_OPENER.match(_norm_quotes(text or ""))
    return m.group(1).lower() if m else ""


# Thanking them for taking part in your own process. Not for HELP — for
# confirming, checking, waiting, holding on: the steps of the workflow.
_TAKING_PART = re.compile(
    r"\bthanks?(?:\s+you)?\s+(?:so\s+much\s+|very\s+much\s+)?for\s+"
    r"(?:that|this|the\s+\w+\s+)?"
    r"(?:confirm|check|wait|hang|hold|bear|explain|look|clarif|shar)", re.I)


def _housekeeping_turn(text: str) -> bool:
    """A turn spent thanking them for participating, with nothing in it.

    THE QUESTION MARK IS THE WHOLE DISCRIMINATOR, and it is the prompt's own
    rule rather than a new one: the ban is on the STANDING form — a thanks that
    IS the turn, or that fronts an announcement — while the same words with the
    question folded into the same breath are exactly what the prompt asks for.
    So "Thanks for checking — is there a waiting list?" is fine and
    "Thanks for confirming that." is a turn the receptionist waited through for
    nothing.

    A FAREWELL IS EXEMPT. Thanking them for explaining the wait list on the way
    out is not housekeeping, it is the close, and flagging it would fire this on
    the happy path — which is how a metric stops being read.

    Measured on the corpus: 45 turns, every one of them content-free or a
    thanks fronting an announcement, and none of the allowed joins.
    """
    t = _norm_quotes(text or "").strip()
    if not _TAKING_PART.search(t):
        return False
    if "?" in t:
        return False
    return not _spoken_farewell(t)


# ── Narrating the reply instead of giving it ─────────────────────────────────
#
# A DIFFERENT FAMILY FROM _ANNOUNCED_ASK, and it has to stay different. That
# one is about a promised QUESTION — the caller is left with nothing to answer,
# and `tool_call_padding` counts it. This is about a promised ANSWER: they asked
# for a date of birth and got "Mm-hmm, one moment while I answer that." The
# caller is not waiting for a question, they are waiting for the fact. Folding
# the two together would make that artifact field stop meaning what it says.
#
# FOUND BY RENDERING, NOT BY READING. The 2026-09-04 prompt rebuild deleted
# every quoted phrase the model had been lifting, and the A/B renders showed
# the verbatim reproductions gone — no more "Sure, no rush.", no more "Of
# course, take your time." What survived was this, generated fresh every time:
#
#   before: "Sure, one moment while I respond to that."   (and the date never
#           arrived in either take)
#   after:  "Okay, let me answer that."
#           "Let me think for a moment."
#           "Let me think about the next step here."
#
# _announced_an_ask catches NONE of those — it is keyed to ask/check verbs —
# so nothing in the process disagreed with them. That is the whole argument for
# a predicate rather than another prompt line: the prompt has banned narration
# in capitals with an operational test attached since it was written.
_NARRATED_REPLY = re.compile(
    r"\b(?:"
    # First person, about to produce the reply rather than producing it.
    # `share` and `give you` are in the list, and were left out of the first
    # cut on a false-positive worry that the closed-loop run then falsified in
    # the other direction: "Sure, let me share that with you." and "Sure, let
    # me give you that now." were 2 of 3 takes on the date-of-birth turn, this
    # did not fire, and the WRONG directive went out in its place. The
    # first-person about-to prefix is what keeps them safe — "thanks for
    # sharing that" has no such prefix and is _TAKING_PART's business.
    r"(?:let me|lemme|i'?ll|i will|i'?m going to|i'?m gonna|gonna)\s+"
    r"(?:just\s+|quickly\s+|first\s+|then\s+){0,2}"
    # `listen` ADDED 2026-09-04, and it is the verb both known leaks used:
    # "Let me listen carefully to what they said and then respond clearly."
    # (call-20260904-1651, the prompt spoken aloud) and "I'll listen for the
    # name so we can pin it down." (call-20260904-1734). It belongs to this
    # family and not to _AGENT_STALL, which exempts any turn containing a "?"
    # — the second of those asked its question in the same breath, so the
    # stall predicate is structurally unable to see it.
    #
    # Announcing that you are listening is never an act that helps them; it is
    # the narration of one. Checked over 1,272 corpus turns: +2 hits, both the
    # turns above, no other turn in any template reaches it.
    r"(?:think|answer|respond|reply|explain|address|share|listen|"
    r"give (?:you|it|that)|get (?:you )?(?:that|it) (?:for|to) you)\b"
    # "one moment while I answer that", "a second while I think"
    r"|\b(?:one|a|just a)\s+(?:moment|minute|second|sec)\s+"
    r"(?:while|and|before)\s+i\b"
    # "give me a second", "let me have a moment"
    r"|(?:give|gimme)\s+me\s+(?:a|one|just a)\s+"
    r"(?:moment|minute|second|sec)\b"
    # "I'm trying to remember", "I'm just thinking"
    r"|i'?m\s+(?:just\s+)?(?:thinking|trying to remember|working (?:it|that) "
    r"out)\b"
    r")", re.I)


def _narrated_the_reply(text: str) -> bool:
    """Said they were about to answer, instead of answering.

    THE CLOSE IS EXEMPT, and it is the one case that matters. This script
    teaches its goodbye as going away to think about it — "let me think about
    it and I might call back" is the happy path and appeared 8 times in the
    corpus. A detector that fires on the close fires on every good call, which
    is how a number stops being read. _spoken_farewell is the same exemption
    _housekeeping_turn takes, for the same reason.
    """
    t = _norm_quotes(text or "").strip()
    if not _NARRATED_REPLY.search(t):
        return False
    return not _spoken_farewell(t)


# ── The instructions arriving in the audio ───────────────────────────────────
#
# call-20260904-1651, the first live call after the patient-behaviour rebuild.
# The caller said "Ah." and the agent said, out loud:
#
#     "Let me listen carefully to what they said and then respond clearly."
#     "Take your time."
#
# The second sentence is correct and is what the prompt asks for. The first is
# the model reading its own instructions back to the receptionist — it is a
# first-person paraphrase of a rule in templates.py:
#
#     "- RESPOND TO WHAT THEY JUST SAID. Their turn decides yours ..."
#
# NOT A DIRECTIVE LEAK, which was the obvious suspect and was checked: the
# re-ask guard whose wording is nearest ("respond to what they actually said")
# prints a console marker when it fires, and that marker is absent from the
# call. It came from the cached prompt.
#
# ── WHY THIS IS DETECTABLE WITHOUT A PHRASE LIST ────────────────────────────
# The prompt and every injected directive are written ABOUT the call, so they
# refer to the caller in the THIRD PERSON — "they", "them", "their question".
# A patient speaking TO that person says "you". So a third-person reference to
# the CALLER'S OWN SPEECH ACT is the instruction register arriving in the
# audio, whatever words carry it, and no list of narration verbs is needed.
#
# THE VERB SET IS CLOSED AND THE PRONOUN IS THE TEST. "she"/"they" meaning the
# DOCTOR or the PRACTICE is how this agent legitimately talks for the whole
# call — "is she taking new patients", "do they have more than one site" — so
# a bare pronoun ban would fire on nearly every turn. What cannot happen in
# natural speech to someone is narrating THAT person's words in the third
# person while talking to them.
#
# Measured over 481 agent turns of patient corpus: ONE hit, the one above, and
# no false positive on any of the legitimate third-person forms.
_LEAKED_INSTRUCTIONS = re.compile(
    r"\b(?:"
    r"what\s+(?:they|he|she)\s+(?:just\s+)?"
    r"(?:said|told|asked|mentioned|answered|meant)"
    r"|(?:they|he|she)\s+(?:just\s+)?"
    r"(?:said|told|asked|answered|mentioned)\s+(?:me|us)\b"
    r"|(?:respond|reply|answer|speak|talk)\s+(?:back\s+)?to\s+(?:them|him|her)\b"
    r"|(?:answer|address)\s+(?:them|their)\b"
    r"|(?:their|his|her)\s+(?:question|answer|reply|response)\b"
    # ── THE FIRST-PERSON HALF, ADDED 2026-09-08 ─────────────────────────────
    # Everything above is the instruction register arriving as a THIRD-PERSON
    # reference to the caller. call-20260907-1814 arrived as first person about
    # the agent's own delivery, seven seconds after its own greeting, with no
    # caller turn in between at all:
    #
    #   "Let me speak quietly and sort out whether you've reached the right
    #    place. Do you know if Dr. Browne..."
    #
    # "speak quietly" is _TONE_PATIENT's opening imperative ("SPEAK NOTICEABLY
    # SLOWLY AND QUIETLY") read back onto the phone. Nothing saw it: this
    # pattern wants they/him/her and the turn has none, and _NARRATED_REPLY's
    # verb class is think/answer/listen/explain — promised ANSWERS — with no
    # verb of speech-manner in it. Two guards, one gap between them.
    #
    # THE ADVERB IS THE WHOLE TEST, and it is why this is structural rather
    # than a phrase list. "Let me speak to my husband about it" is an ordinary
    # patient sentence; "let me speak QUIETLY" is a statement about how one's
    # own voice will sound, which is a thing the instructions say and a thing
    # no caller has ever needed to announce. Requiring a manner word after the
    # verb keeps every real-world "speak/talk/say" out.
    #
    # Measured over 1,349 agent turns of corpus, every template: TWO hits, both
    # genuine and both invisible to the other two guards — the turn above, and
    # call-20260806-2029's "Sorry, I'm speaking fast. Let me say that more
    # clearly.", which is the Pacing & Delivery block narrated the same way.
    # No false positive on any turn in the corpus.
    r"|(?:let me|lemme|i'?ll|i will|i'?m going to|i'?m gonna)\s+"
    r"(?:just\s+|now\s+|first\s+|try to\s+){0,2}"
    r"(?:speak|talk|sound|say (?:this|that|it))\s+"
    r"(?:a bit\s+|a little\s+|more\s+){0,2}"
    r"(?:quiet|quietly|soft|softly|slow|slowly|calm|calmly|clear|clearly|"
    r"brief|briefly|gently|warmly|naturally|plainly|low|lower)\b"
    r")", re.I)


def _leaked_the_instructions(text: str) -> bool:
    """Did this turn narrate the exchange in the register of the prompt?"""
    return bool(_LEAKED_INSTRUCTIONS.search(_norm_quotes(text or "")))


def _stapled_own_detail(text: str) -> bool:
    """Handed over a name or date of birth AND asked something in one breath.

    "A PLAIN FACT IS A PLAIN ANSWER" made checkable. templates.py carries that
    rule in capitals and call-20260904-1651 did this anyway:

        "April 21, 1984.
         Any chance you could tell me if Dr. Browne is taking new patients
         right now?"

    — and the receptionist answered the staple with "Okay.", so the question
    had to be asked again on the next turn. Three turns in 481 across the
    corpus, every one of them genuine.

    _gave_own_detail rather than a date regex, so this stays keyed to the same
    predicate the EHR guards use and cannot drift away from what counts as a
    detail.
    """
    t = _norm_quotes(text or "")
    return bool(_gave_own_detail(t) and "?" in t)


def cadence_directive(kind: str, want: str = "") -> str:
    """What to tell the model when one of the three cadence guards fires.

    ONE DEFINITION, TWO CALL SITES, for the reason closing_directive gives:
    turns.py injects these on a live call and scripts/check_cadence_loop.py
    replays them against the real model, and a hand-copy in the script would
    be exercising a string that no longer goes out.

    `want` IS THE WHOLE DIFFERENCE BETWEEN THIS WORKING AND NOT. Measured by
    the loop script, which seeds an exchange, takes the model's real turn,
    fires the real guard and takes the next turn. The first version of these
    directives named only the failure — "start on the thing you are actually
    saying" — and the corrections that came back were "Okay.", "Let me just
    get the location sorted out with you." and "Thanks for confirming that
    this is Dr. Okafor's office.": nine tries, none of them good. Passing the
    outstanding field produced "Which office does Dr. Okafor see people at?"
    and, on the date-of-birth turn twice over, "January 18, 1977."

    Take `want` from CallObjective.next_spoken(), not missing_spoken(): the
    labels read as clauses, so the article missing_spoken adds would hand the
    model "the whether they're taking new patients" to say.
    """
    # ONE CLAUSE, USED TWICE, so the two directives cannot drift — and it says
    # WHAT to say and nothing about how, because each caller adds its own
    # "start on it" clause afterwards. An earlier cut put "starting on the
    # question itself" in here as well, and the housekeeping directive then
    # went out saying it twice in consecutive sentences.
    _say = (f"Ask them about {want}, in one short sentence." if want else
            "Say the next thing you actually want to say.")
    if kind == "ack_run":
        return (f"(system: you have opened two turns in a row with an "
                f"acknowledgement, which is the cadence of somebody working "
                f"through a form. {_say} Start on the thing itself — no "
                f"opener in front of it, no okay, no thanks, no got it. For "
                f"the rest of this call an acknowledgement is only for "
                f"something that genuinely landed as news.)")
    if kind == "housekeeping":
        return (f"(system: you just spent a whole turn thanking them for "
                f"taking part in your own call. They waited through it and "
                f"learned nothing. {_say} Start on the question itself, with "
                f"no thanks in front of it. Thank them only for help they "
                f"went out of their way for, once, and never as a turn of its "
                f"own.)")
    if kind == "leaked_instructions":
        # NO `want` — steering to the next field would answer a question
        # nobody asked, and the fault is not what they said but that a note to
        # itself went out over the phone.
        return ("(system: that last turn was you talking about the "
                "conversation rather than in it — you described what you were "
                "going to do before doing it, and the person on the phone "
                "heard all of it. Everything you say is spoken to them. Say "
                "the thing itself, and say it to them as 'you', never about "
                "them as 'they'.)")
    if kind == "stapled_detail":
        return ("(system: you gave one of your own details and asked a "
                "question in the same breath. They answered the question and "
                "the detail went past them, so you now have to ask again. A "
                "plain fact is a plain answer: say the fact, and stop. "
                "Whatever you wanted to ask keeps until they have replied.)")
    if kind == "reply_narration":
        # NO `want` HERE, deliberately. They have just asked for something
        # specific and are waiting on it; steering to the next objective field
        # would answer a question nobody asked.
        return ("(system: you just told them you were about to answer instead "
                "of answering. They are waiting on the thing they asked for. "
                "Give it now, plainly, and nothing else. A pause before you "
                "speak is human; a sentence about the pause is you describing "
                "yourself, which a real caller never does.)")
    raise ValueError(f"unknown cadence directive {kind!r}")


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
#
# WIDENED 2026-09-04, and the measurement is the argument. Across the 57
# patient_discovery calls, 31 caller turns are a hold to any human reading
# them and this matched 22 — so nine holds went unrecognised, the 45-second
# stand-down never armed, and _silence_watchdog asked whether they were still
# there while they were doing exactly what they had said they would do. That
# reached the transcript six times as a verbatim "Still with me?", which is
# the single worst-sounding line in the corpus.
#
# The nine split three ways and every one of them is an ordinary front-desk
# sentence:
#   "Give me a few seconds while I pull up her schedule"   x7
#       — `give me a (minute|moment|sec|second)` cannot see "a few seconds",
#         and `while I pull up` was unreachable because this gate rejects
#         first, before _CALLER_WILL_ACT ever runs.
#   "One minute, let me confirm with the doctor"
#       — "one moment" and "one sec" were listed; bare "one minute" was not.
#   "let me give one minute to check whether the doctor is available"
#       — "let me give", with no "me" after the verb.
#
# CONFIRM AND VERIFY ARE THE RISKY ADDITION, so they carry their own exclusion
# below. "I need to confirm one thing from you" is the caller about to ASK
# something, not about to go and look — and reading it as a hold would buy 45
# seconds of dead air on a turn that wants an answer, which is the regression
# _CALLER_WILL_ACT was written to prevent in the first place.
_HOLD_REQUEST = re.compile(
    r"\b(?:(?:let me|lemme|i'?ll|i will|i need to|i have to|i'?m going to|"
    r"gonna)\s+"
    r"(?:just\s+)?(?:check|look|see|find|ask|grab|pull|confirm|verify|give)"
    r"|while\s+(?:i|we)\s+"
    r"(?:just\s+)?(?:check|look|see|find|ask|grab|pull|confirm|verify)"
    r"|(?:give|gimme)\s+(?:me\s+)?(?:just\s+)?(?:a|one)?\s*"
    r"(?:few\s+|couple\s+(?:of\s+)?)?(?:minutes?|moments?|secs?|seconds?)"
    r"|(?:can|could|would)\s+you\s+(?:just\s+|please\s+)*(?:wait|hold|hang on)"
    r"|(?:hold on|hang on|one (?:moment|minute|sec|second)"
    r"|just a (?:minute|moment|sec|second)"
    r"|bear with me))\b", re.I)


# They want something FROM the agent, not time away from them. "I need to
# confirm one thing from you" and "let me check with you" are asks wearing a
# hold's grammar; a 45-second stand-down on either is dead air on a turn that
# is waiting for an answer.
_WANTS_IT_FROM_US = re.compile(
    r"\b(?:confirm|verify|check|ask|clarify)\b[^.?!]{0,30}?"
    r"\b(?:from|with|of)\s+you\b", re.I)


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
    # Asked for BEFORE the cooperative branch below, because "I need to confirm
    # one thing from you" satisfies _CALLER_WILL_ACT ("I ... confirm") and
    # would otherwise be settled as a hold by it.
    if _WANTS_IT_FROM_US.search(t):
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


# Anything that turns naming a place into denying it. Used by the bare-name
# clear in hospital_mismatch, where the whole risk is that "we're NOT Northside"
# contains the name exactly as "this is Northside" does.
_NAME_NEGATED = re.compile(
    r"\b(?:not|no|never|nope|wrong|different|another|other|instead|"
    r"isn'?t|aren'?t|wasn'?t|weren'?t|don'?t|doesn'?t|didn'?t|won'?t|"
    r"used to|no longer|any ?more)\b|n'?t\b", re.I)


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


# How many times we tell the model that re-sending byte-identical arguments
# will not change the answer. Two, matching the cadence directives: a
# correction ignored twice is being ignored, and a third costs transcript
# without buying anything. Deliberately NOT imported from turns.py, which
# imports this package — the number is small enough that one definition per
# side is cheaper than the cycle.
_MAX_REFUSED_REPEAT_NUDGES = 2


__all__ = [
    "_ALREADY_THANKED",
    "_CALLER_WILL_ACT",
    "_CALL_SHAPE_EXITS",
    "_CHOICE_SAVE_TOOLS",
    "_CLAIMS_SAVED",
    "_FACTUAL_ESCALATIONS",
    "_HOLD_REQUEST",
    "_IDENTITY_ASK",
    "_MAX_REFUSED_REPEAT_NUDGES",
    "_MAX_SAVE_REJECTIONS",
    "_NAME_NEGATED",
    "_ORG_WORD",
    "_RETIRED_VOCAB_TEXT",
    "_ACK_OPENER",
    "_LEAKED_INSTRUCTIONS",
    "_NARRATED_REPLY",
    "_SELF_ID",
    "_SELF_ID_WEAK",
    "_SPOKEN_FAREWELL",
    "_TAKING_PART",
    "_WANTS_IT_FROM_US",
    "_STREET_ADDRESS",
    "_STREET_SUFFIX",
    "_claims_saved",
    "_hint_vocabulary",
    "_is_bare_hint_word",
    "_NOT_A_PATIENT",
    "_ack_opener",
    "_agent_stalled",
    "_detail_left_bare",
    "_housekeeping_turn",
    "_leaked_the_instructions",
    "_stapled_own_detail",
    "_narrated_the_reply",
    "_gave_name_and_dob",
    "_gave_own_detail",
    "_said_not_a_patient",
    "_announced_an_ask",
    "_spoken_farewell",
    "cadence_directive",
    "closing_directive",
    "is_hold_request",
]
