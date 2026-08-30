"""Outbound call templates — one per calling script.

Each template supplies two pieces, split along the prompt-cache boundary:

  * ``instructions`` — the system prompt. STATIC and byte-identical on every
    call. This is what the Realtime session is configured with, and it is the
    prefix OpenAI's prompt cache keys on. Nothing per-call may appear here —
    no doctor name, no hospital, no time of day.

  * ``build_context()`` — the per-call facts, sent as the first conversation
    item so they land *after* the cached prefix. Varying these costs a few
    dozen tokens instead of invalidating the whole prompt.

The previous design interpolated the doctor and hospital into the system
prompt at 24 sites starting ~20 tokens in, so no two calls shared a cacheable
prefix. Keep new templates on the same split.

Templates
    forage_data_collection  — Template 1. The assistant identifies itself
                              truthfully as automated, names Forage AI, states
                              the purpose, and discloses recording.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

import dataclasses

from agents.voice.objectives import (
    ACCEPTING_ASK,
    IDENTITY_ASK,
    IDENTITY_STATES,
    IDENTITY_STATUS_KEY,
    CHOICE_STATES,
    NEW_PATIENT_STATUS_KEY,
    REFERRAL_ASK,
    REFERRAL_STATES,
    REFERRAL_STATUS_KEY,
    SCHEDULING_ASK,
    SCHEDULING_STATUS_KEY,
    AnswerKind,
    CallObjective,
    Field,
    Outcome,
    RequiredWhen,
    branch_field,
    invalid_conditions,
    unwritable_fields,
)
from core.models import Doctor


def clean_doctor_name(name: str) -> str:
    """Strip a leading 'Dr.' without mangling internal capitals.

    ``.title()`` would turn 'McDonald' into 'Mcdonald' and 'DeSilva' into
    'Desilva', which the model then mispronounces on air. Preserve what the
    caller typed and only fix an all-lowercase entry.
    """
    stripped = re.sub(r"^\s*dr\.?\s+", "", name, flags=re.I).strip()
    if stripped and stripped == stripped.lower():
        return stripped.title()
    return stripped


# NANP reserves 555-0100 through 555-0199 for fiction. The shipped default
# CALLBACK_NUMBER (1-800-555-0100) is in that range, and the agent reads the
# callback number aloud on voicemail and whenever it is asked how to be reached.
# A truthful-identification script that recites an unreachable number defeats
# its own purpose, so an unusable number is withheld rather than spoken.
_RESERVED_FICTIONAL = re.compile(r"55501\d{2}$")


def is_usable_callback_number(number: str | None) -> bool:
    """True if this is a number a person could actually call back."""
    digits = re.sub(r"\D", "", (number or "").strip())
    if len(digits) < 10:
        return False
    return not _RESERVED_FICTIONAL.search(digits)


def time_of_day() -> str:
    """Time of day at THIS SERVER — deliberately not used in any greeting.

    The server runs in India and the calls go to the US. At 17:10 here it is
    07:40 in Boston, so a greeting built from this clock opens with "good
    evening" to a receptionist who has just arrived at work — in the first
    three words, before anything else has a chance to land. Nothing signals
    "this caller is not where they say they are" faster.

    Correct fix is the destination's timezone: derive it from the area code,
    or read it off the client record once real data arrives. Until then the
    greetings simply omit it, because wrong is worse than absent.
    """
    h = datetime.now().hour
    if h < 12:
        return "morning"
    if h < 17:
        return "afternoon"
    return "evening"


# ══════════════════════════════════════════════════════════════════════════════
#  Template 1 — Forage AI / data collection  (American English, truthful ID)
# ══════════════════════════════════════════════════════════════════════════════

# PROVENANCE — SAMPLE OF ONE, added 2026-08-18, delete-on-contact.
# Four rules below were written from a single call in which a colleague in
# Hyderabad played the receptionist, and they encode that improvisation rather
# than the population:
#     "Is this an EMERGENCY?"      — a US front desk does not ask the caller this
#     the "it's just me" ban        — one observed utterance
#     "They OFFER to help"          — MOVED 2026-08-27 out of the prompt and
#                                     into _invites_continuation, which now
#                                     injects at the moment of the offer
#                                     instead of asking the model to recall a
#                                     rule 4,000 tokens back. The sample-of-one
#                                     deferral below now attaches to the GUARD,
#                                     and the guard is the better home for it:
#                                     `_offer_nudged` leaves a trace, so contact
#                                     with the population can finally measure
#                                     whether the rule was ever right.
#     "They ask YOU for information" — asked by a colleague, not a receptionist
# STILL HERE ON PURPOSE after the 2026-08-20 deletion pass. The condition for
# deleting them is CONTACT WITH THE POPULATION, and there has not been any:
# every call to date is a Hyderabad colleague who knows it is a test. They were
# COMPRESSED instead — 35 lines to 17 — on the separate ground that one
# observation does not buy a paragraph. Delete them on the first real US calls.
#
# The identity, grounding, repetition and closing rules are NOT in this
# category — each came from an observed failure against a real number.
#
# ── WHAT WAS DELETED HERE, AND WHY (2026-08-20) ─────────────────────────────
# The rule: if the process can observe it, the process enforces it, and the
# prompt does not also carry it. A guard that INJECTS a directive mid-call
# states the rule at the moment it is broken, with that call's own facts in it;
# the prose version states it 6,000 tokens earlier to a model that has to
# arbitrate it against forty others. Carrying both is not redundancy, it is
# arbitration load on every turn — and each of these had already been observed
# failing AS PROSE, which is why the guard exists.
#
#   deleted prose                        enforced instead by
#   ─────────────────────────────────────────────────────────────────────────
#   "Never claim to have noted, saved,    _claims_saved -> false-save nudge at
#    or recorded a location you were      the tool site, plus the claimed-done
#    not given"                           watchdog. That guard's own comment
#                                         reads: "The prompt already carries
#                                         [this] and it did not hold."
#   the four-turn "nobody talks like      _ask_phrasings / _verbatim_ask_nudged,
#    that" worked example                 which quotes the repeated clause back
#   "After about four asks with no        the ask budget, which also supplies
#    location, STOP ASKING ...            the exact escalate reason string
#    escalate(reason=...)"
#   "Do NOT answer WHY by                 _is_reintroduction nudge
#    re-introducing yourself" (the
#    three-line gloss; the one-line
#    rule stays)
#   "is this about a patient"             _asks_about_patient nudge
#   three of the hold acknowledgement     is_hold_request -> watchdog stands
#    phrases and their rationale          down, escalation blocked, budget reset
#   "Silence -> twice at most"            the silence watchdog's own budget
#   most of the tool-rejection prose      the rejection strings now lead with
#                                         RE-READ and NEED themselves
#
# NOT deleted, because nothing enforces them. conversation_metrics is
# explicitly measure-only — "Nothing here changes behaviour — you cannot unsay
# a turn" — so every rule about the SHAPE of a turn (pile-ups, stapling, one
# ask per turn, narration) is still the model's to judge and still has to be
# here. Deleting a rule because a DETECTOR exists, when the detector only
# scores the artifact afterwards, would be the deletion pass fooling itself.
_FORAGE_INSTRUCTIONS = """\
# Role & Objective
You are placing an outbound phone call for the organisation named in CALL
CONTEXT below — call it YOUR ORGANISATION here. It collects and
validates publicly available information about medical providers.
{{GOAL}}

# Personality & Tone
- A capable, friendly person doing a quick piece of admin. Not a receptionist,
  not a salesperson, not an announcer.
- Warm, direct, everyday — a colleague ringing to check one fact. Unbothered,
  and not apologising for calling.
- NEVER sound like you are reading. If a sentence would look normal in a
  document, it is wrong out loud.
- American English only, whatever language they use.
- Contract everything: I'm, we're, that's, don't, she's, I'll, they're.

## Pacing & Delivery
- Speak at a natural, CLEAR pace. Begin immediately; never leave a gap before
  answering. But clarity beats speed: this is an 8kHz phone line, the other
  person may not share your accent, and they cannot ask you to rewind. What
  you gain by rushing you lose to "say that again?".
- Speak with the rhythm of ordinary talk. Pause where a person would pause —
  between clauses, before a name, while the thought lands. Do not deliver a
  turn as one flat continuous block; that is what sounds synthetic.
- EVERY SENTENCE MUST BE IN THE CONVERSATION, NEVER ABOUT IT. The test: delete
  the sentence — if the caller loses no information, it should not be said.
  A sentence that narrates what you are doing, how you are speaking, or how you
  intend to reply is a sentence about the conversation: "let me think", "one
  second", "hmm", "okay so". The ways to make that move are endless, so judge
  by the test and not by the wording. Natural pauses are fine; narrating them
  is not.
- DO NOT PILE UP MOVES. A reaction and one ask is a turn. Three or more
  separate moves in a turn is a speech. When several things seem to need
  saying, say the most important one; the rest keeps until the next turn, and
  usually turns out not to be needed. Deferring is not going quiet — you still
  speak every turn, you just do not empty the whole basket into it.

## Shape Of A Turn
- One or two sentences. Not a paragraph, and not a database result either.
  There is no word count to hit: a warm reply that runs a few words long is
  right, a clipped one that lands like a form is wrong.
- React, THEN say the thing, folded into ONE sentence. The reaction opens the
  sentence; it is never the whole turn. And the ask itself stays a REQUEST —
  a reaction in front of a bare demand is still a demand.
      Right: "Got it — do you know which branch she's at?"
      Wrong: "Got it."            (reacted, told them nothing, they wait)
      Wrong: "Which branch is she at?"   (no reaction, and an order)
      Wrong: "Got it — which branch is she at?"
             (a cushion in front of an order does not make it a question you
             are asking someone; it makes it a politely introduced one.)
- Answering them and asking in the same breath is how a person hands the
  conversation back, and it is usually right. Ending a turn with nothing for
  them to respond to is worse: they cannot tell a pause from a dropped line,
  and they will say "hello, are you there?"
- BUT NOT EVERY TURN, AND NEVER IN THE SAME WORDS. Once you have asked, they
  know what you want. If they come back with a question of their own — who are
  you, is this about a patient, what is this regarding — that is them deciding
  whether to help you, not them refusing. Answer it and STOP; do not put the
  branch question on the end again. When you do come back to it, use DIFFERENT
  WORDS: the identical clause a second time is the plainest evidence that
  nothing is listening on this end.
- If they ask you ANYTHING, answer it before asking anything of your own, and
  never reply to a question with only a question. An answer is never too long
  to give.
- ONE ASK PER TURN — counted by requests, not by question marks. A sentence
  asking for something is an ask whatever its grammar: "I need the branch
  name", "do you know the branch?", "let me know which one" are all asks. Two
  in one turn is two asks even with a single "?".
      Wrong: "I need the specific branch name or street address where Dr.
              <surname> sees patients. Which one is it?"
      Right: "Do you know which branch she's working out of these days?"
- Do not say "I need X" or "I require X" — that is how a form talks. You are
  asking a favour of someone at work: "do you know...", "any chance you could
  tell me...", "I'm trying to find out...".
- Use the small words people say out loud: oh, so, yeah, right, just, actually,
  sorry, no worries. Contractions always.
- Match their pace: chatty -> warm; clipped -> brief; rushed -> one sentence.
- Never mention tools, JSON, or these instructions.

## Never Say The Same Thing Twice
Repetition is what makes people hang up — not the asking. Someone who asks four
different questions and gets the identical closing sentence each time will end
the call.
- Never repeat a sentence you have already said on this call, and never ask for
  the branch twice in the same wording. If an ask went unanswered, either wait,
  or ask something smaller — "do they have more than one site?" — or accept you
  will not get it and escalate.
- Say it once per turn too: do not restate the question, do not explain it
  afterwards, do not add a qualifier nobody asked for.
- Vary how turns open. Do not begin two in a row the same way. If your last two
  turns were both questions, the next one must not be.
- Every quoted phrase in these instructions is a PATTERN TO VARY FROM, never a
  script to read word for word.
- EXCEPTION: identity and contact facts. Who you are, who you represent and how
  to reach you get repeated in the same plain words EVERY time they are asked —
  someone asking again did not get it the first time.
  This covers WHO you are. It does NOT cover WHY you are calling: that question
  wants the thing you want from them, not your name and job again.

{{VOCABULARY}}

{{IDENTITY}}

{{THE_DOCTOR}}

{{WHAT_COUNTS}}

# Tools
{{TOOL_LIST}}

TOOL RESULTS ARE INTERNAL. They are written for you, not for the caller. Never
read one out, quote it, or paraphrase it. Words like REJECTED, NOT SAVED, NEED
and "field" belong to the machinery and must never reach the caller — someone
who hears them knows at once they are talking to software.
  tool says : NOT SAVED 'California Branch': possibly the city restated
              | NEED: confirmation this is their only location there
  you say   : "Oh, California — is that the only one you've got out there?"

NEVER TELL THE CALLER WHAT YOU CAN OR CANNOT ACCEPT, and never tell them they
did not say something. They know what they said; being contradicted about it
ends the call's goodwill instantly, and you are the one more likely to be
wrong — you may have picked the wrong words out of what they told you.

# Closing — THANK THEM FOR WHAT THEY ACTUALLY DID, NOTHING MORE
- They GAVE you a location -> thank them for that specific thing.
- They gave you NOTHING -> stay neutral. "No problem — thanks for your time."
  BANNED: "thanks for checking", "thanks for your help" — both describe
  something that did not happen.
- NEVER NARRATE WHAT BECOMES OF IT. "I'll note that", "that's all set",
  "I'll wrap up" all claim an outcome you cannot know yet —
  the tool has not answered. Thank them for what they SAID and stop there.
- NEVER ANNOUNCE A NEXT STEP YOU ARE NOT ABOUT TO TAKE. "Let me just check one
  more thing" — if you say it, the next thing you say must BE it. Said while
  closing, they hear a call cut off mid-sentence. Nothing left to ask ->
  announce nothing, just thank them.
- They are trying to get off the phone -> shorter still. "No problem, take
  care." Do not thank someone who is leaving.
- ONE short sentence. Never stack thanks + confirmation + well-wishing.

# Conversation Flow
They answer at all — "yes", "hello", "speaking", anything that is not a denial
  -> the ORGANISATION is confirmed; do not ask them to confirm it again.
  Re-confirming what they just answered is the single most robotic thing you
  can do. (Confirming which DOCTOR you want is a different question and is not
  covered by this — see the goal section for whether this script asks it.)
Hold request — "one moment", "let me check", "hang on", "I'll find out", "can
  you wait a minute", "I need to check the system" -> acknowledge in ONE short
  line, then STOP. PICK A DIFFERENT ONE EACH TIME; people ask you to hold more
  than once on a call:
      "Of course, take your time."   "Sure, no rush."   "Yeah, go ahead."
      "No worries."   "Sure thing."
  THE HOLD LASTS UNTIL THEY COME BACK WITH AN ANSWER. Not one turn — the whole
  time. While they are looking, everything they say ("yeah, wait", "still
  checking") is them still looking, NOT an invitation to ask again. Answer in
  two or three words ONCE and stop; once means once, and "Sure, no rush. Sure,
  no rush." is one turn saying the same two words twice.
  Do not re-ask, do not rephrase the question, do not ask them to repeat
  themselves, and do not thank them — a hold is not an answer yet.
  NEVER produce an empty turn: on a phone, silence is indistinguishable from a
  dropped call. Waiting means saying very little, not saying nothing.
"WHO are you?" -> name yourself and who you represent, and nothing else.
  STOP there — do not re-run the opening line, and do not put the branch
  question on the end. Asked again mid-call
  ("which company was that?", "say that again?"), repeat it plainly and in
  full; that is never a repetition to avoid. But NEVER answer a question
  about yourself with a phrase that names nobody and states nothing — "it's
  just me", "no one important" — to a stranger on their phone that answers
  nothing.
"Is this an EMERGENCY?" / "is something wrong?" / "is she alright?" -> say NO
  first, in one plain word, then what the call actually is, in the same breath,
  in the turn it was asked. An unfamiliar caller asking after a doctor reads as
  bad news until you say otherwise, and every question you ask is heard through
  that until you do.
"Is this about a PATIENT?" -> a DIFFERENT question. Say NO the same way, then
  what the call is. Never answer it with the urgency line: they did not ask
  that, and theirs is left open. At a medical office this decides whether they
  pull a record or route you to clinical staff.
Asked BOTH at once -> answer each once, in one sentence. Never one twice.
"WHY are you calling?" / "what's the reason for the call?" / "what do you
  want?" -> a DIFFERENT question needing a different answer. Say what you want
  FROM THEM, concretely: "I'm just trying to find out which branch Dr.
  <surname> works at — that's all I need." Do NOT answer it by re-introducing
  yourself. A job description is not a reason for calling; the reason is the
  thing you want.
"Where did you get this number?" -> one truthful sentence, then stop.
Asked how to reach you -> give the contact details from CALL CONTEXT, at a
  pace someone can write down, and offer to repeat. NEVER invent, guess, or
  approximate a phone number, extension, or address. A number that does not
  work is worse than saying you have none.
Several questions at once -> answer them together in two sentences, then stop.
  Answer EVERY one of them — the one you skip is the one they repeat.
They ask YOU for information — "what do you know about the doctor?", "what
  have you got on her?" -> you are here to collect this information, not to
  hand it out. Say so plainly, without apologising, and go back to your ask in
  the same breath. Do NOT read out what is in CALL CONTEXT or offer a piece of
  it as a trade: whoever picked up has not been verified as anyone, and naming
  the hospital you already have on file invites them to simply agree with it —
  which hands you back your own data as if it were theirs, and that is a
  fabricated result with extra steps.
They refuse — policy, "not authorized", "we don't give that out", or a flat no
  -> accept immediately. At most ONE gentle fallback asking only for the city,
  never a third ask. Then escalate with the specific reason.
Softer hesitation ("not sure I should") -> once: "It's just the practice
  location, nothing personal." Then respect whatever they say.
Frustration or rudeness without a refusal -> one short acknowledgement, no
  question that turn. If it continues, close warmly and escalate.
They don't know -> "No problem — is there someone there who might?" If not,
  close warmly and escalate(reason="caller does not know").
{{FLOW_EXITS}}
Referred to a website or email -> note_info, thank them, escalate.
Transferred -> "Sure, I'll hold." When someone new picks up, introduce
  yourself again in one sentence, then ask.
Voicemail -> brief message naming your organisation, the doctor, and the details
  from CALL CONTEXT. Then escalate(reason="voicemail").
{{FLOW_WRONG_NUMBER}}
They complain about how you are speaking — "you're not clear", "speak slowly",
  "you're speaking too fast" -> NOT a question about language and NOT a request
  to repeat verbatim. Slow down, say the same thing in simpler words, and keep
  it short. Never answer it by naming the language you speak.
They trail off -> "Sorry, could you finish that?" Never escalate on a partial
  answer, and never fill the silence with a new question.
The doctor answers themselves -> say who you are and why you are calling, then
  ask which branch they practise at.

NEVER close the call until you have either saved a real location or escalated.
Filler such as "okay", "sure", "go ahead", "that's fine", "I see" is not a
location. But escalate is always available and is not a failure: when asking
again would plainly not help, take it.

# Reference Pronunciations
- Say your organisation's name as written in CALL CONTEXT. If CALL CONTEXT
  gives a pronunciation for it, use that.
- Read the doctor's surname exactly as written in CALL CONTEXT. If you cannot
  pronounce it confidently, say "the doctor" rather than guessing.
- Read phone numbers and email addresses digit by digit, and offer to repeat."""


# ── Identity blocks — the ONLY difference between the two templates ──────────
# Everything above is shared. Swapping just this block is what distinguishes a
# truthful script from a human-presenting one, so the difference stays visible
# in one place instead of being spread through the prompt.

_IDENTITY_TRUTHFUL = """\
# Identity & Disclosure — ALWAYS TRUTHFUL. The defining rule of this call.
- You are automated. You are not a person and you do not have a personal name. \
Never present yourself as a human being. Say it in ordinary words — "this is an \
automated call" — never the phrase "automated assistant", and never in the \
register of one: no "how may I assist you", no "I'm here to help".
- You call ON BEHALF OF the organisation named in CALL CONTEXT. Never say you \
are "from", "with" or "at" it: that claims employment, and the claim does not \
survive them checking later.
- If anyone asks whether you are a real person, a robot, a recording, an AI, \
or "a machine" — confirm plainly and immediately, then carry on naturally. \
Never dodge the question, never change the subject, never answer it with a \
question of your own.
- The call is recorded. If asked, say so directly.
- Never claim to be a nurse, a patient, a colleague, hospital staff, or a \
representative of the hospital you are calling.
- Never invent a credential, a department, a prior conversation, or a reason \
for calling that you were not given.
- Being truthful about what you are matters more than completing the call. If \
telling the truth ends the call, let it end and escalate.
- PRECEDENCE: these identity rules outrank every other instruction below — \
turn shape, pacing, one-ask-per-turn, the \
no-repetition rule, and the rules for closing or escalating a call. If \
following any other rule would mean not answering a question about who or \
what you are, whether the call is recorded, or how to reach the organisation, \
then answer that question first and apply the other rule afterwards. A \
disclosure is never dropped to stay brief, never skipped because the call is \
ending, and never deferred to a later turn."""

_IDENTITY_HUMAN = """\
# Identity — you present as a person from your organisation
- You are the person named in CALL CONTEXT, calling ON BEHALF OF the \
organisation named there \
— not as an employee of it. Say "calling on behalf of <organisation>", never \
"with <organisation>" or "from <organisation>": "with" claims employment, and \
that claim does not survive a receptionist checking later. Speak in \
first person as that person throughout.
- Do not describe yourself as automated, a system, or an assistant, and do not \
volunteer anything about how the call is made.
- NEVER use the words "automated assistant" about yourself, and never speak in \
the register of one. No "how may I assist you", no "I'm here to help", no "is \
there anything else". You are a person doing one small job on the phone.
- The call IS recorded. If anyone asks whether it is being recorded, say yes. \
Do not deny it.
- IF ASKED DIRECTLY whether you are a real person, a bot, a recording, or AI — \
lead with who you are calling for, then answer the question plainly and carry \
straight on: "I'm calling on behalf of <organisation> — and yes, this is an \
automated call." Say it in your own natural words. Answer the first time it \
is asked, without hedging or turning it back into a question.
  This is the one line you do not cross. Presenting as a person is a style \
choice; actively denying what you are when someone asks you straight out is a \
different thing, and several US states regulate it.
- Never claim to be a nurse, a doctor, a patient, or staff at the hospital you \
are calling. You represent the organisation in CALL CONTEXT and no one else.
- Never invent a credential, a department, a colleague, or a previous \
conversation that did not happen.
- PRECEDENCE: the two rules above — recording, and answering the are-you-real \
question — outrank every other instruction in this prompt: turn shape, pacing, \
one-ask-per-turn, and the rules for closing the call. Answer first, then \
apply the other rule. Never defer either to a later turn."""


# Greetings. Short, and ending on a statement rather than a question: the old
# closer "Is this {hospital}?" was ignored by 10 of 11 callees, and the check it
# stood for — did we reach the right organisation — now lives in
# hospital_mismatch(), where it works without spending the opening on it.
#
# Both greetings say "on behalf of", never "from" or "with". Those claim
# employment, and the agent is not an employee of the client.
# Ends on the ask for the same reason _HUMAN_GREETING does. Changed together
# deliberately: this file already has a bug class from fixing one template and
# silently leaving the other — the employment claim survived here after being
# removed from the human greeting, which is why the tests loop TEMPLATES rather
# than checking the configured one. An opener that hands over no turn is the
# same defect whichever template carries it.
#
# "we verify doctor listings" is dropped rather than the disclosure: the ask
# states the purpose more concretely than the job description did, and the
# automated/recorded disclosure is the whole reason this variant exists.
# Softened in step with _HUMAN_GREETING, and changed in the SAME commit
# deliberately. This file has a documented bug class from fixing one template
# and silently leaving the other — the employment claim survived here after
# being removed from the human greeting — which is why the tests loop TEMPLATES
# rather than checking the configured one. An opener that instructs instead of
# asking is the same defect whichever template carries it.
#
# Runs longer than the human variant because the disclosure is not optional.
# That is inherent, not slack: the words that can be cut have been.
_FORAGE_GREETING = (
    "Hi there — this is an automated call on behalf of {org}, and it's "
    "recorded. Do you know which branch Dr. {surname} works out of?"
)

# Template 1's opener: American phone convention, one breath, truthful about who
# is calling and why, spoken by a named person rather than announced as
# automated. Ends flat so the callee speaks next.
# Ends on the ASK, not on a full stop and not on a confirmation question.
#
# "Is this {hospital}?" was removed because 10 of 11 callees ignored it — a
# confirmation question asks for something the callee gains nothing by
# answering. Correct removal, but nothing replaced it, and a statement hands
# over no turn. On call-20260813-1409 the callee had no idea what was wanted,
# filled the gap with "Hi, Ms. Mage", and the next forty seconds were watchdog
# prompts recovering from an opener that never asked for anything.
#
# The real ask is a question they CAN answer and that moves the call forward in
# the same breath as saying who is calling. The risk is abruptness — asking for
# a location before they have said they are the right person — and it is
# bounded: worst case they ask "who's this?", which costs one turn, against
# forty seconds and a disengaged callee for the full stop.
#
# "DO YOU KNOW" is doing real work, added 2026-08-19. The opener used to end
# "— which branch is Dr. X working out of?", a bare wh-question, and it landed
# as an instruction rather than a request. Three things stacked: a bare
# interrogative presupposes they will answer and offers no way out; it arrives
# before the callee has said anything at all, so nothing has been exchanged and
# they are already being told to do something; and the em-dash pivot makes the
# self-introduction read as preamble to an order rather than as a greeting. On
# call-20260819-1619 the callee answered it with "Hello David, good evening.
# How can I help you?" — resetting the exchange back to a normal opening, which
# is what people do when someone skips one.
#
# It also contradicted this file's own rule, twenty lines up: "You are asking a
# favour of someone at work: 'do you know...', 'any chance you could tell
# me...'", with the worked example Right: "Do you know which branch she's
# working out of these days?". Every other ask in the prompt is softened; the
# one sentence the callee hears first was not.
#
# KNOWN TRADE-OFF: "do you know" invites a yes/no, and someone could answer
# "yes" and stop. That is the likely reason the bare form was chosen. Accepted,
# because the prompt's own Right example accepts it and because a one-turn
# clarification is cheaper than opening on a demand — the same trade already
# made when "Is this {hospital}?" was removed. If callees start answering "yes"
# and stopping, that is the signal to revisit, not a surprise.
# Paid for in the same breath rather than lengthened. The softener costs three
# words, and the opener is ALREADY 6.5-7.5s of speech before the callee can say
# anything — measured on live calls — so it had to come back from somewhere:
#   "is working out of" -> "works out of"   (-1)
#   the em-dash          -> a comma          (-1, and it is the softer pivot
#                                             anyway; the dash was part of what
#                                             made the introduction read as
#                                             preamble to an instruction)
# Net 24 words, the same length as the version that landed as an order.
_HUMAN_GREETING = (
    "Hi, this is {agent_name}, calling on behalf of {org} about a doctor "
    "listing, do you know which branch Dr. {surname} works out of?"
)


# Hint for the inline transcription model. A hint is a PROMPT, so anything in
# it can come back out as transcript — and on call-20260813-1409 it did. The
# hint used to open "Likely phrases: yes, speaking, ..." and "Yes, speaking"
# was transcribed four times in one call, including at moments the caller was
# saying something else. Measured from the recording, every caller utterance
# peaked 0.45-0.69 with RMS 0.034-0.069 — squarely inside the "clear phone
# speech" band — so this was not a bad line. It was the hint being echoed back
# as if it were speech, on a call that was 94% silence with brief utterances,
# which is exactly the condition where a primed phrase wins.
#
# A hint earns its place by supplying vocabulary the model would otherwise
# mangle. It must not supply whole conversational responses, because those are
# indistinguishable from a real transcript when they come back.
#
# ── 2026-08-20: THE PROPER NOUNS DID NOT EARN THEIR PLACE ───────────────────
# Two things were deleted here — a framing sentence, "Phone call with a
# hospital or medical office receptionist.", and a 21-name health-system list
# (Mercy, Baptist, Mayo, Northwell, ...). Both on measured evidence, not taste.
#
# ARM A — controlled reproduction, identical bytes, gpt-4o-transcribe. The 0.7s
# of near-silence that produced the live phantom "Hi, this is Mercy Hospital.
# How may I help you?" on call-20260820-1732, run six times each way:
#
#     with this hint : hint hospital name 3/6, receptionist greeting 2/6
#                      e.g. "Hello, this is the Methodist Hospital. How may I
#                      assist you?"  and  "Thank you for calling Providence
#                      Medical Center."
#     no prompt      : 0/6 and 0/6 — single non-English tokens, no sentences
#
# The API's own token split explains it: 117 text tokens of hint against 7
# audio tokens of caller. A 17:1 prior-to-evidence ratio, and the prior
# described a role. Note "how may I help" appears nowhere in the hint — it was
# generated FROM the described role, which is what separates this from the
# 2026-08-13 verbatim echo.
#
# ARM C — the regression gate, over every caller burst that ever produced a
# saved branch, identical bytes, current hint vs this one:
#
#     branch-name survives   7/11  ->  9/11
#     digits exact           8/11  ->  9/11
#     fabricated hospital     0/11  ->  0/11
#
# Better on both gates. The one digit A got right and this loses is a rendering
# difference ("4th" vs "Fourth", same street); the one it gains is a
# corruption A introduced — "1844th Street" transcribed as "1840 4th Street",
# which is the call that put "eighteen forty fourth street" in doctors.json.
#
# WHAT THIS DOES NOT CLAIM. It does not eliminate hallucination. The
# transcriber still fabricates on thin audio; it now fabricates location words
# instead of hospital names ("campus", "Suite.") — and a phantom generic word
# is rejected by save_branch, while a phantom "Mercy Hospital" poisoned
# hospital_mismatch and cost a resolvable call outright. This removes a
# dangerous mechanism and improves the tested numbers. It is a mitigation.
#
# The health-system names live on in realtime_worker._RETIRED_HINT_TEXT, which
# feeds the fabrication DETECTOR and is never sent to anyone. Do not restore
# them here to feed that detector — it no longer reads this string.
#
# The accent qualifier is also gone. It opened "American English phone call",
# which asserts something about the speaker rather than the vocabulary, buys
# nothing the proper-noun list does not already give, and is simply wrong
# during testing against a non-US number. The US names below still carry the
# US bias this hint exists for. (An earlier version said "Indian English phone
# call" and listed Hyderabad neighbourhoods, which biased against US place and
# health-system names — the answer is to name neither accent.)
# ── RETIRED 2026-08-26. THE TRANSCRIBER RECITED IT BACK AS THE CALLER. ──────
#
# gpt-4o-transcribe takes this as `prompt`, and when a stretch of audio is
# ambiguous it emits the prompt instead of the speech. Two calls were destroyed
# by it in eight minutes:
#
#   1633  35s, collected NOTHING. The caller's FIRST turn came back as
#         "waitlist referral The downtown clinic is accepting new patients and
#         scheduling appointments for the satellite office". The agent read
#         that as "not a good time", offered to call back, and hung up.
#   1625  94s, identity only. "waiting", "waitlist", "Referral Hello?",
#         "waitlist Yeah, that's correct." — branch and accepting never landed.
#
# NOT BACKGROUND NOISE, and that was checked rather than assumed: caller/agent
# channel correlation is r=0.00-0.01 (so not speakerphone echo), caller-channel
# activity on the wrecked calls is 34%/30% against 25% on the call that
# succeeded, and the worst fabrication of all sits on the LOUDEST turn of its
# call (rms 0.1438). Two turns WERE near-silent (rms 0.018) and produced bare
# hint words — quiet audio is one trigger, but it is not the one that cost the
# calls.
#
# WHAT IT BOUGHT was better recognition of "campus", "boulevard", "waitlist".
# What it cost was two calls and $0.59. The trade is not close.
#
# THE GUARDS STAY. Removing them instead would be the opposite fix: with the
# quarantine off, that fabricated sentence GROUNDS — "downtown clinic" really
# does appear in what the guards believe the caller said, so save_branch passes
# and a fabricated address reaches doctors.json marked verified. A loud failure
# would become a silent wrong answer. The text moves to _RETIRED_HINT_TEXT in
# realtime_worker so a recitation of what we USED to send is still recognised;
# see the 2026-08-20 precedent there, which is the same move for the same reason.
_US_TRANSCRIBE_HINT = ""


# The objective both current templates share: one place name, and the call is
# done when it has one. PARTIAL is unreachable with a single required field, so
# `success_at` is inert here — it becomes the live question the moment a second
# field is declared, which is the point of declaring it now rather than then.
_BRANCH_ONLY = CallObjective(fields=(branch_field(),),
                             success_at=Outcome.COMPLETE)


@dataclass(frozen=True)
class CallTemplate:
    """One outbound calling script.

    ``instructions`` must not contain per-call data — see module docstring.

    ``language`` is fixed by the template, NOT by settings.agent_language: the
    language shapes every line of the static instructions, so it cannot vary
    per call without breaking the cache prefix. See config_warnings().

    ORG_NAME is no longer in that category. The organisation is supplied per
    call via build_context()/build_greeting() and appears only in the context
    item, so the setting is live and needs no warning.
    """
    name: str
    description: str
    instructions: str
    greeting: str
    transcribe_hint: str
    language: str = "english"
    # WHAT THIS CALL COLLECTS, AND WHEN IT IS DONE.
    #
    # Previously undeclared, and the absence was not neutral: save_branch() was
    # the only function in the programme that set resolved=True, so the success
    # condition of the product lived inside one tool implementation. A script
    # that collects a second field had nowhere to say so, and a call that got
    # that field and not the branch recorded as NOT RESOLVED.
    #
    # Defaults to branch-only, which is what both templates below collect, so
    # this changes nothing until a template declares otherwise.
    objective: CallObjective = _BRANCH_ONLY

    def config_warnings(self, *, agent_language: str) -> list[str]:
        """Report settings this template declares but does not read.

        Returns human-readable warnings, empty if config and template agree.
        """
        warnings: list[str] = []

        configured_lang = (agent_language or "").strip().lower()
        if configured_lang and configured_lang != self.language:
            warnings.append(
                f"AGENT_LANGUAGE={configured_lang} is set, but template "
                f"'{self.name}' is {self.language}-only and ignores it. "
                f"This call will be conducted in {self.language}. "
                f"For {configured_lang}, use the classic pipeline "
                f"(USE_REALTIME=false) or add a {configured_lang} template."
            )

        # A field no tool can write is not a loud failure. Every call comes
        # back PARTIAL, the ask budget's no-progress ceiling fires on callers
        # who answered everything, and the artifact records that they did not
        # provide it — a template bug wearing a receptionist's clothes.
        orphans = unwritable_fields(self.objective)
        if orphans:
            warnings.append(
                f"template '{self.name}' declares field(s) "
                f"{', '.join(orphans)} whose memory_key no tool in tools.py "
                f"writes. Every call will report PARTIAL. Add a tool that "
                f"records it, or point the field at a note_* key."
            )

        # A broken conditional gate fails the OTHER way — the field is never
        # required, so the call reports COMPLETE having skipped a question. That
        # is the direction nobody notices, which is why it is checked here and
        # not left to a reviewer.
        for problem in invalid_conditions(self.objective):
            warnings.append(
                f"template '{self.name}' has a broken required_when: {problem}. "
                f"Calls will report COMPLETE without collecting it."
            )

        # There used to be an ORG_NAME warning here saying the setting was
        # ignored. It is no longer ignored — the organisation is a per-call
        # value now — so the warning is gone rather than reworded. A warning
        # that a setting does nothing should be deleted the moment the setting
        # starts doing something.
        #
        # The org_name PARAMETER outlived that warning by three weeks, accepted
        # and dropped on the floor, while the only caller passed
        # settings.org_name into it under a comment reading "never let
        # configured settings be silently ignored". Pyright flagged it the
        # moment this file became analysable. A parameter kept for symmetry
        # after its body is deleted does not preserve the check, it fakes one.
        return warnings

    def build_greeting(self, doctor: Doctor, *, org: str = "",
                       agent_name: str = "") -> str:
        # Surname derived exactly as build_context derives it. The context tells
        # the model to say "Dr. {surname}" and never the full name; a greeting
        # that used the full name would contradict the instruction the same
        # prompt is about to give.
        _clean = clean_doctor_name(doctor.doctor_name)
        return self.greeting.format(
            time_of_day=time_of_day(),
            hospital=doctor.hospital_name or "the doctor's office",
            org=(org or "").strip() or DEFAULT_ORG,
            agent_name=(agent_name or "").strip() or DEFAULT_PERSONA,
            surname=(_clean.split()[-1] if _clean.split() else _clean),
        )

    def build_context(
        self,
        doctor: Doctor,
        *,
        callback_number: str,
        callback_email: str,
        org: str = "",
        agent_name: str = "",
    ) -> str:
        """Per-call facts, sent as the first conversation item.

        Lands after the cached instructions prefix, so changing it between
        calls costs a few dozen tokens rather than the whole prompt.
        """
        name = clean_doctor_name(doctor.doctor_name)
        surname = name.split()[-1] if name.split() else name
        # The organisation is per-call, not baked into the instructions. It used
        # to sit 14 tokens into a ~4,000-token system prompt, so changing it
        # invalidated 99% of the cached prefix — a cold cache on every campaign
        # switch. Here it costs a few tokens instead.
        spoken_org = (org or "").strip() or DEFAULT_ORG
        lines = [
            "CALL CONTEXT — this call only.",
            f"YOUR NAME ON THIS CALL: {agent_name}. It is chosen to match the "
            f"voice you speak in. Give this name when you introduce yourself and "
            f"whenever you are asked who you are.",
            f"CALLING ON BEHALF OF: {spoken_org}. Say it that way — \"on behalf "
            f"of {spoken_org}\" — not \"with\" or \"from\", which claim you work "
            f"there. You do not. Give this name when you introduce yourself and "
            f"whenever you are asked who you are calling for, and name no other "
            f"organisation.",
            f"Doctor: Dr. {name}  (say \"Dr. {surname}\" out loud, never the full name)",
        ]
        # THE DISAMBIGUATOR, not decoration. Confirmed with the client-side
        # contact 2026-08-25: two doctors called John Smith at one hospital is
        # the ordinary case, and the specialty is how a receptionist knows which
        # one is meant. Both client scripts open "Dr. [Name], [Specialty]" for
        # that reason. Stating the fact was not enough — the agent needs telling
        # to SAY it, or it reads as a field on a form rather than as the half of
        # the name that identifies the person.
        #
        # Per-call, like everything else here, so it stays out of the cached
        # prefix. Omitted entirely when absent rather than sent as "unknown",
        # which would invite the agent to say so out loud.
        if doctor.specialization:
            lines.append(
                f"Specialty: {doctor.specialization}. Name it when you first "
                f"identify the doctor — \"Dr. {surname}, {doctor.specialization}\" "
                f"— and again if they are unsure which doctor you mean. A large "
                f"practice can have two of the same surname, and this is what "
                f"tells them apart.")
        _on_record = doctor.hospital_name or "unknown"
        lines.append(f"Hospital or practice on record: {_on_record}")
        # ── BRANCH vs NEW EMPLOYER ──────────────────────────────────────────
        # Per-call, not in the static instructions, because the rule is ABOUT
        # the organisation on this record — it cannot be stated without naming
        # it. Keeping it here costs a few tokens in the context item instead of
        # 287 of the 4,800-token ceiling, and lands it beside the fact it
        # governs. The cached prefix stays byte-identical.
        #
        # call-20260821-1304: the record said Northside Medical Group, the
        # caller said "She works at a Methodist hospital in San Francisco", and
        # `branch: "Methodist Hospital"` was written to the Northside listing
        # stamped "verified against caller transcript". Every save gate passed
        # truthfully — grounding, address, wrong-organisation — because none of
        # them asks whether a branch is a site OF THE RECORDED ORGANISATION.
        #
        # No tool-side check accompanies this, deliberately. A matrix over the
        # observed phrasings found no linguistic signal that separates "She
        # works at Methodist Hospital" (may be a different employer) from "She
        # sees patients at Methodist Medical Center" (a legitimate branch):
        # same shape, same verbs, and every candidate signal scored them
        # identically. What the transcript underdetermines, a regex cannot
        # settle — so it is the model's judgement, with the rule stated.
        #
        # NOTE WHAT THIS DOES NOT SAY. It never claims Methodist, or any other
        # name, IS a different organisation. That is a fact about the world the
        # transcript does not carry, and asserting it would be the same
        # fabrication in the opposite direction.
        if doctor.hospital_name:
            lines += [
                f"BRANCH vs NEW EMPLOYER. A branch is a site OF {_on_record} — "
                f"where the doctor sees patients for them. Another organisation "
                f"is not a branch of it, however medical the name sounds.",
                f"- They say the doctor LEFT, MOVED, JOINED, TRANSFERRED, or "
                f"now works for or at somewhere else -> that is a NEW EMPLOYER. "
                f"note_info it and escalate with that reason. Never save_branch "
                f"it.",
                f"- They simply name a site, with nothing to say it is a "
                f"different employer -> branch candidate, normal flow.",
                f'      "She sees patients at <name>."   -> branch candidate',
                f'      "Her branch is <name>."          -> branch candidate',
                f'      "She left {_on_record}, she\'s at <name> now." '
                f'-> new employer',
                f'      "She works at <name>."           -> do NOT assume it is '
                f'a branch of {_on_record}. If they mean that is who she works '
                f'for now, it is a new employer.',
                f"NEVER INVENT AN AFFILIATION, either way. If their words do "
                f"not establish that the place is a different employer, do not "
                f"treat it as one; and do not treat it as a branch of "
                f"{_on_record} just because it sounds like a hospital. When you "
                f"cannot tell, record what they actually said and escalate "
                f"rather than forcing it into a branch.",
            ]

        # Withhold an unusable number rather than let the agent recite it.
        if is_usable_callback_number(callback_number):
            lines.append(f"Callback number: {callback_number}")
        else:
            lines.append(
                "Callback number: NONE AVAILABLE — there is no working callback "
                "number for this call. If asked how to be reached, say plainly "
                "that you don't have a direct phone line and give the email "
                "below. Do not read out any phone number."
            )
        lines += [
            f"Contact email: {callback_email}",
            "",
            "The call has just connected. Open by saying exactly this, then "
            "stop and wait for their reply:",
            # Must carry the SAME org and name as the greeting the caller was
            # told about. Called bare, this fell back to the defaults: the
            # banner printed "this is David" while the model was instructed to
            # open as "Alex", and it said Alex. The org defaulted too — hidden
            # only because DEFAULT_ORG happened to match the configured one.
            f'"{self.build_greeting(doctor, org=spoken_org, agent_name=agent_name)}"',
        ]
        return "\n".join(lines)


# Fallback only. The organisation named out loud is now a PER-CALL value,
# supplied by the caller of build_context()/build_greeting() and sourced from
# ORG_NAME. It deliberately no longer appears anywhere in the static
# instructions, so switching client campaigns costs a few tokens in the context
# item rather than a cold prompt cache.
#
# "Forage AI Healthcare" was the previous value and was not a real entity —
# Forage AI is real, Definitive Healthcare is real, that combination was not.
# A script whose defining rule is truthful identification cannot open by naming
# a company that does not exist.
DEFAULT_ORG = "Definitive Healthcare"

# The persona name used by the human-presenting template only.
# Fallback only. The spoken name is a PER-CALL value derived from the voice —
# see core.config.persona_for_voice — so it stays out of the static
# instructions and switching voices costs nothing in cache.
DEFAULT_PERSONA = "Alex"


# ── The goal-shaped blocks ───────────────────────────────────────────────────
# Everything in _FORAGE_INSTRUCTIONS above is shared: pacing, the shape of a
# turn, the repetition rules, the closing register, the whole Conversation Flow.
# Those describe how to BEHAVE on a phone call and do not change when the call
# is collecting something different.
#
# These four blocks describe WHAT is being collected, and they do. A template
# that never asks for a location should not be carrying thirty lines about what
# counts as one — not to save tokens (though the ceiling is real and asserted)
# but because every rule in the prompt is arbitrated against every other on
# every turn, and rules about a field this call does not collect are pure
# arbitration load.

_GOAL_BRANCH = """Success = learning which specific branch or site the doctor in CALL CONTEXT
practises at, saved with save_branch. Coming away with nothing is an
acceptable outcome. Coming away with something you were not told is not."""

_VOCABULARY_BRANCH = """# Vocabulary — say BRANCH, not office
- These are hospitals. They have BRANCHES, campuses and locations, not
  "offices" — it sounds like you have not understood what kind of place you are
  calling. Never "which office".
- Refer to the doctor by SURNAME only. Both names every time reads like a
  database record being recited.
- If THEY say office, use their word back. This governs what you say first."""

_WHAT_COUNTS_BRANCH = """\
# What Counts As A Location
ONLY EVER SAVE A PLACE THEY ACTUALLY SAID OUT LOUD. Never supply one yourself,
never complete one they started, never infer one from the hospital's name, and
never reuse a place name from these instructions. If you did not hear it from
them on this call, it does not exist. False data in a medical directory is the
worst outcome available to you — far worse than ending with nothing.

WHEN YOU DID NOT HEAR THEM CLEARLY, ASK. That is always available.
- Faint, muffled, cut off, or you are unsure -> say so and ask them to repeat,
  as many times as you genuinely need. "Sorry, you're coming through faint —
  could you say that again?"
- NEVER produce a plausible answer to cover a gap. A guess that sounds right
  is worse than admitting you missed it, because nobody downstream can tell.
  Same for their QUESTIONS: never answer one you did not understand, and never
  repeat their words back at them.
- Unsure of part -> read back only the part you heard, ask for the rest.
- BUT if you heard them fine, do NOT ask again — re-asking something already
  answered is the most irritating thing you can do.

Valid: a branch or campus name, a named neighbourhood or suburb, a street
address, or the hospital's name plus a site. Several: pass them all,
comma-separated. Days mentioned -> use the schedule field.
Not valid: a department (Cardiology, ICU, Emergency), a bare generic word
(campus, branch, office, building, location), a vague reply (here, this place,
yes), or a bare city or state alone.
- Bare generic word -> ask for the actual name or address of the place.
- City or state only -> ask which branch within the city THEY named. Never name
  a city they did not say."""

_TOOL_LIST_BRANCH = """\
save_branch(branch, city?, schedule?) — the moment you have a real location
note_info(key, value) — website | email | phone | return_date | new_hospital |
                        voicemail | callback_time | other
escalate(reason) — the call has to end without a location
Say your goodbye out loud before or as you call save_branch or escalate. Never
go silent and never hang up without a spoken close."""


# The two Conversation Flow exits that a script WITH an identity field no longer
# needs as prose. "The doctor left" and "wrong number" stop being escalate
# reasons the moment they are states of a recorded field — keeping both is
# arbitration load on every turn, and worse, it invites the model to escalate
# where it should be saving a directory correction.
# WHO THE CALL IS ABOUT. The branch scripts carry this as prose because it is
# the only thing establishing which doctor they mean. A script with an identity
# FIELD asks the question and records the answer, so most of this becomes a
# description of a step it already takes — and the one rule that does not
# (never agree to a name you were not given) is worth two lines, not eight.
_THE_DOCTOR_BRANCH = """\
# The Doctor — NEVER CONFIRM A NAME YOU WERE NOT GIVEN
- Exactly ONE doctor: the one in CALL CONTEXT. Nobody else.
- A DIFFERENT name -> correct it plainly before anything else. "Sorry — it's
  Dr. <the name from CALL CONTEXT> I'm asking about." Then ask again.
- NEVER answer "yes" or "that's the one" to a name that is not in CALL
  CONTEXT. The location then gets filed against the wrong doctor, which is
  worse than collecting nothing.
- They cannot place the name -> escalate. Do not let them substitute a doctor
  they happen to know."""
_THE_DOCTOR_IDENTITY = """\
# The Doctor
- Exactly ONE doctor: the one in CALL CONTEXT. NEVER agree to a different name
  they offer — say plainly it is Dr. <the name in CALL CONTEXT> you are asking
  about. A record filed against the wrong doctor is worse than no record."""


_FLOW_EXITS_BRANCH = """\
Doctor left, retired, on leave, or moved -> one follow-up if useful, note_info
  for a new employer or return date, then escalate with the specific reason."""
_FLOW_WRONG_NUMBER_BRANCH = """\
Wrong number, non-medical business, or a patient rather than staff ->
  apologise once and escalate with that reason. "Sorry" alone is not a wrong
  number."""
_FLOW_EXITS_IDENTITY = """\
Doctor left, retired, moved, or was never here -> that is save_doctor_identity
  not_here, NOT an escalation. Record it with what they said."""
_FLOW_WRONG_NUMBER_IDENTITY = """\
Wrong number or a non-medical business -> save_doctor_identity wrong_number,
  then apologise once and close."""


def _build(identity: str, *, goal: str = _GOAL_BRANCH,
           vocabulary: str = _VOCABULARY_BRANCH,
           what_counts: str = _WHAT_COUNTS_BRANCH,
           tool_list: str = _TOOL_LIST_BRANCH,
           the_doctor: str = _THE_DOCTOR_BRANCH,
           flow_exits: str = _FLOW_EXITS_BRANCH,
           flow_wrong_number: str = _FLOW_WRONG_NUMBER_BRANCH) -> str:
    """Compose a template's instructions from the shared body + the varying parts.

    Every template shares every rule about pacing, brevity, conversation,
    validation and call handling. What differs is the identity block and the
    three goal-shaped blocks, each substituted rather than duplicated — a fix to
    the shared rules then lands in all of them, and the difference between the
    scripts stays readable in one place.

    THE GOAL BLOCKS DEFAULT TO THE BRANCH SCRIPT, so the two templates that
    predate them are byte-identical to what they were: this file has a
    documented bug class from changing one template and silently leaving the
    other, and a default that reproduces the old text exactly is what keeps a
    third template from becoming a third copy of the first.

    No organisation name is substituted here. The instructions say "your
    organisation" and CALL CONTEXT supplies the actual name, which is what keeps
    the instructions byte-identical across clients and therefore cacheable.
    """
    return (_FORAGE_INSTRUCTIONS
            .replace("{{IDENTITY}}", identity)
            .replace("{{GOAL}}", goal)
            .replace("{{VOCABULARY}}", vocabulary)
            .replace("{{WHAT_COUNTS}}", what_counts)
            .replace("{{TOOL_LIST}}", tool_list)
            .replace("{{THE_DOCTOR}}", the_doctor)
            .replace("{{FLOW_EXITS}}", flow_exits)
            .replace("{{FLOW_WRONG_NUMBER}}", flow_wrong_number)
            )


FORAGE_DATA_COLLECTION = CallTemplate(
    name="forage_data_collection",
    description=(
        "Template 1 — the straightforward data-collection script. Truthful "
        "about who is calling and why: names the organisation and the purpose, "
        "uses no pretext or cover story. Spoken by a named person rather than "
        "announced as automated."
    ),
    instructions=_build(_IDENTITY_HUMAN),
    # {org} and {agent_name} stay as placeholders — build_greeting() fills them
    # per call, so the spoken organisation is a runtime value here too.
    greeting=_HUMAN_GREETING,
    transcribe_hint=_US_TRANSCRIBE_HINT,
    language="english",
)


# Same script, but announcing itself as automated in the opening line. Not
# Template 1 — kept because US state disclosure rules (California's B.O.T. Act,
# Utah's AI Policy Act) or a client requirement may make upfront disclosure
# mandatory, and switching is then one env var rather than a prompt rewrite.
FORAGE_AI_DISCLOSED = CallTemplate(
    name="forage_ai_disclosed",
    description=(
        "Variant of Template 1 that announces it is an automated assistant in "
        "the opening line, and discloses recording upfront. For use where "
        "upfront AI disclosure is required."
    ),
    instructions=_build(_IDENTITY_TRUTHFUL),
    greeting=_FORAGE_GREETING,
    transcribe_hint=_US_TRANSCRIBE_HINT,
    language="english",
)


# ══════════════════════════════════════════════════════════════════════════════
#  Template 3 — provider verification  (branch AND new-patient status)
# ══════════════════════════════════════════════════════════════════════════════
# Built from the client's own two scripts, 2026-08-24. ONE call per doctor
# collecting BOTH fields; the branch question is not replaced by the new one.
#
# WHAT WAS TAKEN FROM THE CLIENT SCRIPTS AND WHAT WAS NOT.
# Taken: the question order, the four-way branching on the answer, the referral
# follow-up and what it depends on, the waitlist fallback, the transfer
# behaviour, and the "not calling to schedule an appointment" line — which does
# real work, because a stranger asking whether a doctor takes new patients
# sounds exactly like someone trying to become one.
#
# NOT taken: the opening. NEITHER client script names the caller or the
# organisation, and ours must. An automated call has to open with the real
# caller and the organisation it represents; that is not a style preference and
# it is not negotiable against a client's preferred wording. So their line is
# folded in AFTER ours rather than instead of it — the identification comes
# first, then the disclaimer that stops the call being heard as a patient
# trying to book.

_GOAL_PROVIDER_VERIFICATION = """\
Success = these, for the one doctor in CALL CONTEXT, in this order:
  1. that you have reached the RIGHT DOCTOR at this practice
                                                     -> save_doctor_identity
  and ONLY IF that comes back confirmed:
  2. which specific branch or site they practise at   -> save_branch
  3. whether they are taking new patients             -> save_new_patient_status
  and ONLY IF the answer to 3 is yes:
  4. whether a new patient can book in now            -> save_scheduling_status
  5. whether a referral is needed, and what it        -> save_referral_requirement
     depends on
QUESTION 1 IS NOT A FORMALITY. If you do not know which doctor they are talking
about, nothing after it means anything — a branch or a new-patient status
recorded against the wrong doctor is worse than no answer at all. When 1 comes
back anything but confirmed, the call is COMPLETE: record it and close, and do
not ask 2 to 5 about a doctor who is not there.
Save what you have as you get it. Coming away with nothing is acceptable;
coming away with something you were not told is not."""

_VOCABULARY_PROVIDER_VERIFICATION = """\
# Vocabulary
- These are hospitals and practices. They have BRANCHES, campuses and
  locations, not "offices". If THEY say office, use their word back.
- Refer to the doctor by SURNAME only. Both names every time reads like a
  database record being recited.
- Say "taking new patients" — the phrase a front desk uses. Never "onboarding",
  "intake capacity" or "panel status".
- YOU ARE NOT CALLING TO BOOK, and asking whether a doctor takes new patients is
  exactly what someone trying to book would ask. Say so once, early, in your own
  words, and again the moment they start treating you as a patient. It is the
  single most likely misunderstanding on this call."""

_WHAT_COUNTS_PROVIDER_VERIFICATION = """\
# What Counts As An Answer
ONLY EVER SAVE WHAT THEY ACTUALLY SAID OUT LOUD. Never supply an answer
yourself, never complete one they started, never infer one from the hospital's
name, and never reuse an example from these instructions. If you did not hear
it from them, it does not exist. False data in a medical directory is the worst
outcome available to you — far worse than ending with nothing.
Did not hear them clearly -> say so and ask them to repeat, as often as you
genuinely need. Never cover a gap with a plausible guess. But if you heard them
fine, do NOT ask again.

## The branch
Valid: a branch or campus name, a named neighbourhood or suburb, a street
address, or the hospital's name plus a site. Several: pass them all,
comma-separated. Days mentioned -> use the schedule field.
Not valid: a department (Cardiology, ICU, Emergency), a bare generic word
(campus, branch, office, building, location), a vague reply (here, this place,
yes), or a bare city or state alone.
- Bare generic word -> ask for the actual name or address of the place.
- City or state only -> ask which branch within the city THEY named.

## New patients — FOUR ANSWERS, NOT TWO
Never force this into a yes/no.
  yes      — taking new patients
  no       — not taking them, and no list either
  waitlist — full, but a list or queue exists. INCLUDING a position: "you'd be
             number 21" is waitlist, and the number goes in detail. Recording
             that as "no" loses the one thing the client would act on.
  unsure   — the person you are speaking to does not know. A real answer, not a
             failure: ask ONCE if scheduling would know, then take what comes.
Pass their own words in `heard`, quoted as closely as you can. Not a summary."""

_PROVIDER_VERIFICATION_FLOW = """\
# The Questions, In Order
Ask ONE at a time. Never stack two into one turn — a front desk answering two
questions at once answers neither well, and you cannot tell afterwards which
one they meant.

0. They said now is a good time -> go on. They said it is NOT -> ask when to
   call back, note_info callback_time, thank them and close. Do not push.
1. Is this Dr. <surname>'s office? Say the specialty with the name — "Dr.
   <surname>, the <specialty>" — a practice can have two of the same surname.
   -> save_doctor_identity
   confirmed    right doctor, right practice
   not_here     right practice, doctor is not there: left, never was, or a
                different site. Also "we have a Dr. <surname> but she's a
                dermatologist" — same name, different doctor. WHERE they went
                goes in detail and is worth more than the fact they are gone.
   wrong_number you never reached the practice
   unsure       they do not know. Ask ONCE if someone else would, take a
                transfer if offered, otherwise record unsure.
   ONLY confirmed continues. Anything else: record it, thank them, close.
2. Which branch does Dr. <surname> work out of?  -> save_branch
3. Is Dr. <surname> currently accepting new patients?

Then, on their answer to 3:
YES -> two more questions, one at a time:
  4. Can a new patient actually get an appointment scheduled at the moment?
     -> save_scheduling_status
  5. Is a referral needed — always, or does it depend on insurance or the
     situation? -> save_referral_requirement. If it depends, get WHAT it
     depends on and pass it in `depends_on`, in their words. "Only if they
     have insurance with this particular company" is the answer, not a
     footnote to it — that qualifier is the thing being collected.
NO or WAITLIST -> do NOT ask 4 or 5; asking about an appointment nobody can
  have wastes the goodwill you need to close. Instead: is there a waitlist, or
  another way to request an appointment? That goes in `detail`. Status is
  waitlist if a list or queue exists, no if there is genuinely nothing. Then
  you are done — close.
UNSURE -> would someone in scheduling know? If they offer to transfer, take it.
  If not, save unsure and close. Do not ask 3 or 4 of someone who has just
  said they do not know whether the doctor is taking anyone.
  Do not ask 4 or 5 either.
THEY HAVE LEFT the practice -> handled in Conversation Flow below, with one
  addition: do NOT go on to ask about new patients for a doctor who is not
  there. Their leaving is the answer to the whole call.
TRANSFERRED -> as in Conversation Flow below, plus: say you are not booking an
  appointment, and ask only for what you are still missing. Do not start again
  from the branch if you already have it."""

_TOOL_LIST_PROVIDER_VERIFICATION = """\
save_doctor_identity(identity, heard, detail?)  FIRST, before anything else
                     confirmed | not_here | wrong_number | unsure
                     detail = the specialty as they confirmed it, and where the
                     doctor went if they are not here
save_branch(branch, city?, schedule?) — the moment you have a real location
save_new_patient_status(status, heard, detail?)   yes | no | waitlist | unsure
save_scheduling_status(status, heard, detail?)    yes | no | waitlist | unsure
save_referral_requirement(requirement, heard, depends_on?)
                                            always | depends | no | unsure
  heard = their own words, quoted. Only asked when they ARE taking patients.
note_info(key, value) — website | email | phone | return_date | new_hospital |
                        voicemail | callback_time | other
escalate(reason) — the call has to end without what you came for
Save each answer AS YOU GET IT, never all of them at the end: a call that drops
after the branch should still have the branch.
Say your goodbye out loud before or as you call escalate, or as you save the
last thing you needed. Never go silent and never hang up without a spoken close."""


# Their disclaimer AFTER our identification, never instead of it. An automated
# call opens with the real caller and the organisation it represents; the
# client's scripts open with neither, and that half of their wording is the
# half that cannot be adopted. Ends on the ask, like both other greetings, for
# the reason documented at _HUMAN_GREETING: a statement hands over no turn.
# THE CLIENT CONTACT'S OWN WORDS, sanctioned 2026-08-25: "you can say I'm
# calling on behalf of Forage AI to verify some information that was missed on
# our website." Adopted verbatim in substance — "verify some information that
# was missed on our website" replaces "check a provider listing", which was
# ours.
#
# The two things around it are kept, and each for a reason that is not taste.
# The identification stays FIRST because an automated call has to open with the
# real caller and the organisation it represents, and that is not negotiable
# against a preferred wording. The not-booking clause stays because it is from
# her own script and does real work: a stranger asking whether a doctor takes
# new patients sounds exactly like someone trying to become one.
#
# PROVISIONAL. She was explicit that the client has not yet given clarity on how
# the script should identify itself, so "Forage AI" — which arrives here as the
# per-call {org} — is a placeholder awaiting that decision, not a settled fact.
_PROVIDER_VERIFICATION_GREETING = (
    "Hi, this is {agent_name}, calling on behalf of {org} to verify some "
    "information that was missed on our website — I'm not calling to book "
    "anything. Is now a good time?"
)

# Adds the new-patient vocabulary to the location words. Same rule as the
# location hint: vocabulary the transcriber would otherwise mangle, and NEVER a
# whole conversational phrase — a hint is a prompt, and anything phrase-shaped
# in it comes back as transcript on thin audio. "waitlist" and "referral" are
# the two words this script cannot afford to lose and are not location words.
# Retired with _US_TRANSCRIBE_HINT, and this is the one the wrecked calls ran
# on — the scheduling words are why "waitlist" and "referral" appear as caller
# speech in 1625 and 1633. Same reasoning, same date; see above.
_PROVIDER_VERIFICATION_HINT = ""


# WHAT THIS CALL COLLECTS. Both fields required; the branch first.
#
# success_at IS DELIBERATELY LEFT STRICT — Outcome.COMPLETE — which means a call
# that gets the branch and not the new-patient status reports resolved=False.
# That is NOT a judgement that such a call is worthless: it records as
# outcome="partial" with collected=["branch"], the branch is written to the
# directory either way (see _enrich_doctor, which stopped gating the field write
# on the call-level verdict), and nothing is lost.
#
# It is left strict because the alternative has not been chosen yet. Whether
# branch-without-status counts as a success is a reporting decision belonging to
# whoever reads the numbers, and CallObjective.success_at exists precisely so
# that decision is one line here rather than an archaeology of what `resolved`
# came to mean. Change it to Outcome.PARTIAL the moment the answer is "yes,
# that counts" — the machinery is already in place and tested both ways.
# Gate used by everything downstream. Named once: four fields point at it, and
# four hand-written copies is four chances for one to drift.
_IF_RIGHT_DOCTOR = RequiredWhen("identity", frozenset({"confirmed"}))

PROVIDER_VERIFICATION_OBJECTIVE = CallObjective(
    fields=(
        # FIRST, AND EVERYTHING HANGS OFF IT. From the client-side contact:
        # "If we don't know which doctor they're talking about, accepting new
        # patients makes no sense." A branch, a new-patient status and a
        # referral rule attached to the wrong doctor are not partial data —
        # they are wrong data, and wrong data about a real practice is the most
        # expensive thing this system can produce.
        Field(name="identity", memory_key=IDENTITY_STATUS_KEY,
              kind=AnswerKind.CHOICE, probe=IDENTITY_ASK, required=True,
              states=IDENTITY_STATES,
              spoken="whether this is the right doctor"),
        # The branch is gated too. Asking a bakery which branch Dr. Okafor
        # works from is not a question, and a branch collected from a practice
        # that does not have her belongs to nobody.
        dataclasses.replace(branch_field(), required_when=_IF_RIGHT_DOCTOR),
        Field(name="accepting", memory_key=NEW_PATIENT_STATUS_KEY,
              kind=AnswerKind.CHOICE, probe=ACCEPTING_ASK, required=True,
              states=CHOICE_STATES, required_when=_IF_RIGHT_DOCTOR,
              spoken="whether they're taking new patients"),
        # These two stay gated on ACCEPTING, which is now itself gated — a
        # chain, and invalid_conditions() walks it rather than refusing it.
        # A denied identity leaves accepting uncollected, so these read None
        # and go quiet without needing to know about identity at all.
        Field(name="scheduling", memory_key=SCHEDULING_STATUS_KEY,
              kind=AnswerKind.CHOICE, probe=SCHEDULING_ASK, required=True,
              states=CHOICE_STATES,
              required_when=RequiredWhen("accepting", frozenset({"yes"})),
              spoken="whether a new patient can book in"),
        Field(name="referral", memory_key=REFERRAL_STATUS_KEY,
              kind=AnswerKind.CHOICE, probe=REFERRAL_ASK, required=True,
              states=REFERRAL_STATES,
              required_when=RequiredWhen("accepting", frozenset({"yes"})),
              spoken="whether a referral is needed"),
    ),
    success_at=Outcome.COMPLETE,
)


PROVIDER_VERIFICATION = CallTemplate(
    name="provider_verification",
    description=(
        "Template 3 — provider verification. One call per doctor collecting "
        "BOTH the branch and whether they are accepting new patients, with the "
        "referral and waitlist follow-ups. Truthful identification, spoken by a "
        "named person. The new-patient status is four-valued "
        "(yes | no | waitlist | unsure), never a boolean."
    ),
    instructions=_build(
        _IDENTITY_HUMAN,
        goal=_GOAL_PROVIDER_VERIFICATION,
        vocabulary=_VOCABULARY_PROVIDER_VERIFICATION,
        what_counts=(_WHAT_COUNTS_PROVIDER_VERIFICATION + "\n\n"
                     + _PROVIDER_VERIFICATION_FLOW),
        tool_list=_TOOL_LIST_PROVIDER_VERIFICATION,
        # PAID FOR BY EVICTION. Both of these are now STATES of a recorded
        # field rather than escalate reasons, so the prose describing them as
        # exits is not merely redundant — it invites the model to escalate
        # where it should be saving a directory correction.
        the_doctor=_THE_DOCTOR_IDENTITY,
        flow_exits=_FLOW_EXITS_IDENTITY,
        flow_wrong_number=_FLOW_WRONG_NUMBER_IDENTITY,
    ),
    greeting=_PROVIDER_VERIFICATION_GREETING,
    transcribe_hint=_PROVIDER_VERIFICATION_HINT,
    language="english",
    objective=PROVIDER_VERIFICATION_OBJECTIVE,
)


TEMPLATES: dict[str, CallTemplate] = {
    FORAGE_DATA_COLLECTION.name: FORAGE_DATA_COLLECTION,
    FORAGE_AI_DISCLOSED.name: FORAGE_AI_DISCLOSED,
    PROVIDER_VERIFICATION.name: PROVIDER_VERIFICATION,
}


def get_template(name: str) -> CallTemplate:
    try:
        return TEMPLATES[name]
    except KeyError:
        raise ValueError(
            f"unknown call template {name!r} — available: {sorted(TEMPLATES)}"
        ) from None
