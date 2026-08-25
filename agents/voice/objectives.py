"""What a call is FOR: the fields it collects, when it is done, and what
shape of answer each ask is entitled to.

Three things that were previously not written down anywhere, and each was a
defect because of it.

1. **What the call collects.** Nothing declared it. `save_branch()` in tools.py
   was the only function in the programme that set ``resolved=True``, so "this
   call succeeded" meant "a branch was recorded" — a product-level definition
   living inside one tool implementation. A call that established
   accepting-new-patients and referral requirements and no branch recorded as
   NOT RESOLVED, and every metric, guard and directory row inherited that.

2. **Partial.** `resolved` is a boolean, so a call that got half of what it
   came for had to be filed as one or the other. `Outcome` is three-valued.
   Whether a given partial counts as success is DECLARED per objective
   (``success_at``) and not decided here — the client has not answered that
   question for branch-without-accepting-status yet, and a boolean would have
   forced an answer by omission.

3. **What kind of answer is expected.** A bare "Yes." is not a place, and it is
   a complete answer to "are you accepting new patients?". Same word, two
   verdicts, and the only thing that separates them is what was asked — so the
   caller's reply cannot be judged on its own words, which is exactly what
   `_is_filler_reply` used to do. See `expected_answers`.

Nothing here imports realtime_worker; the worker imports this. `LOCATION_NOUN`
and the sentence/clause helpers live here for that reason: `_is_location_ask`
in the worker and the branch field's own probe must be the SAME pattern, and a
second copy would rot exactly the way the 41 hand-copied prompt phrases in
tools.py rotted before `_derive_prompt_echoes` replaced them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Optional, Protocol


# ── Text utilities ───────────────────────────────────────────────────────────
# Moved here from realtime_worker (which now imports them back under its own
# private names) because the ask-shape detection below needs the same sentence
# and clause splitting the worker's detectors use, and two copies of a splitter
# is two behaviours.

# The model writes TYPOGRAPHIC apostrophes — "wasn’t", "it’s", "that’s" — and
# every pattern here spells them ASCII. Detectors were blind to the agent's own
# most common output until this existed.
SMART_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def norm_quotes(text: str) -> str:
    """ASCII-ise typographic quotes so the patterns can match what is said."""
    return (text or "").translate(SMART_QUOTES)


# Abbreviations whose full stop does not end a sentence. Without this, "Which
# branch is Dr. Okafor at?" splits into "Which branch is Dr." + "Okafor at?".
ABBREV = re.compile(
    r"\b(Dr|Mr|Mrs|Ms|Prof|St|Ave|Blvd|Rd|Ste|Dept|Inc|Co|approx|no)\.\s",
    re.I)
# A visible sentinel. The empty string is wrong (replacing "" inserts the
# replacement between every character) and a control byte is worse: invisible
# in source, and a literal 0x08 has landed in this repo twice.
ABBREV_MARK = "@@DOT@@"


def sentences(text: str) -> list:
    """Split into sentences without treating "Dr." as the end of one."""
    protected = ABBREV.sub(lambda m: m.group(0).replace(".", ABBREV_MARK), text)
    parts = re.split(r"(?<=[.!?])\s+", protected.strip())
    return [p.replace(ABBREV_MARK, ".").strip() for p in parts if p.strip()]


def clauses(text: str) -> list:
    """Sentences, split again at dashes, semicolons and colons.

    The prompt's own turn shape is "React, THEN say the thing, folded into ONE
    sentence", which produces `reaction — ask`. The ask is therefore the unit
    that carries the question, and it almost never sits at a sentence boundary.
    """
    out = []
    for s in sentences(text):
        for part in re.split(r"\s*[—–-]{1,2}\s+|\s*[;:]\s+", s):
            part = part.strip()
            if part:
                out.append(part)
    return out


# ── What the caller's next turn is allowed to be ─────────────────────────────

class AnswerKind(str, Enum):
    """The shape of answer an ask entitles the caller to give.

    PLACE  — a site name or street address. A bare "yes" is not one, which is
             why the filler filter was right to discard it for a location ask
             and wrong to apply that rule everywhere.
    CHOICE — one of a small closed set. Yes/no is the common case but NOT the
             general one: accepting-new-patients has four states, see
             `ChoiceAnswer`.
    FREE   — anything the caller says. Used for fields where any wording is
             information (e.g. what a referral requirement depends on).
    """
    PLACE = "place"
    CHOICE = "choice"
    FREE = "free"


class ChoiceAnswer(str, Enum):
    """The states a closed-set field can actually come back in.

    FOUR, not two. From the client call on 2026-08-24: accepting-new-patients
    is not a boolean. Offices hold a fixed number of new-patient slots per day
    or per week, and when those are gone the caller is given a QUEUE POSITION —
    "you'd be number twenty-one". That is neither yes nor no, and collapsing it
    to either loses the one fact the client would act on. UNSURE is the fourth
    because a front desk that does not know is a real, recordable answer and
    must not be stored as "no".
    """
    YES = "yes"
    NO = "no"
    WAITLIST = "waitlist"
    UNSURE = "unsure"


# The verbs a practice uses about taking somebody on, and the thing they take
# on. Named once: every pattern below needs them, and three spellings of
# "accepting new patients" is three chances to miss one.
# Inflections included. "We don't take new patients any more" is a refusal
# and the bare stem is how people say it; a pattern that only knows the
# -ing form reads it as no answer at all.
_ACCEPT_VERB = r"(?:tak(?:e|es|ing)|accept(?:s|ing)?|see(?:s|ing)?|admit(?:s|ting)?)"
_NEW_PATIENTS = r"new[- ]patients?"
# Up to three words of slack. THE SLACK IS THE FIX. "taking new patients" was
# written as a fixed phrase, so call-20260824-2014's "she's taking THE new
# patients" matched nothing at all and a clean, unambiguous yes was refused
# three times. Real speech puts determiners and adverbs in there — "taking on
# new", "accepting any new", "not currently taking" — and a pattern with no
# room for them only recognises the sentence its author happened to imagine.
_GAP = r"(?:\w+\s+){0,3}"

# Interjections and discourse markers people open with. The affirmative test was
# anchored with ^\W* — non-word characters only — so "Ah, yes" could not reach
# the "yes": the A is a word character and the anchor stopped dead on it. Same
# defect as _is_filler_reply judging "Yes." on its words alone, one layer down.
_LEAD_IN = r"(?:\W|\b(?:ah|oh|uh+|um+|er+|hmm+|well|so|okay|ok|right|now)\b)*"

# Ordered, and the ORDER IS THE LOGIC. "We're full, but I can put you on the
# list" contains "full" and "no" and "list"; it is a WAITLIST answer, and any
# ordering that reaches NO first records the office as closed to new patients
# when it is not. Same for UNSURE before NO: "I'm not sure, I don't think we
# are" is a front desk guessing, not a policy.
_CHOICE_PATTERNS: tuple[tuple[ChoiceAnswer, "re.Pattern[str]"], ...] = (
    (ChoiceAnswer.WAITLIST, re.compile(
        r"\b(wait ?list|waiting list|on (the|a) list|in (the )?queue|"
        r"you'?(d|ll) be (number|no\.?) ?\w+|number \d+ (in|on)|"
        r"we'?re full|we are full|full (up|right now|at the moment|for now|"
        r"this (week|month))|at capacity|no (open |available )?(slots|spots|"
        r"openings)|slots are (full|gone)|not taking (any )?more (right now|"
        r"at the moment|until)|put you (down|on))\b", re.I)),
    (ChoiceAnswer.UNSURE, re.compile(
        r"\b(not (really )?sure|no idea|do ?n'?t know|does ?n'?t know|"
        r"could ?n'?t say|can'?t say|cannot say|have to (ask|check)|"
        r"you'?d have to|it depends|depends on|it varies|varies by|"
        r"i'?d have to check|let me check)\b", re.I)),
    # NEGATION GETS THE SAME SLACK AS THE AFFIRMATIVE, and it has to, because
    # the two are read in order. "She's not currently taking new patients" used
    # to reach the YES pattern — `not taking` demanded adjacency, "currently"
    # broke it, and the affirmative phrase underneath matched — so a practice
    # REFUSING new patients was classified as accepting them. That is the wrong
    # row this whole system exists to prevent, and it was live.
    (ChoiceAnswer.NO, re.compile(
        rf"^\W*(no|nope|nah)\b"
        rf"|\b(?:not|never|no longer)\s+{_GAP}{_ACCEPT_VERB}\b"
        rf"|\b(?:are ?n'?t|is ?n'?t|do ?n'?t|does ?n'?t|are not|is not)\s+"
        rf"{_GAP}{_ACCEPT_VERB}\b"
        rf"|\b(we'?re not|we are not|closed to new)\b", re.I)),
    (ChoiceAnswer.YES, re.compile(
        rf"^{_LEAD_IN}(?:yes|yeah|yep|yup)\b"
        rf"|\b(we are|we do|we'?re accepting)\b"
        rf"|\b{_ACCEPT_VERB}\s+{_GAP}{_NEW_PATIENTS}\b"
        rf"|\b(?:taking|accepting)\s+(?:on\s+)?new\b"
        rf"|\b(happy to|absolutely|certainly|of course)\b", re.I)),
)

# A negator anywhere earlier IN THE SAME CLAUSE flips an affirmative. Checked
# separately rather than written into the YES pattern because Python has no
# variable-length lookbehind, and because the clause boundary is the part that
# matters: "we're not doing walk-ins, but she is taking new patients" is a YES,
# and a whole-string negation test would call it a NO.
_NEGATOR = re.compile(r"\b(?:not|never|no longer|cannot)\b|n'?t\b", re.I)
_CLAUSE_SPLIT = re.compile(r"[,;.!?]|\bbut\b|\bthough\b|\bhowever\b", re.I)


def _negated_before(text: str, idx: int) -> bool:
    """Is the affirmative at `idx` inside a clause that was already negated?"""
    return bool(_NEGATOR.search(_CLAUSE_SPLIT.split(text[:idx])[-1]))


class ReferralAnswer(str, Enum):
    """Whether a new patient needs a referral. A SEPARATE vocabulary.

    Not ChoiceAnswer with different labels: the client's question is "is a
    referral needed — always, or does it depend?", so the interesting answer is
    the CONDITIONALITY, and "depends" is the state their team acts on. Squeezing
    that into yes/no would throw away the distinction the question exists to
    draw, exactly as recording a queue position as "no" would.
    """
    ALWAYS = "always"
    DEPENDS = "depends"
    NO = "no"
    UNSURE = "unsure"


# Ordered for the same reason _CHOICE_PATTERNS is. "Only for some insurers"
# contains no "depends" but is a DEPENDS answer; "yes, always" must not be read
# as DEPENDS because it contains "insurance" somewhere later in the sentence.
_REFERRAL_PATTERNS: tuple[tuple[ReferralAnswer, "re.Pattern[str]"], ...] = (
    (ReferralAnswer.UNSURE, re.compile(
        r"\b(not (really )?sure|no idea|do ?n'?t know|could ?n'?t say|"
        r"can'?t say|have to (ask|check)|you'?d have to|i'?d have to check)\b",
        re.I)),
    (ReferralAnswer.DEPENDS, re.compile(
        r"\b(depends|depend on|varies|only (for|with|if)|some (plans|insurers|"
        r"insurances)|certain (plans|insurers)|if (their|the) insurance|"
        r"case by case|case-by-case|depending)\b", re.I)),
    (ReferralAnswer.ALWAYS, re.compile(
        r"\b(always|every time|all (new )?patients|in all cases|"
        r"yes,? (they|you|a referral)|required for (all|every)|"
        r"we require|must have (a )?referral|need(s|ed)? a referral)\b", re.I)),
    (ReferralAnswer.NO, re.compile(
        r"^\W*(no|nope|nah)\b|\b(not (needed|required)|no referral|"
        r"do ?n'?t need|does ?n'?t need|without a referral|"
        r"self[- ]refer)\b", re.I)),
)


def classify_referral(text: str) -> Optional[ReferralAnswer]:
    """Which referral state this caller turn expresses, or None."""
    t = norm_quotes(text or "").strip()
    if not t:
        return None
    for state, pattern in _REFERRAL_PATTERNS:
        if pattern.search(t):
            return state
    return None


# A leading BARE POLARITY TOKEN used as a discourse marker, for stripping.
#
# BOTH FAMILIES, not just the affirmative. "No, not at the moment." answered a
# question about SCHEDULING, and with only the yes-family stripped it stood
# alone as a referral NO — because a bare "no" is a valid referral answer in
# that vocabulary. A turn leaning entirely on an opening yes OR an opening no
# asserts nothing about a field nobody asked about.
#
# A DELIMITER IS REQUIRED after the token, so only a discourse marker is taken.
# Without it "no referral needed" lost its "no" and stopped classifying at all —
# there the word is a determiner carrying the meaning, not a preface to it. The
# case this does not cover is an undelimited preface ("Yeah we are full"), which
# is refused on the never-asked path only; the agent then asks, and it grounds
# normally on the next turn.
_BARE_AFFIRM_LEAD = re.compile(
    rf"^{_LEAD_IN}(?:yes|yeah|yep|yup|no|nope|nah)\b"
    rf"(?:\s*[,.;:!?-]+\s*|\s*$)", re.I)


def states_in_its_own_right(text: str, state_value: str,
                            classifier=None) -> bool:
    """Does this turn assert the state WITHOUT leaning on a leading "yes"?

    The discriminator for a turn offered before anyone asked. "Yes, speaking."
    classifies as YES purely on the opening token and asserts nothing about new
    patients; "we are full right now, but I can put you on the list" asserts the
    condition in its own words and would still do so with the "Yeah" removed.

    This replaced a test for whether the turn contained the ASK's vocabulary,
    which was the wrong question: on call-20260825-0915 a textbook waitlist
    answer — full, list, number 21 — contains none of "accepting", "taking
    new", "new patients", and was refused twice for it.
    """
    t = norm_quotes(text or "").strip()
    stripped = _BARE_AFFIRM_LEAD.sub("", t).strip()
    if not stripped:
        return False
    # THE FIELD'S OWN VOCABULARY. Defaulting to classify_choice and leaving it
    # there made the referral guard read "No, not at the moment." — an answer to
    # the SCHEDULING question — as a referral answer, because classify_choice
    # recognises it and classify_referral does not. Each field anchors on its
    # own probe; it has to classify with its own vocabulary too.
    got = (classifier or classify_choice)(stripped)
    return got is not None and got.value == state_value


def classify_choice(text: str) -> Optional[ChoiceAnswer]:
    """Which of the four states this caller turn expresses, or None.

    None means "this turn is not an answer to a closed-set question" — it is
    not a fifth state and must never be recorded as one. A CHOICE field whose
    stored value does not classify is treated as NOT COLLECTED (see
    `Field.present`), which is the same discipline save_branch applies to a
    branch name: a value the process cannot recognise is not evidence.
    """
    t = norm_quotes(text or "").strip()
    if not t:
        return None
    for state, pattern in _CHOICE_PATTERNS:
        m = pattern.search(t)
        if not m:
            continue
        # Only YES inverts under negation. A negated WAITLIST or UNSURE is not
        # a different state, and NO is already the negative.
        if state is ChoiceAnswer.YES and _negated_before(t, m.start()):
            return ChoiceAnswer.NO
        return state
    return None


class IdentityAnswer(str, Enum):
    """Did we reach the right doctor at the right practice?

    THE FIRST QUESTION, and until 2026-08-25 the script had no way to ask it.
    From the client-side contact: "First level of check will be — is this Dr.
    John Smith's office? Then we ask, who is Dr. John Smith, who is a
    cardiologist? Then does he accept new patients. If we don't know which
    doctor they're talking about, accepting new patients makes no sense."

    FOUR STATES, and the split between the two negatives is the point.
    Collapsing them would throw away the more valuable result:

      CONFIRMED    — right doctor, right practice. Everything else may proceed.
      NOT_HERE     — we reached the practice correctly and the doctor is not
                     there: left, never was, or a different site. THE PHONE
                     NUMBER IS FINE and the doctor-to-hospital association on
                     file is wrong. That is a directory correction, and
                     arguably the most valuable negative result this programme
                     can produce — filing it as "wrong number" would send
                     somebody to re-verify a number that was never the problem.
      WRONG_NUMBER — we never reached the practice at all. The number on file
                     is wrong and the row is unusable until it is replaced.
      UNSURE       — the person answering does not know. A locum on the desk is
                     a real, recordable state, not a failure.

    SPECIALTY LIVES IN THIS FIELD, not beside it. The reason specialty is
    collected at all is disambiguation — two doctors called John Smith at one
    hospital, and the specialty is how the receptionist knows which is meant —
    so confirming "Dr. Smith, the cardiologist" IS establishing which Dr.
    Smith. It belongs in this field's `heard` and `detail`, and a specialty
    mismatch ("we have a Dr. Smith, he's a dermatologist") resolves to NOT_HERE
    with the detail saying why, never to CONFIRMED.
    """
    CONFIRMED = "confirmed"
    NOT_HERE = "not_here"
    WRONG_NUMBER = "wrong_number"
    UNSURE = "unsure"


# Ordered like the others, and the order carries the same weight. WRONG_NUMBER
# first because it is the most specific claim; UNSURE before the negatives
# because "I'm not sure he works here" is a person who does not know, not a
# practice denying the doctor; CONFIRMED last so a stray "yes" cannot outrank a
# denial that happens to contain one.
_IDENTITY_PATTERNS: tuple[tuple[IdentityAnswer, "re.Pattern[str]"], ...] = (
    (IdentityAnswer.WRONG_NUMBER, re.compile(
        r"\b(wrong number|wrong (place|office)|you'?ve got the wrong|"
        r"there'?s no such|no such (number|place)|"
        # "This is a bakery." The trailing noun is the whole signal, and an
        # earlier version demanded a word before it, so the commonest phrasing
        # of all matched nothing.
        r"this is (a|an) (\w+\s+){0,2}(shop|store|restaurant|bakery|salon|"
        r"pharmacy|garage|hotel|takeaway)|"
        r"not a (doctor|medical|clinic))\b",
        re.I)),
    (IdentityAnswer.UNSURE, re.compile(
        r"\b(not (really )?sure|do ?n'?t know|no idea|i'?m new|just started|"
        r"i'?d have to check|let me check|have to (ask|check)|"
        # Offering to go and look is NOT a denial. "We have a few doctors but I
        # can check" was landing on NOT_HERE via the qualified-denial shape
        # below — a receptionist being helpful, recorded as the practice
        # denying the doctor. UNSURE is read first, so this settles it.
        r"i'?ll (check|look|find out|ask)|"
        r"i (can|could|will) (check|look|find out|ask)|"
        r"could ?n'?t say|can'?t say)\b", re.I)),
    (IdentityAnswer.NOT_HERE, re.compile(
        r"\b(does ?n'?t work here|do ?n'?t work here|not here|no longer (here|"
        r"with us|works)|has left|he left|she left|they left|no one by that "
        r"name|nobody by that name|never heard of|we do ?n'?t have (a|any|an)|"
        r"there'?s no (dr\.?|doctor)|retired|moved (to|away)|"
        r"different (practice|office|clinic)|not (one of )?our|"
        # THE SPECIALTY MISMATCH, which is a NOT_HERE and reads nothing like
        # one: "we have a Dr. Smith but he's a dermatologist" acknowledges the
        # name and then withdraws it. The contradiction is the signal — this
        # guard cannot know which specialty was wanted, so it recognises the
        # SHAPE of a qualified denial and leaves the reason to `detail`.
        # `[^?!]`, not `[^.?!]`: the gap has to cross "Dr." and a full stop
        # inside an abbreviation is not a sentence boundary — the same problem
        # ABBREV and sentences() exist for. It also lets the two-sentence form
        # through ("We have a Dr. Smith. But he's a dermatologist."), which is
        # how people actually say it.
        r"(we|they) (do )?have\b[^?!]{0,40}\bbut\b)\b", re.I)),
    (IdentityAnswer.CONFIRMED, re.compile(
        r"^\W*(yes|yeah|yep|yup)\b|\b(that'?s (us|right|correct|him|her|them)|"
        r"speaking|this is (he|she|him|her)|you'?ve reached|correct|"
        r"(he|she|they) (works?|practi[cs]es?) here|"
        r"(he|she|they) (is|are) here)\b", re.I)),
)


def classify_identity(text: str) -> Optional[IdentityAnswer]:
    """Which identity state this caller turn expresses, or None."""
    t = norm_quotes(text or "").strip()
    if not t:
        return None
    for state, pattern in _IDENTITY_PATTERNS:
        m = pattern.search(t)
        if not m:
            continue
        # A leading "yes" inside a denial does not confirm anything, the same
        # inversion classify_choice guards against.
        if state is IdentityAnswer.CONFIRMED and _negated_before(t, m.start()):
            return IdentityAnswer.NOT_HERE
        return state
    return None


# Asking whether we have reached the right doctor at the right practice.
#
# NARROW, and it has to be: "office" and "practice" are also LOCATION_NOUN, so
# a loose pattern would make every branch ask look like an identity ask and
# anchor the identity guard on the wrong turn. Every alternative here requires
# the DOCTOR to be named or referred to, which a branch ask does not do.
IDENTITY_ASK = re.compile(
    r"\bhave i reached\b"
    r"|\b(is|was) this\s+(dr\.?|doctor)\b"
    r"|\b(dr\.?|doctor)\s+[\w'-]+(?:'s)?\s+(office|practice|practise|clinic)\b"
    r"|\bright (doctor|practice|place) for\b"
    # "…work HERE", not "…work out of". Without the place-anchor this matched
    # "Which branch does she work out of?" — a BRANCH ask — and would have
    # anchored the identity guard on the wrong turn. `office` and `practice`
    # are LOCATION_NOUN too, so every alternative in this probe has to carry
    # something a branch ask does not.
    r"|\bdoes\s+(dr\.?|doctor|she|he|they)\b[^?!]{0,30}"
    r"\b(work|works|practise|practice|practises|practices|see patients)\s+"
    r"(here|there|at this|with you|for you|out of (here|there))\b",
    re.I)


# The location noun. ONE definition, shared by the worker's `_is_location_ask`
# and by the branch field's probe below.
LOCATION_NOUN = re.compile(
    r"\b(branch|location|office|campus|site|address|practis\w*|practic\w*)\b",
    re.I)

# Asking about taking on new patients. Declared here next to LOCATION_NOUN so a
# template can point a Field at it; nothing reads it until a template does.
ACCEPTING_ASK = re.compile(
    r"\b(accepting|taking (on )?new|new patients|new-patient|"
    r"open to new|seeing new)\b", re.I)

# Asking whether a new patient can actually get on the books right now.
#
# DELIBERATELY NARROW. A bare \bschedul\w+\b matches template 1's "Days
# mentioned -> use the schedule field", which is a note about save_branch's
# third argument and not a question anyone asks a receptionist. The probe gates
# both the ask budget and the promise check, and a template being told it
# promised a question it never asks is a false alarm that teaches people to
# ignore the check. So the scheduling ask has to be ABOUT booking somebody in.
SCHEDULING_ASK = re.compile(
    r"\b(schedul\w+|book\w*|get (them |someone )?in)\b[^.?!]{0,40}"
    r"\b(appointment|new patient|patients?|with (her|him|them|the doctor))\b"
    r"|\bnew patients?\b[^.?!]{0,40}\b(schedul\w+|book\w*|get in|be seen)\b"
    r"|\b(appointments?|openings?)\b[^.?!]{0,30}\bavailable\b", re.I)

# Asking whether a referral is needed, and whether that is unconditional.
#
# THE NOUN ONLY. An earlier version also matched "refer(red)", which fires on
# the shared Conversation Flow's "Referred to a website or email -> note_info" —
# a rule about being fobbed off, not the referral question. Same lesson as
# SCHEDULING_ASK: these probes gate the ask budget, the grounding anchor AND the
# promise check, so a probe that over-matches makes a template look like it is
# asking a question it never asks.
REFERRAL_ASK = re.compile(r"\breferrals?\b", re.I)

# Auxiliaries and modals. A clause that OPENS with one of these and contains no
# wh-word is a polar question — the caller is being asked to pick from a closed
# set, whatever nouns the sentence happens to contain.
_AUX = (r"(?:is|are|was|were|do|does|did|can|could|will|would|have|has|had|"
        r"should|shall|may|might|am)")
_POLAR_OPENER = re.compile(
    rf"^\W*(?:and|but|so|ok|okay|right|just|sorry|then|also|now)?[,\s]*"
    rf"{_AUX}\b", re.I)
# "which office", "what's the address" — a wh-word means an open answer is
# wanted even when the clause opens with an auxiliary: "Do you know WHICH
# office she's at?" is a request for a place, not for a yes.
_WH = re.compile(r"\b(which|what|what'?s|where|whereabouts|who|whom|whose|"
                 r"how many|how much)\b", re.I)
# Tag questions. "That's the only one, right?" is polar however it opens.
_TAG = re.compile(
    r",\s*(right|correct|yes|no|is that (right|correct)|isn'?t it|"
    r"are ?n'?t (you|they)|do ?n'?t you|do you|can you)\s*\??\s*$", re.I)


def is_polar_question(clause: str) -> bool:
    """Is this clause asking the caller to pick, rather than to supply?

    Form, not vocabulary. "Is that your only office there?" names an office and
    expects a yes; "which office is she at?" names one and expects a place. The
    nouns are identical and the shape is not, which is why the branch
    clarification push-back — where the model is told to get "confirmation this
    is their only location there" — was being scored as a location ask and the
    receptionist's "Yes." thrown away as filler.
    """
    c = norm_quotes(clause or "").strip()
    if not c:
        return False
    if _TAG.search(c):
        return True
    return bool(_POLAR_OPENER.match(c)) and not _WH.search(c)


# ── Fields and objectives ────────────────────────────────────────────────────

class _MemoryLike(Protocol):
    def get(self, field: str, default: Any = None) -> Any: ...


@dataclass(frozen=True)
class RequiredWhen:
    """This field is required only when another field came back a certain way.

    DECLARATIVE, NOT A CALLABLE, and the reason is the direction it fails in.
    A predicate over memory would be more general, but a wrong key inside a
    lambda returns None, `holds` goes False, the field is never required, and
    the call reports COMPLETE having asked two of four questions — silently,
    with nothing able to tell that from a call where the condition genuinely
    did not apply. Naming the gate field instead makes that a structural error
    that `invalid_conditions()` catches before a call is placed, the same way
    `unwritable_fields()` catches a memory_key nothing writes.

    It also has to be READABLE, not just checkable: `missing_spoken()` feeds a
    directive the agent says out loud. On a call where the office is not taking
    new patients, the agent must not announce it failed to get the referral
    rule for a question the script never reaches.
    """
    field: str                  # another field in the SAME objective
    is_any_of: frozenset        # values of that field which activate this one

    def holds(self, objective: "CallObjective", memory: "_MemoryLike") -> bool:
        gate = objective.field_named(self.field)
        if gate is None:
            # Structurally invalid; invalid_conditions() reports it loudly.
            # Answering False here would quietly complete the call, so answer
            # True: an unanswerable gate makes the field required, which at
            # worst leaves the call PARTIAL and visible.
            return True
        value = memory.get(gate.memory_key)
        return value is not None and str(value).strip().lower() in self.is_any_of

    def describe(self) -> str:
        """The gate as a phrase, for a log line or an artifact."""
        return f"{self.field} in {{{', '.join(sorted(self.is_any_of))}}}"


def _gate_chain_unresolvable(objective: "CallObjective", field: "Field") -> bool:
    """Can this field's gate chain ever produce an answer?

    THE FAIL-SAFE HALF of allowing chains at all. `invalid_conditions()` reports
    a broken chain to whoever reads warnings; this is what stops a broken chain
    QUIETLY COMPLETING A CALL, and the two have to agree.

    The hole it closes: for a cycle a -> b -> a, `RequiredWhen.holds()` finds
    both gate fields, reads an empty memory value for each, and answers False —
    so both fields become not-required, `missing` empties, and the objective
    reports COMPLETE having collected neither. holds() only fails safe when the
    gate field is ABSENT from the objective; a cycle keeps them all present.

    Unresolvable means: a cycle, a chain that walks out of the objective, or a
    chain whose root is not required at all so nobody ever asks the first
    question. In each case the dependent field stays REQUIRED, the call stays
    PARTIAL, and the fault shows up in `missing` where somebody will see it.
    """
    if field.required_when is None:
        return False
    seen = {field.name}
    walk = objective.field_named(field.required_when.field)
    while walk is not None and walk.required_when is not None:
        if walk.name in seen:
            return True                      # cycle
        seen.add(walk.name)
        walk = objective.field_named(walk.required_when.field)
    if walk is None:
        return True                          # chain leaves the objective
    return not walk.required                 # root nobody ever asks


@dataclass(frozen=True)
class Field:
    """One thing a call is trying to find out.

    ``memory_key`` is where a tool writes it. That coupling is checked, not
    assumed — see `unwritable_fields`: a field nothing can write would leave
    every call permanently PARTIAL, and the failure would look like callers
    refusing rather than like a template declaring something no tool records.
    """
    name: str
    memory_key: str
    kind: AnswerKind
    probe: "re.Pattern[str]"
    required: bool = True
    # The values a CHOICE field may hold. The TOOL canonicalises what the
    # caller said into one of these; this is the membership check that stops
    # anything else satisfying the objective. Empty for PLACE and FREE fields,
    # where any non-empty value is the answer.
    states: frozenset = frozenset()
    # Required only under a condition — see RequiredWhen. None means always
    # required (when `required` is set at all).
    required_when: Optional[RequiredWhen] = None
    # How the field is referred to OUT LOUD, in a directive the model has to
    # turn into a sentence. Defaults to the field name with its underscores
    # opened out, which reads acceptably for "branch" and badly for anything
    # longer — so a template collecting new-patient status is expected to say
    # how it wants that said rather than have the agent read an identifier to a
    # receptionist.
    spoken: str = ""

    @property
    def label(self) -> str:
        return self.spoken or self.name.replace("_", " ")

    def present(self, memory: _MemoryLike) -> bool:
        """Is this field collected?

        A CHOICE field additionally has to hold one of ITS OWN declared states.
        Without that, a tool writing "I'll have to check" into the accepting
        field would satisfy the objective and resolve the call — the same
        false-positive that grounding stops for a branch name, one field type
        later.

        Membership, not classification. The tool is what turns what the caller
        said into a state; by the time a value is stored it should already BE
        one, so anything else is a bug in the tool rather than a sentence to be
        re-read. It also lets each field carry its own vocabulary — referral is
        always/depends/no/unsure, which is not the accepting field's set.
        """
        value = memory.get(self.memory_key)
        if value is None:
            return False
        text = str(value).strip()
        if not text:
            return False
        if self.kind is AnswerKind.CHOICE:
            return text.lower() in (self.states or CHOICE_STATES)
        return True

    def is_required(self, objective: "CallObjective",
                    memory: _MemoryLike) -> bool:
        """Required for THIS call, given what it has established so far."""
        if not self.required:
            return False
        if self.required_when is None:
            return True
        # A gate that can never resolve leaves this field REQUIRED, so the call
        # stays PARTIAL and the fault is visible — never COMPLETE-too-early.
        # Same direction as holds() answering True on a gate it cannot find.
        if _gate_chain_unresolvable(objective, self):
            return True
        return self.required_when.holds(objective, memory)


class GateVerdict(str, Enum):
    """May a value for a gated field be written RIGHT NOW?

    `RequiredWhen` answers a different question — whether the field is required
    — and answering only that was a hole big enough to defeat the identity
    field entirely. On call-20260825-1437 the branch and the new-patient status
    were both saved for a doctor the call never confirmed existed at that
    practice, and both rows reached doctors.json stamped source=voice,
    status=partially_verified. `missing: ["identity"]` sat in the same artifact
    saying so, and nothing read it.

    Relaxing a requirement and permitting a write are opposite directions. A
    field that is not required yet is a question we may skip; a field whose gate
    is unsettled is an answer we may not FILE, because we do not yet know who it
    is about. "Dr. Reyes is at Eastside Clinic, not taking new patients" is not
    partial data when Dr. Reyes was never confirmed to be there — it is wrong
    data about a real practice, which is the most expensive thing this system
    can produce.

    THREE STATES, because the two ways a gate can fail want opposite handling:

      OPEN    — the gate holds. Write it.
      PENDING — the gate field has no answer YET. The caller really did say
                this, so refusing outright would throw away a real answer, and
                that is this project's expensive direction of failure. The value
                is HELD (see tools._defer_save) and applied the moment the gate
                opens.
      CLOSED  — the gate field has an answer, and it is not one that opens this
                field. identity=not_here means the doctor is not at this
                practice; a branch collected there belongs to nobody and no
                later turn can make it belong to somebody. Dropped, and recorded
                as dropped.
    """
    OPEN = "open"
    PENDING = "pending"
    CLOSED = "closed"


def gate_state(objective: "CallObjective", memory: _MemoryLike,
               field: Optional["Field"]) -> tuple:
    """(verdict, gate_field, gate_value) for writing `field` right now.

    An ungated field is always OPEN, which is every field of both branch
    templates — this cannot change a script that declares no conditions.

    AN UNRESOLVABLE GATE CHAIN IS OPEN, and that is the opposite direction from
    `is_required`, deliberately. There, a broken chain keeps the field REQUIRED
    so the call stays PARTIAL and the fault is visible in `missing`. Here, a
    broken chain must not silently refuse every write on every call: the fault
    is in the template, `invalid_conditions()` reports it before a call is
    placed, and destroying the caller's answers is not an acceptable way to
    surface a typo.
    """
    if field is None or field.required_when is None:
        return (GateVerdict.OPEN, None, None)
    if _gate_chain_unresolvable(objective, field):
        return (GateVerdict.OPEN, None, None)
    cond = field.required_when
    gate = objective.field_named(cond.field)
    if gate is None:
        return (GateVerdict.OPEN, None, None)
    value = memory.get(gate.memory_key)
    text = "" if value is None else str(value).strip()
    if not text:
        return (GateVerdict.PENDING, gate, None)
    if text.lower() in cond.is_any_of:
        return (GateVerdict.OPEN, gate, text)
    return (GateVerdict.CLOSED, gate, text)


class Outcome(IntEnum):
    """How much of the objective a call actually got.

    Ordered so a policy can be stated as a threshold (``success_at``) rather
    than as a list of acceptable values.
    """
    NONE = 0
    PARTIAL = 1
    COMPLETE = 2

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class CallObjective:
    """What a template collects, and what counts as done.

    ``success_at`` is the OPEN QUESTION, declared rather than answered. For a
    one-field objective PARTIAL is unreachable and the setting is inert. For
    the branch + accepting-new-patients script it decides whether a call that
    got the branch and not the accepting status is a partial success or a
    failure — and the client lead has not said which. Declaring it as a
    threshold means answering it later is a one-line change in a template,
    with the alternative visible in the type, instead of an archaeology
    expedition through whatever `resolved` came to mean.
    """
    fields: tuple[Field, ...]
    success_at: Outcome = Outcome.COMPLETE

    def field_named(self, name: str) -> Optional[Field]:
        return next((f for f in self.fields if f.name == name), None)

    def kinds(self) -> frozenset:
        """Every answer kind this objective is collecting."""
        return frozenset(f.kind for f in self.fields)

    def collected(self, memory: _MemoryLike) -> tuple:
        return tuple(f.name for f in self.fields if f.present(memory))

    def missing(self, memory: _MemoryLike) -> tuple:
        """Required fields still absent. Empty means the objective is met.

        REQUIRED FOR THIS CALL, not required in the abstract. A script whose
        later questions only apply when an earlier answer came back a certain
        way — ask about scheduling only if they are taking new patients — would
        otherwise leave a correct "no, we're not" call sitting permanently
        PARTIAL, blamed on a receptionist who answered everything she was
        asked. See RequiredWhen.
        """
        return tuple(f.name for f in self.fields
                     if f.is_required(self, memory) and not f.present(memory))

    def not_applicable(self, memory: _MemoryLike) -> tuple:
        """Fields this call will never need, because their condition is unmet.

        Recorded in the artifact rather than merely omitted: "we did not ask"
        and "we asked and got nothing" are different facts about a call, and a
        reviewer counting referral answers needs to know which calls were even
        eligible to have one.
        """
        return tuple(f.name for f in self.fields
                     if f.required and f.required_when is not None
                     and not f.required_when.holds(self, memory))

    def missing_spoken(self, memory: _MemoryLike) -> str:
        """The missing fields as something the agent can say out loud."""
        return ", ".join(f"the {f.label}" for f in self.fields
                         if f.is_required(self, memory) and not f.present(memory))

    def outcome(self, memory: _MemoryLike) -> Outcome:
        if not self.collected(memory):
            return Outcome.NONE
        return Outcome.COMPLETE if not self.missing(memory) else Outcome.PARTIAL

    def is_success(self, memory: _MemoryLike) -> bool:
        """The boolean every existing consumer still reads, derived at last.

        `resolved` is not deleted — CallRecord, doctors.json, master.json, the
        four telephony workers and run_voice all read it — but it is now
        DERIVED from the objective at one site instead of being asserted by
        save_branch.
        """
        return self.outcome(memory) >= self.success_at


# Where save_new_patient_status writes. Named here rather than spelled as a
# string literal in tools.py, templates.py and the tests, because a field whose
# memory_key does not match the key its tool writes is the exact silent failure
# unwritable_fields() exists to catch — and three hand-copied spellings is how
# that mismatch happens.
IDENTITY_STATUS_KEY = "doctor_identity"
NEW_PATIENT_STATUS_KEY = "new_patient_status"
SCHEDULING_STATUS_KEY = "scheduling_status"
REFERRAL_STATUS_KEY = "referral_status"

# The states each CHOICE field may hold. A field carries its own set — see
# Field.present — because these vocabularies are genuinely different and
# collapsing them would lose the distinction each question exists to draw.
IDENTITY_STATES = frozenset(s.value for s in IdentityAnswer)
CHOICE_STATES = frozenset(s.value for s in ChoiceAnswer)
REFERRAL_STATES = frozenset(s.value for s in ReferralAnswer)

# Keys the tools in tools.py can actually write. note_info writes note_<key>
# for any key, so that whole namespace is available; everything else is the
# save_* signatures.
_TOOL_WRITTEN_KEYS = frozenset({
    "branch", "city", "schedule", IDENTITY_STATUS_KEY,
    NEW_PATIENT_STATUS_KEY, SCHEDULING_STATUS_KEY, REFERRAL_STATUS_KEY,
})


def invalid_conditions(objective: CallObjective) -> tuple:
    """Broken RequiredWhen gates. Empty is the healthy answer.

    THE CHECK THAT PAYS FOR THE DECLARATIVE FORM. Each of these would otherwise
    fail silently in the COMPLETE-too-early direction — a call reporting done
    having asked half its questions, indistinguishable from one where the
    condition legitimately did not apply:

      * a gate naming a field that is not in this objective (a typo, or a field
        renamed and the gate not updated)
      * a gate on a field that is not itself unconditionally required, so the
        gate may never resolve at all and the dependent question never fires
      * a gate whose values are not states the gate field can actually hold,
        which makes it unsatisfiable — "accepting in {ye}" is never true
      * a field gated on itself
    """
    problems: list[str] = []
    for f in objective.fields:
        cond = f.required_when
        if cond is None:
            continue
        if cond.field == f.name:
            problems.append(f"{f.name} is gated on itself")
            continue
        gate = objective.field_named(cond.field)
        if gate is None:
            problems.append(
                f"{f.name} is gated on {cond.field!r}, which this objective "
                f"does not declare")
            continue
        # THE CHAIN IS WALKED, not refused on sight.
        #
        # This used to reject any gate whose own field was conditional, on the
        # grounds that "the gate may never resolve". That was stricter than the
        # hazard: with identity -> accepting -> scheduling, a denied identity
        # leaves accepting uncollected, so scheduling's gate reads None and goes
        # False. The chain resolves correctly BY ABSENCE, which is exactly how
        # the script is supposed to behave.
        #
        # What genuinely cannot resolve is a CYCLE, or a chain that never
        # reaches a field somebody is unconditionally required to ask. Both are
        # walked for here.
        #
        # THE FAILURE DIRECTION IS PRESERVED. This loosens a check that failed
        # safe, so the replacement has to fail the same way: an unresolvable
        # chain is REPORTED, and RequiredWhen.holds() answers True on a gate it
        # cannot find — so the dependent field stays required, the call stays
        # PARTIAL, and the fault is visible. Nothing here can make a call
        # COMPLETE that should not be.
        seen = [f.name]
        walk = gate
        while walk is not None and walk.required_when is not None:
            if walk.name in seen:
                problems.append(
                    f"{f.name} sits on a cycle of gates: "
                    f"{' -> '.join(seen + [walk.name])}")
                break
            seen.append(walk.name)
            walk = objective.field_named(walk.required_when.field)
        else:
            if walk is None:
                problems.append(
                    f"{f.name} is gated through {' -> '.join(seen[1:])}, which "
                    f"ends at a field this objective does not declare")
            elif not walk.required:
                problems.append(
                    f"{f.name} is gated through {' -> '.join(seen[1:] + [walk.name])}, "
                    f"and {walk.name!r} is not required at all — the chain can "
                    f"never resolve, so this is never asked")
        if gate.kind is AnswerKind.CHOICE:
            allowed = gate.states or CHOICE_STATES
            unknown = sorted(v for v in cond.is_any_of if v not in allowed)
            if unknown:
                problems.append(
                    f"{f.name} is gated on {gate.name}={unknown}, which "
                    f"{gate.name} can never hold (its states are "
                    f"{sorted(allowed)})")
        if not cond.is_any_of:
            problems.append(f"{f.name} has an empty gate — never required")
    return tuple(problems)


def unwritable_fields(objective: CallObjective) -> tuple:
    """Fields whose memory_key no tool writes. Empty is the healthy answer.

    A template declaring a field nothing records does not fail loudly: every
    call comes back PARTIAL, the give-up path fires on callers who answered
    everything, and the artifact says the caller did not provide it. Cheap to
    check, invisible if not checked.
    """
    return tuple(f.name for f in objective.fields
                 if f.memory_key not in _TOOL_WRITTEN_KEYS
                 and not f.memory_key.startswith("note_"))


def expected_answers(ask_text: str, objective: CallObjective) -> frozenset:
    """Which answer kinds the caller's reply to this turn may take.

    A SET, deliberately. "Are you accepting new patients, and which office is
    that?" is one turn asking two things, and a reply of "Yes" answers it while
    a reply of "the Newark office" answers it too. Returning a single kind
    would discard one of them, and discarding an answer is this project's
    expensive direction of failure — a wrong row can be found later, a thrown
    away answer looks exactly like a receptionist who would not say.

    Being wrong here is bounded by construction: the only replies whose
    treatment depends on the returned set are bare acknowledgement-shaped ones
    (see `_is_filler_reply`). Anything with content in it is an answer either
    way, so a misclassified ask cannot discard a real place name.
    """
    text = norm_quotes(ask_text or "").strip()
    if not text:
        # No ask identified. Accept an answer to anything we are collecting —
        # for a single-field objective that is the historical behaviour
        # exactly, and for a multi-field one it is the safe direction.
        return objective.kinds()

    kinds: set = set()
    for clause in clauses(text) or [text]:
        matched = [f for f in objective.fields if f.probe.search(clause)]
        if is_polar_question(clause):
            # The form decides. A polar question about a location still expects
            # a yes: "is that your only office there?".
            kinds.add(AnswerKind.CHOICE)
        elif matched:
            kinds |= {f.kind for f in matched}
    return frozenset(kinds) or objective.kinds()


# ── The one place success is decided ─────────────────────────────────────────

def record_outcome(memory: Any, objective: CallObjective) -> Outcome:
    """Recompute the call's outcome and write it to memory. Returns it.

    Called after every tool call rather than inside one. save_branch used to
    set ``resolved=True`` itself, which made "the call succeeded" a statement
    one tool was entitled to make about the whole call; note_info and escalate
    could not participate, and a template collecting a second field had nowhere
    to say so.

    Writes the derived boolean as well as the three-valued outcome, so every
    existing reader of memory["resolved"] keeps working and none of them has to
    learn about objectives.
    """
    out = objective.outcome(memory)
    memory.update(
        outcome=out.label,
        resolved=bool(out >= objective.success_at),
        collected=list(objective.collected(memory)),
        missing=list(objective.missing(memory)),
        # "never applied" is a different fact from "asked and got nothing", and
        # only this distinguishes them afterwards. A reviewer counting referral
        # answers needs to know which calls were eligible to have one.
        not_applicable=list(objective.not_applicable(memory)),
    )
    return out


def describe(objective: CallObjective, memory: _MemoryLike) -> str:
    """One line for a log or an artifact: what was collected, what was not."""
    got = objective.collected(memory)
    lost = objective.missing(memory)
    skipped = objective.not_applicable(memory)
    parts = [f"outcome={objective.outcome(memory).label}"]
    if got:
        parts.append("collected=" + ",".join(got))
    if lost:
        parts.append("missing=" + ",".join(lost))
    if skipped:
        parts.append("n/a=" + ",".join(skipped))
    return "  ".join(parts)


def branch_field() -> Field:
    """The field every template so far collects: where the doctor practises."""
    return Field(name="branch", memory_key="branch", kind=AnswerKind.PLACE,
                 probe=LOCATION_NOUN, required=True)


def default_objective() -> CallObjective:
    """The configured template's objective.

    For the classic (non-realtime) pipeline in brain.py, which has no template
    object of its own, and as the fallback for a test session built without
    one.
    """
    from core.config import settings
    from agents.voice.templates import get_template
    return get_template(settings.call_template).objective
