"""What the caller actually said, and whether a value is supported by it.

Every guard in here answers one question — *is there evidence for this?* — over
`sess.turns` and nothing else. None of it touches the WebSocket, the audio
loop, or the session's lifecycle; the only thing it wants from a session is the
transcript and a place to record what it refused.

SPLIT OUT OF realtime_worker.py, which had reached 7,787 lines. Not for
tidiness: pyright stops analysing a function once it is large enough, and when
it gives up it can no longer prove any local inside is read — the editor greys
out dozens of names and stops seeing the calls the module makes. That has bitten
this project before (see _handle_tool_call's docstring, and the week where every
recurring bug lived in the one function pyright had abandoned). This module is
the half that can be reasoned about on its own, so it is the half that moves.

The dependency runs ONE WAY: realtime_worker imports this, never the reverse.
That is checked, not hoped for — the extraction was chosen as the transitive
closure of these guards, and that closure contains no class and nothing from the
transport surface. Keep it that way. If something here starts needing a
RealtimeSession's behaviour rather than its transcript, it belongs on the other
side of the line.

Names are re-exported from realtime_worker so existing callers, and the suite's
`rw._thing` references, keep working unchanged.
"""

import asyncio
import re
import time
from statistics import median
from typing import Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:                       # pragma: no cover
    # THE ONE-WAY RULE, AND WHY THIS DOES NOT BREAK IT. Every `sess` annotation
    # in this file is the string "RealtimeSession", and a string annotation
    # needs a binding for the type checker even though nothing evaluates it at
    # runtime. Without this the module still ran and the whole suite still
    # passed — and Pylance reported fifteen "RealtimeSession is not defined"
    # errors, which is precisely the kind of noise that trains an editor's
    # warnings to be ignored.
    #
    # TYPE_CHECKING is False when Python runs, so no import happens and no
    # cycle exists. The runtime dependency stays exactly one way:
    # realtime_worker imports evidence, never the reverse.
    from agents.voice.realtime_worker import RealtimeSession

from agents.voice.objectives import (
    LOCATION_NOUN as _LOCATION_NOUN,
    norm_quotes as _norm_quotes,
    states_in_its_own_right,
)


# Caller-utterance RMS below which the line is treated as too faint to trust.
# Clear phone speech measures roughly 0.03-0.08; a live call that produced a
# fabricated answer measured 0.004-0.012 throughout.
_LOW_AUDIO_RMS = 0.015


# Turns that MENTION the location without asking for it: acknowledging a value
# just given, or signing off. Everything else that names a location is a request,
# whatever shape it takes.
_NOT_AN_ASK = re.compile(
    r"\b(thanks|thank you|got it|perfect|great|appreciate|have a (good|great)|"
    r"take care|goodbye|bye now|i'?ll (note|record|pass)|i have that|"
    r"that'?s all|no (problem|worries))\b", re.I)


# An acknowledgement together with the location noun it takes as its OBJECT.
#
# _NOT_AN_ASK strips the acknowledgement WORD and leaves the noun it governs,
# so "Thanks for the location." became " for the location." — still a location
# noun, still counted as an ask. Observed on call-20260820-1915: seven
# location_asks against a limit of four, and the verbatim-ask nudge firing to
# tell the agent to "stop stapling it on" about a sentence that asks nothing.
# It cost nothing that call — holds had reset the budget — but an inflated
# count ends a call early on a call without holds.
#
# The distinguishing feature is grammatical, not vocabulary: in the failing
# family the noun is the acknowledgement's object, not part of a fresh request.
# So consume the phrase whole, before the residue test runs.
#
# THE NEGATIVE LOOKAHEAD IS LOAD-BEARING. Without it the two-word gap jumps a
# clause boundary — "Great — and which campus is that" had "and which" eaten
# and the real question with it, which is the expensive direction: a missed ask
# lets the agent pester someone. Words that open a new clause end the object.
_ACK_TAKES_VALUE = re.compile(
    r"\b(thanks|thank you|appreciate|got it|perfect|great)\b"
    r"[,\s—\-]*"
    r"(?:for|on|about)?[,\s]*"
    r"(?:the|that|this|your|those)?\s*"
    r"(?:(?!(?:and|but|so|which|what|where|who|if|when|still|need)\b)\w+\s+){0,2}"
    r"(?:branch|location|office|campus|site|address)\b", re.I)


# Reading back a value the caller already gave.
# READING A VALUE BACK IS NOT ASKING FOR ONE, and the list has to cover the
# agent QUOTING the caller as well as the agent filing the value. On
# call-20260824-2014 the agent said "I heard you say she's taking the new
# patients." — a read-back by any reading — and because that phrasing was
# missing here it scored as an ASK. The grounding anchor moved past every
# caller turn that had answered, the evidence window emptied, and the guard
# stood down and accepted a status it had refused three times. The agent talked
# its own claim into the record.
_CONFIRMS_VALUE = re.compile(
    # PRESENT PROGRESSIVE TOO. The list held "i'll note" and missed "I'm just
    # noting Riverside Campus now" on call-20260827-1010 - the same act, filed
    # as it is said. It matched LOCATION_NOUN on the value it was reading back
    # and scored as a branch ask.
    r"\b(i have that as|i'?ve got that|i'?ll note|i'?m (just )?(noting|"
    r"recording|writing (that|it|this) down)|i'?ve noted|noted as|recorded as|"
    r"i'?ll put (that|it) down|so that'?s|i heard you say|"
    r"you said|what i heard|let me read (that|it) back|"
    # NOT a bare "to confirm". "I'm trying to confirm which branch she works
    # out of" is an ASK, and swallowing it would stop the budget counting the
    # commonest phrasing the agent has. The read-back sense always quotes
    # THEM — "I heard you say", "you said" — and that is the load-bearing part.
    r"i'?ll record (that|it))\b", re.I)


# Reporting that the location was NOT obtained. Names a location noun and reads
# as an ask to the inverted detector, but it is the opposite — it is the agent
# giving up. On call-20260818-1338 "I wasn't able to get the specific branch
# today" was counted as an ask, so a closing line spent a slot of the ask
# budget. Only checked on statements: "I couldn't find the branch — do you know
# it?" carries a question mark and is a genuine ask.
_REPORTS_FAILURE = re.compile(
    r"\b(was ?n'?t able|were ?n'?t able|was not able|could ?n'?t|could not|"
    r"can'?t|cannot|unable|did ?n'?t manage|no luck)\b", re.I)


# PROMISING TO ASK LATER IS NOT ASKING NOW.
#
# The third member of the family above, and the one that got onto a phone.
# call-20260827-1010: the agent said "Thanks for that - I'm just noting
# Riverside Campus now, then I'll ask about new patients." It named the topic,
# carried no question mark, and was neither a read-back nor a closing line, so
# the inverted detector scored it as an ask for the `accepting` field. Nobody
# was asked anything. Three things then ran off that phantom:
#
#   1. `_field_ask_at["accepting"]` was stamped, so the FIRST real ask forty
#      seconds later scored as a RE-ASK, and _field_already_answered went
#      looking for an answer that could only belong to another question. It
#      found "No, I don't have it." - the answer to the street address - and
#      the nudge told the model to record it as the new-patient status.
#   2. `_is_objective_ask` is the gate on the ask budget, so a turn that asked
#      for nothing spent a slot of the budget that ends the call.
#   3. It is the anchor for `_ungrounded_status`. This is the same hole
#      _CONFIRMS_VALUE was cut for on call-20260824-2014, entered from the
#      other tense: an agent turn that is ABOUT the topic while asking nothing
#      moves the evidence window past the turns that answered.
#
# CONSUMED WHOLE, like _ACK_TAKES_VALUE and for the same reason - the promise
# takes its own object, and stripping only the verb would leave the noun behind
# and change nothing. `[^.?!]*` ends the object at the sentence, so a real ask
# in the SAME turn ("...then I'll ask about new patients. Which branch?")
# survives the strip and still counts.
#
# THE DEFERRAL MARKER IS REQUIRED, and it is what keeps this narrow. A missed
# ask lets the agent pester someone, which is the expensive direction, so a
# bare "I'll ask about the branch" - which a receptionist would simply answer -
# is left alone. Only a promise that plainly points at LATER is exempt.
_DEFER = (r"(?:then|next|after (?:that|this)|afterwards?|later|"
          r"in a (?:moment|minute|second|bit)|"
          r"once (?:that'?s|we'?re|that is) (?:done|sorted|out of the way))")
_WILL_ASK = r"i(?:'?ll|'?m going to|'?m gonna| will)\s+(?:then\s+)?ask\b"
#
# `(?=[.!]|$)` IS THE RIGHT EDGE OF THE OBJECT, and it is load-bearing in the
# same way _ACK_TAKES_VALUE's negative lookahead is. `[^.?!]*` alone stops just
# short of a question mark, having already eaten the question: "Then I'll ask
# about new patients - are you taking any?" had the real ask consumed by the
# promise. So the object must END at a full stop or the end of the turn. A
# promise whose own sentence carries a question mark is not stripped at all -
# it is counted as an ask, which is the over-counting side.
_ANNOUNCES_ASK = re.compile(
    # "then I'll ask about new patients" - marker before the promise
    rf"\b{_DEFER}\b[,\s\u2014\-]*{_WILL_ASK}[^.?!]*(?=[.!]|$)"
    # "I'll ask about new patients in a moment" - marker after it
    rf"|\b{_WILL_ASK}[^.?!]*\b{_DEFER}\b[^.?!]*(?=[.!]|$)", re.I)


def _is_ask_for(text: str, probe) -> bool:
    """Is this agent turn asking for the thing `probe` recognises?

    The body of what used to be _is_location_ask, with the noun pattern passed
    in. Parametrised rather than copied when a second field arrived: the
    acknowledgement, read-back and closing exemptions below were each added
    after a live call miscounted, and a second copy of them would have to
    relearn every one of those calls.

    Counts statement-form asks as well as questions. A request phrased politely
    is still a request, and the person on the other end experiences it as one.

    This used to be a whitelist of phrasings requiring a question mark, and it
    scored 0 asks on a call that asked four times — the agent had simply picked
    wordings that were not on the list ("trying to confirm" where the list held
    "trying to find out"). Enumerating phrasings cannot work: the model has more
    ways to ask than anyone can list.

    So it is inverted. Naming the thing IS an ask unless the turn is plainly
    acknowledging or closing. This over-counts a little, which is the safe
    direction for a budget whose purpose is to stop the agent pestering people.
    """
    text = _norm_quotes(text)
    if not probe.search(text):
        return False
    # A promise to ask LATER is not an ask - see _ANNOUNCES_ASK. Stripped
    # before anything else looks at the turn, including the question-mark
    # short-circuit below, because the promise and a real question routinely
    # share a turn and only the promise is exempt.
    text = _ANNOUNCES_ASK.sub("", text)
    if not probe.search(text):
        return False
    # Reading a value back is not asking for one.
    if "?" not in text and (_CONFIRMS_VALUE.search(text)
                            or _REPORTS_FAILURE.search(text)):
        return False
    if "?" in text:
        return True
    # An acknowledgement that goes on to ask for something is still an ask, so
    # only a turn that is ENTIRELY acknowledgement is exempt. Take the
    # acknowledgement's own object with it first — see _ACK_TAKES_VALUE — or
    # "Thanks for the location." leaves a location noun behind and reads as a
    # request for the thing it is thanking them for.
    stripped = _ACK_TAKES_VALUE.sub("", text)
    stripped = _NOT_AN_ASK.sub("", stripped)
    return bool(probe.search(stripped))


def _is_location_ask(text: str) -> bool:
    """Is this agent turn asking where the doctor practises?"""
    return _is_ask_for(text, _LOCATION_NOUN)


# The caller putting a question TO the agent instead of answering theirs.
#
# call-20260819-2121, in sixty seconds:
#   "Sorry, who's calling again?"
#   "Um, is this about a patient or something urgent?"
#   "Is this about patient related?"
#   "How can I help you?"
# Four turns, four questions, no refusal anywhere — a front desk deciding
# whether this call is safe to engage with, which is their job. The ask budget
# counted every one of them as an ask that went unanswered, hit its limit of
# four, and told the agent to escalate. The agent then hung up on "How can I
# help you?" — an open door, and the clearest invitation on the whole call.
#
# `_caller_answered_since` was the wrong instrument to lean on here: it asks
# "did they say something substantive", and a question IS substantive. It just
# is not a refusal, and the budget exists to end calls that are going nowhere,
# not calls where the other person is still working out who they are talking
# to.
#
# Matched by SHAPE, not by a phrase list. Interrogative opener, or an offer of
# help, in a turn that contains no location — an open set of wordings with a
# closed set of shapes.
_VETTING_OPENER = re.compile(
    r"^\W*(?:um+|uh+|er+|so|sorry|okay|ok|alright|yeah|well|hi|hello)?[\s,]*"
    r"(?:who|what|why|which|where|how|is|are|was|were|do|does|did|can|could|"
    r"would|will|may|might|should|sorry)\b", re.I)


# An explicit offer to keep going. Stronger than a screening question: they are
# not deciding whether to engage, they have decided and are waiting on you.
_INVITATION = re.compile(
    r"\bhow\s+(?:can|may|could)\s+i\s+(?:help|assist)\b"
    r"|\bwhat\s+can\s+i\s+(?:do|help)\b"
    r"|\bwhat\s+(?:do|did)\s+you\s+need\b"
    r"|\bwhat(?:'?s| is)\s+(?:this|it)\s+(?:regarding|about|in regard)\b"
    r"|\bgo\s+ahead\b|\bhow\s+can\s+i\s+help\b", re.I)


def _invites_continuation(text: str) -> bool:
    """The caller asking what the agent wants — an open door, not a refusal.

    Blocking escalation on this is the same move as blocking it on a hold
    request. A caller who says "How can I help you?" has told you they are
    willing; ending the call there throws away the one turn most likely to
    produce an answer.
    """
    return bool(_INVITATION.search(_norm_quotes(text or "")))


def _caller_is_vetting(text: str, sess: "RealtimeSession") -> bool:
    """The caller questioning the agent rather than answering, or declining.

    NOT a refusal and NOT an answer — a third thing the budget had no category
    for. Requires the turn to carry no location: "Which branch? The Mission Bay
    one." opens with an interrogative and is plainly an answer, so a shape test
    alone would misread it.
    """
    t = _norm_quotes(text or "").strip()
    if not t:
        return False
    if _invites_continuation(t):
        return True
    if not ("?" in t or _VETTING_OPENER.match(t)):
        return False
    # A turn that NAMES something is an answer however it is phrased. "Which
    # one — the Mission Bay clinic?" opens with an interrogative and is plainly
    # an answer, so the shape test alone would misread it. Same capitalisation
    # signal the grounding checks use, and the same caveat: skip the first word
    # (always capitalised) and skip what we brought to the call ourselves.
    known: set[str] = set()
    known |= _distinctive(getattr(sess.doctor, "hospital_name", "") or "")
    known |= _distinctive(sess.org_name or "")
    known |= {w for w in re.findall(r"[a-z]+",
                                    (getattr(sess.doctor, "doctor_name", "") or "").lower())
              if len(w) > 2}
    if sess.agent_name:
        known.add(sess.agent_name.lower())
    #
    # A proper noun alone is not enough: the first live case was "This is
    # Northside Medical Group and I'm Varun. Sorry, who's calling again?" —
    # which is vetting, and "Varun" is the caller's own name, not a branch. So
    # the word must also sit within two words of a location anchor, the same
    # conjunction _candidate_location uses.
    raw = [w.strip(".,!?-—'\"") for w in t.split()]
    words = [w.lower() for w in raw]
    for i, w in enumerate(words):
        if i == 0 or len(w) <= 2 or not w.isalpha():
            continue
        if (w in known or w in _UNGROUNDED_STOPWORDS or w in _NON_PLACE
                or w in _ORG_STOPWORDS):
            continue
        if not raw[i][:1].isupper():
            continue
        near = words[max(0, i - 2):i] + words[i + 1:i + 3]
        if any(n in _LOCATION_ANCHORS for n in near):
            return False
    return True


# A turn made of nothing but affirmative/negative tokens and punctuation.
#
# THE DISCRIMINATOR FOR A QUESTION MARK THAT IS NOT A QUESTION. The transcriber
# punctuates by intonation, and a receptionist's rising "Yes?" — confirming
# while inviting you to go on — comes back with a "?" on it. _caller_is_vetting
# then fires on the "?" alone, because its only escape hatch is a proper noun
# beside a location anchor and a one-word affirmative has nothing for it to
# find. See the CHOICE call site below for why that mattered.
#
# Deliberately NOT a general "is this interrogative" test. This matches only
# turns with no content beyond the affirmative, so "Yes, that's right?" and
# "Yeah, hi David, how are you?" are untouched — the first because it may be
# echoing our words back for confirmation, the second because it plainly is a
# question. Losing those costs one turn; accepting a real question as an answer
# is the failure _turn_asserts was built for.
_ONLY_AFFIRM = re.compile(
    r"^[\W_]*(?:(?:yes|yeah|yep|yup|no|nope|nah|sure|correct|right|speaking|"
    r"uh|um|oh|ok|okay)\b[\W_]*)+$", re.I)


def _turn_asserts(text: str, sess: "RealtimeSession", *,
                  classifier=None, state: str = "") -> bool:
    """Is this caller turn TELLING us something, or ASKING us?

    Grounding compares a saved value against the caller's own words, and until
    2026-08-20 it did that over one blob of every caller turn — so a value the
    caller ASKED about grounded exactly as well as one they stated.

    call-20260820-1703: the caller said "She's in San Francisco, right?" and
    never confirmed it afterwards. `city: "San Francisco"` was written to the
    directory stamped "verified against caller transcript". They had asked us.
    A receptionist seeking OUR confirmation is not evidence, and we had none to
    give — the record holds an organisation, not a city.

    The distinction already existed in this file. `_caller_is_vetting` was
    built for the ask budget, after the agent hung up on "How can I help you?",
    and it carries the hard part: a capitalised proper noun within two words of
    a _LOCATION_ANCHORS word makes the turn an ANSWER however interrogative its
    shape. That is what keeps "Which one — the Mission Bay clinic?" and "It's
    Mission Bay Clinic, right?" usable. Grounding simply never consulted it.

    THE "?" CONJUNCT IS NOT DECORATION. _caller_is_vetting also fires on
    _VETTING_OPENER alone, with no question mark, and the first cut of this
    predicate threw away "Sorry, Northgate." — a bare answer opening with an
    opener word. That is the case the rule right below defends in its own
    comment: "'Northgate' on its own is a perfectly good answer". Losing a real
    answer is the expensive direction, so an actual question mark is required
    before a turn can be discounted.

    NEGATION IS NOT HANDLED HERE and is tracked separately: "We're not
    Northside Medical Group" still grounds "Northside Medical Group". It is a
    different axis — a denial, not a question — and wants its own predicate.

    ── THE CHOICE ESCAPE HATCH ────────────────────────────────────────────────
    `classifier`/`state` are passed ONLY by _ungrounded_choice, and they exist
    because the paragraph above ("the hard part") is a PLACE test that a CHOICE
    field cannot use. A capitalised proper noun beside a location anchor is what
    rescues "Which one — the Mission Bay clinic?"; a two-bit answer has no
    proper noun to be rescued by, so for those four fields the "?" conjunct had
    nothing standing behind it and decided the question alone.

    Live, on call-20260825-1847: the agent asked "is this Dr. Carol,
    Neurosurgery, at New York Presbyterian?", the caller said "Yes?", and this
    predicate returned False. classify_identity("Yes?") is `confirmed` and was
    never consulted. The save was refused as "they have only asked back, not
    answered", the model was told to get their words, and it asked the identical
    question again nine seconds later — which is the repeat the client reported.
    The same holds for "Yeah?", "Correct?" and "Speaking?".

    So a CHOICE field may rescue a turn, on two conjuncts that must BOTH hold:
    the turn carries no content beyond an affirmative (_ONLY_AFFIRM), and it
    classifies under THAT FIELD'S OWN vocabulary to the state being claimed.
    The second conjunct is what stops this becoming "any '?' turn counts": a
    bare "No?" cannot ground `confirmed`, because classify_identity does not
    read it that way.

    THIS DOES NOT REOPEN THE FABRICATION HOLE, and the reason is that the check
    it relaxes is not the last one. _ungrounded_choice still puts every rescued
    turn through _is_hint_echo against the measured audio level — the gate its
    own docstring calls "the whole test, not a tiebreak" for this field type —
    so a phantom "Yes." on dead air is refused exactly as before. What changes
    is only that a REAL bare affirmative stops being thrown away for carrying a
    mark the caller did not put there.

    Default arguments leave every PLACE caller byte-identical.
    """
    if not ("?" in (text or "") and _caller_is_vetting(text, sess)):
        return True
    if classifier is None or not state:
        return False        # PLACE, and any caller that did not opt in.
    if not _ONLY_AFFIRM.match((text or "").strip()):
        return False        # It said something past the affirmative. Vetting.
    got = classifier(text)
    return got is not None and got.value == state


# Words that anchor a location. A distinctive word sitting next to one of these
# is a candidate place name; the same word anywhere else is just a word. The
# adjacency requirement is what keeps this from firing on every proper noun in
# the call — "Hello, David" has no anchor near it.
_LOCATION_ANCHORS = frozenset({
    "branch", "branches", "campus", "campuses", "clinic", "clinics",
    "office", "offices", "center", "centre", "centers", "centres",
    "hospital", "hospitals", "location", "locations", "site", "sites",
    "building", "tower", "wing", "block", "street", "road", "avenue",
    "boulevard", "lane", "drive", "parkway", "suite", "floor", "area",
})


# Conversational words that will happily sit next to an anchor while naming no
# place at all: "the main branch", "our other office", "which location".
_NON_PLACE = frozenset({
    "main", "other", "another", "same", "this", "that", "these", "those",
    "our", "their", "his", "her", "its", "one", "two", "both", "all", "any",
    "some", "each", "every", "which", "what", "where", "when", "who", "why",
    "here", "there", "yes", "yeah", "yep", "no", "not", "but", "for", "with",
    "from", "about", "only", "just", "also", "still", "sorry", "please",
    "thanks", "thank", "hello", "hey", "okay", "sure", "right", "well",
    "you", "your", "yours", "we", "our", "they", "them", "him", "she", "he",
    "are", "was", "were", "have", "has", "had", "does", "did", "can",
    "could", "will", "would", "should", "need", "needs", "want", "know",
    "tell", "say", "said", "give", "gave", "get", "got", "see", "sees",
    "working", "works", "work", "patients", "patient", "doctor", "doctors",
    "emergency", "call", "calling", "called", "number", "details", "detail",
    "information", "anything", "something", "nothing", "everything",
    "speaking", "moment", "minute", "second", "wait", "hold", "checking",
    # Capitalisation is doing the heavy lifting, so this list only has to
    # cover words that survive it — sentence-initial ones, where every word is
    # capitalised whatever it is.
    "closed", "open", "sorry", "sure", "try", "let", "hang", "just", "look",
    "there's", "thats", "yeah", "well", "actually", "maybe", "probably",
})


# Words that carry no identifying information, so their presence in the
# transcript proves nothing about whether the caller named a real place.
_UNGROUNDED_STOPWORDS = {
    "the", "a", "an", "of", "at", "in", "on", "our", "their", "and",
    "branch", "branches", "office", "offices", "campus", "campuses",
    "clinic", "clinics", "center", "centre", "centers", "centres",
    "hospital", "location", "locations", "site", "sites", "medical",
    "building", "unit", "practice", "city", "street", "road", "avenue",
}


# Words that appear in almost every healthcare organisation's name. Matching on
# these would make "Methodist Medical Center" look like "Northside Medical
# Group", which is exactly the confusion this check exists to catch.
_ORG_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "for", "at", "st", "saint",
    "hospital", "hospitals", "clinic", "clinics", "medical", "medicine",
    "health", "healthcare", "center", "centre", "group", "practice",
    "associates", "physicians", "care", "services", "system", "systems",
    "institute", "department", "dept", "office", "offices", "campus",
})


def _distinctive(name: str) -> set:
    """The tokens in an organisation name that actually identify it."""
    return {w for w in re.findall(r"[a-z]+", (name or "").lower())
            if w not in _ORG_STOPWORDS and len(w) > 2}


# Numbers written as words, mapped to their value. The value is needed to tell
# RENDERING from SUBSTITUTION, which is the whole difficulty here:
#
#   caller "1825 4th Street"   -> "1825 Fourth Street"     rendering. Fine.
#   caller "1844th Street"     -> "eighteen forty fourth"  substitution. Not.
#
# Both replace digits with words. The first keeps a digit the caller gave and
# spells an ordinal that traces back to one ("4th" -> "fourth"); the second
# erases the number entirely and nothing in it traces anywhere. A test that
# just looked for number-words blocked both, and blocking the first throws
# away a correct address — the expensive direction.
#
# NOT a general parser. "eighteen forty fourth" is genuinely ambiguous between
# 1844th, 18 44th and 1840 4th, and picking one would be inventing an address.
# Each word is checked on its own: did the caller say this word, or the digit
# it stands for? That question has an answer without resolving the ambiguity.
#
# "a" and "an" are absent on purpose: articles far more often than quantities,
# and treating them as numbers would reject half of all real branch names.
_NUMBER_WORD_VALUE: dict[str, int] = {
    **{w: i for i, w in enumerate("""
        zero one two three four five six seven eight nine ten eleven twelve
        thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty
    """.split())},
    **{w: v for w, v in zip(
        "thirty forty fifty sixty seventy eighty ninety".split(),
        range(30, 100, 10))},
    "hundred": 100, "thousand": 1000,
    **{w: i + 1 for i, w in enumerate("""
        first second third fourth fifth sixth seventh eighth ninth tenth
        eleventh twelfth thirteenth fourteenth fifteenth sixteenth
        seventeenth eighteenth nineteenth twentieth
    """.split())},
    **{w: v for w, v in zip(
        "thirtieth fortieth fiftieth sixtieth seventieth eightieth "
        "ninetieth".split(), range(30, 100, 10))},
    "hundredth": 100, "thousandth": 1000,
}


def _drop_lost_substance(spoken: str, dropped: str) -> bool:
    """Did muting the second item cost the caller something they needed?

    The one-item guard keeps the FIRST item to produce audio and mutes the
    rest, and it has to: by the time a second item appears the first is already
    on the wire, so there is no choosing between them. But the model does not
    reliably put the substance first, and when it does not the guard deletes
    the answer and keeps the throat-clearing.

    call-20260820-1421, the caller having asked "can you repeat that question
    please?":

        spoken  : "Sure, I'll repeat it clearly."
        muted   : "I'm trying to find out which branch Dr. Okafor works out of."

    They asked for the question and got a promise to give it. Seven seconds of
    silence, then the watchdog. They asked again, the answer was muted again,
    and they hung up at 88s.

    So this is NOT a test of whether to mute — that decision is forced. It is a
    test of whether anything is now OWED, so it can be said in the next turn
    rather than lost. Content words the muted item had and the spoken one did
    not: two or more, and at least half of what it was carrying.

    Deliberately conservative, because the failure of saying it anyway is
    repeating yourself, which this project treats as the thing that makes
    people hang up. "Sure, no rush." muted behind "Sure, no rush." owes
    nothing, and neither does a rephrasing of the ask that was already spoken.
    """
    # The model often REGENERATES the spoken half and appends the rest, so the
    # muted item is a superset rather than a different sentence. Judging the
    # whole thing then dilutes the new part with the repeated one: on
    # call-20260820-1421 the muted item repeated the identity answer and added
    # "Could you tell me which branch Dr. Okafor sees patients at?", and the
    # ask — the only thing the caller had not heard — scored 0.47 and was
    # written off. Strip the repeated head first and judge what is left.
    _n = lambda t: re.sub(r"[^a-z0-9 ]", " ", (t or "").lower()).split()
    _sp, _dr = _n(spoken), _n(dropped)
    owed = _dr[len(_sp):] if _dr[:len(_sp)] == _sp else _dr

    d = {w for w in owed if len(w) > 2 and w not in _UNGROUNDED_STOPWORDS}
    if not d:
        return False
    # If the half they HEARD already asked for the location, a muted second ask
    # owes them nothing — they have the question, and saying it again is the
    # repetition this project treats as what makes people hang up. Word overlap
    # cannot see this on its own: "do you know which branch she works out of
    # these days?" behind "which branch is she at?" shares little vocabulary
    # and is the same request.
    if _is_location_ask(spoken) and _is_location_ask(dropped):
        return False
    said = set(_sp)
    new = d - said
    # Two or more words they have not heard, and at least half of what the
    # muted part was carrying. Both halves matter: the count stops a one-word
    # difference counting, and the fraction stops a REPHRASING of what was
    # already said — "do you know which branch she works out of these days?"
    # behind "which branch is she at?" owes nothing but three stray words.
    return len(new) >= 2 and len(new) / len(d) >= 0.5


# How many times one owed sentence, and one call, may chase a recovery.
#
# NOTHING COUNTED ATTEMPTS, and on call-20260825-1435 that was a livelock. The
# mute in the delta handler is unconditional on a second spoken item; the
# recovery scheduled here is itself a response; the model produced TWO items
# for it; the second was muted, carried the same substance, and set
# `_owed_substance` again. Every pass through the loop looked exactly like the
# first, so nothing could tell it was the fourth. The caller's question was
# never answered.
#
# TWO CAPS, because there are two ways to loop and one counter sees only one of
# them. The per-text cap stops the agent chasing the identical sentence. The
# per-call cap stops the version where the model REGENERATES the owed half a
# little differently each time — same substance, different letters, a per-text
# key can never match it. Both are small on purpose: a recovery that has been
# muted twice is not being muted for a reason a third attempt fixes.
_MAX_OWED_PER_TEXT = 2


_MAX_OWED_PER_CALL = 3


def _owed_key(text: str) -> str:
    """Identity of an owed sentence, for counting attempts at it."""
    return _collapse(text)[:120]


def _owed_refusal(sess: "RealtimeSession", text: str) -> str:
    """Why this owed sentence must NOT be chased again. "" means go ahead.

    A REFUSAL IS NOT A REPAIR. Everything this returns is a call where the
    caller asked something and will not be answered, so it is recorded in
    `owed_abandoned` and printed rather than simply stopping the loop. The bug
    this closes was invisible precisely because giving up and never owing
    anything produced the same artifact.
    """
    n = sess._owed_attempts.get(_owed_key(text), 0)
    if n >= _MAX_OWED_PER_TEXT:
        return f"said {n}x already and muted every time"
    if sess._owed_tried >= _MAX_OWED_PER_CALL:
        return f"{sess._owed_tried} recovery attempts on this call already"
    return ""


def _ungrounded_detail(args: dict, sess: "RealtimeSession", key: str) -> list:
    """Content words in the model's free-text qualifier that nobody said.

    `detail` (and `depends_on`) cannot be fixed by SELECTION the way `heard`
    can: there is no single caller turn that is the qualifier, because the
    field is a summary by construction — "you'd be number 21", "book online or
    call the front desk". Nothing in the transcript is the right thing to copy
    in wholesale.

    So this one is the fallback the selection avoided: check it. Word level,
    not substring, because a summary legitimately reorders and drops words —
    demanding a verbatim substring would reject every honest summary. What it
    catches is the failure actually observed: the model inserting a NOUN nobody
    said. On call-20260824-2116 the qualifier read "Book online or call the
    front desk" when the caller had said only "you need to book through online
    or call" — "desk" appears nowhere in the call.

    Same collapse as the branch check, so "front-desk" and "front desk" are the
    same word to it, and the same stopword list, so ordinary English does not
    have to be grounded.
    """
    value = str(args.get(key) or "").strip()
    if not value:
        return []
    heard = _asserted_caller_text(sess)
    if not heard.strip():
        return []
    out: list = []
    # SPLIT ON NON-LETTERS, not on whitespace. Stripping punctuation off a
    # whitespace token leaves "front-desk" whole, and `.isalpha()` is False for
    # it, so a hyphenated invention was skipped without ever being compared \u2014
    # and "front-desk" is exactly the shape of the word this has to catch.
    for w in re.findall(r"[a-z']+", value.lower()):
        w = w.strip("'")
        if (not w or len(w) <= 2 or w in _UNGROUNDED_STOPWORDS
                or w in _DETAIL_FUNCTION_WORDS or w in out):
            continue
        if _grounded_loosely(w, heard):
            continue
        # A meaning word stands on its CLASS, not on itself. The caller who
        # said "don't" made the negation; the caller who said "as long as"
        # made the condition. Which word the model reached for afterwards is
        # not evidence of anything.
        cls = _meaning_class(w)
        if cls and _class_present(cls, heard):
            continue
        out.append(w)
    return out


# Words whose removal would change what the sentence CLAIMS rather than how it
# reads — grouped into CLASSES, and the grouping is the fix.
#
# These were a flat set, checked word by word against the transcript, and that
# made the guard fire hardest on exactly the answers the client most wants. A
# model paraphrasing the connective is the single most predictable thing it
# does:
#
#   caller "as long as they've got the right insurance"
#   model  "only if they have the right insurance"      -> EMPTIED
#   caller "they need a referral from their primary"
#   model  "only with a referral from their primary care doctor" -> EMPTIED
#
# and the same on the other side:
#
#   caller "we don't take new patients until January"
#   model  "not taking new patients until January"      -> EMPTIED
#
# In every one of those the caller DID negate, or DID make it conditional. The
# model reached for a different word for the same move. Asking whether the
# CALLER SAID "only" is the wrong question; the question is whether the caller
# expressed conditionality at all.
#
# So membership is checked per CLASS: a meaning word counts as grounded when
# ANY member of its class appears in what the caller asserted. An invented
# condition — "only if insured" on a call where nothing was conditional — still
# has no class-mate to stand on, and still drops the whole qualifier.
_MEANING_CLASSES: dict = {
    # Reversing the polarity of the claim.
    "negation": frozenset({
        "not", "never", "without", "cannot", "cant", "dont", "doesnt",
        "isnt", "arent", "wont", "wouldnt", "couldnt", "nor", "none",
        "neither", "no", "nope", "stopped", "closed", "refuse", "refused",
    }),
    # Making the claim conditional — the shape CAQH is after: "yes, but only if
    # you have insurance with this particular company". Necessity words belong
    # here too: "they need a referral" and "only with a referral" are the same
    # move, and a model will swap one for the other without hesitating.
    "condition": frozenset({
        "only", "unless", "except", "provided", "depends", "depending",
        "whether", "case", "long", "need", "needs", "needed", "require",
        "requires", "required", "must", "if", "when", "certain", "some",
    }),
}


# Auxiliaries, copulas and light verbs. Skipped outright in a QUALIFIER, the
# way _UNGROUNDED_STOPWORDS are skipped everywhere: their presence or absence
# says nothing about whether the model invented anything, and checking them
# produced "only if they the right insurance" — a sentence mangled by the
# removal of "have" because the caller had said "got".
_DETAIL_FUNCTION_WORDS = frozenset({
    "are", "is", "was", "were", "been", "being", "have", "has", "had",
    "having", "does", "did", "doing", "will", "would", "can", "could",
    "shall", "should", "get", "gets", "got", "with", "from", "their",
    "them", "they", "your", "you", "our", "its", "it", "that", "this",
    "these", "those", "there", "here", "and", "but", "for", "the", "any",
    "all", "one", "also", "just", "then", "than", "who", "which", "what",
    "about", "into", "onto", "over", "under", "been",
})


def _stem(word: str) -> str:
    """Crude stem, so an inflection is not mistaken for an invention.

    "we don't TAKE new patients" and "not TAKING new patients" are the same
    verb, and reporting "taking" as a word nobody said cost the qualifier on
    call-20260825. Suffix-stripped and de-silent-e'd, so take/takes/taking all
    reduce to "tak".

    Used ONLY for the free-text qualifier. Branch grounding keeps its exact
    comparison: a stem match is a loosening, and the branch field is where a
    loosening costs a wrong address in the directory.
    """
    w = word.lower().strip("'")
    for suf in ("ings", "ing", "ers", "er", "ies", "ied", "es", "ed", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    return w[:-1] if len(w) > 3 and w.endswith("e") else w


def _grounded_loosely(word: str, heard: str) -> bool:
    """Grounded allowing for inflection. Qualifier fields only.

    Two steps, because suffix-stripping only reaches inflections of one lemma.
    "cardiology" and "cardiologists" are the same fact in different parts of
    speech, and no amount of trimming -s and -ing turns one into the other —
    on call-20260825-1226 the caller said "cardiologists", the model wrote
    "cardiology", and the identity qualifier was emptied over the difference.
    A shared long prefix catches that family without reaching anything else:
    the words have to agree for six characters, which ordinary English pairs
    almost never do by accident.

    Qualifier fields ONLY. A prefix rule is a real loosening, and the branch
    field is where a loosening costs a wrong address.
    """
    if _grounded_in(word, heard):
        return True
    target = _stem(word)
    spoken = re.findall(r"[a-z']+", heard.lower())
    if any(_stem(w) == target for w in spoken):
        return True
    return len(word) >= 6 and any(
        len(w) >= 6 and w[:6] == word[:6] for w in spoken)


def _meaning_class(word: str) -> str:
    """Which meaning class this word belongs to, or "" for ordinary content."""
    w = word.replace("'", "").lower()
    for name, members in _MEANING_CLASSES.items():
        if w in members:
            return name
    return ""


def _class_present(name: str, heard: str) -> bool:
    """Did the caller make this KIND of move, in any words at all?

    Whole-word membership, not the substring test the content words use: "if"
    inside "different" is not a condition, and a class check that fired on it
    would ground every qualifier ever written.
    """
    spoken = {w.replace("'", "") for w in re.findall(r"[a-z']+", heard.lower())}
    return bool(spoken & _MEANING_CLASSES[name])


def _collapse(text: str) -> str:
    """Letters and digits only — word boundaries removed, SEQUENCE preserved.

    call-20260824-2113: the caller said "east side clinic" and the model saved
    "Eastside Clinic". `clinic` is a grounding stopword, so `eastside` was the
    only content word left, and it is not a substring of "east side" — the
    space breaks it. Rejected four times, twice while the caller was repeating
    themselves verbatim, and the call recorded "could not obtain the location"
    about a cooperative person who answered immediately and confirmed it.

    THIS IS NOT FUZZY MATCHING, and the difference is the whole reason it is
    allowed where a similarity threshold was not. There is no score and no
    tolerance: every letter must still appear in the same order. It cannot
    rescue "Riverside" from "resides at" — measured, along with every other
    fabrication on record — because those differ in their letters, not in where
    the spaces fall. The (a)/(b) ambiguity that made fuzzy matching unusable
    needs two readings of the same string, and an exact character sequence
    admits only one.

    The model normalising "east side" to "Eastside" is a reasonable thing to do
    with a place name, and the same collapse covers north side/Northside, mid
    town/Midtown, Saint Mary's/St Mary's — a large fraction of real US branch
    names. Measured across all 36 resolved calls: zero rows change.

    DIGITS ARE NOT ROUTED THROUGH THIS. The digit rule keeps its own exact
    comparison of digit RUNS, so "1855" still cannot ground on "1825"; that
    guard exists because a house number nobody said reached the directory, and
    nothing here loosens it.
    """
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _grounded_in(term: str, heard: str) -> bool:
    """Did the caller say this term, allowing for where the spaces fell?"""
    return term in heard or _collapse(term) in _collapse(heard)


def _rode_along(args: dict, sess: "RealtimeSession") -> list:
    """Content words in the saved value that the caller was never heard to say.

    NOT A BLOCK — `_ungrounded_terms` has already decided whether to accept the
    value, and this changes none of that. It answers the narrower question the
    accept does not: WHICH PARTS were actually corroborated.

    The two are different because grounding deliberately accepts on ONE content
    word. "Mission Bay Clinic" grounds if "bay" was said; the digit rule was
    added when that tolerance put a house number nobody said into the directory,
    and this is the alphabetic half of the same hole. On call-20260824-2014 the
    transcriber rendered "Riverside campus" as "She resides at campus", the
    model retried with "Riverside Campus, 1825 4th Street", and it was accepted
    on the street number — correctly, the caller really had said that — while
    "Riverside" itself was never corroborated by anything. The row went to the
    directory stamped "verified against caller transcript", which was true of
    the address and false of the name.

    Blocking that would be wrong: the value was RIGHT, and refusing it costs a
    real answer, which is this project's expensive direction. Recording it costs
    nothing and makes the row's provenance answerable — a reviewer can ask which
    rows contain a token nobody was heard to say, which is exactly the question
    you want to be able to ask after a transcription failure.
    """
    heard = _asserted_caller_text(sess)
    if not heard.strip():
        return []
    out: list = []
    for field in ("branch", "city"):
        value = (args.get(field) or "").strip()
        if not value:
            continue
        # Same tokenizer as _ungrounded_detail, for the same reason: a
        # whitespace split leaves "Mid-Town" as one non-alpha token and skips it.
        for w in re.findall(r"[a-z']+", value.lower()):
            w = w.strip("'")
            if (w and w not in _UNGROUNDED_STOPWORDS
                    and len(w) > 2 and not _grounded_in(w, heard)
                    and w not in out):
                out.append(w)
    return out


# How long a guard may wait for a caller turn that is still transcribing, and
# which tools are worth waiting for.
#
# THE MODEL HEARS AUDIO; THE GUARDS READ TRANSCRIPTS; THE TRANSCRIPT LAGS. The
# Realtime model does not wait for `input_audio_transcription` before acting —
# it works from the audio directly — so every check that reads `sess.turns` can
# be asked its question before the evidence for it exists. That is not a bug in
# any one guard; it is the two halves of this system running on different
# clocks, and it is the same cause as the record-time race _revisit_grounding
# repairs after the fact.
#
# call-20260825-1620, all three inside the same second:
#
#   16:21:37  ⚠️  'eastside' never appeared in the caller transcript
#   16:21:37  🚫 HALLUCINATED BRANCH BLOCKED: {'branch': 'Eastside Clinic'}
#   16:21:37  👤 CALLER : He's at the Eastside clinic.      <- the evidence
#
# The caller said it. The guard called it a fabrication because it asked half a
# second early, and the cost was not the record — the branch was re-saved
# correctly 30 seconds later — it was the conversation. The agent was told it
# had invented the answer, asked for a street address instead, and the next
# half-minute is barge-in wreckage: "new patients", "Actually,", "referral",
# "campus".
#
# The first answer to that was a blocking wait: hold the guard up to 1.5s for
# the words to arrive. THE MEASUREMENT IT ASKED FOR KILLED IT. That comment
# ended "`transcript_waits` records every wait so the ceiling can be set from
# measurement rather than from this comment", and across 119 call artifacts the
# distribution came back:
#
#   n=14   timeout 12   landed 0   discarded 0
#   waited_s: 1.5 1.5 1.5 1.5 1.5 1.5 1.51 1.52 1.52 1.53 1.53 1.53 1.53 1.53
#
# A spike on the ceiling and not one early return. The wait never once did its
# job, and it was not a self-inflicted deadlock either — the poll loop yielded
# every 50ms, so the reader task was free to deliver a transcript that simply
# never came that fast. It cost 1.5s of latency on every save that hit it; on
# call-20260827-1010 that is the whole of `ours 1.53s` in a 3.44s reply.
#
# So the wait is gone and the answer is entirely the DEFERRAL: a save whose
# guard objects while the words are still in flight is held, and judged on the
# transcript event itself — see _resolve_deferred_save, and the comment at that
# call site, which is the moment the wait was standing in for. The predicate
# below is what both of them asked; now only one caller is left.


# THE CALLER ASKING TO HANG UP. Deliberately narrow.
#
# call-20260826-1656 ran 193s and the last 23 of them were this, verbatim:
#
#     caller  Thank you bye Cut the call.
#     AGENT   Take care.
#     caller  How many times you will tell me bro?
#     AGENT   Right, ending here - bye.
#     caller  Bye-bye.          AGENT  Bye.
#     caller  Bye, I said.      AGENT  Goodbye.
#     caller  Hahaha.           AGENT  Take care.
#
# The model was not misbehaving - it said goodbye every single time. Nothing
# in the application could act on it, because sess.done had exactly two
# triggers and neither was reachable by the caller.
#
# EXPLICIT TOKENS ONLY, and no sentiment. "How many times you will tell me
# bro?" is the clearest statement of intent on that whole call and it is NOT
# matched here: reading frustration is a judgement, and a judgement that
# hangs up on a caller mid-sentence is the expensive direction. A farewell
# or a direct instruction is a fact.
#
# "by the way" does not match: \bbye\b requires the e.
_CALLER_ENDS_CALL = re.compile(
    r"\b(?:"
    r"bye\s*[-,]?\s*bye|bye|goodbye|good\s?bye"
    r"|cut the call|end the call|hang up|disconnect the call"
    r"|stop the call|that\'?s all,?\s*(?:thanks|thank you)"
    r")\b", re.I)


def _caller_ends_call(text: str) -> bool:
    """Did the caller just ask to end the call?

    Used to set sess.done, which is the only thing that reaches the hangup.
    Nothing else about the call is changed by it: fields already collected
    stay collected, and the outcome is computed from memory exactly as it
    would have been. This decides WHEN to stop, never WHAT was learned.
    """
    return bool(_CALLER_ENDS_CALL.search(_norm_quotes(text or "")))

def _transcript_pending(sess: "RealtimeSession") -> bool:
    """Is the newest caller turn's transcript still in flight?

    THE ONE QUESTION "are the words still coming?", asked by every caller that
    needs it. It was shared with a blocking wait until that wait was deleted
    for never once succeeding (see the comment above); the deferral is now the
    only answer, so this predicate is the only place the definition lives.

    Two conditions, both load-bearing:
      - the newest turn is a caller placeholder, and
      - the transcriber has not already answered for it.

    The second is what separates "still transcribing" from "answered, and what
    came back was thrown away" — a hint echo stripped to nothing, a fabricated
    turn on silence. Those leave the placeholder standing and look identical
    from the outside. call-20260825-1712 spent the old wait's ceiling twice on
    exactly this — the transcriber had replied, the reply was junk, and the
    placeholder stayed standing.

    WHY A PREDICATE AND NOT A STRING MATCH. The caller needs to tell "the
    evidence has not arrived" apart from "the evidence arrived and contradicts
    you", and the guards express the difference in English prose. Matching that
    prose would put the fix one reworded message away from silently deferring a
    real contradiction — which is the one thing that must never be deferred.
    """
    if not sess.turns:
        return False
    last = sess.turns[-1]
    return (last.role == "caller" and last.text.strip() == "[...]"
            and sess._transcript_at < sess._placeholder_at)


def _ever_transcribed(sess: "RealtimeSession") -> bool:
    """Has ANY caller turn on this call ever rendered into words?

    The difference between "they have not answered yet" and "this call has no
    transcription". A guard may refuse the first; refusing the second would
    block every save on a line where nothing renders, which is a lost row for a
    reason the caller had nothing to do with.
    """
    return any(t.role == "caller" and t.text.strip() != "[...]"
               for t in sess.turns)


def _grounding_verdict(rode: list, heard_any: bool,
                       contested: Sequence = ()) -> str:
    """The `grounding` sentence for a save. One speller, two callers.

    CONTESTED IS NOT THE SAME AS VERIFIED, and until call-20260827-1010 the
    artifact could not tell them apart. The model tried to save "Riverside
    Campus"; the guard refused it because the transcript read "Private site
    campus"; the escalation guard then pushed the model to save the
    transcript's wording, and THAT save stamped a bare "verified against
    caller transcript". The row reached doctors.json as status="verified" with
    no surviving sign that a value had been rejected on the way — the console
    said 🚫 and 🛑 and nothing durable did.
    """
    if not heard_any:
        return ("SKIPPED — no caller speech was transcribed on this call, so "
                "the saved location could not be checked against anything the "
                "caller actually said")
    base = "verified against caller transcript" + (
        f" EXCEPT {', '.join(repr(w) for w in rode)}, which the caller was "
        f"never transcribed saying" if rode else "")
    if not contested:
        return base
    _tried = ", ".join(repr(str(r.get("value", ""))) for r in contested)
    return (f"CONTESTED — {len(contested)} earlier branch value(s) were "
            f"rejected as ungrounded on this call ({_tried}), so the saved "
            f"wording may be the transcriber's rather than the caller's; "
            f"{base}")


def _revisit_grounding(sess: "RealtimeSession") -> None:
    """Re-decide the grounding verdict against the FINISHED transcript.

    THE VERDICT WAS WRITTEN ONCE, AT SAVE TIME, AND NEVER LOOKED AT AGAIN — and
    the transcript it reads is not finished at save time. Transcription lands
    after the audio, so a caller turn can still be the `[...]` placeholder when
    the tool call it caused is already running, and `_asserted_caller_text`
    skips placeholders. The evidence is not absent; it has not arrived.

    call-20260825-1425. The caller said "Same at Riverside campus 7th street"
    at 14:26:06. save_branch ran at ~14:26:07 with that turn still a
    placeholder, so `riverside` was recorded as a word nobody was heard to say
    — in the same artifact that carries the sentence, timestamped one second
    before the save. Not a logic error and not fixable by loosening the
    comparison: it is a race, and the answer to a race is to ask again once the
    racing is over.

    RE-DECIDED, NOT RELAXED. This recomputes the same verdict from the same
    function against a complete transcript, so it can move in both directions —
    a term that arrived is dropped from the exception list, and a term that
    never arrived stays in it. Only the reading changes, never the rule.

    Both readings are kept. `grounding_at_save` preserves what was believed
    while the call was live, because a verdict that silently improves after the
    fact is a verdict you cannot audit: the question "did the guard fire during
    the call" has to stay answerable.
    """
    before = sess.memory.get("grounding")
    if not before:
        return              # nothing was ever saved; there is no verdict
    args = {"branch": sess.memory.get("branch") or "",
            "city": sess.memory.get("city") or ""}
    if not (args["branch"] or args["city"]):
        return
    heard_any = any(t.role == "caller" and t.text.strip() != "[...]"
                    for t in sess.turns)
    rode = _rode_along(args, sess)
    after = _grounding_verdict(rode, heard_any,
                               getattr(sess, "branch_rejections", ()))
    if after == before:
        return
    sess.memory.update(grounding=after, grounding_at_save=before,
                       rode_along=rode or None)
    print(f"[Realtime] ⏪ GROUNDING RE-READ on the finished transcript — the "
          f"turn that corroborated it landed after the save", flush=True)
    print(f"[Realtime]    at save: {before}", flush=True)
    print(f"[Realtime]    final  : {after}", flush=True)


def _asserted_caller_text(sess: "RealtimeSession") -> str:
    """Everything the caller ASSERTED, lowercased, as one blob.

    Extracted so the grounding check and the rode-along report read the same
    evidence. Two copies of "what did the caller actually tell us" is two
    answers to that question the first time one of them is edited.
    """
    return " ".join(
        t.text.lower() for t in sess.turns
        if t.role == "caller" and t.text.strip() != "[...]"
        and _turn_asserts(t.text, sess))


def _ungrounded_terms(args: dict, sess: "RealtimeSession") -> str:
    """Return a description of any branch/city term the caller never said.

    Empty string means everything checks out. Compares against the caller's
    own transcribed words only — the agent's words are excluded, or the model
    could ground a fabrication in its own earlier hallucination.

    If no caller speech was transcribed at all (every turn still a `[...]`
    placeholder) the check is skipped rather than blocking every save, since
    absence of transcript is not evidence of fabrication.
    """
    # A caller turn is not the caller — it is a model's guess at the caller, and
    # the transcription hint is a prompt to that model. On call-20260813-1409
    # "Yes, speaking" (the second phrase in the hint's old "Likely phrases"
    # list) came back four times. The hint still names health systems and
    # location words, because those are what make REAL answers transcribe
    # correctly — which means the vocabulary that constitutes a valid branch
    # answer is exactly the vocabulary that can be echoed. If that echo lands in
    # a caller turn it becomes grounding evidence, and a fabricated location
    # gets written to the directory looking verified. That is worse than any
    # wasted turn: the check built to stop fabrication would be certifying it.
    #
    # Two independent signals have to fail before a turn is discounted, because
    # neither is sufficient alone:
    #
    #   1. The turn is EXACTLY the term and nothing else. A hint echo arrives
    #      bare; a real answer usually comes with surrounding words ("she's at
    #      the Mercy campus", "that'd be north campus I think"). But
    #      "Northgate" on its own is a perfectly good answer, so this cannot
    #      stand alone.
    #   2. The audio carried no real signal. Loudest-300ms window, never the
    #      mean — the mean is dominated by the gaps between words and once told
    #      an audible caller they were faint.
    #
    # A bare one-word answer on strong audio still grounds. A bare one-word
    # answer on near-silence does not.
    _usable = []
    for t in sess.turns:
        if t.role != "caller" or t.text.strip() == "[...]":
            continue
        _usable.append(t)
    # ASSERTIVE TURNS ONLY. A value the caller ASKED us about is not
    # evidence that they told us it — see _turn_asserts. _usable is left
    # whole because the hint-echo check below asks a different question
    # ("was this term only ever a bare echo on dead air"), which a
    # question-shaped turn answers just as well as a statement.
    heard = _asserted_caller_text(sess)
    if not heard.strip():
        # Nothing transcribed, or nothing ASSERTED — cannot judge either
        # way, so do not block. Same conservative direction as before.
        return ""

    missing = []
    for field in ("branch", "city"):
        value = (args.get(field) or "").strip()
        if not value:
            continue
        terms = [w.strip(".,!?-—'\"") for w in value.lower().split()]

        # ── DIGITS MUST MATCH EXACTLY ───────────────────────────────────────
        # The word rule below is deliberately lenient: one content word
        # matching is enough, because transcription is imperfect and a real
        # answer is worth more than a blocked one. That tolerance is right for
        # words and exactly wrong for numbers.
        #
        # call-20260819-1716: the caller said "1825 4th Street". The agent
        # saved "Mission Bay Clinic, 1855 Fourth Street" and grounding PASSED
        # it — because "bay" appeared, and one word was enough. A four-digit
        # house number nobody said went into the client directory.
        #
        # That is the worst failure this whole system exists to prevent. Not an
        # empty row and not an obviously wrong one, but a PLAUSIBLE one: no
        # reviewer spots it, and someone sent to 1855 Fourth Street finds the
        # wrong building. A misheard street name is recoverable; a misheard
        # street number is a wrong address that looks right.
        #
        # So numbers get no tolerance at all. Every digit run in the value must
        # appear verbatim in what the caller actually said.
        # A NUMBER SAID AS A WORD IS STILL THAT NUMBER. This is normalisation,
        # not tolerance: the rule's zero-tolerance is about numbers the caller
        # never GAVE, not about which notation they arrived in.
        #
        # call-20260825-1226: the caller said "Riverside Campus Seventh Street"
        # twice, the model wrote "7th", and the digit rule reported "number 7
        # not in what the caller said" — three refusals, and the branch that
        # finally saved was a bare "Riverside" with the campus and the street
        # both lost. The map was already there and already knew seventh -> 7;
        # only the caller's side of the comparison was not consulting it. The
        # reverse direction — value spelled out, caller said digits — has been
        # handled since the spelled-number bypass was closed, so this is the
        # missing half of a check that was always meant to be symmetric.
        _said_nums = set(re.findall(r"\d+", heard))
        _said_nums |= {str(_NUMBER_WORD_VALUE[w])
                       for w in re.findall(r"[a-z]+", heard)
                       if w in _NUMBER_WORD_VALUE}
        _value_nums = set(re.findall(r"\d+", value))
        _invented = sorted(_value_nums - _said_nums)
        if _invented:
            missing.append(
                f"{field}={value!r} (number{'s' if len(_invented) > 1 else ''} "
                f"{', '.join(_invented)} not in what the caller said)")
            continue

        # ── AND THE SAME TOLERANCE FOR NUMBERS SPELLED OUT ──────────────────
        # The rule above only inspects digit runs, so a value carrying NO
        # digits skips it entirely — _value_nums is empty, nothing is compared,
        # and the check passes vacuously. Spelling the number in words is
        # therefore a complete bypass of the strictest guard in this file.
        #
        # call-20260820-1321 walked straight into it, and the guard drove it
        # there. The caller said "It's Mission Bay Clinic, 1844th Street."
        #   1st try  'Mission Bay Clinic, 18 4th Street'    -> REJECTED, rightly
        #   2nd try  'Mission Bay Clinic, 18 4th Street'    -> REJECTED, rightly
        #   3rd try  'mission bay clinic, eighteen forty fourth street' -> SAVED
        # and it reached doctors.json as "partially_verified" with grounding
        # "verified against caller transcript". Nothing verified it: there were
        # no digits to check. Two rejections reading "NEED: wording the caller
        # used out loud" taught the model that digits were the problem, so it
        # wrote them as words and the guard waved it through — a guard that
        # trains the model to evade it is worse than no guard, because the
        # result carries a verification stamp.
        #
        # Same zero tolerance, then, but applied per word so that RENDERING
        # still passes. A number-word is grounded if the caller said that word
        # ("Seven Hills Clinic"), or said the digit it stands for
        # ("4th Street" -> "Fourth Street"). Neither, and it was substituted.
        _heard_words = set(re.findall(r"[a-z]+", heard))
        _value_numwords = {p for t in terms for p in t.split("-")
                           if p in _NUMBER_WORD_VALUE}
        _spelled = sorted(
            w for w in _value_numwords
            if w not in _heard_words
            and str(_NUMBER_WORD_VALUE[w]) not in _said_nums)
        if _spelled:
            missing.append(
                f"{field}={value!r} (numbers as words: "
                f"{', '.join(_spelled)} | caller did not say them)")
            continue

        content = [w for w in terms if w and w not in _UNGROUNDED_STOPWORDS]
        if not content:
            continue
        # One content word appearing is enough — transcription is imperfect and
        # we would rather let a real answer through than block it. See the digit
        # rule above for where this tolerance had to stop.
        if not any(_grounded_in(w, heard) for w in content):
            missing.append(f"{field}={value!r}")
            continue
        # It appears. Check it did not appear ONLY as a bare echo on dead air.
        _support = [t for t in _usable
                    if any(_grounded_in(w, t.text.lower()) for w in content)]
        _level = _caller_speech_level(sess)
        if _support and all(_is_hint_echo(t, content, _level) for t in _support):
            missing.append(
                f"{field}={value!r} (only heard as a bare term on silent audio)")
    return " and ".join(missing)


def _ungrounded_choice(args: dict, sess: "RealtimeSession", *,
                       arg: str, probe, classifier, states,
                       label: str, since_at_least: int = 0,
                       floor_reason: str = "") -> str:
    """Grounding for a closed-set field. Empty string means it checks out.

    PARAMETRISED OVER THE VOCABULARY, not copied per field. `probe` is the
    pattern that recognises the ask this answer belongs to — it is what anchors
    the search, and it is the same Field.probe the objective and the ask budget
    already use, so a template cannot end up with three different opinions
    about what counts as asking the question.

    WHY THIS IS NOT `classify_choice(heard)` OVER THE CALLER BLOB, which is the
    obvious reading of "do for CHOICE what _ungrounded_terms does for PLACE".
    That would be strictly weaker than the location check, not equivalent, for
    three reasons that all come from the same place: a location is a
    high-entropy proper noun and a status is two bits.

    1. THE BLOB IS THE WRONG SCOPE. "Northgate" appearing anywhere in a call is
       evidence somebody said Northgate. "Yes" appearing anywhere is evidence of
       nothing — callers say it constantly for other reasons. "Yes, speaking."
       at pickup would ground a YES for a call where the accepting question was
       never answered at all. So the evidence must come from the turns AFTER we
       asked, not from the call.
    2. THE BARE-TERM CONJUNCT COLLAPSES. _is_hint_echo requires two signals to
       fail together: the turn is nothing but the term, AND the audio carried no
       signal. For a location the first is a real discriminator, because a
       genuine answer usually arrives with surrounding words. A genuine status
       answer IS "Yes." — bare is the normal shape — so that conjunct is
       satisfied by every true answer and the audio measurement is doing all of
       the work alone. Which means it needs the rms check MORE than the location
       check does, not less: it is the only signal left.
    3. THE TRANSCRIBER FABRICATES EXACTLY THIS. 0.7s of near-silence produced a
       whole receptionist greeting on call-20260820-1732, and the hint that did
       it has since been cut — but the old hint's own "Likely phrases: yes,
       speaking" came back four times on call-20260813-1409. A phantom "Yes." on
       dead air is squarely inside what this transcriber does, and unlike a
       phantom place name there is no distinctiveness left to catch it.

    So: anchored to the ask, asserted rather than asked back, classified to the
    state being claimed, and rejected when its only support is a bare token on
    silent audio. When there was no ask to anchor to, the turn must additionally
    be ABOUT new patients — see the check below, which is where reason 1 would
    otherwise creep back in.
    """
    status = str(args.get(arg) or "").strip().lower()
    if status not in states:
        return ""      # tools.py rejects this on its own terms; not our call.

    # ANCHORED. Only turns after the most recent ASK about THIS field count.
    #
    # `_is_ask_for`, not a bare probe match, and the difference is a false
    # accept. The probe recognises the TOPIC; an agent turn can be about the
    # topic while asking nothing — a read-back, an acknowledgement, a closing
    # line. On call-20260824-2014 the agent's own "I heard you say she's taking
    # the new patients" matched the probe, advanced the anchor past every
    # caller turn that had answered, and left the window empty; the guard then
    # took its own "no evidence since the ask" branch and ACCEPTED a status it
    # had just refused three times. The model cannot be allowed to move the
    # goalposts by talking, which is the same principle as _ungrounded_terms
    # excluding the agent's words from `heard`.
    since = 0
    asked = False
    for i, t in enumerate(sess.turns):
        if t.role == "agent" and _is_ask_for(t.text or "", probe):
            since = i + 1
            asked = True

    # A FLOOR UNDER THE ANCHOR, for a caller that supersedes its own earlier
    # answer. Only identity passes one, and only after the name was spelled
    # out — see _name_mismatch.
    #
    # `heard` is provenance: it is the sentence a reviewer reads as the reason
    # this row says what it says. On call-20260825-1620 identity saved with
    #
    #   heard: "Yes, Dr. Rayaz is our oncologist."
    #
    # which is the turn the guard REFUSED, three turns before the agent spelled
    # R-E-Y-E-S and the caller said "Yes, the same doctor." The confirmation
    # rests entirely on that later turn; the stored quote names a doctor this
    # call decided was the wrong one. Selection picks the fullest matching turn
    # and the mangled one is simply longer, so length chose it.
    #
    # Same reason the mismatch scan moves: turns before the letters are about a
    # name the line mangled, turns after them are about ours. Evidence and
    # provenance have to agree on where the question was settled.
    floored = since_at_least > since
    if floored:
        since = since_at_least
    usable = [t for t in sess.turns[since:]
              if t.role == "caller" and t.text.strip() != "[...]"]
    if not usable:
        if floored:
            # NOT the same as nothing being transcribed. We put a specific
            # question to them — "is this the name you have?" — and they have
            # not answered it. Standing down here would accept a confirmation
            # whose only support is the turn the spelling was performed to
            # supersede, which is the defect this floor exists to close.
            return (f"{label}={status!r} — {floor_reason} | NEED: their answer "
                    f"to THAT question, in their own words")
        # NOTHING TRANSCRIBED SINCE WE ASKED.
        #
        # THIS STOOD DOWN, AND ON call-20260825-1731 THAT CONFIRMED THE DOCTOR
        # ON NO EVIDENCE AT ALL. The agent asked "Is this Dr. Reyes, Oncology,
        # at Lakeview Medical?", the wait for the transcript timed out, this
        # branch stood down, and identity saved CONFIRMED with the model's own
        # unchecked string — `heard: "Okay."` — as its provenance. Nothing the
        # caller said was ever consulted; `classify_identity("Okay.")` is None,
        # so that quote could not have grounded anything had selection run at
        # all. The `detail` guard on the SAME tool call reported "'reyes',
        # 'oncology' never appeared in the caller transcript" and the identity
        # was accepted anyway.
        #
        # It cost more than one bad row. Three seconds later the caller said
        # "Yes, Dr. Rayef is our oncologist." — the real answer, with the
        # surname mangled — and because identity was already CONFIRMED,
        # _wrong_doctor_named never ran on it and the spell-and-confirm repair
        # never fired. `name_mismatches` is null on a call that contained one.
        # One permissive branch disabled the whole chain beneath it.
        #
        # THE FUNCTION WAS ALREADY INCONSISTENT WITH ITSELF. Twenty lines below,
        # "they spoke and none of it was an answer" REFUSES, and the reason
        # given is that a status has no second gate under it — tools.py accepts
        # any of the four by definition, so this is the only thing between a
        # model's guess and the directory. That argument applies with more
        # force, not less, when they said nothing whatsoever. The weaker
        # evidence was being refused and the weakest accepted.
        #
        # The lag this stood down for is now handled where it belongs, by the
        # deferral: a save the guard objects to while the words are in flight
        # is held and re-judged on the transcript event. So "nothing since the
        # ask" here means nothing came, and the cost of refusing is one more
        # turn: the model saves again when the transcript lands, which is
        # exactly the turn on which 1731 would have caught "Rayef".
        if _ever_transcribed(sess):
            return (f"{label}={status!r} — nothing has been transcribed since "
                    f"you asked | NEED: their answer to THAT question, in "
                    f"their own words")
        # ...UNLESS NOTHING HAS EVER RENDERED ON THIS CALL. Then transcription
        # is not working — a line too poor for it, or a model without it — and
        # refusing here would block every save for the rest of the call for a
        # reason the caller had no part in. Stand down, as before, but do not
        # pretend the quote was checked: `heard` is model-authored on this path
        # and selection never runs to replace it.
        sess.unverified_quotes.append(
            {"field": label, "value": status,
             "heard": str(args.get("heard") or "")[:120]})
        return ""

    # ASSERTED, not asked back. "Is she accepting new patients?" repeated by a
    # receptionist checking what we want is not them telling us. Same predicate
    # the location check uses, for the same reason.
    #
    # WITH THIS FIELD'S VOCABULARY, which the location caller cannot pass and
    # this one must. The predicate's PLACE rescue is a proper noun beside a
    # location anchor, and a two-bit answer has none — so without this, a bare
    # "Yes?" was discarded on the transcriber's punctuation while
    # `classifier("Yes?")` said `confirmed`. See _turn_asserts.
    asserted = [t for t in usable
                if _turn_asserts(t.text, sess,
                                 classifier=classifier, state=status)]
    if not asserted:
        # THEY SPOKE AND NONE OF IT WAS AN ANSWER. This is NOT the same as the
        # silence above and must not share its verdict.
        #
        # The location guard can afford to stand down here, because a saved
        # branch still has to survive the blob check — its words must appear in
        # what the caller said, so there is a second gate underneath. A status
        # has no second gate: it is two bits, tools.py accepts any of the four
        # by definition, and this function is the only thing standing between a
        # model's guess and the directory. Standing down when the caller has
        # demonstrably only asked questions back would mean any status could be
        # saved at that moment, which is precisely the fabrication case.
        said = "; ".join(t.text.strip()[:60] for t in usable[-2:])
        return (f"{label}={status!r} — since you asked, they have only asked "
                f"back, not answered | THEY SAID: {said!r}")

    matching = []
    for t in asserted:
        heard_state = classifier(t.text)
        if heard_state is None or heard_state.value != status:
            continue
        # NEVER ASKED -> THE TURN MUST BE ABOUT NEW PATIENTS.
        #
        # Without an ask there is no anchor, so `since` is 0 and every caller
        # turn from pickup onwards is in scope — which reopens reason 1 above
        # in the one case it bites hardest. "Yes, speaking." is the single most
        # common opening utterance in this corpus (it is the phrase the retired
        # transcription hint echoed four times on call-20260813-1409), it
        # classifies as YES on a bare affirmative, and with no ask to anchor
        # against it would ground a new-patient status nobody was ever asked
        # for. The permissive branch was contradicting this function's own
        # docstring.
        #
        # Volunteering the answer is still honoured, which is what the anchor
        # was relaxed for in the first place: a receptionist who says "we're
        # not taking new patients right now" while you are still on the branch
        # question has told you, and that turn is ABOUT the thing. A bare "yes"
        # is not. The cost of being wrong here is one turn — the agent asks,
        # they repeat, it grounds normally — against a wrong directory row that
        # nobody can spot afterwards.
        # NEVER ASKED -> THE TURN MUST STAND ON ITS OWN.
        #
        # This tested whether the turn contained the ASK's vocabulary, and that
        # was the wrong question. On call-20260825-0915 the caller said "we are
        # full right now, but I can put you on the list. You would be number
        # 21." — a textbook waitlist answer — while the agent was still asking
        # about the BRANCH, so nothing had matched ACCEPTING_ASK and the
        # never-asked path applied. That sentence contains none of "accepting",
        # "taking new" or "new patients", so the vocabulary test threw it out.
        # Refused twice, and the queue position the client most wanted never
        # reached the record.
        #
        # What the rule was actually defending against is a BARE AFFIRMATIVE
        # with no anchor: "Yes, speaking." at pickup classifies YES on its
        # opening token alone and asserts nothing about new patients. So test
        # for that directly — strip the leading yes and see whether what remains
        # still says the same thing. A turn that states the condition in its own
        # words survives; one that was only ever a "yes" does not.
        if not asked and not states_in_its_own_right(
                t.text, status, classifier):
            continue
        matching.append(t)
    if not matching:
        said = "; ".join(t.text.strip()[:60] for t in asserted[-3:])
        return (f"{label}={status!r} — nothing the caller said since you asked "
                f"reads as that answer | THEY SAID: {said!r}")

    # The only support is a bare token on audio that carried nothing. For this
    # field that is the whole test, not a tiebreak — see 2. above.
    level = _caller_speech_level(sess)
    tokens = [w for t in matching
              for w in re.findall(r"[a-z']+", t.text.lower())]
    if all(_is_hint_echo(t, tokens, level) for t in matching):
        return (f"{label}={status!r} — only heard as a bare word on silent "
                f"audio, which is what a transcription artefact looks like")

    # ── SELECTION, NOT VALIDATION ───────────────────────────────────────────
    # `heard` is supposed to be what the caller said. It arrives model-authored
    # and, until now, entirely unchecked — while all three tool schemas told the
    # model it was "checked against the call transcript". On
    # call-20260824-2116 the model inserted clauses nobody uttered:
    #
    #   caller : "Yeah, definitely, you can reach out to them."
    #   heard  : "Yeah, definitely, they're taking new patients also. You can
    #             reach out to them."
    #   caller : "Yeah, you need to book through online or call. Please do that."
    #   heard  : "...call FROM THE FRONT DESK. Please do that."
    #
    # A fabricated quote is worse than a wrong status, because it reads as
    # verbatim to whoever audits the row and there is nothing in the record to
    # say it is not.
    #
    # So the model's string is not checked — it is DISCARDED. This function has
    # already identified the caller turn that corroborated the status; that
    # turn's real text is what gets stored, and the model's version becomes
    # irrelevant rather than merely suspect. Checking would leave the failure
    # mode in place with a detector in front of it; selection removes it.
    #
    # WHICH matching turn, when there are several.
    #
    # Last-wins was the first rule and it is wrong. On call-20260825-0915 the
    # caller circled back and the VAD split their final answer, so the last
    # turn classifying as WAITLIST was the fragment "The status waitlist is" —
    # a mid-sentence scrap that went into the record as the quotation
    # justifying the state, while "we are full right now, but I can put you on
    # the list. You would be number 21." sat further up.
    #
    # FIRST-WINS IS NOT THE ANSWER EITHER, and the reason is worth stating
    # because it is the tempting flip: a fragment can just as easily arrive
    # first, and on a call where the caller corrects themselves the earliest
    # match would be the superseded answer.
    #
    # LONGEST WINS, and the justification is that there is no correctness
    # dimension left to trade. `matching` is already filtered to turns whose
    # classification EQUALS the state being saved — every candidate asserts the
    # same thing — so recency buys nothing, and the only remaining question is
    # which of several agreeing sentences is the fullest statement of it. A
    # truncated fragment is short by construction; a complete answer carries its
    # qualifiers with it, which is also how "number 21" survives into the record
    # rather than being summarised away.
    #
    # Ties go to the later turn: same length, same claim, prefer the one they
    # most recently stood behind.
    args["heard"] = max(
        enumerate(matching),
        key=lambda pair: (len(pair[1].text.strip()), pair[0]),
    )[1].text.strip()
    return ""


# A doctor named in speech: "Dr. Kapoor", "Doctor Smith", "Dr Okafor's".
_NAMED_DOCTOR = re.compile(r"\b(?:dr\.?|doctor)\s+([a-z][a-z'-]{2,})", re.I)


# A possessive on the end of a captured surname: "Okafor's", "Jones'".
# A SUFFIX, matched as one — `.rstrip("'s")` looks like it removes this and does
# not: rstrip takes a SET OF CHARACTERS, so it eats every trailing apostrophe
# and every trailing s. "Reyes" came back "reye".
#
# Live, on call-20260825-1625: the caller said "Dr. Reyes is an oncologist" —
# transcribed perfectly, the right doctor, the right practice — and the guard
# reported "they named 'reye', and the doctor on this call is 'reyes'". It
# refused a correct confirmation and spent a turn spelling a name nobody had
# got wrong. Every surname ending in s is affected: Reyes, Jones, Hayes,
# Brooks, Sanders, Rivers. The 1226 fixtures are Okafor, Kapoor and Smith, so
# none of them could show it.
_POSSESSIVE = re.compile(r"'s?$")


def _surnames_named(text: str) -> list:
    """Every surname the caller attached to a doctor title, lowercased."""
    return [_POSSESSIVE.sub("", m.group(1).lower())
            for m in _NAMED_DOCTOR.finditer(_norm_quotes(text or ""))]


def _our_surname(sess: "RealtimeSession") -> str:
    """The surname on record for this call, lowercased. "" if we have none."""
    from agents.voice.templates import clean_doctor_name
    full = clean_doctor_name(getattr(sess.doctor, "doctor_name", "") or "")
    return (full.split()[-1] if full.split() else full).lower()


def _spell_out(surname: str) -> str:
    """"reyes" -> "R-E-Y-E-S". What the agent is told to say."""
    return "-".join(c.upper() for c in surname if c.isalpha())


def _spelled_out(text: str, surname: str) -> bool:
    """Did this agent turn spell our surname letter by letter?

    THE REPAIR HAS TO BE DETECTABLE, not merely requested. The rejection asks
    the model to spell the name; whether it did is what decides that the
    caller's next answer is evidence about OUR doctor rather than about
    whatever the transcriber produced. Asking and assuming is how a repair
    becomes a way of accepting anything.

    Requires a real separator between every letter, so the plain name does not
    count as having spelled it: "Reyes" and "R-E-Y-E-S" are different sounds on
    the line and only the second one survives a transcriber that mangles names.
    """
    letters = [c for c in (surname or "") if c.isalpha()]
    if len(letters) < 3:
        return False
    pat = r"[\s.\-]+".join(re.escape(c) for c in letters)
    return re.search(pat, text or "", re.I) is not None


def _wrong_doctor_named(text: str, sess: "RealtimeSession") -> str:
    """A surname in this turn that is NOT the doctor on record. "" if fine.

    THE CHECK THE IDENTITY FIELD EXISTED FOR AND DID NOT HAVE. On
    call-20260825-1226 the record said Dr. Okafor, the caller said "that's
    right, Dr. Kapoor is one of our cardiologists", and identity saved
    CONFIRMED — because the guard classified the affirmative and never looked
    at the name. Okafor and Kapoor are not the same person. The field exists to
    answer "are we talking about the right doctor", and it answered yes about a
    different one.

    STRICT, AND DELIBERATELY THE OPPOSITE OF THE BRANCH RULE. Branch grounding
    is lenient because refusing a real answer costs a lost row; here leniency
    costs a row CONFIRMED against the wrong person, attached to a real
    practice, which is the worst output this system has. So a named surname
    must match the one in CALL CONTEXT — collapsed for spacing and hyphens,
    which preserves the letter sequence, and nothing fuzzier. If the
    transcriber mangled our doctor's name we refuse a correct confirmation:
    that is the right way round, because "the ASR mangled it" and "they named
    someone else" produce the same string and only one of them is safe to
    assume.

    Silence about the name is not a mismatch. A caller who says "yes, speaking"
    names nobody, and this returns "" — the classification stands on its own.
    """
    named = _surnames_named(text)
    if not named:
        return ""
    ours = _our_surname(sess)
    if not ours:
        return ""
    if any(_collapse(n) == _collapse(ours) for n in named):
        return ""
    return ", ".join(sorted(set(named)))


def _name_mismatch(sess: "RealtimeSession", other: str, said: str) -> str:
    """Record a wrong surname, and return the rejection that repairs it.

    THE FAILURE HAS TO BE VISIBLE FIRST. Before this, the only trace was
    `wrong_doctor_named` in CallMemory, which nothing read and no artifact
    carried — so calls 1433 and 1437 ended UNCONFIRMED with nothing anywhere
    saying the name was the reason. "Never asked" and "asked, and the name came
    back wrong three times" are different results and looked identical.

    THEN THE REPAIR, AND IT CANNOT BE STRING MATCHING. Every mismatch on those
    calls came from the transcriber, not from the receptionist: "Dr. Riaz",
    "Dr. Yes", "Dr. Ayers" are all the line mangling "Reyes". Fuzzy matching
    was measured against them and cannot work — soundex and metaphone catch
    Riaz->Reyes but not Ayers->Reyes, edit distance catches none of the three,
    and every threshold loose enough to catch Ayers also matches Okafor to
    Kapoor, which is the confirmed-against-the-wrong-doctor bug this guard was
    written for. There is no rescue in comparing the strings harder.

    So stop comparing and ASK, in the one form a bad line cannot corrupt: spell
    the name a letter at a time and have them confirm those letters. The
    caller's next answer is then evidence about OUR doctor rather than about
    whatever the transcriber produced, which is why `_name_spelled_at` moves
    the scan forward rather than merely being recorded — see the loop below.

    Once, though. If they name someone else after being given the letters, that
    is the receptionist and not the line, and asking a third time spends the
    call on a question already answered.
    """
    ours = _our_surname(sess)
    entry = {"heard": other, "ours": ours, "said": said.strip()[:120],
             "after_spelling": bool(sess._name_spelled_at)}
    if entry not in sess.name_mismatches:
        sess.name_mismatches.append(entry)
    sess.memory.update(wrong_doctor_named=other)
    head = (f"identity='confirmed' — they named {other!r}, and the doctor on "
            f"this call is {ours!r} | THEY SAID: {said.strip()[:70]!r}")
    if not sess._name_spelled_at:
        return (f"{head} | SPELL IT: say the name one letter at a time — "
                f"\"{_spell_out(ours)}\" — and ask whether that is the name "
                f"they have. Do not save identity until they answer THAT.")
    return (f"{head} | You already spelled it out and they named someone else "
            f"again, so this is the practice answering and not the line. Stop "
            f"asking about the name. If they said our doctor is not at this "
            f"practice save identity='not_here'; otherwise escalate with their "
            f"words. Do not ask about the branch or new patients.")


# A caller turn is "quiet" relative to how loudly THIS caller has been
# speaking, not against a fixed number. _LOW_AUDIO_RMS alone is an absolute
# threshold on a quantity that has no absolute meaning: line gain, handset,
# carrier and distance all move it, so one constant cannot be right for two
# different calls.
#
# Measured on call-20260818-1338, where the transcriber emitted "Mercy Medical
# Center" — a phrase assembled from _US_TRANSCRIBE_HINT, which names Mercy
# first among health systems and "medical center" among location words. The
# caller never said it:
#
#     real  "why are you collecting"    0.0954
#     real  "Los Angeles, California"   0.1532
#     real  "It is Los Angeles only."   0.0465
#     FAKE  "Mercy Medical Center."     0.0174     <- cleared _LOW_AUDIO_RMS (0.015)
#
# The hallucination sat just above the constant while being a quarter of this
# caller's own median level. Every fraction from 0.25 to 0.50 separates the
# four cleanly; 0.35 is the middle of that band. Checked against
# call-20260818-1112, where all four caller turns are believed genuine: none
# is flagged.
#
# RE-DERIVED 2026-08-18 against the Twilio recordings, after the accusation
# this was built on turned out to be false and after audio_rms itself was found
# to be under-reporting. Method: for each of 30 calls with a dual-channel
# recording, take the N loudest caller-channel bursts where N is the number of
# transcribed caller turns, and compute min/median over them — i.e. how quiet a
# GENUINE turn gets relative to that caller's own typical level.
#
#     lowest 0.291   p10 0.458   p25 0.662   median 0.766
#     calls with a genuine turn below median*0.35 :  2/30
#     calls with a genuine turn below median*0.20 :  0/30
#
# 0.35 was too aggressive: on ~7% of calls it would classify a real caller turn
# as quiet, and a bare one-word branch name is exactly the shape that then gets
# rejected — "'Northgate' on its own is a perfectly good answer".
#
# BE CLEAR ABOUT WHAT THIS NOW BUYS. The case it was written for (the "Mercy
# Medical Center" turn) was retracted — that audio is real. With no confirmed
# positive case and a safe calibration, the adaptive term only acts on turns
# between the absolute floor and median*0.20, which is a narrow band. It is
# kept because the reasoning still holds — an absolute constant on a
# level-dependent quantity cannot be right for two different lines — not
# because it is known to catch anything. Do not widen it without a confirmed
# fabrication to widen it against.
_QUIET_FRACTION = 0.20


# Below this many measured turns the median is not a median. One turn's
# "median" is itself, which can never be a fraction of itself, so the adaptive
# test would silently never fire — the failure mode this file keeps relearning.
_MIN_TURNS_FOR_ADAPTIVE = 3


def _caller_speech_level(sess: "RealtimeSession") -> Optional[float]:
    """This caller's typical loudest-300ms level, or None if not yet knowable.

    Median, not mean: it survives one hallucinated near-silent turn among
    several real ones, which is precisely the population being judged.
    """
    vals: list[float] = []
    for t in sess.turns:
        if t.role != "caller" or t.text.strip() == "[...]":
            continue
        r = getattr(t, "audio_rms", None)
        if r is not None:
            vals.append(float(r))
    if len(vals) < _MIN_TURNS_FOR_ADAPTIVE:
        return None
    return float(median(vals))


def _is_hint_echo(turn, content_words: list, speech_level: Optional[float] = None) -> bool:
    """True if this caller turn looks like the transcriber echoing its own hint.

    Both signals must fail: the turn is nothing but the term, AND the audio it
    came from carried no real signal. See _ungrounded_terms for why neither is
    sufficient on its own. An unmeasured turn (audio_rms None) is given the
    benefit of the doubt — absence of measurement is not evidence of
    fabrication, the same rule the transcript check already follows.
    """
    rms = getattr(turn, "audio_rms", None)
    # Absolute floor OR a fraction of this caller's own level, whichever is
    # higher. The absolute alone let a hallucination through at 0.0174; the
    # relative alone would collapse toward zero on a call where every turn is
    # quiet, since a fraction of nothing is nothing.
    quiet_below = _LOW_AUDIO_RMS
    if speech_level is not None:
        quiet_below = max(quiet_below, speech_level * _QUIET_FRACTION)
    if rms is None or rms >= quiet_below:
        return False
    bare = [w.strip(".,!?-—'\"") for w in turn.text.lower().split()]
    bare = [w for w in bare if w and w not in _UNGROUNDED_STOPWORDS]
    return bool(bare) and set(bare) <= set(content_words)


# THE INTERFACE, DECLARED. Underscore-prefixed names read as private, and a
# checker cannot see that realtime_worker re-exports every one of them — so
# each definition here was reported as never accessed. Ten standing false
# hints are worse than none: they are how a REAL "not accessed" gets scrolled
# past. The leading underscores stay because the suite addresses these as
# `rw._thing` and renaming them is a change to callers, not a tidy-up.
__all__ = [
    "_ACK_TAKES_VALUE",
    "_CONFIRMS_VALUE",
    "_DETAIL_FUNCTION_WORDS",
    "_INVITATION",
    "_LOCATION_ANCHORS",
    "_LOW_AUDIO_RMS",
    "_MAX_OWED_PER_CALL",
    "_MAX_OWED_PER_TEXT",
    "_MEANING_CLASSES",
    "_MIN_TURNS_FOR_ADAPTIVE",
    "_NAMED_DOCTOR",
    "_NON_PLACE",
    "_NOT_AN_ASK",
    "_NUMBER_WORD_VALUE",
    "_ORG_STOPWORDS",
    "_POSSESSIVE",
    "_QUIET_FRACTION",
    "_REPORTS_FAILURE",
    "_UNGROUNDED_STOPWORDS",
    "_VETTING_OPENER",
    "_asserted_caller_text",
    "_transcript_pending",
    "_caller_ends_call",
    "_caller_is_vetting",
    "_caller_speech_level",
    "_class_present",
    "_collapse",
    "_distinctive",
    "_drop_lost_substance",
    "_ever_transcribed",
    "_grounded_in",
    "_grounded_loosely",
    "_grounding_verdict",
    "_invites_continuation",
    "_is_ask_for",
    "_is_hint_echo",
    "_is_location_ask",
    "_meaning_class",
    "_name_mismatch",
    "_our_surname",
    "_owed_key",
    "_owed_refusal",
    "_revisit_grounding",
    "_rode_along",
    "_spell_out",
    "_spelled_out",
    "_stem",
    "_surnames_named",
    "_turn_asserts",
    "_ungrounded_choice",
    "_ungrounded_detail",
    "_ungrounded_terms",
    "_wrong_doctor_named",
]
