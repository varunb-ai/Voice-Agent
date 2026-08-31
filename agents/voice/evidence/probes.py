"""Predicates over TEXT. No session, no turns, no call.

Everything here answers a question about one string: is this an ask, is this
word distinctive, does this term appear in what was heard. They are the
matchers that `window` applies across a transcript and that `guards` reasons
with.

THE BOUNDARY IS THE ARGUMENT LIST. If a function here ever needs `sess` it
belongs in `window` or `guards` instead. That is what keeps this file
testable one string at a time, and it is the line that stops the pattern
layer beneath growing a dependency on the state of a call.

`_is_hint_echo` takes a TURN rather than a bare string, which is the one
exception. It stays because it still judges a single utterance in isolation:
nothing about the rest of the call reaches it except a speech level its
caller computes and passes in.
"""
import re
from typing import Optional

from agents.voice.objectives import (
    LOCATION_NOUN as _LOCATION_NOUN,
    norm_quotes as _norm_quotes,
)
from agents.voice.evidence.patterns import (
    _ACK_TAKES_VALUE,
    _ANNOUNCES_ASK,
    _CALLER_ENDS_CALL,
    _CONFIRMS_VALUE,
    _INVITATION,
    _LOW_AUDIO_RMS,
    _MEANING_CLASSES,
    _NOT_AN_ASK,
    _ORG_STOPWORDS,
    _QUIET_FRACTION,
    _REPORTS_FAILURE,
    _UNGROUNDED_STOPWORDS,
)

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



def _invites_continuation(text: str) -> bool:
    """The caller asking what the agent wants — an open door, not a refusal.

    Blocking escalation on this is the same move as blocking it on a hold
    request. A caller who says "How can I help you?" has told you they are
    willing; ending the call there throws away the one turn most likely to
    produce an answer.
    """
    return bool(_INVITATION.search(_norm_quotes(text or "")))



def _distinctive(name: str) -> set:
    """The tokens in an organisation name that actually identify it."""
    return {w for w in re.findall(r"[a-z]+", (name or "").lower())
            if w not in _ORG_STOPWORDS and len(w) > 2}



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



def _owed_key(text: str) -> str:
    """Identity of an owed sentence, for counting attempts at it."""
    return _collapse(text)[:120]



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



def _caller_ends_call(text: str) -> bool:
    """Did the caller just ask to end the call?

    Used to set sess.done, which is the only thing that reaches the hangup.
    Nothing else about the call is changed by it: fields already collected
    stay collected, and the outcome is computed from memory exactly as it
    would have been. This decides WHEN to stop, never WHAT was learned.
    """
    return bool(_CALLER_ENDS_CALL.search(_norm_quotes(text or "")))



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


__all__ = [
    "_caller_ends_call",
    "_class_present",
    "_collapse",
    "_distinctive",
    "_drop_lost_substance",
    "_grounded_in",
    "_grounded_loosely",
    "_invites_continuation",
    "_is_ask_for",
    "_is_hint_echo",
    "_is_location_ask",
    "_meaning_class",
    "_owed_key",
    "_stem",
]
