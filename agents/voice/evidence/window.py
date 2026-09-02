"""WHERE in the transcript the evidence is allowed to come from.

The anchoring layer. These walk `sess.turns` and answer questions about
position and provenance rather than about meaning: which turns are in scope
for a field, whether a turn asserts something or asks back, whether anything
has been transcribed at all, how loud the caller has been.

THE ANCHOR HAS TWO ENDS, and it took a lost identity to learn the second.
`since` was a floor with nothing over it, so on call-20260831-1048 an answer
about the BRANCH grounded an IDENTITY eight turns after the doctor had been
confirmed. `_other_field_probes` is what closes the window: an ask for any
other field ends this one's. Both ends are applied in `_ungrounded_choice`,
which lives in `guards`.
"""
import re
from statistics import median
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:                       # pragma: no cover - typing only
    # A binding for the string annotations below, and nothing more.
    # TYPE_CHECKING is False when Python runs, so no import happens and the
    # one-way rule this package was extracted under still holds:
    # realtime_worker imports us, never the reverse.
    from agents.voice.realtime_worker import RealtimeSession

from agents.voice.objectives import (
    norm_quotes as _norm_quotes,
)
from agents.voice.evidence.patterns import (
    _LOCATION_ANCHORS,
    _MIN_TURNS_FOR_ADAPTIVE,
    _NON_PLACE,
    _ONLY_AFFIRM,
    _ORG_STOPWORDS,
    _UNGROUNDED_STOPWORDS,
    _VETTING_OPENER,
)
from agents.voice.evidence.probes import (
    _distinctive,
    _invites_continuation,
)

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
    # BY WORD, like the two lines above it. `known.add(name.lower())` put the
    # whole string in as one key, and the loop below tests SINGLE words — fine
    # while every persona was one first name ("Sarah"), silently useless the
    # moment a template introduced itself with two. patient_discovery speaks as
    # a synthetic person, so "emile keswick" went in whole and "Keswick" was
    # matched by nothing: our own invented surname, one word from a location
    # anchor, read as the caller naming a place.
    known |= _distinctive(sess.agent_name or "")
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



def _other_field_probes(sess: "RealtimeSession", probe) -> tuple:
    """Every OTHER field's ask-probe on this call's objective.

    Read off the objective rather than listed here, for the same reason `probe`
    is passed into _ungrounded_choice rather than chosen inside it: a template
    that adds a field would otherwise get an evidence ceiling that cannot see
    the new field's ask, and that one field would silently go back to having no
    upper bound at all.

    IDENTITY, not equality. The probes are module-level singletons in
    objectives.py, so `is not` says exactly what is meant — "every ask that is
    not the one anchoring this window" — and two fields sharing one pattern
    would be a template bug rather than something for this to guess around.

    DEFENSIVE, because this module is routinely handed a namespace carrying
    only `turns` — see `double()` in the suite, and _objective_of's docstring
    for the same argument on the other side of the line. A guard that raises on
    a test double is a guard that stops being tested, and no objective means no
    ceiling, which is precisely the behaviour that shipped before this existed.
    """
    try:
        obj = getattr(sess, "objective", None)
        if obj is None or not getattr(obj, "fields", None):
            from agents.voice.objectives import default_objective
            obj = default_objective()
        return tuple(f.probe for f in obj.fields
                     if f.probe is not None and f.probe is not probe)
    except Exception:                       # pragma: no cover - double safety
        return ()



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


__all__ = [
    "_asserted_caller_text",
    "_caller_is_vetting",
    "_caller_speech_level",
    "_ever_transcribed",
    "_other_field_probes",
    "_transcript_pending",
    "_turn_asserts",
]
