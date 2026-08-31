"""Whose name was said, and what to do when it is not ours.

The spell-and-confirm repair, kept together because it is one behaviour:
recognise that the caller named a DIFFERENT doctor, refuse the confirmation,
and offer our doctor's surname one letter at a time so that the next answer
is about a name the line has not mangled.

Split out because it is the one part of this package about a proper noun
rather than about evidence in general, and because `_POSSESSIVE` has its own
history worth keeping visible: `.rstrip("'s")` looks like it strips a suffix
and does not. It takes a SET of characters, so "Reyes" came back "reye" and
the guard accused our own doctor on a perfectly good confirmation.
"""
import re
from typing import TYPE_CHECKING

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
    _NAMED_DOCTOR,
    _POSSESSIVE,
)
from agents.voice.evidence.probes import (
    _collapse,
)

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


__all__ = [
    "_name_mismatch",
    "_our_surname",
    "_spell_out",
    "_spelled_out",
    "_surnames_named",
    "_wrong_doctor_named",
]
