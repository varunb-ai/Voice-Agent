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



def _edit_distance(a: str, b: str) -> int:
    """Plain Levenshtein. Surnames are short; nothing here needs to be clever."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# A near-miss is ONE character, on a name long enough for that to be small, with
# the first letter intact.
_NEAR_MISS_MIN_LEN = 4
_NEAR_MISS_MAX_EDITS = 1


def _near_miss(heard: str, ours: str) -> bool:
    """Is this our surname with one character mangled by the line?

    THE DOCSTRING BELOW SAYS FUZZY MATCHING CANNOT WORK, AND IT IS RIGHT ABOUT
    THE CLAIM IT MAKES. Measured against 1437: soundex catches Riaz->Reyes but
    not Ayers->Reyes, edit distance catches neither, and every threshold loose
    enough for Ayers also matches Kapoor->Okafor — a row confirmed against the
    wrong doctor, which is the worst output this system has. Nothing here
    disturbs that finding: this rescues NONE of Riaz, Ayers or Yes.

    What it rescues is the much narrower class the September corpus is actually
    made of, which nobody had counted. Of the 17 name refusals across 51 calls:

        brown   vs browne   d=1   x6      the silent 'e', dropped by the line
        abul    vs abel     d=1   x3
        okofor  vs okafor   d=1   x2
        april   vs abel     d=3   x2      <- still refused, still gets the letters
        rokofor vs okafor   d=2   x1      <- still refused
        okopher vs okafor   d=4   x1      <- still refused

    Eleven of eighteen are a SINGLE character. And the threshold that clears
    them is provably below every documented false positive: Kapoor/Okafor is 3,
    Yes/Reyes is 2, Ayers/Reyes is 3, Riaz/Reyes is 4. One is the only edit
    budget that separates the two sets, so one is what this takes — the rest
    keep the spell-and-confirm repair, which is the guard working.

    TWO EXTRA CONSTRAINTS, both cheap and both load-bearing at the margin:
    length >= 4 so a single character is a small proportion of the name (it
    keeps "Tim" from matching "Kim"), and the same first letter, because a
    genuinely different surname usually differs at the front.

    THE RESIDUAL RISK, STATED RATHER THAN HIDDEN: two real doctors at one
    practice whose surnames differ by one interior character — Hall and Hill,
    Shaw and Shah — are accepted as the same person. That is a real hole and it
    is narrower than the one it replaces, where a receptionist saying our own
    doctor's name got a spelling interrogation 17 times in 51 calls. Every
    rescue is written to `name_mismatches` with `near_miss` set, so the row
    says what was assumed and a reviewer can find it.
    """
    if not heard or not ours:
        return False
    if len(ours) < _NEAR_MISS_MIN_LEN:
        return False
    if heard[:1] != ours[:1]:
        return False
    return _edit_distance(heard, ours) <= _NEAR_MISS_MAX_EDITS


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
    which preserves the letter sequence — or be one character off it under
    `_near_miss`, and nothing looser than that. If the transcriber mangled our
    doctor's name by more than a character we still refuse a correct
    confirmation: that is the right way round, because "the ASR mangled it" and
    "they named someone else" produce the same string and past one edit only
    one of them is safe to assume.

    AMENDED 2026-09-04, and the paragraph above is the reason the amendment is
    exactly one character wide rather than a similarity score. Measured over 51
    September calls this returned a name on 17 saves, 11 of which were our own
    doctor with a single letter changed — six of them "Brown" for "Browne". The
    cost was not the row: it was the spell-and-confirm repair firing on a
    receptionist who had just said the right name, and the model re-asking a
    question already answered. See `_near_miss` for the threshold, the four
    documented false positives it stays below, and the residual risk it takes.

    Silence about the name is not a mismatch. A caller who says "yes, speaking"
    names nobody, and this returns "" — the classification stands on its own.
    """
    named = _surnames_named(text)
    if not named:
        return ""
    ours = _our_surname(sess)
    if not ours:
        return ""
    _ours_c = _collapse(ours)
    if any(_collapse(n) == _ours_c for n in named):
        return ""
    # ONE CHARACTER OFF IS THE LINE, NOT A DIFFERENT DOCTOR — see _near_miss for
    # the measurement and for the false positives this deliberately does not
    # reach. Recorded rather than waved through: a rescue is an ASSUMPTION about
    # what was said, and an assumption nothing writes down is indistinguishable
    # from a match.
    _near = next((n for n in named if _near_miss(_collapse(n), _ours_c)), "")
    if _near:
        entry = {"heard": _near, "ours": ours, "said": (text or "").strip()[:120],
                 "after_spelling": bool(sess._name_spelled_at),
                 "near_miss": True}
        if entry not in sess.name_mismatches:
            sess.name_mismatches.append(entry)
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

    THE LETTERS ARE THE MECHANISM; THE MANNER IS SEPARATE, and this message
    has to carry both or it buys the repair at the cost of the persona. Asking
    for the letters alone produced "Just to check, Dr. Abel, A-B-E-L — is that
    the doctor there?" on call-20260903-2121: correct, and audibly a procedure
    being executed. A prospective patient who did not catch a name does not
    verify it, they say they did not catch it — so the instruction now names
    the STANCE (you misheard) before the content (the letters), and marks the
    letters as the only part that is not the model's to reword. Telling them
    they said it wrong is banned here for the same reason the shared body bans
    it everywhere: they know what they said, and we are the more likely to be
    wrong about it.

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
        return (f"{head} | ASK AS SOMEONE WHO MISHEARD, NOT AS A CHECK: you "
                f"did not catch the name, so say so and offer the letters — "
                f"\"Sorry, just to make sure I heard you right — Dr. "
                f"{ours.title()}? {_spell_out(ours)}?\" Word it your own way, "
                f"but SAY THE LETTERS: they are the part of this a bad line "
                f"cannot mangle. Never tell them they said it wrong. Do not "
                f"save identity until they answer THAT.")
    return (f"{head} | You already spelled it out and they named someone else "
            f"again, so this is the practice answering and not the line. Stop "
            f"asking about the name. If they said our doctor is not at this "
            f"practice save identity='not_here'; otherwise escalate with their "
            f"words. Do not ask about the branch or new patients.")


def _note_name_heard(sess: "RealtimeSession", said: str) -> str:
    """Record a surname that is not ours, and ask NOTHING of the call.

    THE GUARD THAT ONLY RUNS WHILE IT CAN ACT. `_wrong_doctor_named` is
    consulted from inside `save_doctor_identity`'s grounding, so once identity
    is CONFIRMED nothing looks at the doctor's name again for the rest of the
    call. On call-20260903-1126 the record said "Dr. Pediatric" — a specialty
    typed into the name field — and the receptionist said "Dr. Abel" at 11:27,
    79 seconds after identity was confirmed at 11:26. `name_mismatches` is null
    on a call whose transcript contains the correct surname, twice.

    PASSIVE ON PURPOSE, and this is the whole design. The active guard exists
    to stop a save against the wrong doctor and pays for that with a re-ask;
    re-opening a settled identity to argue about the name would spend the call
    on a question already answered, which is the failure `_name_spelled_at`
    was added to prevent. This one writes a row and returns. What it catches is
    OUR bad input, and the place to fix that is the record, not the call.

    Returns the surname recorded, or "" when there is nothing to record.
    """
    other = _wrong_doctor_named(said, sess)
    if not other:
        return ""
    entry = {"heard": other, "ours": _our_surname(sess),
             "said": said.strip()[:120],
             "after_spelling": bool(sess._name_spelled_at),
             "passive": True}
    # DEDUPED AGAINST THE ACTIVE ROW, not just against other passive ones. The
    # same sentence can reach both paths — the guard on the save, this on the
    # transcript — and two rows for one utterance is the double-count that
    # made the check_refusals corpus unreadable.
    _same = {k: v for k, v in entry.items() if k != "passive"}
    for _e in sess.name_mismatches:
        if {k: v for k, v in _e.items() if k != "passive"} == _same:
            return other
    sess.name_mismatches.append(entry)
    return other


__all__ = [
    "_name_mismatch",
    "_near_miss",
    "_note_name_heard",
    "_our_surname",
    "_spell_out",
    "_spelled_out",
    "_surnames_named",
    "_wrong_doctor_named",
]
