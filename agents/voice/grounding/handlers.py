"""What each tool is allowed to write, and the reason when it is not.

One function per tool, plus the sess-reading helpers they share. Every
guard here answers the same question - is this value supported by what the
caller actually said - and returns the result the layers above report on.

TWO OF THE THREE ARE PURE APART FROM `sess`: they read the transcript and
write the session's own records, and touch neither the socket nor the
call's lifecycle. _guard_escalate is the exception and says so in its own
docstring; a blocked escalation answers the tool call itself, so it takes
the socket and returns a stop flag.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:                    # pragma: no cover - typing only
    # A binding for the string annotations below. TYPE_CHECKING is False
    # at run time, so no import happens and the one-way rule holds: the
    # worker imports this package, never the reverse.
    from agents.voice.realtime_worker import RealtimeSession

from agents.voice.evidence import (
    _LOCATION_ANCHORS,
    _NON_PLACE,
    _ORG_STOPWORDS,
    _UNGROUNDED_STOPWORDS,
    _transcript_pending,
    _caller_speech_level,
    _distinctive,
    _grounding_verdict,
    _invites_continuation,
    _is_hint_echo,
    _meaning_class,
    _rode_along,
    _ungrounded_detail,
    _ungrounded_terms,
)
from agents.voice.tools import run_tool
from agents.voice.grounding.vocabulary import (
    _CALL_SHAPE_EXITS,
    _CHOICE_SAVE_TOOLS,
    _FACTUAL_ESCALATIONS,
    _ORG_WORD,
    _SELF_ID,
    _SELF_ID_WEAK,
    _STREET_ADDRESS,
    _is_bare_hint_word,
    is_hold_request,
)

def _ungrounded_escalation(reason: str, sess: "RealtimeSession") -> str:
    """Reject an escalation reason asserting something the caller never said.

    A live call ended with escalate(reason="doctor deceased") after the caller
    said only "actually, he's not working right now". Nobody said died, passed
    away, or deceased. save_branch was guarded against exactly this and
    escalate was not — so a fabricated claim about a named real person went
    into the record, where a reviewer would read it as fact.

    Only claims ABOUT THE DOCTOR are checked. Reasons describing the call
    itself ("declined to share", "wrong number", "no response") are the agent's
    own observation and need no corroboration.
    """
    heard = " ".join(t.text.lower() for t in sess.turns
                     if t.role == "caller" and t.text.strip() != "[...]")
    if not heard.strip():
        return ""
    low = reason.lower()
    for claim, markers in _FACTUAL_ESCALATIONS.items():
        if claim in low and not any(m in heard for m in markers):
            return (f"reason {reason!r} states the doctor is {claim}, which "
                    f"nobody said on this call")
    return ""


def _candidate_location(sess: "RealtimeSession") -> str:
    """A place the CALLER named that was never saved. Empty if there is none.

    The mirror image of _ungrounded_terms: that one asks "did the caller say
    this?", this one asks "did the caller say ANYTHING, when we are about to
    record that they said nothing?".

    Deliberately conservative — it gates an escalation, and a detector that
    fires on ordinary conversation would trap the agent on a call it cannot
    end. A word counts only if it is distinctive (not a stopword, not a filler,
    not already on our own record), it sits within two words of a location
    anchor, and it survives the same hint-echo test a saved branch has to pass.
    """
    usable = [t for t in sess.turns
              if t.role == "caller" and t.text.strip() != "[...]"]
    if not usable:
        return ""

    # Words we already had before the call started cannot be an answer FROM the
    # call. The hospital on record is the whole reason the fabrication happened
    # — the model reshaped it into a branch name — so hearing it echoed back is
    # not the caller naming a site.
    known: set[str] = set()
    known |= _distinctive(getattr(sess.doctor, "hospital_name", "") or "")
    known |= _distinctive(sess.org_name or "")
    known |= {w for w in re.findall(r"[a-z]+", (sess.doctor.doctor_name or "").lower())
              if len(w) > 2}
    # BY WORD, like the two lines above it — see the same change in
    # evidence/window.py. `known.add(name.lower())` keyed the whole string
    # while the loop below tests SINGLE words, so a two-token spoken name
    # ("Emile Keswick") protected neither token. Inert for the one-word
    # personas: _distinctive("Sarah") is {"sarah"}.
    known |= _distinctive(sess.agent_name or "")

    for t in usable:
        raw = [w.strip(".,!?-—'\"") for w in t.text.split()]
        words = [w.lower() for w in raw]
        # A place name is a PROPER NOUN, and the transcriber capitalises it.
        # That is the strongest signal available and lowercasing throws it
        # away: without it "the office is closed" and "hospital, how can I
        # help" both read as candidates, because "closed" and "how" are simply
        # words no stoplist thought to name. Enumerating English is not a
        # strategy. Capitalisation cuts the space in one move.
        #
        # It is a CONJUNCTION with the stoplists, never a replacement —
        # sentence-initial words are capitalised regardless of what they are,
        # which is what the stoplists are still for.
        #
        # If a turn came back with no case information at all (all lower, all
        # upper), capitalisation says nothing about any word in it, so fall
        # back to the stoplists alone rather than silently detecting nothing.
        # Same rule the grounding check follows: absence of a signal is not
        # evidence, and a degraded transcript must not quietly disable a guard.
        cased = t.text != t.text.lower() and t.text != t.text.upper()
        for i, w in enumerate(words):
            if len(w) <= 2 or not w.isalpha():
                continue
            if (w in _UNGROUNDED_STOPWORDS or w in _NON_PLACE
                    or w in _ORG_STOPWORDS or w in known):
                continue
            if cased and not raw[i][:1].isupper():
                continue
            near = words[max(0, i - 2):i] + words[i + 1:i + 3]
            if not any(n in _LOCATION_ANCHORS for n in near):
                continue
            # Same defence a real save has to clear: a bare term on dead air is
            # the transcriber echoing its own hint, not the caller speaking.
            if _is_hint_echo(t, [w], _caller_speech_level(sess)):
                continue
            return f"{raw[i]!r} — they said: {t.text.strip()!r}"
    return ""


def _discarded_location(reason: str, sess: "RealtimeSession") -> str:
    """Block an escalation claiming nothing was given when something was.

    Returns a rejection description, or "" to allow the escalation.

    THE TRANSCRIPT DECIDES, NOT THE MODEL'S WORDING. This used to run
    _candidate_location only when the reason matched _NO_LOCATION_CLAIMS — a
    phrase whitelist checked against text the model composes freely. On
    call-20260821-1152 the caller said "She works at Mission Bay clinic in San
    Francisco, but I'm not sure which location that is", the model escalated
    with "caller could not provide...", and the list holds "did not provide"
    but not "could not provide". One word, guard silent, and a branch that
    grounds cleanly — it saved on the previous call — was thrown away.
    Enumerating the model's phrasings cannot work; _is_location_ask was
    inverted for the same reason and says so in its own docstring.
    """
    if any(m in reason.lower() for m in _CALL_SHAPE_EXITS):
        return ""
    # Reaching the WRONG ORGANISATION is a legitimate exit even when a place
    # was named, and the place named is usually the wrong organisation itself.
    # Detected structurally rather than by another phrase list.
    if hospital_mismatch(sess):
        return ""
    return _candidate_location(sess)


def _address_offered(sess: "RealtimeSession") -> Optional[str]:
    """A street address the caller gave, or None. The latest one wins."""
    found = None
    for t in sess.turns:
        if t.role != "caller" or t.text.strip() == "[...]":
            continue
        for m in _STREET_ADDRESS.finditer(t.text):
            found = f"{m.group(1)} {' '.join(m.group(2).split())}"
    return found


def _address_dropped(args: dict, sess: "RealtimeSession") -> Optional[str]:
    """The caller gave a street address and this save leaves it out."""
    addr = _address_offered(sess)
    if not addr:
        return None
    saved = " ".join(str(args.get(f) or "") for f in ("branch", "city")).lower()
    number = addr.split()[0]
    # Keyed on the NUMBER, not the words: "Fourth Street" can legitimately be
    # absent from a value that already names the site, but the house number is
    # either recorded or it is lost.
    return None if number in saved else addr


def hospital_mismatch(sess: "RealtimeSession") -> str:
    """The caller answered as a DIFFERENT organisation than the one on record.

    A branch saved against the wrong hospital is corrupt data, and it is the one
    failure the grounding guard cannot see: every word can be genuinely quoted
    from the caller and the record still ends up wrong, because the call reached
    the wrong place.

    On a live call the record said "Northside Medical Group" and the caller
    answered "Thank you for calling the Methodist Medical Center." Nothing
    noticed, and the agent went on to invent an address for it.

    Fires only on a POSITIVE mismatch — a recognisable different name in an
    answering phrase. Silence is the norm, not a signal: most people answer
    without naming the place, and treating that as suspicion would block almost
    every call. Empty string means no conflict found.
    """
    recorded = getattr(getattr(sess, "doctor", None), "hospital_name", "") or ""
    on_record = _distinctive(recorded)
    if not on_record:
        return ""
    # LATER EVIDENCE CAN CORRECT EARLIER EVIDENCE. This used to return on the
    # first differing claim, so a mismatch raised at pickup could never be
    # resolved however the rest of the call went.
    #
    # call-20260820-1440: the caller answered "Hi, this is North Medical Group",
    # the record said "Northside Medical Group", and the save was blocked with
    # "NEED: which place this call actually reached". The agent asked. The
    # caller answered "This is Northside Medical Group." Nothing consumed it —
    # the agent escalated one second later with a reason the caller had just
    # contradicted, and a genuine branch (Mission Bay Clinic, 1825 Fourth
    # Street, grounding clean) was thrown away.
    #
    # Note what this does NOT do: it never decides whether "North" and
    # "Northside" are the same name. That question is unanswerable from a
    # transcript and normalising them would be inventing data. It answers the
    # question the rejection actually asked, and only that one.
    #
    # NOT "the last utterance wins" either. The clear requires a positive
    # SELF-IDENTIFICATION as the recorded organisation — "this is X", "you've
    # reached X" — which _SELF_ID/_SELF_ID_WEAK already distinguish from merely
    # naming it. "We're not Northside Medical Group" and "Dr. Okafor isn't at
    # Northside any more" both contain the name and neither qualifies.
    #
    # And it must come AFTER the differing claim. A confirmation at pickup
    # followed by a different organisation later is a transfer, not a
    # correction, and that mismatch must stand.
    _mismatch = ""
    for turn in sess.turns:
        if turn.role != "caller" or not turn.text:
            continue
        claims = list(_SELF_ID.findall(turn.text))
        claims += [c for c in _SELF_ID_WEAK.findall(turn.text) if _ORG_WORD.search(c)]
        if _mismatch:
            # Only a positive self-ID as the place on record clears it.
            if any(_distinctive(c) & on_record for c in claims):
                return ""
            continue
        # If the recorded name appears anywhere in this turn, they are the right
        # place however else they phrase it. "Northside, this is Amy."
        if on_record & _distinctive(turn.text):
            continue
        for claimed in claims:
            said = _distinctive(claimed)
            # Overlap of even one distinctive token means the same place under a
            # slightly different name — "Northside Medical Center" vs "Group".
            if said and not (said & on_record):
                _mismatch = (f"caller answered as {claimed.strip()!r}, but this "
                             f"call is recorded against {recorded!r}")
                break
    return _mismatch


def _strip_ungrounded_detail(args: dict, sess: "RealtimeSession",
                             key: str) -> tuple:
    """Drop the words nobody said, KEEP the rest. Returns (dropped, reason).

    Discarding the whole string was the first rule and it traded badly. On
    call-20260825-0922 the caller said "you will be the number 21", the model
    wrote "you would be number 21", and one mismatched verb tense — will against
    would — emptied the field and took "number 21" with it. On a waitlist call
    the queue position is the most valuable thing in the record; losing it to a
    tense is not a defensible price for tidiness.

    So the ungrounded words go and the remainder stays. Three things bound that:

      * A NEGATOR IS NEVER STRIPPED. See _MEANING_WORDS. Removing one rewrites
        the claim instead of trimming it, so an ungrounded negator drops the
        whole qualifier — the one case where discarding is still right.
      * DIGITS ARE NEVER STRIPPED, because they are never checked: the token
        pattern is alphabetic, so "21" is not a candidate and cannot be dropped.
        That is what carries a queue position through.
      * A REMAINDER WITH NOTHING IN IT IS NOT KEPT. If stripping leaves no digit
        and no content word, the field is emptied — but RECORDED as emptied,
        never silently, because a quietly blank field reads exactly like a
        caller who volunteered nothing.
    """
    value = str(args.get(key) or "").strip()
    if not value:
        return (), ""
    dropped = _ungrounded_detail(args, sess, key)
    if not dropped:
        return (), ""

    # A meaning word only reaches `dropped` when its ENTIRE CLASS was absent
    # from the transcript — see _ungrounded_detail. So this is no longer "the
    # model used a word they did not", it is "the model made a move they never
    # made": invented a negation, or invented a condition. That still rewrites
    # the claim rather than trimming it, and still drops the whole qualifier.
    risky = [w for w in dropped if _meaning_class(w)]
    if risky:
        args[key] = ""
        return tuple(dropped), (
            "dropped whole - " + ", ".join(repr(w) for w in risky)
            + " changes what it claims, and trimming a negator rewrites the "
              "sentence")

    # Keep each whitespace word unless its alphabetic core was ungrounded, so
    # punctuation and digits ride along with the words that survive.
    kept = []
    for word in value.split():
        core = "".join(re.findall(r"[a-z']+", word.lower())).strip("'")
        if core and core in dropped:
            continue
        kept.append(word)
    remainder = " ".join(kept).strip()
    # Trim the danglers a deletion leaves behind. Removing "desk" from "call
    # the front desk" leaves "call the", which is not wrong so much as visibly
    # broken, and a reviewer who sees that stops trusting the field. Only
    # trailing function words go, and only from the end — nothing in the middle
    # is touched, so this cannot change what the remainder says.
    while True:
        _stripped = remainder.rstrip(" .,;:-")
        _tail = _stripped.rsplit(" ", 1)[-1].lower() if " " in _stripped else ""
        if _tail in {"the", "a", "an", "or", "and", "of", "to", "for", "with",
                     "from", "at", "in", "on", "by", "your", "their"}:
            remainder = _stripped[: _stripped.rfind(" ")].rstrip(" .,;:-")
            continue
        remainder = _stripped
        break

    informative = bool(re.search(r"\d", remainder)) or any(
        w for w in re.findall(r"[a-z']+", remainder.lower())
        if len(w) > 2 and w not in _UNGROUNDED_STOPWORDS)
    if not informative:
        args[key] = ""
        return tuple(dropped), "dropped whole - nothing informative survived"

    args[key] = remainder
    return tuple(dropped), "trimmed to " + repr(remainder)


def _guard_save_branch(name: str, args: dict,
                       sess: "RealtimeSession") -> dict:
    """
    Run save_branch behind the location guards, and say what came of it.

    PURE APART FROM `sess`. It reads the transcript and writes the session's
    own records — branch_rejections, the deferred save, the nudge flags — and
    it touches neither the socket nor the call's lifecycle. That is what makes
    it liftable: the dispatcher keeps every await, so the order in which
    things reach OpenAI is unchanged by the move.

    Returns the tool result the dispatcher goes on to report, defer or refuse.
    """
    # Grounding check. On a live call the model called save_branch
    # with {'branch': 'Riverside Clinic', 'city': 'Atlanta'} when
    # the caller had said only "Hello" and "Okay, next slide,
    # please". "Riverside Campus" was an EXAMPLE in the prompt; the
    # model reshaped it into a fabricated result and hung up.
    # Nothing downstream could tell that record from a real one.
    #
    # So a location may only be saved if the caller actually said
    # it. Verified against the transcript, not the model's claim.
    # The check switches itself off when nothing was
    # transcribed — correct, since absence of transcript is not
    # evidence of fabrication and blocking would kill genuine
    # saves on a bad line. But that is exactly the condition
    # that produces fabrications: bad line -> no transcript ->
    # guard off -> a location the model may have inferred gets
    # written as fact. And with the out-of-band whisper
    # fallback removed there is no second path to a transcript.
    #
    # So record it. A save that could not be verified must not
    # be indistinguishable downstream from one that was.
    # THE CALLER ANSWERED. Whatever happens to the value below, the model
    # only reaches here because it believed it heard a place — so the ask
    # budget, which exists to stop the agent pestering someone who will not
    # engage, has no business counting this call against them.
    #
    # It did, and it cost call-20260821-1931. The caller said "Mission Bay
    # Clinic, 1825 4th Street"; the live transcript mangled it to "Ford
    # Street"; grounding rejected the model's correct reading of it; the
    # agent asked again, that re-ask hit the 4-ask limit, and the give-up
    # directive fired. The caller then repeated the address cleanly — and
    # _ungrounded_terms passes on that transcript, verified — but the agent
    # had already been told to stop, so it said goodbye instead of
    # retrying. The recovery path existed and the budget closed it.
    #
    # Safe because the two budgets measure different things and only this
    # one is being reset: a model that keeps offering bad values is still
    # bounded by _MAX_SAVE_REJECTIONS, which counts up while this counts
    # down. Charging a rejected save to both is double jeopardy, and the
    # person paying it is the caller who answered.
    heard_any = any(t.role == "caller" and t.text.strip() != "[...]"
                    for t in sess.turns)
    # QUALIFIED, not just asserted. Grounding accepts on one content word,
    # so "verified against caller transcript" was being stamped on values
    # whose distinctive part nobody was heard to say — see _rode_along.
    _rode = _rode_along(args, sess)
    sess.memory.update(grounding=_grounding_verdict(
        _rode, heard_any, getattr(sess, "branch_rejections", ())))
    if _rode:
        sess.memory.update(rode_along=_rode)
        print(f"[Realtime] ⚠️  grounded on other words — "
              f"{', '.join(repr(w) for w in _rode)} never appeared in the "
              f"caller transcript", flush=True)
    ungrounded = _ungrounded_terms(args, sess)
    # FIRST, because it is the only gate that can see this one. A bare
    # hint word passes grounding (it is in the transcript), passes the
    # address check, and passes the organisation check — all three were
    # measured false on call-20260821-1705 while "Suite" sat in args.
    _val = str(args.get("branch") or "")
    _bears_an_answer = True
    if _is_bare_hint_word(_val, getattr(sess, "transcribe_hint", "") or ""):
        sess.memory.update(untrusted_location=_val)
        # A generic hint word on a turn that carried no speech. Nothing was
        # said, so nothing was answered.
        _bears_an_answer = False
        result = {
            "ok": False,
            "error": (
                f"NOT SAVED — {_val!r} is one generic location word, not "
                f"the name of a place | LIKELY a transcription artifact "
                f"on a turn that carried no speech "
                f"| NEED: the site name in full, as they said it"
            ),
        }
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
              f"🎣 HINT ECHO BLOCKED: {_val!r} came from our own "
              f"transcription prompt", flush=True)
    elif ungrounded and _transcript_pending(sess):
        # ── THE WORDS ARE STILL IN FLIGHT ───────────────────────────
        # The same hold the choice fields have had since 2026-08-26,
        # which this path was missed out of. call-20260827-0942 is what
        # that omission costs, twice on one call:
        #
        #   waited 1.50s for the transcript and it never came
        #   BLOCKED {"branch": "Riverside campus"}
        #   CALLER : He works out at Riverside Campus.   <- one line later
        #
        #   waited 1.50s ... never came
        #   BLOCKED {"branch": "Riverside campus, 1477 10th Street"}
        #           (numbers 10, 1477 not in what the caller said)
        #   CALLER : I think it's 1477 10th Street.      <- one line later
        #
        # The caller heard the question a third time and said "Oh, no,
        # that's the specific address. I already told that." Then
        # escalate was refused - correctly - because the discard guard
        # could see they HAD given a location. Every guard was right on
        # its own terms and the call still ended with branch = null.
        #
        # Nothing is written here. _resolve_deferred_save re-runs
        # _ungrounded_terms against the real words the moment they land.
        sess._deferred_save = {
            "name": name, "args": dict(args),
            "why": ungrounded, "at": time.monotonic(),
            "asked_turns": len(sess.turns),
        }
        result = {"ok": True, "pending": True, "note": (
            "Held — their words are still being transcribed and this will "
            "be checked against them. Do not ask again and do not apologise; "
            "carry on.")}
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
              f"⏸️  BRANCH HELD FOR EVIDENCE: {args}", flush=True)
        print(f"          the transcript for that turn is still in flight "
              f"— judging it when it lands, not now", flush=True)
    elif ungrounded:
        # Terse fragment, not an English imperative. The old
        # wording ("Never save a location you were not told.
        # Ask them for it...") was fluent prose, so relaying it
        # produced a grammatical sentence — and on
        # call-20260818-1112 the agent said, out loud, "Sorry,
        # I can't use that unless you've actually said the
        # place name" to a caller who HAD just said one.
        #
        # RE-READ comes first because that is the actual fix
        # nine times in ten: the caller said "office Abadan
        # branch" and the model tried to save "Northside
        # Branch", reshaped from the hospital name on its own
        # record. The answer was already on the call. Telling
        # it to ask is what sent that call to escalation with
        # the location sitting in the transcript.
        # SAY WHICH PART IS WRONG. _ungrounded_terms has always computed a
        # specific reason — which field, which value, which number — and
        # this site discarded it and sent a generic line instead. Same
        # shape as 5aed263, where the failure reason was in every event and
        # was thrown away: the diagnosis existed and never reached anyone.
        #
        # It cost a real call. On call-20260820-1321 two rejections in a
        # row said only "NEED: wording the caller used out loud", so the
        # model could not tell that its NUMBER was the problem — and "out
        # loud" reads as "as spoken", which is an active nudge toward
        # spelling digits into words. It did exactly that on the third try
        # and bypassed the digit guard entirely.
        #
        # The reason text is built terse and non-speakable for the same
        # reason the rest of these are: it is machinery, and on
        # call-20260818-1112 the agent read one of these out to a caller.
        result = {
            "ok": False,
            "error": (
                f"REJECTED — {ungrounded} "
                f"| RE-READ: caller turns, verbatim; a valid "
                f"location is often already among them "
                f"| NEED: their own words, any number in digits"
            ),
        }
        # DURABLE, not just printed. This project's recurring defect is
        # a guard that acts and leaves no trace, and this was the last
        # branch guard still failing that way: on call-20260827-1010 the
        # block appears nowhere in the artifact — "HALLUCINAT", "REJECTED"
        # and "blocked" all score 0 against the JSON — while doctors.json
        # carries status="verified". Whoever reads that row later cannot
        # tell it from a clean one, which is the whole point of the row.
        sess.branch_rejections.append({
            "value": str(args.get("branch") or ""),
            "why": ungrounded,
            "at": datetime.now().strftime("%H:%M:%S"),
        })
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
              f"🚫 HALLUCINATED BRANCH BLOCKED: {args}", flush=True)
        # UNGROUNDED IS TWO DIFFERENT EVENTS AND ONLY ONE OF THEM IS
        # "they did not answer".
        #
        # call-20260821-1931: the caller said "Mission Bay Clinic, 1825
        # 4th Street", the transcript mangled it to "Ford Street", and
        # grounding rejected the model's CORRECT reading. They had
        # answered; only the rendering was wrong, and charging that to
        # the budget is what fired the give-up on a caller who then
        # repeated the address cleanly into a call already told to stop.
        #
        # call-20260902-1716: six location asks, and the caller named
        # nowhere at all.
        #
        # _candidate_location is exactly this question already asked and
        # answered elsewhere -- "did the caller say ANYTHING, when we are
        # about to record that they said nothing?" It finds Mission on
        # the 1931 turns and nothing on the 1716 ones.
        _bears_an_answer = bool(_candidate_location(sess))
    elif (_dropped := _address_dropped(args, sess)) and not sess._address_nudged:
        # ONE-SHOT. The value being saved is CORRECT, only less complete
        # than what they said, so this must never be able to stop the call
        # finishing — a true-but-thin record beats no record at all.
        #
        # The rejection points at the transcript rather than at the caller:
        # they already supplied it, and a wording that sends the agent back
        # to ask again is how call-20260818-1112 lost an answer that was
        # already on the call.
        sess._address_nudged = True
        sess.memory.update(address_offered=_dropped)
        result = {"ok": False, "error": (
            f"NOT SAVED — a street address was given and this value omits "
            f"it | THEY SAID: {_dropped!r} | RETRY: save_branch with both, "
            f"comma-separated | ALREADY SUPPLIED, nothing further needed "
            f"from them")}
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
              f"📍 ADDRESS DROPPED — they gave {_dropped!r}; asking for it "
              f"to be saved too", flush=True)
    elif (mismatch := hospital_mismatch(sess)):
        # Every word can be genuinely quoted from the caller and
        # the record still be wrong, because the call reached
        # the wrong organisation. Grounding cannot see this.
        sess.memory.update(hospital_mismatch=mismatch)
        result = {
            "ok": False,
            "error": (
                f"NOT SAVED — wrong organisation: {mismatch} "
                f"| NEED: which place this call actually reached"
            ),
        }
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
              f"🏥 WRONG ORGANISATION: {mismatch}", flush=True)
    else:
        result = run_tool(name, sess.memory, args, sess.objective)
    # CHARGED ON THE GUARD'S VERDICT, NOT THE MODEL'S CLAIM.
    #
    # The reset used to stand ABOVE the guard, on the mere fact that the model
    # had put a value in the argument. That reads the model's belief that it
    # heard an answer as evidence that one was given, and the guard one line
    # below exists precisely because that belief is sometimes false. So every
    # time the guard caught a hallucinated save it also, silently, bought the
    # model a fresh budget to hallucinate into again.
    #
    # call-20260902-1716: six location asks answered with 'But man, I know,
    # right?', 'My rabbit' and 'Would the bulbous man'. The model twice claimed
    # an identity nobody gave it -- once lifted verbatim from 'I am Salome
    # speaking, how can I help you?' -- and both claims reset the budget
    # seconds before the guard rejected them with 'nothing the caller said
    # since you asked reads as that answer'. no-progress ended the call at 2 of
    # 8 and nothing ever gave up.
    #
    # This is the failure reset_ask_budget's own docstring names -- a guard
    # silently undoing another guard's work -- running the other way.
    #
    # ONLY THE UNGROUNDED VERDICTS MEAN 'they did not answer'. Every other
    # refusal here is about the SHAPE of a value the caller really did give --
    # an address left out, the wrong organisation -- and the double jeopardy
    # argument that put the reset up top still holds for those: the person
    # paying it would be the caller who answered.
    if str(args.get("branch") or "").strip():
        if _bears_an_answer:
            sess.reset_ask_budget("caller named a place")
        else:
            print("[Realtime] budget NOT reset - that value was refused as "
                  "ungrounded, so nothing the caller said bought it",
                  flush=True)
    return result



def _guard_choice_save(name: str, args: dict,
                       sess: "RealtimeSession") -> dict:
    """
    Run one of the four CHOICE saves behind its own grounding guard.

    ONE FUNCTION FOR FOUR TOOLS, because the only thing that differs
    between them is the row in _CHOICE_SAVE_TOOLS: the argument the value
    arrives in, the guard that judges it, the phrasing of what is needed,
    and the memory key a refusal is recorded under. Everything else -
    the budget reset, the deferral, the refusal record - is identical, and
    four copies of it is four chances for one to drift.

    Same contract as _guard_save_branch: reads the transcript, writes the
    session's own records, touches neither the socket nor the lifecycle.
    """
    _arg, _guard, _need, _gkey = _CHOICE_SAVE_TOOLS[name]
    _bears_an_answer = True
    ungrounded_choice = _guard(args, sess)
    # ── THE EVIDENCE HAS NOT ARRIVED YET IS NOT THE SAME AS THE EVIDENCE
    #    CONTRADICTS YOU, and until now both ended in the same refusal.
    #
    # call-20260826-1422: six saves, six waits, six timeouts, zero landed.
    # Every rejection was followed within the same second by the caller
    # transcript containing the answer. "nothing has been transcribed since
    # you asked" was true when asked and false a heartbeat later.
    #
    # The old comment on this path argued the cost of refusing was "one
    # more turn: the model saves again when the transcript lands". That is
    # not what the model did. It apologised — "I'm just making sure I heard
    # you clearly, since phone audio can clip a bit" — and re-asked a
    # question already answered, twice, on a 151s happy path.
    #
    # So when the guard objects and the words are STILL IN FLIGHT, hold the
    # decision instead of taking it. Nothing is saved here. The same guard
    # runs again the instant the transcript lands, against the real words.
    # If they never land, nothing is ever written — which is the behaviour
    # this branch has always had, reached by waiting rather than by
    # guessing.
    if ungrounded_choice and _transcript_pending(sess):
        sess._deferred_save = {
            "name": name, "args": dict(args),
            "why": ungrounded_choice, "at": time.monotonic(),
            "asked_turns": len(sess.turns),
        }
        # ok=True with nothing written is deliberate and it is the whole
        # point. ok=False is what produced the apology and the re-ask; the
        # model needs to hear "this is in hand, carry on", and the promise
        # is kept by _resolve_deferred_save, which will inject a correction
        # if the words do not bear it out.
        result = {"ok": True, "pending": True, "note": (
            "Held — their words are still being transcribed and this will "
            "be checked against them. Do not ask again and do not apologise; "
            "carry on.")}
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
              f"⏸️  HELD FOR EVIDENCE: {name}({args})", flush=True)
        print(f"          the transcript for that turn is still in flight "
              f"— judging it when it lands, not now", flush=True)
    elif ungrounded_choice:
        sess.memory.update(**{_gkey: f"BLOCKED — {ungrounded_choice}"})
        # Terse fragments, no fluent imperative. A rejection the model can
        # paraphrase into a grammatical sentence is a rejection it will read
        # out loud — see _reject's docstring in tools.py, which exists
        # because a live call relayed one to a receptionist verbatim.
        # RE-READ REACHES ALL FIVE FIELDS NOW, not just the branch.
        # save_branch has carried this fragment since call-20260820-1321,
        # where two bare rejections left the model unable to tell WHAT was
        # wrong and it rephrased into a worse answer. The four choice
        # fields never had it — they got the reason and nothing about where
        # to look — and the prompt was carrying the difference in prose
        # ("RE-READ WHAT THEY ACTUALLY SAID BEFORE YOU ASK AGAIN") for all
        # five. This is a tool result, not the cached prefix, so unifying
        # it costs nothing against the prompt ceiling and lets that prose
        # go: the instruction now arrives at the moment it applies, on the
        # field it applies to.
        result = {"ok": False, "error": (
            f"NOT SAVED — {ungrounded_choice} "
            f"| RE-READ: their turns, verbatim; the answer is often "
            f"already among them "
            f"| NEED: {_need}")}
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
              f"🚫 UNGROUNDED ANSWER BLOCKED: {name}({args})", flush=True)
        # The guard has just said, of this very turn, that nothing the caller
        # said reads as this answer. That is the plainest evidence there is
        # that the ask went unanswered, and it must not buy a fresh budget.
        _bears_an_answer = False
    else:
        sess.memory.update(**{_gkey: "verified against caller transcript"})
        # THE QUALIFIER IS DROPPED, NOT THE SAVE. The status is grounded and
        # worth recording; a summary carrying a word nobody said is not, and
        # refusing the whole call over it would throw away a verified answer
        # to protect a footnote. Recorded either way — a field silently
        # emptied is the same invisibility the fabricated version had.
        for _dkey in ("detail", "depends_on"):
            _was = args.get(_dkey)
            _bad, _what = _strip_ungrounded_detail(args, sess, _dkey)
            if _bad:
                sess.memory.update(**{
                    f"{_gkey}_{_dkey}_as_written": _was,
                    f"{_gkey}_dropped_words": list(_bad)})
                print(f"[Realtime] ⚠️  {_dkey}: "
                      f"{', '.join(repr(w) for w in _bad)} never "
                      f"appeared in the caller transcript — {_what}",
                      flush=True)
        result = run_tool(name, sess.memory, args, sess.objective)
    # CHARGED ON THE GUARD'S VERDICT, NOT THE MODEL'S CLAIM -- see the note at
    # the foot of _guard_save_branch for the call this comes from. A HELD save
    # still resets: the words are in flight, not absent, and refusing to credit
    # a caller whose answer is merely late is the expensive direction.
    if str(args.get(_arg) or "").strip():
        if _bears_an_answer:
            sess.reset_ask_budget(f"caller answered: {name}")
        else:
            print(f"[Realtime] budget NOT reset - {name} was refused as "
                  f"ungrounded, so nothing the caller said bought it",
                  flush=True)
    return result



async def _guard_escalate(name: str, args: dict,
                          sess: "RealtimeSession", oai_ws,
                          call_id: str,
                          _pending_tools: dict) -> tuple[dict, bool]:
    """
    Decide whether the call may be given up on, and on what stated reason.

    THE ONE BRANCH THAT IS NOT PURE, which is why it takes the socket and
    the pending-tool table when its three siblings take neither. A blocked
    escalation does not merely produce a refusal: it ANSWERS the tool call
    on the spot, sends the model a nudge saying what to do instead, and
    ends the handler early — because everything after it in the dispatcher
    reports and acts on an escalation that is not going to happen.

    So it returns (result, stop). `stop` True means the tool call has
    already been answered here and the dispatcher must return immediately;
    False means the ordinary path continues with `result`.

    A tuple rather than an exception, and rather than the _ToolOutcome the
    old inline code built: the outcome carries three of the dispatcher's
    own locals, and handing those out so a branch can rebuild them is how
    a function that was supposed to be liftable acquires a second job.
    """
        # Clearing sess._give_up_sent stops us RE-SENDING the
        # directive; it cannot unsay it. Once injected, the model has
        # "stop asking and escalate" in its context and will act on
        # it whatever the caller says next — which on a live call was
        # "can you please give me a minute? I just need to check".
        # So the block has to be here, at the tool call, the same way
        # a fabricated branch is blocked.
    last_caller = next((t.text for t in reversed(sess.turns)
                        if t.role == "caller" and t.text
                        and t.text != "[...]"), "")
    # Two shapes of "not a refusal", blocked the same way. A hold request
    # is "wait, I'm getting it"; an invitation is "what do you want?" —
    # and on call-20260819-2121 the agent answered the second by hanging
    # up. The caller had asked three screening questions, the budget
    # counted all three, the give-up directive went out, and then they
    # said "How can I help you?" — the most willing thing anyone said on
    # that call — and the agent closed on it.
    _blocked = ""
    if not sess.memory.get("branch"):
        # THE ONE TOOL THE DELETED WAIT COVERED THAT CANNOT DEFER.
        # _TRANSCRIPT_READING_TOOLS held six tools; five of them are saves
        # whose guard hands an objection to _resolve_deferred_save when the
        # words are still in flight. escalate has no such path — it ends
        # the call — so removing the wait would leave _discarded_location
        # reading a transcript that has not caught up, and the answer the
        # caller just gave would be invisible to the guard whose entire job
        # is noticing it. One turn of grace instead, and ONE-SHOT like every
        # other injected directive here: the placeholder resolves either
        # way within a turn, and a guard that can refuse forever is a call
        # that cannot be ended.
        if _transcript_pending(sess) and not sess._escalation_held:
            _blocked = "in flight"
        elif is_hold_request(last_caller):
            _blocked = "hold"
        elif _invites_continuation(last_caller):
            _blocked = "invitation"
    if _blocked:
        if _blocked == "in flight":
            sess._escalation_held = True
            result = {"ok": False, "error": (
                "NOT ESCALATED — their last turn is still transcribing "
                "| NEED: wait one turn; the answer may be in it")}
            _line = ("🚪 ESCALATION HELD — their last words are still "
                     "transcribing")
            _say = ("(system: hold on before ending the call. They have "
                    "just said something that has not reached you yet. "
                    "Wait for it — say nothing new and do not end the "
                    "call.)")
        elif _blocked == "hold":
            result = {"ok": False, "error": (
                "NOT ESCALATED — caller is mid-lookup, not refusing "
                "| NEED: a two-word hold acknowledgement, then "
                "silence until they return")}
            _line = "⏳ ESCALATION BLOCKED — caller is checking"
            _say = ("(system: disregard the earlier instruction to "
                    "stop and escalate. They are looking the branch up "
                    "right now. Wait for them.)")
        else:
            result = {"ok": False, "error": (
                "NOT ESCALATED — caller just asked what you need "
                "| NEED: tell them plainly, in one sentence, which "
                "doctor and that you want the branch")}
            _line = "🚪 ESCALATION BLOCKED — caller asked what you need"
            _say = ("(system: disregard the earlier instruction to "
                    "stop and escalate. They have just asked what you "
                    "want, which means they are willing to help and "
                    "have not refused anything. Answer them: name the "
                    "doctor and say you are trying to find out which "
                    "branch they work out of. One sentence, then "
                    "wait.)")
        # The budget put us here, and it was wrong: they were engaging
        # the whole time. Reset it or the very next ask escalates again.
        #
        # NOT FOR "in flight". That one says nothing about whether they
        # were engaging — the budget may have run out for perfectly good
        # reasons and we are only asking it to wait for words already on
        # the way. Resetting there would buy a stall a fresh budget.
        if _blocked != "in flight":
            sess.reset_ask_budget("escalation blocked — caller is engaging")
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
              f"{_line}: {last_caller[:60]!r}", flush=True)
        await oai_ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": _say}]},
        }))
        _pending_tools.pop(call_id, None)
        await oai_ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {"type": "function_call_output",
                     "call_id": call_id,
                     "output": json.dumps(result)},
        }))
        return result, True
    _reason = args.get("reason", "")
    # The inverse guard. Recorded whether or not it blocks:
    # blocking is one-shot, but a discarded answer must never
    # leave the call invisible. Without this the artifact says
    # only "never provided a location", which is the false
    # claim itself, and nothing downstream can tell.
    discarded = _discarded_location(_reason, sess)
    if discarded:
        sess.memory.update(discarded_location=discarded)
    bad = _ungrounded_escalation(_reason, sess)
    if discarded and not sess._discard_blocked:
        # ONE-SHOT, like every other injected directive here. A
        # guard that can refuse forever is a call that cannot be
        # ended: the detector is deliberately conservative, but
        # "conservative" is not "never wrong", and the failure
        # mode of blocking twice is an agent stuck on the phone
        # with a receptionist it has already thanked.
        sess._discard_blocked = True
        result = {"ok": False, "error": (
            f"NOT ESCALATED — reason asserts no location was "
            f"given; the transcript has one "
            f"| CALLER SAID: {discarded} "
            f"| NEED: save_branch with THEIR wording, or an "
            f"escalation reason that is true")}
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
              f"↩️  DISCARDED ANSWER — escalation blocked: "
              f"{discarded[:80]}", flush=True)
    elif bad:
        result = {"ok": False, "error": (
            f"REJECTED — {bad} | NEED: a reason drawn from this "
            f"call's events, not an inference about the doctor "
            f"| FALLBACK: 'could not obtain the location'")}
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
              f"🚫 UNGROUNDED ESCALATION BLOCKED: {args}",
              flush=True)
    else:
        result = run_tool(name, sess.memory, args, sess.objective)
    return result, False


__all__ = [
    "_address_dropped",
    "_address_offered",
    "_candidate_location",
    "_discarded_location",
    "_guard_choice_save",
    "_guard_escalate",
    "_guard_save_branch",
    "_strip_ungrounded_detail",
    "_ungrounded_escalation",
    "hospital_mismatch",
]
