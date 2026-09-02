"""The answers: is this value supported by what the caller actually said?

The top of the layering. Each function takes a value the model wants to save
and returns the reason it may not be - an empty string means it checks out.
They compose everything beneath: `patterns` for vocabulary, `probes` to judge
a string, `window` to decide which strings are in scope at all.

WHY THE CHOICE GUARD IS NOT THE PLACE GUARD WITH A DIFFERENT PATTERN. A
location is a high-entropy proper noun; a status is two bits. The blob check
that grounds "Northgate" grounds nothing for "yes", which callers say
constantly for other reasons - so the choice guard is anchored to the ask,
bounded above by the next one, and leans on the audio measurement in a way
the place guard never has to. `_ungrounded_choice` argues this at length in
its own docstring; it is the longest thing in this package for that reason.
"""
import re
from typing import Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:                       # pragma: no cover - typing only
    # A binding for the string annotations below, and nothing more.
    # TYPE_CHECKING is False when Python runs, so no import happens and the
    # one-way rule this package was extracted under still holds:
    # realtime_worker imports us, never the reverse.
    from agents.voice.realtime_worker import RealtimeSession

from agents.voice.objectives import (
    states_in_its_own_right,
)
from agents.voice.evidence.patterns import (
    _DETAIL_FUNCTION_WORDS,
    _MAX_OWED_PER_CALL,
    _MAX_OWED_PER_TEXT,
    _NUMBER_WORD_VALUE,
    _UNGROUNDED_STOPWORDS,
)
from agents.voice.evidence.probes import (
    _class_present,
    _grounded_in,
    _grounded_loosely,
    _number_grounded_indices,
    _is_ask_for,
    _is_hint_echo,
    _meaning_class,
    _owed_key,
)
from agents.voice.evidence.window import (
    _asserted_caller_text,
    _caller_speech_level,
    _ever_transcribed,
    _other_field_probes,
    _turn_asserts,
)

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
    _words = [w.strip("'") for w in re.findall(r"[a-z']+", value.lower())]
    # A NUMBER THEY GAVE IN FIGURES IS STILL A NUMBER THEY GAVE. Computed over
    # the whole list rather than per word, because a tens/unit pair only
    # grounds as a pair - see _number_grounded_indices for the call where
    # "twenty second" against their "22nd" cost the field its only fact.
    _numeric = _number_grounded_indices(_words, heard)
    for i, w in enumerate(_words):
        if (not w or len(w) <= 2 or w in _UNGROUNDED_STOPWORDS
                or w in _DETAIL_FUNCTION_WORDS or w in out):
            continue
        if i in _numeric:
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
    # BOUNDED ABOVE AS WELL, and the ceiling is the half that was missing.
    #
    # `since` is a floor with nothing over it, so once an ask matched, every
    # later caller turn stayed evidence for that field for the rest of the
    # call. On call-20260831-1048 the agent asked "could you confirm this is
    # Dr. Jennifer, Cardiology, at New York Baptist Hospital?" at turn 2 and
    # the caller confirmed; six turns later it asked "which branch does Dr.
    # Jennifer work out of?" and the caller said "I don't know the branch
    # name." `classify_identity` reads "don't know" as UNSURE — correctly, in
    # isolation — and that turn was still inside the identity window, so an
    # answer about a BUILDING overwrote an identity that had been grounded and
    # confirmed eight turns earlier. It cost more than the row: identity is the
    # gate every other field hangs off (see _IF_RIGHT_DOCTOR), so flipping it
    # to `unsure` made branch, accepting, scheduling and referral all
    # not-required, the objective went COMPLETE, and the call hung up on the
    # question it had just asked.
    #
    # IDENTITY_ASK's lookahead deliberately refuses to match a branch ask, so
    # the identity guard cannot be anchored ON a branch turn — which is right,
    # and which also meant a branch ask could never move the anchor OFF
    # identity. The window had no way to close. So an ask for ANY OTHER FIELD
    # closes it: turns from the other question onwards belong to that question.
    _others = _other_field_probes(sess, probe)
    since = 0
    asked = False
    until: Optional[int] = None
    for i, t in enumerate(sess.turns):
        if t.role != "agent":
            continue
        _text = t.text or ""
        if _is_ask_for(_text, probe):
            # OUR OWN ASK FIRST, and `continue` rather than falling through.
            # One turn routinely names two fields ("thanks — and which branch
            # is she at?"), and a turn that asks for THIS field re-opens the
            # window; it must never be able to close it in the same breath.
            since, asked, until = i + 1, True, None
            continue
        if until is None and any(_is_ask_for(_text, p) for p in _others):
            until = i

    # NO CEILING WHEN WE NEVER ASKED. The never-asked path below exists to
    # honour an answer the caller VOLUNTEERED, and on call-20260825-0915 that
    # arrived ("we are full right now, but I can put you on the list") while
    # the agent was still asking about the branch. Bounding an unanchored
    # window at the first other-field ask would throw away exactly the turn the
    # path was widened for. `states_in_its_own_right` is that path's gate and
    # stays the only one.
    if not asked:
        until = None

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
    # THE FLOOR OUTRANKS A CEILING IT HAS ALREADY PASSED. The floor marks a
    # fresher and more specific question than any ask — we spelled the name out
    # and put it to them directly — so a topic change from before it is not
    # bounding anything any more. Without this the two anchors could cross and
    # leave an empty window with the wrong reason attached to it.
    if until is not None and until <= since:
        until = None
    # `sess.turns[since:None]` is `sess.turns[since:]`, so the unbounded case
    # needs no branch of its own.
    usable = [t for t in sess.turns[since:until]
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
        if until is not None:
            # THE WINDOW CLOSED WITH NOTHING IN IT. We asked, then moved on to
            # a different question before they said anything — so the turns
            # that exist belong to the other question and there is no evidence
            # for this field at all. Distinct from the silence below, and it
            # has to be: telling the model "nothing has been transcribed" when
            # the caller has been talking the whole time invites it to re-ask
            # the wrong thing.
            return (f"{label}={status!r} — you moved on to another question "
                    f"before they answered this one | NEED: ask about "
                    f"{label} again, and use what they say to THAT")
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


__all__ = [
    "_grounding_verdict",
    "_owed_refusal",
    "_revisit_grounding",
    "_rode_along",
    "_ungrounded_choice",
    "_ungrounded_detail",
    "_ungrounded_terms",
]
