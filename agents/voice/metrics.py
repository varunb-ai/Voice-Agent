"""The shape of a finished conversation. Measure-only.

Split from realtime_worker 2026-08-26, verbatim.

- Detects stapling, pile-ups and repeats, and changes nothing: it scores the
  artifact after the call. It is not a guard, cannot reject a turn, and no
  prompt rule may be deleted because it exists. Only a check that INJECTS a
  directive mid-call or REJECTS a tool call replaces prose.
- It could not move with latency.py: it reads _is_filler_reply and
  _norm_clause, which were in the worker then. They are in turns.py now.
"""
from __future__ import annotations

import logging
import re

from agents.voice.evidence import _is_location_ask
from agents.voice.objectives import clauses as _clauses, sentences as _sentences
from agents.voice.turns import _is_filler_reply, _norm_clause

log = logging.getLogger(__name__)


# A comma before a coordinating conjunction - the one sentence-internal seam
# `clauses()` deliberately does not cut, because an ask almost never sits on
# it. A repeated span very much can: see the third pass in conversation_metrics.
_COMMA_CLAUSE = re.compile(r",\s+(?:but|and|so|then|though|yet|because)\b\s*",
                           re.I)


# Thanking somebody for a FIELD NAME. The nouns are the ones the call actually
# collects plus the generic stand-ins the model reaches for; "help" and
# "checking" are deliberately absent because they describe an ACT the person
# performed rather than a slot on a form — and the prompt already bans those
# two by name a few lines below the rule this watches.
# Discourse markers people open a turn with. Same closed list objectives.py
# uses to decide whether a bare "yes" is an answer — one vocabulary, so the
# two cannot drift apart on what counts as an opener.
_OPENER_LEAD = re.compile(
    r"^(?:\W|\b(?:ah|oh|uh+|um+|er+|hmm+|well|so|okay|ok|right|now|sure|yeah)\b)+",
    re.I)


def _opening_clause(sentence: str) -> str:
    """The first clause of one SENTENCE, with any leading marker removed.

    "Okay, thanks for explaining that — ..." and "Thanks for explaining that —
    ..." open with the same words; only a marker separates them, and a
    comparison that keeps it is measuring the marker rather than the repeat.
    """
    for clause in _clauses(sentence):
        return _norm_clause(_OPENER_LEAD.sub("", clause.strip()))
    return ""


_CATEGORY_ECHO = re.compile(
    r"\b(?:thanks?(?:\s+you)?|thank you)\s+(?:so much\s+|very much\s+)?"
    r"for\s+(?:the|that|this)\s+"
    r"(?:location|information|info|details?|branch|address|name|"
    r"confirmation|clarification|update|answer)\b", re.I)


def _double_ask(text: str) -> bool:
    """Two requests for the same thing inside one turn.

    Counted by requests, not by question marks. "I need the specific branch name
    or street address where Dr. Okafor sees patients. Which one is it?" carries
    one "?" and asks twice — a statement-form request followed by a question.
    The trailing question is also vaguer than the statement it repeats, so it
    reads as asking the caller to choose between the options just listed.
    """
    parts = _sentences(text)
    # "Which one is it?" names nothing, so it is not an ask by content — it is an
    # ask by context, pointing back at the request before it. That is precisely
    # what makes it vague. So the shape to catch is a statement-form request
    # followed by a separate question, not two recognisable location asks.
    statement_asks = [p for p in parts if "?" not in p and _is_location_ask(p)]
    questions      = [p for p in parts if "?" in p]
    if statement_asks and questions:
        return True
    return sum(1 for p in questions if _is_location_ask(p)) > 1

def conversation_metrics(turns: list, raw: "list | None" = None) -> dict:
    """Count the conversational failures that prose rules keep failing to stop.

    Three attempts at fixing the same behaviour by writing more forceful
    instructions, each one ignored, is evidence the marginal rule is doing
    less each time. These are constraints on the SHAPE of a turn rather than
    its content, which makes them detectable in code even though the prompt
    cannot reliably enforce them.

    Nothing here changes behaviour — you cannot unsay a turn. The point is to
    have a number, so the next prompt edit can be evaluated against the last
    one instead of by reading a transcript and forming an impression.

      stapled_questions  — agent asked a question in the same turn it answered
                           one of theirs. Six of these in one 111s call.
      back_to_back_asks  — agent asked again WITHOUT BEING ANSWERED in between.
      repeated_sentences — same agent sentence said more than once.

    `raw` IS THE UNMERGED TRANSCRIPT, and it exists because save() hands this
    function the MERGED one — which collapses consecutive agent turns inside
    5s into a single turn. Every counter above wants that: a reply flushed in
    two fragments is one thing the caller heard. `stacked_agent_turns` is the
    one counter that wants the opposite, because the merge is precisely what
    it is looking for, and reading it off `turns` would report zero on every
    call that has the defect.

    That is not hypothetical here. The verbatim-repeat carve-out a few lines
    into save()'s merge exists because the fragment rule deleted a live
    🔁 REPEATED SENTENCE before `repeated_sentences` could count it, and the
    artifact then read zero on the fault it was written to measure. Same trap,
    one counter over; the way past it is to keep both transcripts rather than
    to add a second carve-out.

    Defaults to `turns` so every existing call site — and the whole suite,
    which passes hand-built lists that are already unmerged — keeps meaning
    exactly what it did.
    """
    agent = [t for t in turns if t.role == "agent"]
    stapled = back_to_back = 0
    prev_agent_asked = False

    for i, turn in enumerate(turns):
        if turn.role != "agent":
            # A CALLER ANSWER BREAKS THE RUN, and until 2026-08-24 it did not:
            # this loop skipped straight past caller turns, so prev_agent_asked
            # carried across them and any two AGENT turns that both asked
            # something counted — however well the call was going.
            #
            # call-20260824-1604 scored 1 on a flawless exchange: greeting asked
            # for the branch, the caller gave it, the agent asked the second
            # scripted question. That was tolerable while the script had one
            # question, because a second ask usually WAS a re-ask. On a
            # four-question script every healthy call trips it on every adjacent
            # pair, and a metric that fires on the good case is worse than no
            # metric — it is the number people stop reading.
            #
            # The defect actually worth counting is asking again into silence,
            # so silence is what has to persist the run. Filler is silence:
            # "Hello." after a barge-in truncation is the case the ask budget
            # was built around.
            if turn.role == "caller" and turn.text.strip() != "[...]" \
                    and not _is_filler_reply(turn.text):
                prev_agent_asked = False
            continue
        asks = "?" in turn.text
        prev_caller = next((turns[j] for j in range(i - 1, -1, -1)
                            if turns[j].role == "caller"), None)
        # Did the caller's most recent turn, immediately before this one, ask
        # something? Then answering AND asking in one breath is the failure.
        if asks and prev_caller is not None and i > 0 and turns[i - 1] is prev_caller \
                and "?" in prev_caller.text:
            stapled += 1
        if asks and prev_agent_asked:
            back_to_back += 1
        prev_agent_asked = asks

    seen: dict[str, int] = {}
    for t in agent:
        for sentence in _sentences(t.text):
            key = _norm_clause(sentence)
            if len(key.split()) >= 4:
                seen[key] = seen.get(key, 0) + 1

    # Sentence-level repeats first, then clause-level ones that are NOT already
    # inside a sentence counted above. Saying one sentence twice is ONE
    # repetition, not one for the sentence plus one for each of its clauses.
    #
    # Counting clauses INSTEAD of sentences was the first attempt and it lost
    # repeats: a five-word sentence splits into two sub-threshold clauses and
    # vanishes. Checked against the whole call history — that swap silently
    # dropped a real repeat on call-20260806-2029 while fixing three others.
    # Both levels, largest unit wins.
    #
    # NOTE: values are NOT comparable with calls analysed before 2026-08-18.
    # The old figure counted sentences only and was structurally too low, so a
    # rise across that date is the metric being fixed, not the agent worsening.
    repeated = sum(n - 1 for n in seen.values() if n > 1)
    _covered = {k for k, n in seen.items() if n > 1}
    clause_seen: dict = {}
    for t in agent:
        for sentence in _sentences(t.text):
            if _norm_clause(sentence) in _covered:
                continue    # already counted as a whole-sentence repeat
            for clause in _clauses(sentence):
                key = _norm_clause(clause)
                if len(key.split()) >= 4:
                    clause_seen[key] = clause_seen.get(key, 0) + 1
    repeated += sum(n - 1 for n in clause_seen.values() if n > 1)

    # ── A repeated span INSIDE the sentences, which the two passes above miss ─
    #
    # call-20260902-1544, the turn a listener hears as the most robotic thing
    # on the call: "I'm not in your system yet, but my name is Simone Hallam.
    # I'm not in your system yet, but it's December 27, 1989." Seven words said
    # twice, four seconds apart, and this function scored the call 0 repeats -
    # so the artifact said the turn was clean.
    #
    # Neither pass could see it. The two sentences differ, so the sentence pass
    # counts nothing; `clauses()` splits on dashes, semicolons and colons and
    # never on a comma before a conjunction, so each sentence is also a single
    # clause and the clause pass counts nothing either. The repeated span is
    # not a unit at any level this function looks at.
    #
    # SPLIT HERE AND NOT IN clauses(). That function decides what an ASK is -
    # `_ask_clauses`, `_is_objective_ask`, the verbatim re-ask guard - and
    # widening it would change which turns those guards fire on. This is
    # measure-only and must stay that way, so the extra split lives here.
    #
    # Largest unit still wins: a sentence or clause already counted as a repeat
    # is skipped, so one repeated sentence is one repetition and not one for
    # the sentence plus one for each fragment inside it.
    _covered |= {k for k, n in clause_seen.items() if n > 1}
    comma_seen: dict = {}
    for t in agent:
        for sentence in _sentences(t.text):
            if _norm_clause(sentence) in _covered:
                continue
            for clause in _clauses(sentence):
                if _norm_clause(clause) in _covered:
                    continue
                for part in _COMMA_CLAUSE.split(clause):
                    key = _norm_clause(part)
                    if len(key.split()) >= 4:
                        comma_seen[key] = comma_seen.get(key, 0) + 1
    repeated += sum(n - 1 for n in comma_seen.values() if n > 1)

    # Adjacent agent turns that are word-for-word identical.
    #
    # Separate from `repeated_sentences` because it needs no length floor. The
    # ≥4-word floor above is there so "Got it." said in six different turns is
    # not counted as five repetitions — across a call, a short stock phrase
    # recurring is normal speech. Back to back it is not: "Sure, no rush. Sure,
    # no rush." has no innocent reading, and the floor was the only reason
    # call-20260819-2044 scored zero on a repeat the live detector had already
    # flagged in the console.
    back_to_back_repeats = sum(
        1 for a, b in zip(agent, agent[1:])
        if _norm_clause(a.text) and _norm_clause(a.text) == _norm_clause(b.text))

    # Denominators, so counts can be compared across calls of different
    # difficulty. A hostile caller who answers nothing gives the agent six
    # chances to staple; a cooperative one gives it one. Raw counts make the
    # easy call look better when it may simply have had fewer opportunities.
    caller = [t for t in turns if t.role == "caller"]
    caller_questions = sum(1 for t in caller if "?" in t.text)

    # ── Unsolicited PII dumps ────────────────────────────────────────────────
    # The synthetic identity (synthetic_identity, seeded per doctor) is an
    # ANSWER, and the artifact must show a question before it. On
    # call-20260902-1245-5dce the agent gave name AND date of birth four
    # seconds in, unprompted, in response to a ghost turn ("OpenAI.", 0.0217
    # RMS — see the gray-zone note at _SILENT_AUDIO_RMS in audio.py) — and no
    # metric recorded it, so the failure was invisible until a person read the
    # transcript. Measure-only: it changes no behavior and cannot reject a
    # turn; it is the number that says whether the details gate held.
    #
    # The detector keys on the DOB YEAR rather than the name, because the name
    # is not in the artifact — the year range is synthetic_identity's own
    # (1960 + seed % 45), so anything it matches on an agent turn is almost
    # certainly the persona's date of birth being read aloud.
    #
    # ASKED means ANY prior caller turn, not the immediately preceding one: on
    # the live call the real ask ("What is your name and date of birth?") sat
    # four caller turns upstream of the compliant answer, behind two stalls and
    # a re-prompt — a last-turn-only window false-flags the disclosure the
    # guard wanted while missing nothing.
    # ── Thanking them for the CATEGORY instead of the thing ─────────────────
    # "Got it, thanks for the location — is she taking new patients right now?"
    # A person says "thanks, Northgate". Naming the field back is slot-labelling
    # and it is the single most audible tell that a form is being filled in.
    #
    # WHY IT IS COUNTED RATHER THAN GUARDED. The fix is a WORD CHOICE, which is
    # the one thing only the model can make — there is nothing here to reject
    # and nothing to inject that would not itself be a prose rule. So this is
    # measure-only by nature, not by convenience, and the rule it watches
    # (templates.py, "say the PLACE back, never the word 'location'") stays in
    # the prompt where it belongs.
    #
    # IT EXISTS BECAUSE THE REWORD NEEDS A VERDICT. Measured before the change:
    # 38 of 937 agent turns, 4.1% overall — forage 4.9%, provider_verification
    # 4.7%, patient_discovery 1.3% — with "location" 21 of the 38. A prompt
    # rule with no number against it is a rule nobody can tell has stopped
    # working, which is how the intake disclaimer reached nine calls unnoticed.
    category_noun_echoes = sum(1 for t in agent if _CATEGORY_ECHO.search(t.text))

    # ── Two agent turns in a row opening with the same clause ───────────────
    # call-20260903-1259, the last two things the agent said:
    #   "Okay, thanks for explaining that — I just need to know how to get on it."
    #   "Thanks for explaining that — take care."
    # Word for word the same opener, four words long, three seconds apart. All
    # three passes above scored it zero.
    #
    # WHY THEY ALL MISSED IT. The sentences differ, so the sentence pass sees
    # nothing. `clauses()` splits on the dash, giving "Okay, thanks for
    # explaining that" and "Thanks for explaining that" — which are not equal,
    # because of a single leading "Okay,". And the comma pass only splits
    # before a coordinating conjunction, which "thanks" is not. A discourse
    # marker in front of a repeat hides it from every existing pass.
    #
    # STRIPPED HERE, NOT IN _norm_clause. That function is read by the verbatim
    # re-ask guard in turns.py; widening it would change which turns a GUARD
    # fires on, and this is measure-only. Same reason the comma split lives in
    # this file rather than in clauses().
    #
    # PER SENTENCE, NOT PER TURN, and that correction came from the same call.
    # A first pass compared adjacent agent TURNS and scored -1259 zero — the
    # two utterances were flushed three seconds apart but reached the
    # transcript as ONE turn, so the repeat is inside a turn and not across
    # two. Keying on sentences catches it either way, which is what the
    # measurement has to survive: how the turns happen to be recorded is not
    # something a listener can hear.
    #
    # ITS OWN COUNTER, not folded into repeated_sentences. That number means
    # "a span said twice anywhere in the call" and carries a ≥4-word floor
    # tuned for it; this one is specifically about OPENERS, where three words
    # already sound like a loop. Merging them would make both mean less.
    _openers: dict = {}
    for t in agent:
        for sentence in _sentences(t.text):
            _o = _opening_clause(sentence)
            if _o and len(_o.split()) >= 3:
                _openers[_o] = _openers.get(_o, 0) + 1
    shared_opening_clauses = sum(n - 1 for n in _openers.values() if n > 1)

    # ── Two agent turns with no caller speech between them ──────────────────
    # call-20260903-1422, the last four seconds of the call:
    #   14:24:37 agent  "Thanks, that helps - I'm just going to think about it
    #                    for now."
    #   14:24:40 agent  "Okay, thanks for explaining that."
    # Two turns, nobody between them, and the second thanks her for the second
    # time. The teardown had asked for a goodbye because the response carrying
    # the completing tool spoke no audio - see the "spoken" branch in
    # _decide_close, which is the fix.
    #
    # NOTHING ELSE SCORED IT. shared_opening_clauses compares openers and these
    # differ ("Thanks, that helps" vs "thanks for explaining that"); the
    # sentence passes compare sentences and these differ too. And the
    # transcript PRINTER merges consecutive agent turns onto one line, so
    # afterwards there was not even anything to look at. The structure is what
    # is wrong, so the structure is what is counted.
    #
    # OFF `raw`, NOT `turns`, AND THAT IS THE WHOLE COUNTER. save() merges
    # consecutive agent turns within 5s into one before it gets here, so the
    # pair this exists to find is a single turn by the time `turns` arrives
    # and this would read zero on every call that has the defect. See the
    # `raw` paragraph in the docstring.
    #
    # "[...]" IS NOT A CALLER TURN. It is their transcript still in flight -
    # counting it would say somebody spoke between two of ours when the whole
    # question is whether anybody did. Same exclusion the close walks make.
    _real = [t for t in (turns if raw is None else raw)
             if (t.text or "").strip() and (t.text or "").strip() != "[...]"]
    stacked_agent_turns = sum(
        1 for a, b in zip(_real, _real[1:])
        if a.role == "agent" and b.role == "agent")

    unsolicited_pii_dumps = 0
    for i, t in enumerate(turns):
        if t.role != "agent" or not re.search(r"\b(19\d{2}|200[0-4])\b", t.text):
            continue
        asked = any(
            pt.role == "caller" and ("?" in pt.text or re.search(
                r"\b(name|birth|dob|address)\b", pt.text.lower()))
            for pt in turns[:i])
        if not asked:
            unsolicited_pii_dumps += 1

    return {
        # How many times it asked where the doctor practises. On the call that
        # exposed this it was six, with no location offered between any of
        # them — the number that says "it would not let go".
        "location_asks": sum(1 for t in agent if _is_location_ask(t.text)),
        # Two requests for the same fact inside one turn. The prompt's rule was
        # "EXACTLY ONE question mark per turn", which a statement-form request
        # followed by a question passes with one "?" — the same blind spot the
        # ask DETECTOR had. On a live call: "I need the specific branch name or
        # street address where Dr. Okafor sees patients. Which one is it?"
        "double_asks": sum(1 for t in agent if _double_ask(t.text)),
        # Moves stacked into one turn, counted as sentences and needing no
        # vocabulary at all — the banned-phrase list for thinking-narration
        # missed 2 of the 3 wordings actually used, because "ways to narrate"
        # is an open set. Sentence count is structural and cannot rot.
        # The greeting is excluded: it is a fixed line, not a pile-up, and it
        # would otherwise dominate the count.
        "piled_turns": sum(1 for t in agent[1:] if len(_sentences(t.text)) >= 3),
        "longest_turn_sentences": max((len(_sentences(t.text)) for t in agent[1:]),
                                      default=0),
        "longest_turn_words": max((len(t.text.split()) for t in agent[1:]), default=0),
        "agent_turns": len(agent),
        "caller_turns": len(caller),
        "caller_questions": caller_questions,
        "question_turns": sum(1 for t in agent if "?" in t.text),
        "stapled_questions": stapled,
        # The rate is the comparable figure: of the times they asked something,
        # how often did the agent answer and ask back in the same breath?
        "staple_rate": round(stapled / caller_questions, 2) if caller_questions else None,
        "back_to_back_asks": back_to_back,
        "repeated_sentences": repeated,
        "back_to_back_repeats": back_to_back_repeats,
        "unsolicited_pii_dumps": unsolicited_pii_dumps,
        # Turns that thanked them for a field name instead of the thing they
        # said. The number the "say the PLACE back" reword has to move; without
        # it the reword is a guess nobody can settle.
        "category_noun_echoes": category_noun_echoes,
        # Adjacent agent turns opening on the same clause. Non-zero is the
        # agent reaching for the same stock phrase twice running — audible as
        # a loop even when every existing repeat pass scores the call clean.
        "shared_opening_clauses": shared_opening_clauses,
        # Agent turns spoken straight onto another agent turn. Non-zero is the
        # agent talking to itself: the caller had no gap to speak into and
        # whatever the second turn says arrives as an afterthought on the
        # first. The closing pair is where it shows up worst.
        "stacked_agent_turns": stacked_agent_turns,
    }


__all__ = [
    "_double_ask",
    "conversation_metrics",
]
