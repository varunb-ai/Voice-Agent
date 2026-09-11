# Voice Agent — project context

Outbound voice agent (OpenAI Realtime + Twilio) that phones medical offices as
a **prospective patient** and records which branch a doctor practises at,
whether they are taking new patients, and either the waiting-list answer or
where the doctor moved to.

```bash
python run_twilio.py --doctor "Jane Okafor" --specialty "Pediatric" \
                     --hospital "Mercy General" --to "+1..."
python test_realtime_protocol.py     # fully offline, ~3,099 checks, no network
```

**Two checks are RED on a clean tree** — the `check_refusals` corpus manifest
rows (`unexpected: call-20260904-1306-42d4 identity`, `15 findings for 14
distinct gaps`). They were red before your change. Do not "fix" them reflexively.

---

## Live configuration — AND IT LIVES IN A GITIGNORED FILE

`.env` is ignored (`.gitignore:2`) and `.env.example` **does not match it**. A
fresh clone runs a materially different agent: template 1, ambience off, VAD at
400 ms. These are the values production actually runs, and why:

| setting | live | `.env.example` | why it is not the default |
|---|---|---|---|
| `CALL_TEMPLATE` | `patient_discovery` | `forage_data_collection` | Template 4 is the one under active work |
| `REALTIME_MODEL` | `gpt-realtime-2` | *(absent)* | |
| `REALTIME_VOICE` | `marin` | *(absent)* | marin takes steering; cedar barely does |
| `REALTIME_SILENCE_MS` | **`700`** | *(absent — code default is `400`)* | **Do not lower without evidence.** 400 ms cut callers off mid-sentence |
| `REALTIME_VAD_EAGERNESS` | `medium` | *(absent)* | |
| `REALTIME_BACKCHANNELS` | `true` | *(absent)* | clips under the caller's speech; the audio layer owns listening noises, not the LLM |
| `REALTIME_AMBIENCE` | `true` | `false` | on since 2026-09-04 |
| `REALTIME_AMBIENCE_DB` / `_DUCK_DB` | `-45` / `-50` | same | the duck target IS a live control; -54 read as the room going away |
| `REALTIME_NOISE_REDUCTION` | `near_field` | *(absent)* | |
| `REALTIME_ECHO_GATE` | `pass` | *(absent)* | |

If `.env` is ever lost, restore from this table. Syncing `.env.example` to it is
a separate decision — it changes what a clone runs.

---

## The prompt has a hard ceiling, and it is the main design constraint

`patient_discovery` sits at **6,850 / 6,900 tokens** and **28,643 / 28,800
chars**. The chars bind first. Ceilings live in `_PROMPT_CEILINGS` in
`test_realtime_protocol.py`.

**To add a rule you must evict one.** That is the entire point of the number —
raising it silently is how four earlier behaviour problems each got answered by
appending a paragraph. Prefer replacing text that is redundant, contradictory,
low-value, already enforced by code, or overly procedural.

Two evictions that have already paid for themselves, and generalise:

- **A quoted phrase the model could say becomes a phrase library.** Measured
  over 470 turns: banned *words* → 0 said; banned context-specific *sentences*
  → 0 said; quoted reusable *openers* → **166 said**, including 23 sittings of
  one string filed under `BANNED` in capitals. Name no sound, quote no line the
  agent could utter, give no example turn.
- **A rule the code enforces is prose the model can drop.** Ordering, the
  outcome label, the one-item-per-response rule and the disclaimer are all
  enforced in `objectives.py` / `grounding/` / `turns.py`.

---

## THE standing result — read this before proposing a prompt fix

**~14 code guards hold. 7 prompt rules did not.** Every measured attempt to fix
a behaviour by writing a better instruction has come back null:

| attempt | measurement |
|---|---|
| join rule (reaction + ask are one turn) | 0/8 vs 0/8 |
| stake / motivation rewrite of `_TONE_PATIENT` | **0/6 vs 0/6**, arms textually indistinguishable |
| `_TONE_PATIENT` as a whole vs an empty slot | no measure separates them (n=8, marin + cedar) |
| owed-substance recovery directive | ignored 4× running; moved into a guard, then held |

Only the extreme rate/loudness/pitch arm ever moved commanded measures (6/6,
p=0.016). **Take-to-take variance swamps prompt effects 2–8×**, so any A/B under
~14 takes per arm measures the draw, not the prompt.

Corollary: when a prose rule and a guard disagree about the same behaviour, fix
the guard. Prose that a guard can enforce belongs in the guard.

---

## Known issues, current

Verified this session by calling the predicates — not inferred:

1. **The patient still sounds workflow-oriented.** Bad news comes back as a
   receipt: *"Okay, thanks for letting me know—let me just ask one quick thing
   about that."* Confirmed live and 12/12 offline across both prompt arms.
   **It is not the muting artefact** — on `call-20260908-1628` the bad-news turn
   emitted a single item (`dropped_second_items` holds only `'Thanks.'` and
   `'How would I get on that list?'`, from other turns).
   **Not fixed by the two counter repairs below**, and do not expect it to be:
   the A/B measured 0/6 reactions in both arms with no directive in play.
2. Response/tool timing: tool turns are 1.24 s speak-first vs 2.59 s tool-first.
   The cost cannot be removed, only hidden.
3. Backchannel behaviour and ambience continuity both need live validation.

### Closed 2026-09-09 — both were counters, neither was the prompt

Kept because the *shape* of each recurs, not as a changelog.

- **The cadence detector booked the emotion as the tic.**
  `_ack_opener` lists `oh` and `i see`, the only two disappointment markers this
  model reaches for in text, so `"Oh, that's a shame — is there a waitlist?"`
  scored as the enquiry-bot cadence and drew a directive banning openers for the
  rest of the call. Fixed by `_reacted_to_news` (`grounding/vocabulary.py`).
  `_ack_opener` is untouched and still lexical; the judgement moved to the call
  sites, which drop an earned opener from the run **in both positions**.
  10 turns of 1,184 exempted, 5 of 425 runs suppressed, 0 lost.
  The discriminator is **receipt vs reaction**, NOT the question mark — the
  cadence *is* ack-then-field-question, so `"?"` would gut it.
- **The announced-question detector had a hole.** `if "?" in t: return False`
  ran *before* `_ANNOUNCED_ASK`, discarding a match the pattern had already
  made — a guard-clause defect that reads exactly like a vocabulary gap.
  A second, independent cause: the placeholder allowed one modifier, so
  `"one more quick thing"` matched nothing. Fixed by `_PERMISSION_TO_ASK` plus
  `{1,2}`. 3 turns newly detected, 0 lost.
  **Check what runs before a pattern before widening the pattern.**

---

## Constraints

- **Do not spend renders or credits without explicit approval.** Credits are
  company-funded. `probe_delivery_ceiling.py` has no resume path and one
  invocation is 70 renders, not 28.
- **Do not raise the prompt ceiling by appending text.** Evict first.
- **Do not lower `REALTIME_SILENCE_MS` from 700 without evidence.**
- **Do not hardcode Template 4 field names in generic objective logic.** A
  template declares what it collects (`CallObjective`); checkers must not grow
  a list of template names. Same shape as `names_org`.
- **Separate diagnosis from implementation.** Show the conflicting rule, what
  can be evicted, the exact replacement and the token delta — then stop for
  approval. Do not edit and report in one motion.
- **The repo is LF.** `Path.write_text` on Windows rewrites the whole file with
  CRLF; write bytes, or pass `newline="\n"`. A bash heredoc turns a regex `\b`
  into a backspace.
- **Assert on what you found, never on the absence of one spelling.** A check
  that passes by finding nothing is not a check. Pair every absence assertion
  with a positive control, and mutation-test it.
- **The repo stays private** — transcripts, colleague names, the client name.

---

## Safety rules that are NOT bugs

- **The patient declines a real waitlist sign-up** (13/13, freshly worded each
  time). The caller is synthetic; accepting puts a non-person on a real
  practice's list ahead of real people. This is the EHR guard reaching its end.
  The *delivery* is fair game; the decision is not.
- **A detail never leaves without a not-a-patient line standing in front of
  it.** A name plus a DOB is enough to open a medical record. Enforced by
  `_detail_left_bare`, not by prose.
- **Recording disclosure and the are-you-a-bot answer outrank everything**,
  including the persona.

---

## Longer-form history

Per-defect write-ups live in the user's memory directory, indexed by
`MEMORY.md` — start with `where-the-project-stands.md`. This file is the
orientation that must survive a fresh clone; that directory is the detail.
