# Voice-Agent

An outbound voice agent that phones medical offices to verify **which branch a
doctor practises at**, and writes the answer back into a directory record.

Speech-to-speech over the OpenAI Realtime API, carried on a Twilio phone line.
A call takes about 75 seconds and costs roughly **$0.11 per minute** all-in
(model + telephony), measured.

```
run_twilio.py ──▶ Twilio places the call
                     │
                     ├─ POST /answer      → TwiML: <Connect><Stream>
                     └─ WS /stream/<sid>  → media stream opens
                                              │
              agents/voice/realtime_worker.py ┤ bridges Twilio μ-law 8 kHz
                                              │ ↔ OpenAI Realtime (gpt-realtime-2)
                                              │
                                              ├─ guards run on every tool call
                                              ├─ call artifact → data/3 cases jsons/
                                              └─ enrichment   → data/3 cases jsons/doctors.json
```

---

## The problem this is actually solving

Getting a location out of a receptionist is easy. Getting one that is **true**
is not.

A language model on a phone line will happily produce a plausible branch name
it was never told — reshaped from the hospital name already in its context, or
echoed back from the transcription hint. That output is indistinguishable from
a real answer once it reaches a database. For a directory product, a fabricated
row is worse than an empty one: the empty one is visibly a gap.

So most of the engineering here is not conversational. It is a set of guards
that sit between the model and the record, each one written against a specific
failure observed on a real call.

| guard | what it stops |
|---|---|
| **grounding** | a location the caller never said — checked against the transcript, not the model's claim |
| **hint echo** | the transcriber emitting a phrase from its own vocabulary hint on near-silent audio |
| **bare city** | `branch="Los Angeles", city="Los Angeles"` — a city is not a branch |
| **wrong organisation** | every word genuinely quoted, but the call reached a different hospital |
| **discarded answer** | escalating *"never provided a location"* when the transcript contains one |
| **ask budget** | asking a fifth time, counted by answers received rather than seconds elapsed |
| **false save claim** | telling the caller it was saved when the tool then rejected it |
| **identity** | answering "are you a bot?" truthfully, first time, every time |

Tool results are deliberately written as terse machine fragments rather than
fluent English — a model that paraphrases `NOT SAVED 'X': possibly the city
restated | NEED: site name` produces something visibly wrong, instead of
something plausible it can read down the phone.

---

## What a call produces

Every call writes a JSON artifact with the full transcript, per-turn audio RMS,
token usage and cost, plus the outcome of each guard. A resolved call also
upserts the enriched record:

```json
{
  "doctor_name": "Dr. Jane Okafor",
  "hospital_name": "Northside Medical Group",
  "branch": "Downtown Branch",
  "city": "Los Angeles",
  "source": "voice",
  "status": "partially_verified",
  "missing_for_complete": ["specialization"],
  "enriched_at": "2026-08-18T10:44:35+00:00",
  "enriched_by": "call-20260818-1613-6a8d"
}
```

`status` is `verified`, not `complete` — a successful call confirms a branch
from a second source, which is not the same as the record having every required
field. `missing_for_complete` names what is still absent rather than hiding it
behind a boolean.

---

## Layout

```
agents/voice/realtime_worker.py   the Realtime bridge, guards, artifacts
agents/voice/twilio_worker.py     webhooks, media stream, call routing
agents/voice/templates.py         call scripts (prompt + greeting + hint)
agents/voice/tools.py             save_branch / note_info / escalate
core/models.py                    Doctor, CallRecord, TranscriptTurn
core/config.py                    every tunable, with why it is set that way
test_realtime_protocol.py         the test suite (see below)
run_twilio.py                     place a call
check_realtime.py                 probe the live API without spending a call
```

`agents/email/` is an earlier email-first path, and `run_voice_local.py` drives
a local Piper/Whisper pipeline. Neither is on the phone path.

---

## Tests

```bash
python test_realtime_protocol.py
```

539 checks, fully offline — fake Twilio and fake OpenAI sockets, no API key, no
phone call, no cost. It drives complete scripted calls and asserts on the
actual wire traffic.

Two conventions worth knowing before adding to it:

**Every guard is written against a real failure, and mutation-proven.** Break
the fix, and the test that covers it must fail. Several guards in this repo
were once correct, live, and aimed just past where the bug was — a rejection
test that only enumerated one of two sources, a `/status` check that probed a
registry one code path never wrote, a repeat detector that compared sentences
when the repeated unit was the clause after the dash. Each passed cleanly while
the bug shipped.

**Find the population, prove you found it, judge every member.** Never assert
the absence of one exact string; that check passes by finding nothing. Where a
test inspects source, it enumerates by parsing rather than grepping, and
asserts a lower bound on what it found before judging.

---

## Running it

Setup is in **[SETUP.md](SETUP.md)** — Python, Twilio, ngrok, and the
environment variables in `.env.example`.

```bash
python run_twilio.py --doctor "Dr. Jane Okafor" \
                     --hospital "Northside Medical Group" \
                     --to "+1..."
```

`check_realtime.py` validates the model, the audio configuration and the
template against the live API without creating a response, so it costs nothing.

### Voice models

`data/voices/*.onnx` are Piper weights used only by the local pipeline
(`USE_REALTIME=false`). They are not committed — 345 MB of redistributable
weights, two of them over GitHub's 100 MB file limit. Fetch from
[rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) and drop the
`.onnx` and matching `.onnx.json` into `data/voices/`.

---

## Known limits

- **~1.9–3.1 s reply gap.** Measured floor is ~1.08 s of inference plus
  round-trip from India to OpenAI's US endpoints, on top of the VAD silence
  threshold. Only the threshold is tunable. Sub-2 s is the realistic target
  from here; sub-1 s needs a US-region server.
- **8 kHz μ-law.** The phone line is the ceiling on how natural any voice can
  sound, and no prompt or model change moves it.
- **The transcription hint is double-edged.** The proper nouns that make real US
  health-system names transcribe correctly are the same ones the transcriber
  can emit unprompted. The quiet-audio guard catches the near-silent case; a
  loud mis-transcription would still pass.
- **`is_complete()` requires a specialization** the CLI never supplies, so every
  record this agent resolves reports `partially_verified`. Surfaced in the
  artifact rather than silently downgraded, pending a decision on whether
  specialization is genuinely required.
