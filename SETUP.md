# Setup Guide — from zero to a real phone call

This project makes **outbound phone calls to medical offices** and asks a
receptionist where a doctor practises, then writes the verified answer into a
directory record. Everything below is written for someone who has never seen
the repo.

There are two ways to run it. Do the first one.

| | **Golden path** *(this is the system)* | **Local/offline** *(legacy)* |
|---|---|---|
| speech | OpenAI Realtime API, `gpt-realtime-2` | Whisper + Piper, on your machine |
| phone | Twilio | Twilio, or nothing |
| needs | an OpenAI key and a Twilio trial | ~4 GB of models and a lot of patience |
| setting | `USE_REALTIME=true` *(the default)* | `USE_REALTIME=false` |
| section | **Steps 0–6**, below | [Local/offline experimentation](#localoffline-experimentation-use_realtimefalse), at the bottom |

> **If you read one thing:** the golden path does **not** need Ollama, Qwen,
> PostgreSQL, Redis, LiveKit, or Piper. Earlier versions of this guide told you
> to install all of them. They are not on the live path any more, and
> installing them will not get you a phone call.

---

## How a call actually flows

```
  you                run_twilio.py            Twilio              the office
   |                       |                    |                     |
   |--- run the command -->|                    |                     |
   |                       |--- place call ---->|------ rings ------->|
   |                       |                    |                     |
   |              +--------+--------+           |                 picks up
   |              | FastAPI :8000   |<-- POST /answer <----------------|
   |              |  (your machine) |--> TwiML <Connect><Stream> ----->|
   |              +--------+--------+           |                     |
   |                       |<=== WebSocket: 8 kHz u-law audio ========>|
   |                       |                    |                     |
   |              +--------+--------+                                 |
   |              | OpenAI Realtime |   speech in -> speech out        |
   |              | gpt-realtime-2  |   + tool calls (save_branch, ...)|
   |              +--------+--------+                                 |
   |                       |                                          |
   |            eight guards sit between the model and the record     |
   |                       |
   |            data/3 cases jsons/call-....json   transcript, cost, verdicts
   |            data/3 cases voice/call-....wav    both sides
   +----------- data/3 cases jsons/doctors.json    the enriched row
```

Twilio has to reach **your laptop** over the public internet, which is what the
tunnel in Step 3 is for. That is the only genuinely fiddly part.

---

## STEP 0 — Open the right window

Windows: press `Win`, type **PowerShell**, press Enter.
macOS/Linux: open **Terminal**.

Then go to the project folder and make a virtual environment:

```powershell
cd path\to\conversational_ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux: `source .venv/bin/activate`

Your prompt should now start with `(.venv)`. If it does not, nothing below will
work — everything installs into that environment.

---

## STEP 1 — Install the Python libraries

```powershell
pip install -r requirements.txt
```

Two of these are worth knowing about, because both fail in a confusing way:

- **`python-multipart`** — FastAPI needs it to read Twilio's webhook, which
  arrives as a form post. Without it, `/answer` returns a 400 and the callee
  hears silence on a call that otherwise looks fine. It is imported lazily, so
  nothing complains until a real call is already in flight.
- **`websockets`** — the transport for both legs (Twilio's media stream and the
  Realtime API). An old version connects and then drops frames.

---

## STEP 2 — Your two keys (`.env`)

Copy the template and open it:

```powershell
copy .env.example .env
notepad .env
```

macOS/Linux: `cp .env.example .env` then `nano .env`

The golden path needs **five values**. Everything else in that file belongs to
the legacy path and can stay empty.

| variable | where it comes from |
|---|---|
| `OPENAI_API_KEY` | platform.openai.com → API keys. **Must have Realtime API access.** |
| `TWILIO_ACCOUNT_SID` | Twilio Console, front page |
| `TWILIO_AUTH_TOKEN` | Twilio Console, front page |
| `TWILIO_FROM_NUMBER` | the number you bought, in `+15551234567` form |
| `SERVER_PUBLIC_URL` | your tunnel URL — **Step 3 fills this in for you** |

Twilio gives about $15 of trial credit without a card. A call costs roughly
**$0.11–0.15 per minute all-in** (OpenAI audio tokens plus Twilio minutes), so
trial credit is worth on the order of a hundred short calls.

> **Trial accounts can only call verified numbers.** Twilio Console → Phone
> Numbers → Verified Caller IDs → add the number you want to call, and answer
> the confirmation call. Skip this and the call fails with error 21219 before
> any of our code runs.

---

## STEP 3 — The tunnel (Twilio has to reach your laptop)

Twilio posts a webhook to your machine and then opens a WebSocket to it. Your
laptop has no public address, so a tunnel provides one.

**Install ngrok**, then in a **second terminal window** — leave it running:

```powershell
ngrok http 8000
```

Now, back in your first window:

```powershell
python update_ngrok_url.py
```

That reads the live tunnel URL and writes it into `SERVER_PUBLIC_URL` for you.
Do not copy it by hand.

> **The URL changes every time you restart the tunnel.** A stale
> `SERVER_PUBLIC_URL` fails *silently*: the call connects, Twilio's webhook goes
> to a dead host, and the callee hears nothing. `run_twilio.py` refuses to place
> a call when it detects a stale URL, which is the only reason that mistake is
> survivable. Re-run `update_ngrok_url.py` after every tunnel restart.

`cloudflared` works too — `update_ngrok_url.py` handles both. It is supported
because Windows Defender has been known to delete `ngrok.exe` on sight.

---

## STEP 4 — Preflight (free, no phone call)

```powershell
python check_realtime.py
```

This resolves your settings, renders the call template, checks the prompt-cache
split, then opens one WebSocket to the Realtime model and sends a single
`session.update`. It never asks for a response, so **no audio is generated and
nothing is billed** beyond the connection itself.

If this is green, your key, your model access, and your prompt are all fine.
Everything after this point is telephony.

---

## STEP 5 — Make the call

```powershell
python run_twilio.py --doctor "Dr. Jennifer" --hospital "New York Baptist Hospital" --specialty "Cardiology" --to "+15551234567"
```

| flag | required | why |
|---|---|---|
| `--doctor` | yes | the doctor to ask about |
| `--hospital` | yes | used to confirm you reached the right practice |
| `--specialty` | **effectively yes** | how a receptionist tells two doctors of the same name apart. Without it a resolved record can never reach COMPLETE — it files as PARTIALLY_VERIFIED however well the call went. |
| `--to` | yes | the number to ring, with `+` and country code |
| `--port` | no | defaults to 8000; must match your tunnel |

The command starts the FastAPI server **and** places the call. Watch the
terminal: it prints every turn, every guard verdict, the reply latency, and a
full cost breakdown when the call ends.

---

## STEP 6 — What you get back

| where | what |
|---|---|
| `data/3 cases jsons/call-....json` | the artifact: transcript, per-turn audio RMS, token and cost breakdown, every guard verdict, every refusal |
| `data/3 cases voice/call-....wav` | both sides of the call, on separate channels |
| `data/3 cases voice/twilio-call-....mp3` | Twilio's own recording, for adjudicating audio problems |
| `data/3 cases jsons/doctors.json` | the enriched directory row |

**These are not in git.** Call artifacts are outputs, and every transcript is a
receptionist who did not agree to be in a repository. They stay on your machine.

---

## The feedback loop — when the agent mishears

This is the part people miss, and it is the most useful tool in the repo.

The agent's guards refuse anything the caller did not demonstrably say. That is
the point: a fabricated directory row is worse than an empty one, because the
empty one is visibly a gap. But a guard can also be *too strict* and throw away
a real answer, and when that happens the call looks fine and the field is
simply missing.

**Do not guess at which phrasing broke.** Run the sweep:

```powershell
python check_refusals.py
```

It reads every artifact you have and reports refusals the call itself went on
to contradict. Two verdicts:

| verdict | meaning |
|---|---|
| **COST** | the field was refused and never landed. The guard did not delay the answer, it destroyed it. |
| **PREMATURE** | the field was refused and landed *later*. The caller was made to repeat themselves in words the probe finally recognised. |

Each finding hands you the caller's exact words. That string is the deliverable:

```
  COST      call-20260827-1130-ed9f   field=referral   refused after the words landed
      caller said : "It's depend upon situation"
      guard said  : referral='depends' - nothing the caller said since you asked ...
```

The workflow is fixed, and please follow it in order:

1. Run the sweep. Read the `caller said` string.
2. Widen the pattern in `agents/voice/objectives.py` so it reads that phrasing.
3. **Pin the exact phrase in `test_realtime_protocol.py`, and check that the
   test fails before your fix and passes after.** A guard written past the real
   bug ships silently broken — that has happened here more than once, which is
   why this step is not optional.
4. Re-run the sweep. The line stops appearing.

Useful flags:

```powershell
python check_refusals.py --since 20260831
python check_refusals.py --dir "some/other/dir"
```

Exit code is 1 when anything is flagged, so it can gate a batch.

---

## The offline test suite

```powershell
python test_realtime_protocol.py
```

Around three minutes, ~2,100 assertions, **no network and no API cost**. It
drives complete scripted calls through fake Twilio and OpenAI sockets and
asserts on the actual wire traffic. Exit code 0 means green.

Run it before every commit. If you have no call artifacts on disk it will
report `corpus sweep NOT run` — that is deliberate. The sweep needs real calls,
and a check that measures nothing while printing nothing is worse than one that
says so out loud.

---

## Choosing what the agent asks

`CALL_TEMPLATE` in `.env` picks the script:

| template | what it collects |
|---|---|
| `forage_data_collection` *(default)* | the branch only |
| `forage_ai_disclosed` | the same, and announces up front that it is automated |
| `provider_verification` | identity → branch → accepting new patients → scheduling → referral. Everything after the first is gated on identity coming back `confirmed`. |

A template declares its *objective* — the fields, their probes, and when each
one is required. The objective decides when a call is finished; no tool decides
that by name.

---

## If something breaks

| what you see | what to do |
|---|---|
| callee hears silence, call otherwise fine | stale tunnel URL. Re-run `python update_ngrok_url.py`, then `python check_realtime.py`. |
| Twilio error 21219 | trial account, unverified number. See Step 2's note. |
| `/answer` returns 400 | `python-multipart` missing. Re-run Step 1. |
| connection refused in `check_realtime.py` | your key has no Realtime API access — a normal OpenAI key does not get it automatically. |
| a field is missing from a call that clearly answered it | `python check_refusals.py`. That is exactly what it is for. |
| the agent repeats a question | look at `save_refusals` in the artifact. A refusal with no repair is a re-ask. |
| suite red on `corpus sweep NOT run` | you have no artifacts on disk. Expected on a fresh clone. |

**Golden rule:** `check_realtime.py` proves the model and the prompt.
`test_realtime_protocol.py` proves the code. If both are green, the problem is
telephony or `.env`.

---

## Local/offline experimentation (`USE_REALTIME=false`)

> **Legacy. Not the production path.** Kept because it runs without an OpenAI
> key and without a network, which is occasionally useful when working on turn
> logic. It is slower, noticeably worse at conversation, and none of the
> latency or cost numbers quoted anywhere in this repo describe it.

The classic pipeline is **VAD → Whisper → a local LLM → Piper**, in
`agents/experiment/`. Set `USE_REALTIME=false` and supply:

| what | how |
|---|---|
| **Ollama + a Qwen model** | install from ollama.com, then `ollama pull qwen3:8b`. Set `LLM_BASE_URL=http://localhost:11434/v1`, `LLM_API_KEY=ollama`, `LLM_MODEL=qwen3:8b`. |
| **Whisper** | `WHISPER_MODEL=small` is fine on a CPU. |
| **Piper voices** | ~345 MB of weights, **not in git** — they made the repo unpushable and were purged from history. Fetch from <https://huggingface.co/rhasspy/piper-voices>. |
| **PostgreSQL** *(optional)* | `DATABASE_URL=...`. Without it, `core/db.py` falls back to JSON files in `data/db/`, which is fine for one machine. |
| **Redis** *(optional)* | `REDIS_URL=...`. Without it, call memory is in-process. |

Run it with `python run_voice_local.py`, or `python run_voice.py --selftest`.

Related, also not on the live path:

- **`agents/email/`** — an earlier email-first approach to the same problem.
  `python run_email.py --selftest`.
- **`monitoring/`** — Prometheus and Grafana dashboards.
- **`archive/experiment_telephony/`** — the provider survey that lost to Twilio
  (Exotel, Telnyx, Vonage, LiveKit) plus the local-audio edges. Nothing imports
  it. See `archive/README.md` for what each one was and why it is kept.
