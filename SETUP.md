# 🧸 Setup Guide — From Zero to a Working System (Kid-Level)

This guide turns the code into a real, running system on your Windows 11 PC.
Follow it **top to bottom**. After every step there's a ✅ **"Did it work?"** check
so you never move forward broken.

Think of it like building a robot:
- 🧠 **Ollama** = the robot's brain (the AI)
- 🗄️ **PostgreSQL** = the robot's notebook (where it writes facts)
- ⚡ **Redis** = the robot's short-term memory (what it's thinking *right now*)
- ☎️ **LiveKit** = the robot's telephone (so it can call hospitals)
- 🐍 **Python** = the glue that holds the robot together

You do **not** need all of them at once. The system already works with **none** of
them (it uses pretend stand-ins). You turn each piece "real" one at a time.

---

## STEP 0 — Open the right window

1. Press the **Windows key**, type `powershell`, press **Enter**.
2. A blue/black window opens. This is the **terminal** — where you type commands.
3. Go into the project folder by copy-pasting this and pressing Enter:

```powershell
cd C:\Users\salom\OneDrive\Desktop\conversational_ai
```

✅ **Did it work?** The start of the line should now end with `conversational_ai>`.

> 💡 To paste in PowerShell: **right-click**. That's the paste button here.

---

## STEP 1 — Python libraries (the glue)

The robot's glue needs some extra parts. Install them all at once:

```powershell
pip install -r requirements.txt
```

This takes a few minutes. Lots of text scrolls by — that's normal.

Then install the pretend web-browser the crawler uses:

```powershell
python -m playwright install chromium
```

✅ **Did it work?** Run this:

```powershell
python run_validation.py --selftest
```

You should see a table with green ✅ marks. **If you see the table, the glue works!**
🎉 The system is already alive — it's just using pretend stand-ins for now.

---

## STEP 2 — The Brain 🧠 (Ollama + Qwen)

This lets the robot actually *think* about messy text.

1. Open your web browser, go to **https://ollama.com/download**
2. Click **Download for Windows**, run the installer, click Next → Next → Finish.
3. Ollama now runs quietly in the background (look for its icon near the clock).
4. Back in PowerShell, download the AI brain (this is a big file — be patient):

```powershell
ollama pull qwen3:8b
```

> 💡 We use `qwen3:8b` (the small brain) because it fits on a normal PC.
> The design's `qwen3:32b` (big brain) needs a very powerful computer. Same family,
> just smaller. You can change this later in the `.env` file.

✅ **Did it work?** Run:

```powershell
ollama run qwen3:8b "say hello in 3 words"
```

If it replies with a few words, **the brain is on!** Type `/bye` to exit.

---

## STEP 3 — The Notebook 🗄️ (PostgreSQL)

This is where verified doctors get permanently saved.

1. Go to **https://www.postgresql.org/download/windows/**
2. Click **"Download the installer"**, pick the newest version, run it.
3. During install it asks for a **password** — type `postgres` and **remember it**.
   (Keep every other setting default. Port stays **5432**.)
4. Finish the install. (You can skip "Stack Builder" at the end — click Cancel.)

Now make a notebook called `doctors`. In PowerShell:

```powershell
& "C:\Program Files\PostgreSQL\17\bin\createdb.exe" -U postgres doctors
```

> 💡 If your version isn't 17, change the number to match the folder inside
> `C:\Program Files\PostgreSQL\`. It will ask for the password (`postgres`).

✅ **Did it work?** Run:

```powershell
& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -d doctors -c "\dt"
```

If it says **"Did not find any relations"** — that's perfect! It means the empty
notebook exists. (The tables get created automatically when you run the pipeline.)

---

## STEP 4 — Tell the robot your secrets (.env file)

The robot reads its settings from a file called `.env`. Make it from the example:

```powershell
copy .env.example .env
notepad .env
```

Notepad opens. Change these lines to match what you set up:

```
LLM_MODEL=qwen3:8b
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/doctors
```

(If you used a different Postgres password, put it where the second `postgres` is.)

Press **Ctrl+S** to save, then close Notepad.

✅ **Did it work?** Run the whole pipeline for real now:

```powershell
python run_pipeline.py --selftest
```

Look at the top line. If it says `db backend: postgres` (not `json`) — **your robot
is now writing to the real notebook!** 🎉

✅ **Double-check the notebook has doctors in it:**

```powershell
& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -d doctors -c "SELECT doctor_name, branch, confidence_score FROM doctors;"
```

You should see John Smith, Jane Doe, Robert Lee with their branches. **The robot
remembered them!**

---

## STEP 5 — Short-term memory ⚡ (Redis) — *optional*

Only needed for **real phone calls** and **running many hospitals at once**. Skip
this if you're not making live calls yet.

Windows doesn't have official Redis, so use **Memurai** (Redis for Windows):

1. Go to **https://www.memurai.com/get-memurai**, download the free Developer edition.
2. Run the installer, click Next → Next → Finish. It starts automatically.

✅ **Did it work?** Run:

```powershell
python run_queue.py --selftest
```

If the top line says `celery mode: distributed (Redis)` instead of `EAGER` —
**short-term memory is connected!**

---

## STEP 6 — Email 📧 (so the robot can send real emails) — *optional*

1. Get a Gmail App Password (full steps in the chat history):
   - Turn on 2-Step Verification: https://myaccount.google.com/security
   - Make an app password: https://myaccount.google.com/apppasswords
   - Copy the 16-letter code (remove spaces).
2. Open `.env` again (`notepad .env`) and fill in:

```
EMAIL_ADDRESS=youremail@gmail.com
EMAIL_PASSWORD=your16lettercode
```

Save and close.

✅ **Did it work?** Run a real send:

```powershell
python run_email.py --to youremail@gmail.com --live
```

Check that Gmail inbox — if a "Verifying doctor branch" email arrived, **the robot
can send mail!**

---

## STEP 7 — The Telephone ☎️ (LiveKit) — *the hard one, do last*

This is the only advanced part. It lets the robot make **actual phone calls**.
You need three things working together: LiveKit (the switchboard), a SIP trunk (the
phone line), Whisper (ears) and CosyVoice (voice).

> ⚠️ **Honest heads-up:** this step is genuinely complex and costs a little money
> (a phone-line provider). Do the first 6 steps first and make sure everything else
> works. Then tackle this when you're ready. Here's the path:

1. **Make a LiveKit account (free tier):** https://cloud.livekit.io
   - It gives you a `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`.
   - (Or self-host: https://docs.livekit.io/home/self-hosting/ — advanced.)
2. **Get a phone line (SIP trunk)** from a provider like Twilio or Telnyx, and
   connect it to LiveKit following: https://docs.livekit.io/sip/
3. **Install the voice libraries:**

```powershell
pip install livekit-agents livekit-plugins-silero faster-whisper
```

4. **Install CosyVoice 2** (the open-source voice) by following its repo:
   https://github.com/FunAudioLLM/CosyVoice
5. **Add the LiveKit keys** to `.env`:

```
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
```

6. **Open `agents/voice/livekit_adapters.py`** and look for the `TODO(livekit)`
   notes. The exact function names in the LiveKit library change between versions,
   so you (or I, in a new session) confirm them against the version you installed by
   checking: https://docs.livekit.io/agents/

✅ **Did it work?** Start the voice worker:

```powershell
python -m agents.voice.worker
```

If it connects to LiveKit and waits for calls without crashing, **the telephone is
wired up.** Real calls then flow: phone → Whisper → Qwen → CosyVoice → phone.

---

## STEP 8 — Watch it work 📊 (Prometheus + Grafana) — *optional, fun*

This draws live charts of what the robot is doing.

1. **Prometheus** (collects numbers): download from
   https://prometheus.io/download/ → unzip → in that folder run:

```powershell
.\prometheus.exe --config.file="C:\Users\salom\OneDrive\Desktop\conversational_ai\monitoring\prometheus.yml"
```

2. **Tell the robot to publish numbers** (in a separate PowerShell window):

```powershell
cd C:\Users\salom\OneDrive\Desktop\conversational_ai
python run_queue.py --serve-metrics
```

3. **Grafana** (draws charts): download from https://grafana.com/grafana/download
   → install → open http://localhost:3000 (login admin/admin) →
   add a Prometheus data source pointing at `http://localhost:9090` →
   Dashboards → Import → upload `monitoring/grafana_dashboard.json`.

✅ **Did it work?** You'll see boxes like "Doctors Processed" and "Verified %"
filling up as you run the pipeline. **You can now watch your robot live!**

---

## 🏁 The finish line — how to run the real thing

Once Steps 1–4 are done, this one command runs the **entire system for real**,
crawling a hospital and saving doctors to PostgreSQL:

```powershell
python run_pipeline.py https://some-hospital.com/our-doctors
```

## 🆘 If something breaks

| What you see | What to do |
|---|---|
| `db backend: json` (wanted postgres) | PostgreSQL isn't running, or `.env` `DATABASE_URL` is wrong. Re-check Step 3 & 4. |
| `celery mode: EAGER` (wanted Redis) | Redis/Memurai isn't running. Re-check Step 5. |
| Crawler finds 0 doctors | The page may need a different URL (look for the hospital's "Our Doctors" page), or it blocks bots. |
| `ollama` command not found | Close and reopen PowerShell after installing Ollama. |
| Anything else | Run that agent's `--selftest` — if the self-test passes, the code is fine and the problem is the service/config. |

**Golden rule:** every part has a `--selftest`. If the self-test is green, the code
is healthy — any problem is a service that's off or a setting in `.env`.
