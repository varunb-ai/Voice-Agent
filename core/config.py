"""Central configuration, loaded once from the .env file.

Every agent imports `settings` from here instead of reading os.environ directly,
so there is a single place that documents what the system needs.
"""
from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Accepted by the Realtime API. marin/cedar are gpt-realtime-2 only.
REALTIME_VOICES = {
    "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse",
    "marin", "cedar",
}

# The persona name must match the voice the callee actually hears. Switching to
# cedar while the script still said "this is Sarah" derailed an entire call:
#   caller: "Oh, why is your name Sarah? I think you're a boy, right?"
#   caller: "But first answer me this, why are you keeping a girl and being a boy?"
# Three of six caller turns went on it and the branch never came up. Voice and
# name were two independent settings with nothing checking they agreed.
#
# Derived from the voice rather than configured separately, so they cannot drift
# apart. Names chosen to be common, unremarkable and clear at 8kHz — the point of
# a persona name is that nobody asks about it.
VOICE_PERSONA = {
    "marin":   "Sarah",
    "coral":   "Sarah",
    "shimmer": "Sarah",
    "sage":    "Sarah",
    "cedar":   "David",
    "ash":     "David",
    "echo":    "David",
    "ballad":  "David",
    "verse":   "David",
    "alloy":   "Alex",   # neutral voice, neutral name
}


def persona_for_voice(voice: str) -> str:
    """The name the agent gives, matched to the voice the callee hears."""
    return VOICE_PERSONA.get((voice or "").strip().lower(), "Alex")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM (Qwen3-32B via Ollama, OpenAI-compatible)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen3:32b"

    # PostgreSQL
    database_url: str = "postgresql://postgres:postgres@localhost:5432/doctors"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Email (Agent 3)
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    imap_host: str = "imap.gmail.com"
    imap_port: int = 993
    email_address: str = ""
    email_password: str = ""

    # Voice Agent — LiveKit (telephony + agents framework)
    livekit_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""

    # Vonage / Nexmo (India phone calls — $2 free trial, no card, no KYC)
    vonage_api_key: str = ""             # Dashboard -> API Settings -> API Key
    vonage_api_secret: str = ""          # Dashboard -> API Settings -> API Secret
    vonage_application_id: str = ""      # Dashboard -> Applications -> App ID
    vonage_private_key_path: str = "private.key"  # downloaded when creating app
    vonage_from_number: str = ""         # your Vonage number e.g. +914000000000

    # Exotel (India phone calls — free trial, no card needed)
    exotel_account_sid: str = ""        # Dashboard -> API Credentials -> Account SID
    exotel_api_key: str = ""            # Dashboard -> API Credentials -> API Key
    exotel_api_token: str = ""          # Dashboard -> API Credentials -> API Token
    exotel_from_number: str = ""        # your ExoPhone Indian number e.g. +914020000000
    exotel_app_id: str = ""             # App Bazaar -> your voicebot app ID

    # Twilio (real phone calls — $15 free trial, no card needed)
    twilio_account_sid: str = ""        # Console -> Account SID
    twilio_auth_token: str = ""         # Console -> Auth Token
    twilio_from_number: str = ""        # your Twilio trial number e.g. +12125550100
    # Verify X-Twilio-Signature on every inbound webhook. Leave ON — the public
    # endpoints are otherwise callable by anyone who learns the tunnel URL.
    # Only turn off for local replay testing with fabricated webhook payloads.
    twilio_validate_webhooks: bool = True

    # SignalWire (real phone calls — $5 free trial, no card needed)
    signalwire_space: str = ""          # e.g. myproject.signalwire.com
    signalwire_project_id: str = ""     # Dashboard -> API -> Project ID
    signalwire_api_token: str = ""      # Dashboard -> API -> API Token
    signalwire_from_number: str = ""    # number you bought in SignalWire

    # Telnyx (real phone calls — SIP/PSTN)
    telnyx_api_key: str = ""            # from telnyx.com dashboard → API Keys
    telnyx_from_number: str = ""        # your Telnyx number e.g. +12125550100
    telnyx_connection_id: str = ""      # Call Control Application ID (Telnyx dashboard)

    # Agent identity — shown to receptionists who ask
    org_name: str = "Definitive Healthcare"
    callback_number: str = "1-800-555-0100"
    callback_email: str = "directory@definitivehc.com"

    # Public server URL — set to your ngrok URL while testing
    # e.g.  SERVER_PUBLIC_URL=https://abc123.ngrok-free.app
    server_public_url: str = "https://your-ngrok-url.ngrok-free.app"

    # Voice models (open source)
    whisper_model: str = "small"        # CPU-friendly default; "large-v3" for prod/GPU
    cosyvoice_model_dir: str = ""        # local path to a downloaded CosyVoice 2 model

    # OpenAI API key — used for primary LLM (gpt-4o-mini), STT (whisper-1), and TTS (nova)
    openai_api_key: str = ""

    # Groq API key — used for fallback LLM (llama-3.3-70b) when OpenAI is unavailable
    groq_api_key: str = ""

    # Use OpenAI Realtime API instead of the classic STT→LLM→TTS pipeline.
    # Measured on live calls: ~2s agent response latency (range 1.9-3.4s),
    # $0.06-0.12 per completed call. Earlier comments here claimed
    # ~300-500ms; that figure was never measured and is contradicted by
    # every call we have recorded.
    # Defaults describe the path this project actually runs. They previously
    # described the retired classic pipeline, so a fresh clone booted into a
    # different system than the one being tested.
    use_realtime: bool = True

    # What to do with caller audio while the agent is speaking.
    #   "pass"   — forward it. The caller can interrupt, because OpenAI's VAD
    #              only fires on audio it receives. Relies on near_field noise
    #              reduction and semantic_vad to separate speech from line echo,
    #              neither of which existed when this gate was written.
    #   "energy" — forward only frames above realtime_echo_rms; real speech
    #              passes, quiet line echo does not.
    #   "drop"   — discard them. No echo, but the agent CANNOT be interrupted:
    #              no audio reaches OpenAI, so speech_started never fires and
    #              the barge-in handler is unreachable.
    realtime_echo_gate: str = "pass"

    # Emit a short "mm-hm" while the caller is mid-utterance, injected straight
    # into the Twilio stream rather than generated by the model. See
    # agents/voice/backchannel.py.
    #
    # OFF BY DEFAULT, deliberately. It has never run on a phone line, and there
    # is a specific untested interaction: a callee on SPEAKERPHONE may have
    # their mic pick our backchannel back up. realtime_echo_gate is "pass", so
    # caller frames are forwarded to OpenAI unconditionally, and backchannels
    # do not set sess.agent_speaking — so nothing would suppress the echo and
    # the agent could transcribe its own noise as something the caller said.
    #
    # It is off so that the next call tests ONE new thing (the hint-echo
    # quarantine) instead of two. Turn it on for the call after, with one
    # variable changed, and listen for the agent answering its own "mm-hm".
    realtime_backchannels: bool = False
    realtime_echo_rms: float = 0.020

    # How many times the agent may ask for the location before it must give up.
    #
    # A live call asked six times in 111 seconds and never got an answer. That
    # was not a phrasing failure — the caller engaged throughout but never
    # refused, never said they did not know, was not a wrong number and was not
    # voicemail, so NONE of the prompt's escalation triggers matched. The
    # standing instruction is "never close until you have saved a location or
    # escalated", and with no exit condition the only thing left to do was ask
    # again. The agent behaved exactly as specified.
    #
    # A budget is the missing condition, and unlike a phrasing rule it is
    # enforceable rather than hoped for.
    realtime_max_location_asks: int = 4

    # Which calling script to run — see agents/voice/templates.py
    call_template: str = "forage_data_collection"

    # Realtime model. Set explicitly rather than probing a fallback list, so the
    # cost breakdown below actually corresponds to the model that served the call.
    # Confirm the account has it: `python check_realtime.py --probe`.
    realtime_model: str = "gpt-realtime-2"

    # Cap on a single spoken response. The agent is meant to say one or two short
    # sentences per turn; without a cap a runaway response is billed as audio out.
    #
    # RAISED 400 -> 1500 on 2026-08-20. The cap counts AUDIO tokens as well as
    # text, which is easy to forget: audio runs ~20 tok/s of speech, so 400 is
    # not "a long paragraph", it is roughly one ordinary spoken turn plus its
    # transcript. On call-20260820-1230 it truncated a live response mid-turn:
    #
    #   [12:31:28] AGENT: I'm calling on behalf of Definitive Healthcare, and
    #                     yes, this is an automated call. ...
    #   [Realtime] ⚠️  response incomplete: max_output_tokens
    #
    # That was the answer to "are you a real person or is this a recording?" —
    # the disclosure the prompt calls the one line you do not cross, and the
    # one several US states regulate. A cap that can cut it off is the most
    # expensive setting in this file.
    #
    # 1500 is picked against the longest LEGITIMATE turn, not against the
    # average. That is the voicemail message: organisation, doctor, purpose,
    # and an email address read out character by character — call it 25s of
    # speech, ~500 audio tokens plus transcript, so ~650. 1500 clears it with
    # margin while still bounding a runaway to roughly a minute of audio, which
    # the barge-in and one-spoken-item guards would catch long before.
    realtime_max_response_tokens: int = 1500

    # Transcription model for the written transcript. NOT in the conversational
    # path — the agent hears the caller's audio directly — but it IS what the
    # grounding check compares a saved location against, so accuracy matters.
    # whisper-1 hallucinates confident text on quiet audio: a caller saying
    # "hello, how can I help you" at low volume was transcribed as "Okay, next
    # slide, please", a stock phrase from its training data. gpt-4o-transcribe
    # is markedly more reliable on telephony-grade input. Fall back to
    # "whisper-1" if the account lacks access.
    realtime_transcribe_model: str = "gpt-4o-transcribe"

    # ── Audio path ───────────────────────────────────────────────────────────
    # "pcmu"  — g711 μ-law, the format Twilio already speaks. Passes through
    #           untouched: no μ-law decode, no 8k->24k resample in, no 24k->8k
    #           resample out. Two fewer resamples per 20ms frame.
    # "pcm"   — PCM16 24kHz. Requires converting every frame in both directions.
    # Confirm the account/API accepts pcmu with: python check_realtime.py --audio-probe
    realtime_audio_format: str = "pcmu"

    # "near_field" | "far_field" | "off".
    # Untested. The earlier justification ("a handset is a near-field mic") was
    # a guess: the model never receives handset audio, it receives 8kHz μ-law.
    # Which setting actually helps on telephony is an empirical question.
    realtime_noise_reduction: str = "near_field"

    # "server_vad"   — pure silence timing. silence_duration_ms is additive dead
    #                  time on every turn, and too low cuts people off mid-word.
    # "semantic_vad" — the model scores whether the speaker has actually
    #                  finished, so it can respond fast without truncating a
    #                  pause. Designed for exactly this trade-off.
    # Probe confirmed the account accepts semantic_vad. Switched: server_vad
    # forced a choice between cutting people off (360ms interrupted a caller
    # mid-sentence) and adding dead time to every turn (550ms). Semantic
    # detection removes the trade-off rather than tuning a threshold.
    realtime_turn_detection: str = "server_vad"
    # server_vad only: silence after the caller stops before a response starts.
    # Additive dead time on every turn, but NOT the whole gap and not the
    # dominant term. 360ms cut people off mid-sentence; 550ms was the
    # over-correction.
    #
    # The interpolation that used to live here was wrong, and wrong in a way
    # worth naming: from "500ms -> 0.71s, 1000ms -> 1.47s" it concluded 700ms
    # lands near 1.0s. That treats the gap as proportional to this setting.
    # It is not. The gap is
    #
    #     silence_ms  +  model inference  +  round trip India->US
    #
    # and only the first term moves when you change this number. Measured on
    # live calls at 700ms, the agent's first audio arrived 1.19s
    # (call-20260818-1112) and 2.44s (call-20260818-1338) after
    # response.create — and that is AFTER this silence has already elapsed. So
    # the observed caller-stops-to-agent-speaks gap is ~1.9-3.1s, two to three
    # times what the interpolation predicted.
    #
    # Consequence for tuning: 700 -> 400 buys about 300ms off a ~2.5s gap. Real,
    # worth having, and nowhere near enough to reach 1s. The fixed component
    # dominates and is not tunable from here — it is the floor for this
    # architecture until the server sits closer to the callee. Sub-2s is the
    # honest target, not sub-1s.
    #
    # Same class as the "~300-500ms" latency banner already corrected: a number
    # derived by reasoning rather than measured, which then gets trusted.
    # check_realtime.py can measure variants against the live API for free.
    realtime_silence_ms: int = 700
    # semantic_vad only: "low" | "medium" | "high" | "auto".
    # low = gives the speaker more thinking time, high = chunks quickly.
    # "low" chosen because callers here pause mid-answer while looking
    # something up, and being interrupted is worse than a slightly late reply.
    realtime_vad_eagerness: str = "low"

    # ── Realtime pricing, USD per 1M tokens ──────────────────────────────────
    # Set for gpt-realtime-2, from developers.openai.com/api/docs/pricing,
    # checked 2026-08-05. The old code hardcoded preview-era rates
    # ($100/$200 audio, $5/$20 text) regardless of which model actually served
    # the call — that is roughly 3x the real audio rate.
    #
    # gpt-realtime-2 and gpt-realtime differ in ONE value:
    #     text output   gpt-realtime-2 $24.00   gpt-realtime $16.00
    # Everything else is identical across the two. If REALTIME_MODEL changes,
    # change price_text_out to match.
    price_audio_in: float = 32.0
    price_audio_in_cached: float = 0.40
    price_audio_out: float = 64.0
    price_text_in: float = 4.00
    price_text_in_cached: float = 0.40
    price_text_out: float = 24.0   # gpt-realtime-2; use 16.0 for gpt-realtime
    # Telephony, USD per minute — DESTINATION-DEPENDENT, not a global rate.
    # Currently set for US -> India mobile, which is the quality-test target.
    # US -> US is roughly a third of this. Change it when the target changes,
    # or the per-minute figure in the cost breakdown will be wrong.
    price_telephony_per_min: float = 0.0165

    # Agent's spoken voice — OpenAI Realtime API (USE_REALTIME=true)
    #
    # marin  — brighter female, professional register   } gpt-realtime-2 ONLY.
    # cedar  — warm mid-range male, professional        } OpenAI recommends
    #                                                     these for best quality
    # Legacy catalog (works on any Realtime model, consumer-casual register):
    #   alloy, ash, ballad, coral, echo, sage, shimmer, verse
    #
    # marin/cedar are trained for this model and handle natural pauses and
    # fillers far better than the legacy voices. They will FAIL on older
    # Realtime models — check_realtime.py flags the mismatch.
    realtime_voice: str = "marin"

    # Language Sarah speaks during the call
    # Options: english | hindi | telugu
    agent_language: str = "english"

    # Agent's spoken voice (Piper TTS — only used when USE_REALTIME=false)
    piper_voice: str = "en_US-lessac-high"  # see agents/voice/tts_local.py for options
    # "random"        = pick a random voice each call
    # "random_female" = random female voice
    # "random_male"   = random male voice

    # Legacy pyttsx3 settings (no longer used)
    voice_gender: str = "female"
    voice_age: str = "adult"
    voice_rate: int = 0
    voice_volume: float = 1.0


    @field_validator("realtime_voice")
    @classmethod
    def _check_voice(cls, v: str) -> str:
        """Reject an unusable voice at import, not mid-call.

        An empty or misspelled REALTIME_VOICE is accepted everywhere locally and
        then rejected by session.update AFTER the callee has already picked up —
        they get a connected call with silence, and it costs a real call to find
        out. Fail before anything dials.
        """
        v = (v or "").strip()
        if v not in REALTIME_VOICES:
            raise ValueError(
                f"REALTIME_VOICE={v!r} is not a valid Realtime voice. "
                f"Choose one of: {', '.join(sorted(REALTIME_VOICES))}. "
                f"marin and cedar require REALTIME_MODEL=gpt-realtime-2."
            )
        return v


settings = Settings()
