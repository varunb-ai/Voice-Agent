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
    # Latency: ~300-500ms vs ~2s. Cost: ~$0.06/min vs ~$0.01/min.
    use_realtime: bool = False

    # Which calling script to run — see agents/voice/templates.py
    call_template: str = "forage_data_collection"

    # Realtime model. Set explicitly rather than probing a fallback list, so the
    # cost breakdown below actually corresponds to the model that served the call.
    # Confirm the account has it: `python check_realtime.py --probe`.
    realtime_model: str = "gpt-realtime-2"

    # Cap on a single spoken response. The agent is meant to say one or two short
    # sentences per turn; without a cap a runaway response is billed as audio out.
    realtime_max_response_tokens: int = 400

    # Transcription model for the written transcript. NOT in the conversational
    # path — the agent hears the caller's audio directly — but it IS what the
    # grounding check compares a saved location against, so accuracy matters.
    # whisper-1 hallucinates confident text on quiet audio: a caller saying
    # "hello, how can I help you" at low volume was transcribed as "Okay, next
    # slide, please", a stock phrase from its training data. gpt-4o-transcribe
    # is markedly more reliable on telephony-grade input. Fall back to
    # "whisper-1" if the account lacks access.
    realtime_transcribe_model: str = "gpt-4o-transcribe"

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
