# archive/

Code that was load-bearing during R&D and is not on any live path now. Kept
rather than deleted because each of these was written against a real provider
integration and the notes in them are the record of why that provider was not
chosen.

Nothing here is imported by `agents/`, `core/`, or the entry points. It is not
on `sys.path` for any run, and the offline suite does not touch it. If
something here starts being needed again, move it back rather than importing
across this boundary.

## experiment_telephony/

The pre-Twilio provider survey and the local-audio pipeline's edges. The
project standardised on **Twilio + OpenAI Realtime (gpt-realtime-2)** on
2026-08, and these stopped being reachable then.

| file | what it was |
|---|---|
| `exotel_worker.py` | Exotel media-stream worker |
| `telnyx_worker.py` | Telnyx media-stream worker |
| `vonage_worker.py` | Vonage media-stream worker |
| `livekit_adapters.py` | LiveKit STT/TTS/LLM adapter shims |
| `worker.py` | the LiveKit-hosted agent entry point |
| `mic.py` | local microphone capture, for desk testing |
| `outbound.py` | the pre-Twilio outbound placement path |
| `tts_cosyvoice.py` | CosyVoice TTS backend |

Archived 2026-08-31. `agents/experiment/` keeps the four modules the classic
(`USE_REALTIME=false`) pipeline still reaches — `brain`, `prompts`,
`stt_whisper`, `tts_local` — and `memory.py` / `audio_utils.py` were promoted
to `core/` in the same pass, because the LIVE Realtime path depends on both:
`CallMemory` is the call sheet every save tool writes through, and
`audio_utils` supplies the μ-law codec and resampler on the Twilio leg.
