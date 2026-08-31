"""Layers 1 & 2 — Telephony (LiveKit SIP) + Voice Framework (LiveKit Agents).

This is the integration shell that turns the offline-testable VoiceBrain into a
real phone agent. It wires:

    SIP call ──▶ LiveKit room ──▶ [VAD ─ Whisper STT ─ Qwen LLM ─ CosyVoice TTS]

LiveKit handles streaming, interruptions, and turn-taking (so we don't hand-write
listen()/detect_stop()/play_voice()). Qwen runs via the OpenAI-compatible plugin
pointed at Ollama; STT/TTS use our open-source adapters.

RUN (after `pip install livekit-agents livekit-plugins-silero livekit-plugins-openai`
and a running LiveKit server + SIP trunk):

    python -m agents.experiment.worker dev

Inbound calls: configure a LiveKit SIP inbound trunk + dispatch rule to route the
PSTN number into a room this worker joins. Outbound (call a hospital): use the
LiveKit SIP CreateSIPParticipant API with the hospital's number.

NOTE: livekit-agents APIs differ across versions; this targets the 1.x AgentSession
API. Confirm method names against the installed version before first run.
"""
from __future__ import annotations

import os

from core.config import settings
from core.models import Doctor
from core.memory import CallMemory
from agents.experiment.prompts import build_system_prompt, build_greeting


def build_session(doctor: Doctor):
    """Construct a LiveKit AgentSession wired to our open-source models."""
    from livekit.agents import AgentSession  # type: ignore[import-not-found]  # optional runtime dep
    from livekit.plugins import openai, silero  # type: ignore[import-not-found]  # optional runtime dep

    # Custom STT/TTS adapters wrapping Whisper + CosyVoice are registered here.
    # For first bring-up you can start with openai-compatible STT/TTS pointed at
    # local servers, then swap in the dedicated adapters below.
    from agents.experiment.livekit_adapters import WhisperSTT, CosyVoiceTTS

    return AgentSession(
        vad=silero.VAD.load(),
        stt=WhisperSTT(),
        llm=openai.LLM(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        ),
        tts=CosyVoiceTTS(),
    )


async def entrypoint(ctx):
    """LiveKit Agents entrypoint — one call = one room = one doctor lookup."""
    from livekit.agents import Agent  # type: ignore[import-not-found]  # optional runtime dep

    await ctx.connect()

    # The doctor to ask about is passed via room metadata (set when the call is
    # dispatched). Falls back to a placeholder for manual testing.
    doctor = _doctor_from_metadata(ctx)
    memory = CallMemory(call_id=ctx.room.name)
    memory.update(
        doctor=doctor.doctor_name,
        specialization=doctor.specialization,
        hospital=doctor.hospital_name,
    )

    session = build_session(doctor)
    agent = Agent(instructions=build_system_prompt(doctor))
    await session.start(agent=agent, room=ctx.room)
    await session.say(build_greeting(doctor))
    # LiveKit now drives the loop; tool calls (save_branch/escalate) update memory.
    # On hangup, a shutdown hook should push memory -> validation -> database.


def _doctor_from_metadata(ctx) -> Doctor:
    import json
    meta = getattr(ctx.room, "metadata", "") or "{}"
    try:
        data = json.loads(meta)
    except Exception:
        data = {}
    return Doctor(
        doctor_name=data.get("doctor_name", "the doctor"),
        specialization=data.get("specialization"),
        hospital_name=data.get("hospital_name"),
    )


def main() -> None:
    from livekit.agents import cli, WorkerOptions  # type: ignore[import-not-found]  # optional runtime dep
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


if __name__ == "__main__":
    main()
