"""Shared data models — the contract that flows between agents.

The discovery agent produces `Doctor` records; verification, email, voice, and
validation agents read and enrich the same shape; the database agent persists it.
Keeping one schema here prevents each agent from inventing its own dict format.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import ClassVar, Optional

from pydantic import BaseModel, Field


class Source(str, Enum):
    """Where a piece of information came from — drives confidence scoring."""
    WEBSITE = "website"
    EMAIL = "email"
    VOICE = "voice"


class DoctorStatus(str, Enum):
    DISCOVERED = "discovered"            # found on a website, not yet verified
    COMPLETE = "complete"                # all required fields present
    MISSING_BRANCH = "missing_branch"    # needs follow-up (email -> voice)
    VERIFIED = "verified"                # confirmed by >=1 extra source
    PARTIALLY_VERIFIED = "partially_verified"


class Doctor(BaseModel):
    """One physician discovered from a hospital website."""
    doctor_name: str
    specialization: Optional[str] = None
    branch: Optional[str] = None
    city: Optional[str] = None

    hospital_name: Optional[str] = None
    source_url: Optional[str] = None
    source: Source = Source.WEBSITE

    status: DoctorStatus = DoctorStatus.DISCOVERED
    confidence: int = 0                  # 0-100, filled by the validation agent
    # datetime.utcnow is deprecated since 3.12 AND returns a NAIVE datetime, so
    # a UTC value compared against a naive local one differs silently by the
    # server's offset — 5.5 hours here. Same class as the datetime.now().hour
    # bug that greeted a US morning with "good evening".
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # When a later agent (voice, email) last wrote to this record. None means
    # nothing has enriched it since discovery.
    enriched_at: Optional[datetime] = None

    #: Fields is_complete() requires. Named once so the check and the
    #: explanation below can never drift apart. ClassVar, or pydantic takes it
    #: for a model field and refuses the class outright.
    REQUIRED_FOR_COMPLETE: ClassVar[tuple[str, ...]] = (
        "doctor_name", "specialization", "branch")

    def is_complete(self) -> bool:
        """Required fields for a usable directory entry."""
        return not self.missing_for_complete()

    def missing_for_complete(self) -> list[str]:
        """Which is_complete() requirements this record still fails.

        is_complete() answers with a bare bool, so a record that fails it gives
        no clue why — and that matters here more than it looks. run_twilio.py
        never passes a specialization, so EVERY doctor this voice agent
        resolves fails on that one field: a flawless call that confirms the
        branch still reads as incomplete, and anything downstream gating on
        is_complete() would score it as a failure.

        Naming the missing fields turns that from an invisible downgrade into
        something visible in the artifact. OPEN QUESTION for the client: is
        specialization genuinely required for a usable directory entry, or
        should it come out of this list? Until that is answered the voice path
        records PARTIALLY_VERIFIED with the reason attached rather than
        guessing either way.
        """
        return [f for f in self.REQUIRED_FOR_COMPLETE if not getattr(self, f, None)]


class DiscoveryResult(BaseModel):
    """What Agent 1 returns for a single crawled hospital page."""
    hospital_name: Optional[str] = None
    source_url: str
    doctors: list[Doctor] = Field(default_factory=list)
    error: Optional[str] = None


class TranscriptTurn(BaseModel):
    role: str                       # "agent" | "caller"
    text: str
    timestamp: Optional[str] = None  # "HH:MM:SS" wall-clock time of the turn
    # Loudest-300ms RMS of the audio this turn was transcribed from, for caller
    # turns only. Evidence that a HUMAN said these words rather than the
    # transcription model supplying them: a transcription hint is a prompt, and
    # its contents can surface as transcript on near-silent audio. The grounding
    # check trusts caller turns, so it needs to know which ones carried signal.
    # None for agent turns and for callers when the audio was not measured.
    audio_rms: Optional[float] = None


class CallRecord(BaseModel):
    """One recorded phone call — maps to the `calls` table (Agent 5 / Agent 8)."""
    call_id: str
    doctor_name: str
    hospital_name: Optional[str] = None
    branch: Optional[str] = None
    resolved: bool = False
    duration_seconds: Optional[int] = None
    cost_usd: Optional[float] = None
    transcript: list[TranscriptTurn] = Field(default_factory=list)
    summary: Optional[str] = None
    audio_path: Optional[str] = None
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
