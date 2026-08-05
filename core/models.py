"""Shared data models — the contract that flows between agents.

The discovery agent produces `Doctor` records; verification, email, voice, and
validation agents read and enrich the same shape; the database agent persists it.
Keeping one schema here prevents each agent from inventing its own dict format.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

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
    discovered_at: datetime = Field(default_factory=datetime.utcnow)

    def is_complete(self) -> bool:
        """Required fields for a usable directory entry."""
        return bool(self.doctor_name and self.specialization and self.branch)


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
    recorded_at: datetime = Field(default_factory=datetime.utcnow)
