"""Agent 3 — Email Agent (LangGraph orchestration).

Why LangGraph: the email flow is a state machine, not a straight line. A doctor
moves EMAIL_SENT -> WAITING_REPLY -> (parsed -> COMPLETED) or
(no reply -> FOLLOWUP_SENT -> ... -> VOICE_CALL). Encoding that as nested ifs rots
fast; LangGraph makes the states and transitions explicit.

    generate ─▶ send ─▶ check_reply ─┬─reply──▶ parse ─▶ done(COMPLETED)
                                      │
                                      └─no reply─▶ followup? ─┬─yes─▶ send (loop)
                                                              └─no──▶ escalate(VOICE_CALL)

`check_reply` calls the mailer; in dry-run / no-credentials it returns nothing, so
the flow naturally walks to follow-up then voice escalation — exactly the design's
Day1/Day2/Day3 path, collapsed for testing. A `reply_provider` hook lets tests and
the real scheduler inject replies.
"""
from __future__ import annotations

from enum import Enum
from typing import Callable, Optional

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from core.models import Doctor, DoctorStatus, Source
from agents.email.generation import generate
from agents.email.mailer import send_email, fetch_replies_from, IncomingReply
from agents.email.replyparser import parse_reply

MAX_FOLLOWUPS = 1            # Day1 initial + Day2 follow-up, then escalate on Day3


class EmailStatus(str, Enum):
    EMAIL_SENT = "EMAIL_SENT"
    WAITING_REPLY = "WAITING_REPLY"
    FOLLOWUP_SENT = "FOLLOWUP_SENT"
    VOICE_CALL = "VOICE_CALL"        # escalate to Agent 4
    COMPLETED = "COMPLETED"


# A reply_provider returns the reply body for a doctor, or None if none yet.
ReplyProvider = Callable[[Doctor], Optional[str]]


class EmailState(BaseModel):
    doctor: Doctor
    to_addr: str
    status: EmailStatus = EmailStatus.EMAIL_SENT
    followups_sent: int = 0
    use_llm: bool = False
    dry_run: bool = True
    found_branch: Optional[str] = None
    log: list[str] = Field(default_factory=list)

    # Non-serialized hook for injecting replies (tests / scheduler). Excluded from dumps.
    model_config = {"arbitrary_types_allowed": True}


# ── Nodes ────────────────────────────────────────────────────────────────

def _make_nodes(reply_provider: ReplyProvider):

    def node_send(state: EmailState) -> EmailState:
        subject, body = generate(state.doctor, use_llm=state.use_llm)
        send_email(state.to_addr, subject, body, dry_run=state.dry_run)
        if state.followups_sent == 0:
            state.status = EmailStatus.EMAIL_SENT
            state.log.append(f"sent initial email to {state.to_addr}")
        else:
            state.status = EmailStatus.FOLLOWUP_SENT
            state.log.append(f"sent follow-up #{state.followups_sent}")
        return state

    def node_check_reply(state: EmailState) -> EmailState:
        state.status = EmailStatus.WAITING_REPLY
        body = reply_provider(state.doctor)
        if body:
            branch, city = parse_reply(body, state.doctor.doctor_name, use_llm=state.use_llm)
            state.found_branch = branch
            if city:
                state.doctor.city = city
            state.log.append(f"reply received; parsed branch={branch!r}")
        else:
            state.log.append("no reply yet")
        return state

    def node_parse_done(state: EmailState) -> EmailState:
        state.doctor.branch = state.found_branch
        state.doctor.source = Source.EMAIL
        state.doctor.status = DoctorStatus.COMPLETE
        state.status = EmailStatus.COMPLETED
        state.log.append("branch confirmed via email -> COMPLETED")
        return state

    def node_followup(state: EmailState) -> EmailState:
        state.followups_sent += 1
        state.log.append(f"scheduling follow-up #{state.followups_sent}")
        return state

    def node_escalate(state: EmailState) -> EmailState:
        state.status = EmailStatus.VOICE_CALL
        state.doctor.status = DoctorStatus.MISSING_BRANCH
        state.log.append("no reply after follow-ups -> escalate to Voice Agent")
        return state

    return node_send, node_check_reply, node_parse_done, node_followup, node_escalate


# ── Edges (routing) ──────────────────────────────────────────────────────

def _route_after_check(state: EmailState) -> str:
    if state.found_branch:
        return "parse_done"
    if state.followups_sent < MAX_FOLLOWUPS:
        return "followup"
    return "escalate"


def build_graph(reply_provider: ReplyProvider):
    send, check, parse_done, followup, escalate = _make_nodes(reply_provider)

    g = StateGraph(EmailState)
    g.add_node("send", send)
    g.add_node("check_reply", check)
    g.add_node("parse_done", parse_done)
    g.add_node("followup", followup)
    g.add_node("escalate", escalate)

    g.set_entry_point("send")
    g.add_edge("send", "check_reply")
    g.add_conditional_edges(
        "check_reply",
        _route_after_check,
        {"parse_done": "parse_done", "followup": "followup", "escalate": "escalate"},
    )
    g.add_edge("followup", "send")        # follow-up loops back to send
    g.add_edge("parse_done", END)
    g.add_edge("escalate", END)
    return g.compile()


def _default_reply_provider(doctor: Doctor) -> Optional[str]:
    """Real mode: read the mailbox for a reply from this doctor's hospital."""
    replies: list[IncomingReply] = fetch_replies_from(doctor.hospital_name or "", dry_run=False)
    return replies[0].body if replies else None


def run_email_flow(
    doctor: Doctor,
    to_addr: str,
    *,
    use_llm: bool = False,
    dry_run: bool = True,
    reply_provider: Optional[ReplyProvider] = None,
) -> EmailState:
    """Run the email state machine for one doctor to a terminal state."""
    provider = reply_provider or (lambda d: None) if dry_run else (reply_provider or _default_reply_provider)
    graph = build_graph(provider)
    initial = EmailState(doctor=doctor, to_addr=to_addr, use_llm=use_llm, dry_run=dry_run)
    final = graph.invoke(initial)
    # langgraph returns a dict-like; coerce back to the model.
    return EmailState(**final) if not isinstance(final, EmailState) else final
