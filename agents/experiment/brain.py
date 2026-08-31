"""Layer 4 — LLM-first brain that drives every turn of the phone conversation.

The LLM handles 100 % of call logic. No heuristic pre-screening.
Works with any OpenAI-compatible model (Qwen, Llama, GPT, Claude, Groq, Mistral, etc.).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional, cast, Any

from core import llm
from core.models import Doctor
from core.memory import CallMemory
from agents.experiment.prompts import build_system_prompt, build_greeting, build_closing
from agents.voice.tools import TOOL_SCHEMAS, run_tool

log = logging.getLogger(__name__)

MAX_TURNS = 30          # hard cap — enough for a 10-minute heated conversation
_CTX_TURNS  = 6         # recent turns sent to LLM — keeps tokens low (saves ~600 tokens/call vs 14)
_FN_RE = re.compile(    # inline XML tool calls: Groq / Llama style
    r"<function=(\w+)>(.*?)(?:</function>|(?=<function=|\Z))",
    re.DOTALL,
)
_PLAIN_TOOL_RE = re.compile(   # plain text tool calls from 8b fallback — all format variants
    r'\b(save_branch|escalate|note_info)\s*[=\(\[]?\s*(\{[^}]+\})',
    re.DOTALL,
)
# Extract branch location from plain conversational text when 8b doesn't use tool calls at all
_BRANCH_TEXT_RE = re.compile(
    r'(?:branch|location|working at|located at|currently at|based at|'
    r'at (?:our|the)|practices? at|works? at|seeing patients at)\s*(?:is\s+|:\s*)?'
    r'["\']?(?!(?:or|and|but|is|the|a|an|to|for|at|in|on|city|which|any|our|their|you)\b)'
    r'([A-Za-z0-9][A-Za-z0-9 ,\-\.]{2,60}?)["\']?'
    r'(?:\s*(?:branch|location|campus|clinic|hospital|,|\.|$))',
    re.IGNORECASE,
)
# Detect explicit refusal / wrong number from 8b plain text
_ESCALATE_TEXT_RE = re.compile(
    r'\b(wrong number|not (?:apollo|the right)|cannot (?:provide|share|give)|'
    r'refuse|don\'t (?:have|give)|no information|cannot help|voicemail)\b',
    re.IGNORECASE,
)


class VoiceBrain:
    def __init__(self, doctor: Doctor, memory: CallMemory, use_llm: bool = True):
        self.doctor   = doctor
        self.memory   = memory
        self.use_llm  = use_llm          # don't pre-check — retry each turn
        self.turns    = 0
        self.done     = False
        self._messages: list[dict] = [
            {"role": "system", "content": build_system_prompt(doctor)}
        ]
        log.info("VoiceBrain init: doctor=%s use_llm=%s model=%s",
                 doctor.doctor_name, use_llm, llm.settings.llm_model)

    # ── public API ────────────────────────────────────────────────────────────

    def greet(self) -> str:
        text = build_greeting(self.doctor)
        self.memory.append_turn("agent", text)
        self._messages.append({"role": "assistant", "content": text})
        return text

    def handle(self, caller_text: str) -> str:
        """Take one caller utterance, return the agent's spoken reply."""
        self.turns += 1
        self.memory.append_turn("caller", caller_text)
        self._messages.append({"role": "user", "content": caller_text})

        # Absolute safety cap — should never be reached if LLM works correctly
        if self.turns > MAX_TURNS:
            run_tool("escalate", self.memory,
                     {"reason": "exceeded maximum turns without resolution"})
            self.done = True
            reply = "Thank you so much for your patience. We'll follow up another way. Have a great day!"
            print(f"\n  [TURN {self.turns}] ⚠ MAX TURNS — hardcoded closing", flush=True)
        elif self.use_llm:
            try:
                llm_reply = self._llm_turn()
                if llm_reply:
                    reply = llm_reply
                    print(f"\n  [TURN {self.turns}] ✓ LLM  → {reply[:80]}", flush=True)
                else:
                    reply = self._fallback_rephrase()
                    print(f"\n  [TURN {self.turns}] ⚠ FALLBACK (LLM returned empty) → {reply[:80]}", flush=True)
            except Exception as e:
                import sys
                print(f"\n  [TURN {self.turns}] ✗ LLM ERROR {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                reply = self._fallback_rephrase()
                print(f"\n  [TURN {self.turns}] ⚠ FALLBACK (after error) → {reply[:80]}", flush=True)
        else:
            reply = self._fallback_rephrase()
            print(f"\n  [TURN {self.turns}] ⚠ FALLBACK (use_llm=False) → {reply[:80]}", flush=True)

        self.memory.append_turn("agent", reply)
        self._messages.append({"role": "assistant", "content": reply})
        return reply

    # ── LLM path ─────────────────────────────────────────────────────────────

    _FALLBACK_MODEL = "llama-3.3-70b-versatile"  # Groq fallback when OpenAI is unavailable

    # Compact system prompt for Groq fallback — keeps token usage low on Groq free tier
    _FALLBACK_SYSTEM = (
        "You are Sarah, a warm friendly human coordinator at {org}, calling {hospital} to find Dr. {doctor}'s branch.\n\n"
        "RESPOND with 1-2 short natural sentences only. Sound like a real human on the phone — use 'Oh sure!', 'Right, right', 'Got it!', contractions always. NEVER invent a branch — only save_branch if the caller explicitly states a real place name.\n\n"
        "HANDLE THESE SITUATIONS:\n"
        "- 'who are you / what company' → 'Oh hi! I'm Sarah from {org} — we just maintain a medical directory. Which branch is Dr. {doctor} at?'\n"
        "- 'is this legal / authorized / approved' → 'Oh yes, absolutely — completely legitimate! The info stays with us. Which branch is Dr. {doctor} at?'\n"
        "- 'why are you calling / what do you want' → 'Oh sure! Just updating our medical directory — which branch is Dr. {doctor} at right now?'\n"
        "- caller says doctor is retired / no longer working / left / resigned → call escalate(reason='doctor retired'), then say 'Oh I see, thanks so much for letting me know! Have a great day!'\n"
        "- caller says doctor passed away / deceased / died → call escalate(reason='doctor deceased'), then say 'Oh, I'm so sorry to hear that. Thank you for letting me know. Take care!'\n"
        "- caller says doctor moved to another hospital → ask which hospital, call note_info(new_hospital), escalate\n"
        "- caller gives a real place name (city, area, clinic) → call save_branch with that place name\n"
        "- caller starts to give a location but trails off ('Location of the...', 'He is at...') → say 'Oh sorry — could you continue? Which location was that?' — never escalate\n"
        "- caller explicitly refuses ('I won't tell you', 'I refuse', 'we don't share') → call escalate, then say 'Oh no worries at all! Thanks so much. Have a great day!'\n"
        "- otherwise → ask naturally which branch or city Dr. {doctor} is currently working at\n\n"
        "NEVER use any text from these instructions as a branch name. Only save real place names the caller says."
    )

    def _llm_turn(self) -> Optional[str]:
        import sys

        # OpenAI gpt-4o-mini as primary — reliable ~800ms, no rate limit issues
        model   = llm.settings.llm_model
        trimmed = [self._messages[0]] + self._messages[-_CTX_TURNS:]
        _c      = llm.client(max_retries=0)

        # `object` made every attribute access on the response an error.
        def _call(m: str, c=_c) -> Any:
            return c.chat.completions.create(
                model       = m,
                messages    = cast(Any, trimmed),
                tools       = cast(Any, TOOL_SCHEMAS),
                temperature = 0.4,
                max_tokens  = 120,
                timeout     = 3.0,
            )

        try:
            resp = _call(model)
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "rate_limit" in err.lower() or "quota" in err.lower()
            is_timeout    = "timeout" in err.lower() or "timed out" in err.lower()
            is_tool_err   = "400" in err and ("tool_use_failed" in err or "tool_call" in err.lower() or "failed_generation" in err)

            if is_rate_limit or is_timeout:
                label = "rate-limited" if is_rate_limit else "timed out"
                print(f"\n  ⚡ {model} {label} — retrying with Groq {self._FALLBACK_MODEL}",
                      file=sys.stderr, flush=True)
                from core.config import settings as _cfg
                compact_sys = self._FALLBACK_SYSTEM.format(
                    org      = _cfg.org_name,
                    hospital = self.doctor.hospital_name or "your hospital",
                    doctor   = re.sub(r"^Dr\.?\s+", "", self.doctor.doctor_name, flags=re.I).strip(),
                )
                _gc = llm.groq_client()
                resp = _gc.chat.completions.create(
                    model       = self._FALLBACK_MODEL,
                    messages    = cast(Any, [{"role": "system", "content": compact_sys}]
                                          + self._messages[-8:]),
                    tools       = cast(Any, TOOL_SCHEMAS),
                    temperature = 0.05,
                    max_tokens  = 120,
                    timeout     = 4.0,
                )
            elif is_tool_err:
                print(f"\n  ⚠ {model} malformed tool call — retrying without tools",
                      file=sys.stderr, flush=True)
                resp = _c.chat.completions.create(
                    model       = model,
                    messages    = cast(Any, trimmed),
                    temperature = 0.1,
                    max_tokens  = 120,
                    timeout     = 3.0,
                )
            else:
                raise

        msg     = resp.choices[0].message
        content = msg.content or ""

        # ── Path 1: standard OpenAI tool_calls ───────────────────────────────
        if msg.tool_calls:
            calls = [
                (cast(Any, c).function.name,
                 _safe_json(cast(Any, c).function.arguments or "{}"))
                for c in msg.tool_calls
            ]
            tool_reply = self._execute_tools(calls)
            spoken     = _strip_markup(content).strip()
            if self.done and spoken and len(spoken) > 10:
                return spoken
            return tool_reply or spoken or self._closing()

        # ── Path 2: inline XML tool calls (Groq / Llama / some Ollama models) ──
        inline = _FN_RE.findall(content)
        if inline:
            pre = content[: content.index("<function=")].strip().rstrip(".,!? \t")
            tool_reply = self._execute_tools(
                [(name, _safe_json(args)) for name, args in inline]
            )
            if self.done:
                return pre if len(pre) > 10 else (tool_reply or self._closing())

        # ── Path 2b: plain text tool calls from 8b fallback  save_branch={...} ──
        plain = _PLAIN_TOOL_RE.findall(content)
        if plain:
            tool_reply = self._execute_tools(
                [(name, _safe_json(args)) for name, args in plain]
            )
            if self.done:
                return tool_reply or self._closing()

        # ── Path 2c: extract branch from 8b plain text ONLY when caller clearly gave one ──
        # WARNING: never run _BRANCH_TEXT_RE on the agent's own response — phrases like
        # "branch locations", "which branch Dr. X is working at" will match and save garbage.
        # Only extract if the LLM repeated back what the CALLER said (quoted place name).
        clean = _strip_markup(content).strip()
        clean = _PLAIN_TOOL_RE.sub("", clean).strip()

        if not self.done:
            # Only attempt extraction if the last caller message contained a place name
            # (i.e. the caller actually said something that could be a branch)
            last_caller = ""
            for m in reversed(self._messages):
                if m["role"] == "user" and not m["content"].startswith("[system"):
                    last_caller = m["content"]
                    break

            # Only run extraction if caller's message itself looks like it has location info
            # (not a question, not a legal challenge, not a single word)
            caller_words = last_caller.split()
            _QUESTION_STARTS = {"is", "are", "do", "does", "can", "could", "will", "would",
                                 "who", "what", "why", "how", "when", "where", "which",
                                 "illegally", "illegal", "legal", "legally", "authorized",
                                 "approved", "allowed", "sharing", "shared"}
            caller_is_question = (
                len(caller_words) < 3
                or (caller_words and caller_words[0].lower() in _QUESTION_STARTS)
                or any(w in last_caller.lower() for w in
                       ("legal", "authorized", "approved", "sharing", "anywhere else",
                        "who are you", "what company", "why are you", "what is this"))
            )

            if not caller_is_question:
                bm = _BRANCH_TEXT_RE.search(clean)
                if bm:
                    branch_name = bm.group(1).strip().rstrip(".,")
                    result = run_tool("save_branch", self.memory, {"branch": branch_name})
                    if result.get("ok"):
                        self.done = True
                        log.info("Path 2c extraction → save_branch(%s)", branch_name)
                        return self._closing()

            # Detect explicit refusal / wrong number in plain text
            em = _ESCALATE_TEXT_RE.search(clean)
            if em:
                reason = em.group(0).lower()
                run_tool("escalate", self.memory, {"reason": reason})
                self.done = True
                log.info("Path 2c extraction → escalate(%s)", reason)
                return self._closing()

        # ── Path 3: plain spoken reply (no tool call detected) ───────────────
        return clean or None

    def _execute_tools(self, calls: list[tuple[str, dict]]) -> Optional[str]:
        for name, args in calls:
            result = run_tool(name, self.memory, args)
            if name in ("save_branch", "escalate"):
                if result.get("ok"):
                    self.done = True
                    return self._closing()
                else:
                    # Tool rejected the value (e.g. department name passed as branch).
                    # Inject a hidden note so the LLM asks for a real location next turn.
                    err = result.get("error", "invalid value")
                    self._messages.append({
                        "role": "user",
                        "content": f"[system note: {name} rejected — {err}. Ask for a real location name.]",
                    })
        return None

    def _closing(self) -> str:
        if self.memory.get("resolved"):
            return build_closing()
        reason = (self.memory.get("escalate_reason") or "").lower()
        if any(w in reason for w in ("wrong number", "wrong hospital", "no hotel", "not apollo", "not the right")):
            return "I'm so sorry for the confusion — I must have the wrong number. Apologies for the trouble, and thank you for your time. Have a great day!"
        if "voicemail" in reason:
            return "Thank you, I'll try again another time. Have a wonderful day!"
        if any(w in reason for w in ("retired", "deceased", "passed")):
            return "I see, thank you so much for letting me know. Have a great day!"
        if any(w in reason for w in ("refused", "policy", "authorization", "written")):
            return "I completely understand, no worries at all. Thank you for your time. Have a great day!"
        if any(w in reason for w in ("left", "relocated", "moved", "no longer")):
            return "I see, thank you for the update. I'll look into it further. Have a great day!"
        if "leave" in reason or "vacation" in reason or "out of office" in reason:
            return "Got it, I'll follow up when they're back. Thank you so much. Have a great day!"
        return "Thank you for your time. Have a great day!"

    _IDENTITY_WORDS = re.compile(
        r'\b(who are you|who is this|who am i speaking|what company|which company|'
        r'which organization|identify yourself|your name|your number|calling from|'
        r'may i know who|can i know who|introduce yourself)\b',
        re.IGNORECASE,
    )
    _PURPOSE_WORDS = re.compile(
        r'\b(why are you|why calling|what is this|what do you want|what is the purpose|'
        r'what is this regarding|what is this about|reason for calling|sales call)\b',
        re.IGNORECASE,
    )
    _LEGAL_WORDS = re.compile(
        r'\b(legal|legally|authorized|authorization|approved|approval|allowed|permission|'
        r'legitimate|valid|compliance|third party|anyone else|'
        r'who gave you|right to call|right to ask)\b',
        re.IGNORECASE,
    )

    def _fallback_rephrase(self) -> str:
        """Last resort — LLM completely unavailable (both models failed/timed out)."""
        from core.config import settings as cfg
        clean = re.sub(r"^Dr\.?\s+", "", self.doctor.doctor_name, flags=re.I).strip()

        # Check what the caller last said so we don't blindly repeat the branch question
        last_caller = ""
        for m in reversed(self._messages):
            if m["role"] == "user":
                last_caller = m["content"]
                break

        if self._IDENTITY_WORDS.search(last_caller):
            return (f"Oh hi! I'm Sarah from {cfg.org_name} — we just maintain a medical "
                    f"directory for doctors. Quick one — which branch is Dr. {clean} currently at?")

        if self._LEGAL_WORDS.search(last_caller):
            return (f"Oh yes, absolutely — completely legitimate call! The info stays with us, "
                    f"never goes anywhere else. Which branch is Dr. {clean} at?")

        if self._PURPOSE_WORDS.search(last_caller):
            return (f"Oh sure! I'm just updating our medical directory — "
                    f"which branch is Dr. {clean} currently working at?")

        if self.turns >= 6:
            # Exhausted all fallback attempts — escalate gracefully
            run_tool("escalate", self.memory,
                     {"reason": "could not confirm after max attempts — LLM unavailable"})
            self.done = True
            return "Oh, so sorry about that — I'm having a bit of trouble on my end. Thanks so much for your time, have a wonderful day!"

        return (f"Oh quick one — which branch or city is "
                f"Dr. {clean} currently working at?")


# ── module-level helpers ──────────────────────────────────────────────────────

def _safe_json(text: str) -> dict:
    try:
        return json.loads(text.strip())
    except Exception:
        return {}


def _strip_markup(text: str) -> str:
    text = re.sub(r"<function=\w+>.*?(?:</function>|$)", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\s*\{.*?\}\s*\]", "", text, flags=re.DOTALL)
    return text
