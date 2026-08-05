"""Thin wrapper around the open-source Qwen3 model served by Ollama.

Ollama exposes an OpenAI-compatible API, so we reuse the `openai` client purely
as an HTTP client pointed at localhost. Swapping to the 32B tag (or a hosted
open-source endpoint) later means changing only `.env`, not this code.
"""
from __future__ import annotations

import json
from typing import Optional

from openai import OpenAI

from core.config import settings

_client: Optional[OpenAI] = None
_groq_client: Optional[OpenAI] = None

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def client(max_retries: int = 2) -> OpenAI:
    global _client
    if max_retries != 2:
        return OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key,
                      max_retries=max_retries)
    if _client is None:
        _client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    return _client


def groq_client(max_retries: int = 0) -> OpenAI:
    """OpenAI-compatible client pointed at Groq — used as fallback brain."""
    global _groq_client
    if _groq_client is None:
        _groq_client = OpenAI(base_url=_GROQ_BASE_URL, api_key=settings.groq_api_key,
                              max_retries=max_retries)
    return _groq_client


def chat(prompt: str, system: str = "", temperature: float = 0.0) -> str:
    """Single-turn completion. Returns the model's text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = client().chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def chat_json(prompt: str, system: str = "", temperature: float = 0.0) -> dict:
    """Completion that must return JSON. Asks for json_object format and parses it.

    Used by verification / extraction agents only as a *fallback* when the cheap
    deterministic rules can't decide — never on the happy path.
    """
    full_system = (system + "\nRespond with valid JSON only, no prose.").strip()
    resp = client().chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": full_system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Last resort: pull the first {...} block out of the text.
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            return json.loads(raw[start : end + 1])
        raise


def is_available() -> bool:
    """True if the configured LLM endpoint answers."""
    import logging
    try:
        client().models.list()
        logging.getLogger(__name__).info(
            "LLM available: %s @ %s", settings.llm_model, settings.llm_base_url
        )
        return True
    except Exception as e:
        logging.getLogger(__name__).warning(
            "LLM not available (%s @ %s): %s",
            settings.llm_model, settings.llm_base_url, e,
        )
        return False
