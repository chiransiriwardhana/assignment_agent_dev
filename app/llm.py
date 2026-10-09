"""Groq LLM access with retry + model fallback + JSON helpers.

Model rationale (see README): openai/gpt-oss-120b for reasoning/answer synthesis,
openai/gpt-oss-120b for cheap, fast sub-tasks (routing, map steps, fact extraction).
"""
import asyncio
import json
import logging
import re
from typing import AsyncIterator

from app.config import settings
from app.logging_setup import log_event

log = logging.getLogger(__name__)


class LLMUnavailable(Exception):
    """Raised when every model/retry failed. Callers degrade gracefully."""


def _model(name: str, json_mode: bool = False, temperature: float = 0.1):
    if not settings.groq_api_key:
        raise LLMUnavailable("GROQ_API_KEY is not configured")
    from langchain_groq import ChatGroq
    m = ChatGroq(model=name, temperature=temperature, api_key=settings.groq_api_key,
                 max_retries=0, timeout=40)
    return m.bind(response_format={"type": "json_object"}) if json_mode else m


def _chain(fast: bool) -> list[str]:
    return [settings.fast_model, settings.primary_model] if fast else [settings.primary_model, settings.fast_model]


async def _invoke(messages, fast: bool, json_mode: bool):
    last: Exception | None = None
    for name in _chain(fast):
        for attempt in range(2):
            try:
                return await _model(name, json_mode).ainvoke(messages)
            except LLMUnavailable:
                raise
            except Exception as exc:  # network, rate limit, 5xx ...
                last = exc
                log_event(log, "llm_error", level=logging.WARNING, model=name, attempt=attempt, error=str(exc)[:200])
                await asyncio.sleep(0.6 * (attempt + 1))
    raise LLMUnavailable(str(last))


def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


async def llm_json(system: str, user: str, fast: bool = True) -> dict:
    """Ask for a JSON object. (Groq JSON mode requires the word JSON in the prompt.)"""
    resp = await _invoke([("system", system + "\nRespond with a single valid JSON object."), ("human", user)],
                         fast=fast, json_mode=True)
    try:
        return parse_json(resp.content)
    except Exception as exc:
        raise LLMUnavailable(f"invalid JSON from model: {exc}") from exc


async def llm_text(system: str, user: str, fast: bool = False) -> str:
    resp = await _invoke([("system", system), ("human", user)], fast=fast, json_mode=False)
    return resp.content


async def llm_stream(system: str, user: str) -> AsyncIterator[str]:
    """Stream tokens. Falls back to the next model only if nothing has been emitted yet."""
    last: Exception | None = None
    for name in _chain(fast=False):
        emitted = False
        try:
            async for chunk in _model(name, temperature=0.2).astream([("system", system), ("human", user)]):
                if chunk.content:
                    emitted = True
                    yield chunk.content
            return
        except LLMUnavailable:
            raise
        except Exception as exc:
            last = exc
            log_event(log, "llm_stream_error", level=logging.WARNING, model=name, emitted=emitted, error=str(exc)[:200])
            if emitted:
                raise LLMUnavailable(f"stream interrupted: {exc}") from exc
    raise LLMUnavailable(str(last))
