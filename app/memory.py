"""Conversation memory.

Design (3 tiers):
  1. Working memory   - last N turns verbatim (exact wording matters for follow-ups like "what about the second one?").
  2. Rolling summary  - older turns are compressed by the fast LLM so context stays bounded across long sessions.
  3. Long-term memory - durable user facts/preferences ("I'm on the payments SRE team") persisted per user (JSON file
                        here; a Pinecone namespace / Postgres in production) and recalled by keyword overlap.
Sessions are keyed by "<username>:<session_id>" so one user can never read another user's history.
"""
import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings
from app.llm import LLMUnavailable, llm_json, llm_text
from app.retrieval.corpus import tokenize

WINDOW_TURNS = 6          # messages kept verbatim (3 user + 3 assistant)
SUMMARIZE_AFTER = 10      # compress when more than this many messages are stored
MAX_FACTS = 25
_FACT_TRIGGER = re.compile(r"\b(i am|i'm|i work|my team|my role|i prefer|remember|call me|i manage|i lead)\b", re.I)


@dataclass
class Session:
    messages: list[tuple[str, str]] = field(default_factory=list)
    summary: str = ""


class MemoryManager:
    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()
        Path(settings.memory_dir).mkdir(parents=True, exist_ok=True)

    # ---- short-term
    def history_text(self, key: str) -> str:
        s = self.sessions.get(key)
        if not s:
            return ""
        parts = []
        if s.summary:
            parts.append(f"[Summary of earlier conversation] {s.summary}")
        parts += [f"{role}: {text[:700]}" for role, text in s.messages[-WINDOW_TURNS:]]
        return "\n".join(parts)

    def add_turn(self, key: str, question: str, answer: str) -> None:
        s = self.sessions.setdefault(key, Session())
        s.messages += [("User", question), ("Assistant", answer)]

    async def maybe_summarize(self, key: str) -> bool:
        s = self.sessions[key]
        if len(s.messages) <= SUMMARIZE_AFTER:
            return False
        old, s.messages = s.messages[:-WINDOW_TURNS], s.messages[-WINDOW_TURNS:]
        text = "\n".join(f"{r}: {t[:500]}" for r, t in old)
        try:
            s.summary = await llm_text("Summarise this conversation in <=120 words, keeping names, services, "
                                       "document IDs and open questions.", f"{s.summary}\n{text}", fast=True)
        except LLMUnavailable:  # degrade: crude truncation rather than losing context entirely
            s.summary = (s.summary + " " + text)[-800:]
        return True

    # ---- long-term
    def _path(self, user: str) -> Path:
        return Path(settings.memory_dir) / f"{re.sub(r'[^a-z0-9_-]', '', user.lower())}.json"

    def facts(self, user: str) -> list[str]:
        p = self._path(user)
        return json.loads(p.read_text()) if p.exists() else []

    def recall(self, user: str, query: str, k: int = 3) -> list[str]:
        q = set(tokenize(query))
        scored = sorted(self.facts(user), key=lambda f: -len(q & set(tokenize(f))))
        return scored[:k]

    async def learn(self, user: str, message: str) -> list[str]:
        """Extract durable facts only when the message looks like it contains one (saves an LLM call)."""
        if not _FACT_TRIGGER.search(message):
            return []
        try:
            out = await llm_json('Extract durable facts or preferences about the USER (role, team, focus area, '
                                 'preferences) from the message. Format: {"facts": ["..."]}. Empty list if none.',
                                 message, fast=True)
        except LLMUnavailable:
            return []
        new = [f.strip()[:200] for f in out.get("facts", []) if isinstance(f, str) and f.strip()]
        async with self._lock:
            existing = self.facts(user)
            added = [f for f in new if f not in existing]
            if added:
                self._path(user).write_text(json.dumps((existing + added)[-MAX_FACTS:], indent=1))
        return added
