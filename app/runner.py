"""Runs one graph turn (or resumes after human approval) and yields UI-facing events."""
import asyncio
import logging
import uuid
from typing import AsyncIterator

from langgraph.types import Command

from app.auth import Principal
from app.config import settings
from app.logging_setup import log_event
from app.services import Services

log = logging.getLogger(__name__)


async def run_turn(s: Services, principal: Principal, session_id: str, message: str | None = None,
                   approve: bool | None = None) -> AsyncIterator[dict]:
    key = f"{principal.username}:{session_id}"
    if approve is None:
        thread_id = f"{key}:{uuid.uuid4().hex[:8]}"
        s.pending.pop(key, None)
        graph_input = {"session_key": key, "username": principal.username, "role": principal.role,
                       "question": message, "attempts": 0, "tool_results": [], "contexts": []}
    else:
        thread_id = s.pending.get(key)
        if not thread_id:
            yield {"type": "error", "message": "No pending approval for this session."}
            return
        graph_input = Command(resume=approve)

    config = {"configurable": {"thread_id": thread_id}, "run_name": "crest_assistant_turn",
              "tags": ["crest-assistant", f"role:{principal.role}"],
              "metadata": {"user": principal.username, "role": principal.role, "session_id": session_id}}
    log_event(log, "turn_start", user=principal.username, role=principal.role, resume=approve is not None)

    interrupted = False
    try:
        async with asyncio.timeout(settings.turn_timeout_sec):
            async for mode, chunk in s.graph.astream(graph_input, config, stream_mode=["updates", "custom"]):
                if mode == "custom":
                    yield chunk
                    continue
                for node, _update in chunk.items():
                    if node == "__interrupt__":
                        s.pending[key] = thread_id
                        interrupted = True
                        yield {"type": "approval_required", "payload": chunk[node][0].value}
                        continue          # keep draining; do NOT return
                    yield {"type": "node", "node": node, "status": "completed"}
        if not interrupted:               # keep the pending thread when waiting for approval
            s.pending.pop(key, None)
            
    except TimeoutError:
        yield {"type": "error", "message": "The request timed out. Please try a narrower question."}
    except Exception as exc:
        log_event(log, "turn_failed", level=logging.ERROR, error=str(exc)[:300])
        yield {"type": "error", "message": "Something went wrong while processing your request. Please try again."}
