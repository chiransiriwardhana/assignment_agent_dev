"""Tool layer: schema validation + RBAC + timeout + error normalisation, in one choke point.

The LLM only ever *proposes* a tool call. ToolExecutor.execute() is the single place where a call is allowed to
happen, so an agent cannot bypass authorization, send malformed parameters, or hang the turn.
"""
import asyncio
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from app.auth import Principal
from app.config import settings
from app.logging_setup import log_event

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*a, **k):
        return lambda f: f

log = logging.getLogger(__name__)
DocType = Literal["incident", "runbook", "architecture", "policy", "product_spec", "meeting_notes"]
SAFE_TEXT = re.compile(r"^[\w\s\-\.@]{0,80}$")


# ------------------------------------------------------------------ argument schemas
class KnowledgeSearchArgs(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    department: str | None = Field(None, pattern=r"^[a-z\-]{2,30}$")
    document_type: DocType | None = None
    date_from: date | None = None
    date_to: date | None = None
    top_k: int = Field(6, ge=1, le=10)


class AnalysisArgs(BaseModel):
    operation: Literal["count_by", "timeline", "keyword_frequency"]
    group_field: Literal["department", "document_type", "access_level"] = "department"
    keywords: list[str] = Field(default_factory=list, max_length=10)
    department: str | None = Field(None, pattern=r"^[a-z\-]{2,30}$")
    document_type: DocType | None = None
    date_from: date | None = None
    date_to: date | None = None


MCP_PARAMS = {  # allow-list of tools and their permitted parameters
    "employee_lookup": {"name", "department"},
    "service_catalog": {"service"},
    "incident_records": {"service", "severity"},
}


class McpArgs(BaseModel):
    server_tool: Literal["employee_lookup", "service_catalog", "incident_records"]
    params: dict[str, str] = Field(default_factory=dict)

    def model_post_init(self, _ctx) -> None:
        bad = set(self.params) - MCP_PARAMS[self.server_tool]
        if bad:
            raise ValueError(f"unknown parameters for {self.server_tool}: {sorted(bad)}")
        for k, v in self.params.items():
            if not SAFE_TEXT.match(v):
                raise ValueError(f"unsafe characters in parameter '{k}'")


class NoArgs(BaseModel):
    pass


# ------------------------------------------------------------------ results / registry
@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str | None = None
    code: str | None = None  # permission_denied | invalid_args | timeout | unavailable | tool_error


@dataclass
class ToolSpec:
    name: str
    args_model: type[BaseModel]
    handler: Callable[[Principal, Any], Awaitable[Any]]


class ToolExecutor:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    @traceable(run_type="tool", name="tool_execute")
    async def execute(self, principal: Principal, name: str, raw_args: dict) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(False, error=f"unknown tool '{name}'", code="invalid_args")
        if not principal.can_use(name):  # RBAC - cannot be bypassed by the agent
            log_event(log, "tool_denied", level=logging.WARNING, user=principal.username, role=principal.role, tool=name)
            return ToolResult(False, error=f"role '{principal.role}' is not allowed to use '{name}'", code="permission_denied")
        try:
            args = spec.args_model(**raw_args)
        except (ValidationError, ValueError) as exc:
            return ToolResult(False, error=f"invalid parameters: {str(exc)[:200]}", code="invalid_args")
        try:
            data = await asyncio.wait_for(spec.handler(principal, args), settings.tool_timeout_sec)
            log_event(log, "tool_ok", user=principal.username, tool=name)
            return ToolResult(True, data=data)
        except asyncio.TimeoutError:
            return ToolResult(False, error=f"timed out after {settings.tool_timeout_sec}s", code="timeout")
        except ConnectionError as exc:
            return ToolResult(False, error=str(exc), code="unavailable")
        except Exception as exc:
            log_event(log, "tool_error", level=logging.ERROR, tool=name, error=str(exc)[:300])
            return ToolResult(False, error=str(exc)[:200], code="tool_error")


# ------------------------------------------------------------------ handlers
async def _knowledge_search(principal: Principal, a: KnowledgeSearchArgs):
    from app.services import get_services
    res = await get_services().store.search(principal, a.query, a.top_k, a.department, a.document_type, a.date_from, a.date_to)
    return [{"doc_id": h.chunk.doc_id, "title": h.chunk.title, "score": round(h.score, 3),
             "excerpt": h.chunk.text[:300]} for h in res.hits]


async def _python_analysis(principal: Principal, a: AnalysisArgs):
    from app.services import get_services
    corpus = get_services().corpus

    def run():
        docs = corpus.list_docs(principal, a.department, a.document_type, a.date_from, a.date_to)
        if a.operation == "count_by":
            return dict(Counter(d[a.group_field] for d in docs))
        if a.operation == "timeline":
            return dict(sorted(Counter(d["date"][:7] for d in docs).items()))
        out = {}
        for kw in a.keywords:
            ids = [d["id"] for d in docs if kw.lower() in (corpus.docs[d["id"]].body.lower())]
            out[kw] = {"documents": len(ids), "doc_ids": ids}
        return {"scanned": len(docs), "keywords": out}

    return await asyncio.to_thread(run)


async def _mcp_query(principal: Principal, a: McpArgs):
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:
        raise ConnectionError("MCP client library not installed") from exc
    params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"])
    try:
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                res = await session.call_tool(a.server_tool, a.params)
    except Exception as exc:
        raise ConnectionError(f"MCP server unreachable: {exc}") from exc
    texts = [c.text for c in res.content if getattr(c, "text", None)]
    parsed = []
    for t in texts:
        try:
            parsed.append(json.loads(t))
        except json.JSONDecodeError:
            parsed.append(t)
    return parsed


async def _admin_reindex(principal: Principal, _a: NoArgs):
    from app.services import get_services
    s = get_services()
    s.corpus.reload()
    await s.store.rebuild(s.corpus.chunks)
    return {"documents": len(s.corpus.docs), "chunks": len(s.corpus.chunks)}


def build_executor() -> ToolExecutor:
    ex = ToolExecutor()
    ex.register(ToolSpec("knowledge_search", KnowledgeSearchArgs, _knowledge_search))
    ex.register(ToolSpec("python_analysis", AnalysisArgs, _python_analysis))
    ex.register(ToolSpec("mcp_query", McpArgs, _mcp_query))
    ex.register(ToolSpec("admin_reindex", NoArgs, _admin_reindex))
    return ex
