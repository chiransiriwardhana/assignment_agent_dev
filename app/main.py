"""FastAPI entrypoint: auth, rate limiting, SSE streaming chat, HITL approval, feedback."""
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app import auth
from app.config import settings
from app.logging_setup import log_event, setup_logging
from app.rate_limit import RateLimited
from app.runner import run_turn
from app.services import build_services, get_services

setup_logging()
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await build_services()
    yield


app = FastAPI(title="Crest Assistant API", lifespan=lifespan)


class LoginRequest(BaseModel):
    username: str = Field(max_length=50)
    password: str = Field(max_length=100)


class ChatRequest(BaseModel):
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{4,64}$")
    message: str = Field(min_length=1, max_length=4000)  # stricter 2000-char/guard check happens in the graph


class ApprovalRequest(BaseModel):
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{4,64}$")
    approved: bool


class FeedbackRequest(BaseModel):
    session_id: str
    rating: int = Field(ge=-1, le=1)
    comment: str = Field("", max_length=500)
    question: str = Field("", max_length=2000)
    answer: str = Field("", max_length=8000)
    trace_id: str | None = None


def current_user(authorization: str = Header(default="")) -> auth.Principal:
    try:
        return auth.authenticate(authorization.removeprefix("Bearer ").strip())
    except auth.AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc))


@app.exception_handler(RateLimited)
async def rate_limited_handler(_req, exc: RateLimited):
    return JSONResponse(status_code=429, content={"detail": str(exc), "retry_after": round(exc.retry_after, 1)},
                        headers={"Retry-After": str(int(exc.retry_after) + 1)})


@app.get("/health")
async def health():
    s = get_services()
    return {"status": "ok", "documents": len(s.corpus.docs), "vector_backend": "pinecone" if s.store.primary else "local"}


@app.post("/login")
async def login(req: LoginRequest):
    try:
        token, principal = auth.login(req.username, req.password)
    except auth.AuthError:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"token": token, "username": principal.username, "role": principal.role}


async def _sse(events):
    try:
        async for ev in events:
            yield f"data: {json.dumps(ev, default=str)}\n\n"
    except Exception as exc:  # last-resort guard so the client always gets a well-formed stream
        log_event(log, "stream_error", level=logging.ERROR, error=str(exc)[:200])
        yield f"data: {json.dumps({'type': 'error', 'message': 'Stream failed unexpectedly.'})}\n\n"


@app.post("/chat")
async def chat(req: ChatRequest, principal: auth.Principal = Depends(current_user)):
    s = get_services()
    await s.limiter.acquire(principal.username, principal.role)  # raises RateLimited -> 429
    return StreamingResponse(_sse(run_turn(s, principal, req.session_id, message=req.message)),
                             media_type="text/event-stream")


@app.post("/approve")
async def approve(req: ApprovalRequest, principal: auth.Principal = Depends(current_user)):
    s = get_services()
    return StreamingResponse(_sse(run_turn(s, principal, req.session_id, approve=req.approved)),
                             media_type="text/event-stream")


@app.post("/feedback")
async def feedback(req: FeedbackRequest, principal: auth.Principal = Depends(current_user)):
    record = {"ts": time.time(), "user": principal.username, **req.model_dump()}
    with open(settings.feedback_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    # Also attach to the LangSmith trace so quality can be analysed next to execution data.
    if req.trace_id:
        try:
            from langsmith import Client
            Client().create_feedback(run_id=req.trace_id, key="user_rating", score=req.rating, comment=req.comment)
        except Exception as exc:
            log_event(log, "langsmith_feedback_failed", level=logging.WARNING, error=str(exc)[:120])
    return {"status": "recorded"}
