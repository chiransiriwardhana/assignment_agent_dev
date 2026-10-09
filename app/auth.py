"""Authentication (Option A: hardcoded users) and Role Based Access Control.

RBAC is defined in ONE place (ROLE_TOOLS / ROLE_ACCESS) and enforced at three layers:
  1. Supervisor strips tools the role cannot use (so the plan is sane),
  2. ToolExecutor re-checks on every call (the LLM cannot bypass it),
  3. Retrieval filters are built from the Principal, never from LLM output.
"""
import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass

from app.config import settings

TOKEN_TTL_SEC = 8 * 3600

# Which tools each role may invoke.
ROLE_TOOLS = {
    "viewer": {"knowledge_search"},
    "analyst": {"knowledge_search", "python_analysis", "mcp_query"},
    "admin": {"knowledge_search", "python_analysis", "mcp_query", "admin_reindex"},
}
# Which document sensitivity levels each role may read.
ROLE_ACCESS = {
    "viewer": {"public"},
    "analyst": {"public", "internal"},
    "admin": {"public", "internal", "confidential"},
}
# Pinecone namespaces (physical isolation of confidential content).
ROLE_NAMESPACES = {
    "viewer": ["general"],
    "analyst": ["general"],
    "admin": ["general", "restricted"],
}


@dataclass(frozen=True)
class Principal:
    username: str
    role: str

    @property
    def access_levels(self) -> set[str]:
        return ROLE_ACCESS[self.role]

    @property
    def namespaces(self) -> list[str]:
        return ROLE_NAMESPACES[self.role]

    def can_use(self, tool: str) -> bool:
        return tool in ROLE_TOOLS[self.role]

    def can_see(self, metadata: dict) -> bool:
        return metadata.get("access_level") in self.access_levels


def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


# Demo only. A real deployment would use Keycloak/OIDC (Option B).
USERS = {
    "viewer1": {"pw": _hash(settings.demo_password), "role": "viewer"},
    "analyst1": {"pw": _hash(settings.demo_password), "role": "analyst"},
    "admin1": {"pw": _hash(settings.demo_password), "role": "admin"},
}
_TOKENS: dict[str, tuple[Principal, float]] = {}


class AuthError(Exception):
    pass


def login(username: str, password: str) -> tuple[str, Principal]:
    user = USERS.get(username)
    # compare_digest on both branches to avoid user-enumeration timing differences
    ok = hmac.compare_digest(_hash(password), user["pw"] if user else _hash("x"))
    if not user or not ok:
        raise AuthError("Invalid credentials")
    principal = Principal(username, user["role"])
    token = secrets.token_urlsafe(32)
    _TOKENS[token] = (principal, time.time() + TOKEN_TTL_SEC)
    return token, principal


def authenticate(token: str) -> Principal:
    entry = _TOKENS.get(token)
    if not entry or entry[1] < time.time():
        raise AuthError("Invalid or expired token")
    return entry[0]
