"""Security & guardrails.

Layers (defence in depth):
  INPUT      check_user_input()      - length/charset validation + prompt-injection, exfiltration, tool-abuse patterns
  RETRIEVED  sanitize_retrieved()    - strips instruction-like sentences from documents (indirect injection)
  OUTPUT     validate_response()     - hallucinated-citation check, grounding check, secret/PII redaction, brand rules
Pattern matching is a cheap first line; the structural controls (RBAC in ToolExecutor, filters built from the
Principal, documents wrapped as untrusted data in the prompt) do not depend on the LLM behaving.
"""
import re
import unicodedata
from dataclasses import dataclass, field

MAX_INPUT_CHARS = 2000

_INJECTION = {
    "instruction_override": [
        r"ignore (all |any |the |your )?(previous|prior|above|earlier) (instructions|rules|prompts?|directions)",
        r"disregard .{0,40}(instructions|rules|guidelines)",
        r"you are now (dan|in developer mode|unrestricted)",
        r"(developer|god|jailbreak) mode",
        r"pretend (you have|there are) no (rules|restrictions|guidelines)",
        r"forget (everything|all) (you|that)",
    ],
    "data_exfiltration": [
        r"(reveal|show|print|repeat|output|leak|display).{0,40}(system prompt|hidden instructions|initial prompt|api[ _-]?key|secret|credentials|password)",
        r"(send|post|forward|upload|email).{0,60}(https?://|webhook|@[\w.-]+\.\w+)",
    ],
    "tool_abuse": [
        r"(call|run|execute|invoke|use).{0,30}(all|every|any) tools?",
        r"admin_reindex.{0,20}(without|skip|bypass)",
        r"(drop|truncate|delete) (table|database)",
        r"rm -rf|os\.system|subprocess|__import__|eval\(|exec\(",
        r"(bypass|skip|disable).{0,20}(approval|authori[sz]ation|permission|guardrail|validation)",
    ],
}
_COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in _INJECTION.items()}

REFUSALS = {
    "instruction_override": "I can't change my operating rules or ignore my instructions. I'm glad to help with questions about Crestline Commercial Bank's policies, runbooks, incidents and product documentation.",
    "data_exfiltration": "I can't share internal configuration, credentials, or send data to external destinations. Is there a document or procedure I can help you find instead?",
    "tool_abuse": "I can't run that request - tool use follows your role's permissions and approved workflows. Tell me the business goal and I'll use the appropriate approved tool.",
    "invalid": "Please send a non-empty question of up to 2,000 characters.",
}


@dataclass
class GuardResult:
    ok: bool
    cleaned: str = ""
    category: str = ""
    reason: str = ""


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]", "", text)  # zero-width / bidi tricks
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text.strip()


def check_user_input(text: str) -> GuardResult:
    text = _clean(text or "")
    if not text or len(text) > MAX_INPUT_CHARS:
        return GuardResult(False, category="invalid", reason="empty or too long")
    for category, patterns in _COMPILED.items():
        for p in patterns:
            if p.search(text):
                return GuardResult(False, category=category, reason=f"matched {category} pattern")
    return GuardResult(True, cleaned=text, reason="clean")


_DOC_INJECTION = re.compile(
    r"(ignore|disregard|forget)[^.\n]{0,40}(instructions|rules|prompt)"
    r"|system prompt|you must now|new instructions?:|reveal [^.\n]{0,30}(key|secret|password)"
    r"|send [^.\n]{0,40}https?://",
    re.I,
)


def sanitize_retrieved(text: str) -> tuple[str, bool]:
    """Neutralise instruction-like sentences inside retrieved content. Returns (text, was_flagged)."""
    flagged = False
    out = []
    for sentence in re.split(r"(?<=[.!?\n])\s+", _clean(text)):
        if _DOC_INJECTION.search(sentence):
            flagged = True
            out.append("[removed: suspicious instruction found in document]")
        else:
            out.append(sentence)
    return " ".join(out), flagged


# ---------------------------------------------------------------- output guards
CITE_RE = re.compile(r"\[([A-Z]{2,6}-[A-Z0-9-]+)\]")
_SECRETS = [
    re.compile(r"gsk_[A-Za-z0-9]{20,}"), re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"), re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4}\b"),
]
_BRAND = re.compile(r"guaranteed returns?|risk[- ]free investment|you will definitely (make|earn)", re.I)
_PROMPT_LEAK = re.compile(r"you are crest assistant|answer rules:|<documents>", re.I)


@dataclass
class ValidationResult:
    ok: bool
    issues: list[str] = field(default_factory=list)
    cleaned: str = ""


def validate_response(text: str, allowed_ids: set[str], require_citations: bool) -> ValidationResult:
    issues: list[str] = []
    cleaned = text.strip()
    if len(cleaned) < 5:
        return ValidationResult(False, ["empty response"], cleaned)

    cited = set(CITE_RE.findall(cleaned))
    invented = cited - allowed_ids
    if invented:
        issues.append(f"hallucinated citations: {sorted(invented)}")
        cleaned = CITE_RE.sub(lambda m: m.group(0) if m.group(1) in allowed_ids else "", cleaned)
    if require_citations and not (cited & allowed_ids):
        issues.append("answer is grounded in documents but cites none")
    if _PROMPT_LEAK.search(cleaned):
        issues.append("possible system prompt leakage")
        cleaned = _PROMPT_LEAK.sub("[redacted]", cleaned)
    if _BRAND.search(cleaned):
        issues.append("brand/compliance: promissory financial language")
        cleaned = _BRAND.sub("[removed]", cleaned)
    for pat in _SECRETS:
        if pat.search(cleaned):
            issues.append("sensitive token redacted")
            cleaned = pat.sub("[REDACTED]", cleaned)
    return ValidationResult(not issues, issues, cleaned)
