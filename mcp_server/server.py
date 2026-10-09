"""Dummy enterprise MCP server (stdio). Run standalone: python -m mcp_server.server"""
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("crest-enterprise-data")

EMPLOYEES = [
    {"name": "Priya Fernando", "department": "payments", "title": "Payments Platform Lead", "oncall": True},
    {"name": "Marcus Chen", "department": "payments", "title": "Site Reliability Engineer", "oncall": False},
    {"name": "Amara Silva", "department": "digital-banking", "title": "Mobile Engineering Manager", "oncall": False},
    {"name": "Daniel Okafor", "department": "security", "title": "Security Operations Lead", "oncall": True},
]
SERVICES = [
    {"service": "payments-core", "owner": "payments", "tier": 1, "runbook": "RB-001", "slo": "99.95%"},
    {"service": "card-gateway", "owner": "payments", "tier": 1, "runbook": "RB-003", "slo": "99.9%"},
    {"service": "mobile-bff", "owner": "digital-banking", "tier": 2, "runbook": "RB-004", "slo": "99.5%"},
]
INCIDENTS = [
    {"id": "INC-2026-002", "service": "payments-core", "severity": "SEV2", "status": "resolved"},
    {"id": "INC-2026-005", "service": "card-gateway", "severity": "SEV3", "status": "open"},
    {"id": "INC-2025-029", "service": "payments-core", "severity": "SEV2", "status": "resolved"},
]


def _filter(rows: list[dict], **crit) -> list[dict]:
    return [r for r in rows if all(v is None or str(r.get(k, "")).lower() == str(v).lower() for k, v in crit.items())]


@mcp.tool()
def employee_lookup(name: str | None = None, department: str | None = None) -> list[dict]:
    """Look up employees by exact name and/or department."""
    return _filter(EMPLOYEES, name=name, department=department)


@mcp.tool()
def service_catalog(service: str | None = None) -> list[dict]:
    """Look up services in the service catalog (owner, tier, runbook, SLO)."""
    return _filter(SERVICES, service=service)


@mcp.tool()
def incident_records(service: str | None = None, severity: str | None = None) -> list[dict]:
    """Live incident records, optionally filtered by service/severity."""
    return _filter(INCIDENTS, service=service, severity=severity)


if __name__ == "__main__":
    mcp.run()
