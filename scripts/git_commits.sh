#!/usr/bin/env bash
# Creates a readable, incremental commit history. Run once in a fresh repo: bash scripts/git_commits.sh
set -e
git init -q 2>/dev/null || true
c() { git add "${@:2}" && git commit -qm "$1"; }
c "chore: project scaffold, config, structured logging"            requirements.txt .env.example .gitignore app/__init__.py app/config.py app/logging_setup.py
c "feat(security): auth, RBAC, token-bucket rate limiter"          app/auth.py app/rate_limit.py
c "feat(security): prompt-injection, retrieved-content and output guardrails" app/security.py
c "feat(rag): corpus, chunking, hybrid Pinecone store, local fallback, reranker" app/retrieval scripts/ingest.py
c "feat(data): synthetic bank documents generator"                 scripts/generate_sample_docs.py
c "feat(llm): Groq Llama wrapper with retry and model fallback"    app/llm.py
c "feat(tools): validated, RBAC-enforced tool executor + MCP server" app/tools.py mcp_server
c "feat(memory): session window, rolling summary, long-term facts" app/memory.py
c "feat(rlm): sandboxed python search plans, batching, recursive reduce" app/rlm.py
c "feat(agents): LangGraph multi-agent graph with HITL approval"   app/agents app/services.py app/runner.py
c "feat(api): FastAPI SSE chat, approval, feedback endpoints"      app/main.py
c "feat(ui): Streamlit chat with live agent activity panel"        ui
c "test: core unit tests"                                          tests
c "build: Dockerfile and docker-compose"                           Dockerfile docker-compose.yml
c "docs: README with architecture, security, assumptions"          README.md
