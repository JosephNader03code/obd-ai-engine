---
title: OBD Vision AI Engine
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# OBD Vision — AI Diagnostic Engine

FastAPI service that turns raw OBD-II telemetry into an Arabic-first diagnostic report.

## How it runs

Hugging Face Spaces picks up `Dockerfile.hf`; rename or symlink it to `Dockerfile` at the repo root when uploading to the Space.

- Health: `GET /api/health`
- Main endpoint: `POST /api/analyze`

## Env vars

| Name | Default | What it does |
|------|---------|--------------|
| `LLM_PROVIDER` | `mock` | `mock` / `anthropic` / `openai` / `ollama` |
| `CORS_ALLOWED_ORIGINS` | `*` | Comma-separated list of allowed frontends |

## Usage

```bash
curl -X POST https://<username>-obd-ai.hf.space/api/analyze \
  -H "Content-Type: application/json" \
  -d @payload.json
```
