# RoadGuard AI Copilot

> **Auto Insurance AI Copilot** — Accurate. Grounded. Cited. Never Fabricated.

A production-grade conversational AI for auto insurance covering coverage Q&A, repair-cost lookups, total-loss decisions, FNOL guidance, and tier-aware rental + roadside benefits.

- **Backend:** FastAPI + LangGraph + FAISS (RAG) + OpenAI
- **Frontend:** Streamlit
- **Deploy:** Docker Compose (local + AWS EC2)

---

## Architecture (high level)

```
User -> Streamlit UI -> FastAPI -> LangGraph
                                     |
   Input Guardrail --> Intent Router --> Tools (RAG + CSV) --> LLM Synthesis --> Output Guardrail
   (PII / inj.)       (gpt-4o-mini)     (FAISS + Pandas)        (gpt-4o)        (legal/fault/citation)
```

Full design: see `Architecture_PLAN.txt`.

---

## Prerequisites

- Python 3.11
- Docker + Docker Compose (for containerized run)
- OpenAI API key
- LangSmith API key (for tracing — optional but recommended)

---

## Quickstart (Docker — recommended)

```bash
# 1. Configure secrets
cp .env.example .env
# edit .env and set OPENAI_API_KEY and LANGCHAIN_API_KEY

# 2. Build + start
docker compose build
docker compose up -d

# 3. One-time: build the FAISS index from PDFs
docker compose exec api python -m ingestion.build_index

# 4. Verify
curl http://localhost:8000/health
# UI: http://localhost:8501
# API docs: http://localhost:8000/docs
```

## Quickstart (local Python)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # then add real API keys

# Build the index
python -m ingestion.build_index

# Start API (terminal 1)
uvicorn api.main:app --reload --port 8000

# Start UI (terminal 2)
streamlit run ui/app.py
```

---

## Project Layout

```
auto_insurance_copilot/
├── agent/            # LangGraph orchestrator + 8 tools + memory + prompts
├── api/              # FastAPI app, routes, schemas, middleware, services
├── data/             # PDFs, CSVs, persisted FAISS index
├── evaluation/       # 30-question DeepEval suite + 4 adversarial tests
├── guardrails/       # Input + output guardrails
├── ingestion/        # PDF chunker, CSV loader, embedder, build_index entrypoint
├── observability/    # Structured logger + LangSmith tracer
├── rag/              # FAISS store, metadata-filtered retriever, reranker, citations
├── scripts/          # ingest.sh, run_eval.sh, deploy_aws.sh
├── ui/               # Streamlit components
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.ui
└── requirements.txt
```

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/chat` | Main chat endpoint |
| DELETE | `/api/v1/session/{id}` | Clear session memory |
| GET | `/health` | Liveness — verifies FAISS + CSVs loaded |
| GET | `/ready` | Readiness — chunk count + active sessions |
| GET | `/docs` | Swagger UI |

### Sample request

```json
POST /api/v1/chat
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "ACV $8,000, repair $6,500 in IL — total loss?",
  "policy_tier": "premium",
  "state_code": "IL",
  "acv": 8000,
  "repair_cost": 6500
}
```

---

## Evaluation

```bash
# Run the 30-question DeepEval suite + 4 adversarial guardrail tests
bash scripts/run_eval.sh
```

Datasets: `evaluation/test_dataset/`. Metrics thresholds (Faithfulness ≥0.85, Answer Relevancy ≥0.80, Contextual Recall ≥0.75, Contextual Precision ≥0.70) are defined in `evaluation/metrics.py`.

---

## AWS Deployment

```bash
# On a fresh Ubuntu 22.04 EC2 (t3.medium recommended)
git clone <repo> && cd auto_insurance_copilot
cp .env.example .env && nano .env
bash scripts/deploy_aws.sh
```

Open ports: `80` (Nginx — optional), `8000` (API), `8501` (UI). Restrict SSH (`22`) to known IPs only.

---

## Configuration

Every threshold, model name, path, and limit is an env var with a default in `api/config.py`. Tune behavior by editing `.env` — never edit code.

Key flags:

- `SIMILARITY_THRESHOLD=0.65` — anti-hallucination floor
- `RERANKER_ENABLED=true` — cross-encoder reranking
- `MAX_CITATIONS_PER_RESPONSE=4`
- `SESSION_TTL_MINUTES=30`
- `RATE_LIMIT_PER_MINUTE=30`
- `FABRICATED_COST_TOLERANCE_PCT=10`

---

## Engineering Standards

- All FastAPI routes + tools are `async`
- Tools return uniform `ToolResult` — no exceptions reach the user
- Comments explain **why**, not what
- All API keys via env vars (`.env` is gitignored)
- Structured JSON logs (`structlog`) with correlation IDs
- LangSmith trace URL returned in every API response

---

## License & Contact

Capstone Project — Group 02 Auto Insurance AI Copilot.
For claim-related issues: **1-800-555-0601**.
