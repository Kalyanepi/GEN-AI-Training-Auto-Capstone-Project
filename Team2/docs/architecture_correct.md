# RoadGuard AI Copilot — Correct Architecture

## Request Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────────────────────────────┐
│   Browser   │────▶│  Streamlit   │────▶│           FastAPI /chat/stream          │
└─────────────┘     └──────────────┘     └─────────────────────────────────────────┘
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           LangGraph Agent (Orchestrator)                            │
│                                                                                     │
│   ┌─────────────┐                                                                   │
│   │  Preflight  │◄─────────────────────────────────────────────────────────────────┤
│   │   Node      │                                                                   │
│   └──────┬──────┘                                                                   │
│          │                                                                          │
│     ┌────┴────┐                                                                     │
│     │         │  Parallel Execution (asyncio.gather)                               │
│     ▼         ▼                                                                     │
│ ┌────────┐  ┌──────────┐                                                           │
│ │Input   │  │  Intent  │                                                           │
│ │Guardrail│  │  Router  │                                                           │
│ │~200ms  │  │  ~300ms  │                                                           │
│ └────┬───┘  └────┬─────┘                                                           │
│      │            │                                                                 │
│      └────┬───────┘                                                                 │
│           │                                                                         │
│           ▼                                                                         │
│   ┌───────────────┐                                                                 │
│   │  Route Check  │                                                                 │
│   │  (Conditional)│                                                                 │
│   └───────┬───────┘                                                                 │
│           │                                                                         │
│     ┌─────┼─────┐                                                                   │
│     │     │     │                                                                   │
│     ▼     ▼     ▼                                                                   │
│ ┌──────┐ ┌──────┐ ┌──────────┐                                                     │
│ │Block │ │Tools │ │  Bypass  │                                                     │
│ │      │ │      │ │(greeting/│                                                     │
│ │      │ │      │ │clarify)  │                                                     │
│ └──────┘ └──────┘ └──────────┘                                                     │
│            │                                                                        │
│            ▼                                                                        │
│   ┌───────────────────┐                                                           │
│   │   Tool Execution  │                                                           │
│   │    ~1200ms        │                                                           │
│   │                   │                                                           │
│   │  ┌─────────────┐  │                                                           │
│   │  │ FAISS Search│  │                                                           │
│   │  │ ~200ms      │  │                                                           │
│   │  └─────────────┘  │                                                           │
│   │  ┌─────────────┐  │                                                           │
│   │  │  Reranker   │  │                                                           │
│   │  │  ~400ms     │  │                                                           │
│   │  └─────────────┘  │                                                           │
│   │  ┌─────────────┐  │                                                           │
│   │  │ CSV Lookup  │  │                                                           │
│   │  │  ~100ms     │  │                                                           │
│   │  └─────────────┘  │                                                           │
│   └─────────┬─────────┘                                                           │
│             │                                                                       │
│             ▼                                                                       │
│   ┌───────────────────┐                                                           │
│   │   LLM Synthesis   │                                                           │
│   │    ~3000ms        │◄──────────────────┐                                        │
│   │                   │                     │                                       │
│   │  (GPT-4o-mini)   │                     │                                       │
│   └─────────┬─────────┘                     │                                       │
│             │                               │                                       │
│             ▼                               │                                       │
│   ┌───────────────────┐                     │                                       │
│   │  Output Guardrail │                     │                                       │
│   │    ~200ms         │                     │                                       │
│   └─────────┬─────────┘                     │                                       │
│             │                               │                                       │
│             ▼                               │                                       │
│   ┌───────────────────┐                     │                                       │
│   │   Memory Update   │                     │                                       │
│   │    ~50ms          │                     │                                       │
│   └─────────┬─────────┘                     │                                       │
│             │                               │                                       │
│             └───────────────┬───────────────┘                                       │
│                             │                                                       │
│                             ▼                                                       │
│                    ┌────────────────┐                                               │
│                    │  SSE Stream    │                                               │
│                    │  Response      │                                               │
│                    └────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │    Browser     │
                    └────────────────┘
```

## Why This Architecture?

### 1. Parallel Guardrail + Router (~400ms total)
Both use GPT-4o-mini and run simultaneously via `asyncio.gather()`

### 2. Conditional Routing
- **Block**: If input guardrail detects PII/injection/jailbreak
- **Tools**: Normal flow (coverage, FNOL, repair, total loss)
- **Bypass**: GREETING or CLARIFICATION_NEEDED (skip tools + synthesis)

### 3. Tool Execution (~1200ms)
All tools run in parallel:
- FAISS retrieval: ~200ms
- Cross-encoder reranker: ~400ms  
- CSV lookups: ~100ms
- LLM calls within tools: ~500ms

### 4. LLM Synthesis (~3000ms) - BOTTLENECK
Single GPT-4o-mini call generating the final answer. This is where 60% of time is spent.

### 5. Streaming Response
Tokens stream via SSE as they're generated, improving perceived latency.

## Latency Breakdown

| Stage | Component | Time | Parallel? |
|-------|-----------|------|-----------|
| 1 | Input Guardrail | 200ms | ✅ with Router |
| 1 | Intent Router | 300ms | ✅ with Guardrail |
| 2 | Tool Execution | 1200ms | ✅ Internal parallel |
| 3 | **LLM Synthesis** | **3000ms** | ❌ Sequential |
| 4 | Output Guardrail | 200ms | ❌ Sequential |
| 5 | Memory Update | 50ms | ❌ Sequential |
| **Total** | | **~4950ms** | |

## Optimization Opportunities

1. **Reduce synthesis tokens** (900→500) ✅ Done
2. **Limit chunks in prompt** (all→top 3) ✅ Done
3. **Shorter history** (10→3 turns) ✅ Done
4. **Response caching** (common queries) 🔄 Pending
5. **Rule-based templates** (FNOL, coverage) 🔄 Pending

## Tech Stack

| Layer | Technology |
|-------|------------|
| UI | Streamlit 1.40 |
| API | FastAPI + Uvicorn |
| Agent Framework | LangGraph 0.2.50 |
| LLM | OpenAI GPT-4o-mini |
| Vector DB | FAISS (60 chunks) |
| Embeddings | text-embedding-3-small |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Guardrails | Presidio (PII) + LLM (scope) |
