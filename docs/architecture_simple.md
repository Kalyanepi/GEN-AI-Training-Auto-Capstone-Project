# RoadGuard AI Copilot — Architecture

## System Flow

```mermaid
flowchart LR
    A[User] --> B[Streamlit UI]
    B --> C[FastAPI]
    C --> D[LangGraph Agent]
    D --> E[Response]

    subgraph Agent[LangGraph Agent]
        direction TB
        D1[Input Guardrail] --> D2[Intent Router]
        D2 --> D3[Tool Execution]
        D3 --> D4[LLM Synthesis]
        D4 --> D5[Output Guardrail]
        D5 --> D6[Memory Update]
    end

    subgraph Backend[Backend Services]
        F1[FAISS Vector DB]
        F2[Cross-Encoder Reranker]
        F3[CSV Data]
        F4[OpenAI GPT-4o-mini]
    end

    D3 --> F1
    D3 --> F2
    D3 --> F3
    D4 --> F4
```

## Latency Breakdown

| Component | Time |
|-----------|------|
| Input Guardrail | ~200ms |
| Intent Router | ~300ms |
| Tool Execution | ~1200ms |
| **LLM Synthesis** | **~3000ms** |
| Output Guardrail | ~200ms |
| Memory Update | ~50ms |
| **Total** | **~4950ms** |

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Agent**: LangGraph
- **LLM**: GPT-4o-mini
- **Vector DB**: FAISS
- **Embeddings**: text-embedding-3-small
- **Reranker**: cross-encoder/ms-marco-MiniLM

## Optimizations Applied

1. Parallel Guardrail + Router
2. Reduced synthesis tokens (900→500)
3. Top-3 chunks only in prompt
4. Truncated history (10→3 turns)
