# RoadGuard AI Copilot — Full Architecture Diagram

```mermaid
flowchart TB
    subgraph Client["Client Layer"]
        USER["User"]
        STREAMLIT["Streamlit UI"]
        subgraph UI_Components["UI Components"]
            SIDEBAR["Sidebar"]
            CHAT["Chat Panel"]
            METRICS["Metrics Panel"]
            RIGHT["Right Panel"]
        end
    end

    subgraph API["API Layer"]
        ROUTES["Routes"]
        subgraph Middleware["Middleware"]
            RATE["Rate Limiter"]
            LOG["Logging"]
        end
        CHAT_SERVICE["ChatService"]
    end

    subgraph Agent["LangGraph Agent"]
        direction TB
        PREFLIGHT["Preflight Node"]
        
        subgraph Guardrail["Guardrails"]
            INPUT_GR["Input Guardrail"]
            OUTPUT_GR["Output Guardrail"]
        end
        
        ROUTER["Intent Router"]
        
        subgraph Decisions["Intent Decisions"]
            OUT_OF_SCOPE["OUT_OF_SCOPE"]
            CLARIFY["CLARIFICATION_NEEDED"]
            TOOLS_PATH["TOOL_INTENT"]
        end
        
        TOOL_EXEC["Tool Execution"]
        
        subgraph Tools["8 Tools"]
            COV["Coverage ID"]
            RAG["Policy RAG"]
            FNOL["FNOL Guide"]
            UM["UM/UIM"]
            RENTAL["Rental"]
            ROAD["Roadside"]
            REPAIR["Repair Cost"]
            TOTAL["Total Loss"]
        end
        
        SYNTH["LLM Synthesis"]
        MEMORY["Memory Update"]
    end

    subgraph RAG_System["RAG System"]
        EMBED["Embedder"]
        FAISS["FAISS Store"]
        RETRIEVER["Retriever"]
        RERANKER["Reranker"]
        CITATION["Citations"]
    end

    subgraph Data["Data Layer"]
        REPAIR_CSV["Repair CSV"]
        TOTAL_CSV["Total Loss CSV"]
        PDFS["Policy PDFs"]
        FAISS_IDX["FAISS Index"]
    end

    subgraph External["External"]
        OPENAI["OpenAI API"]
        LANGSMITH["LangSmith"]
    end

    subgraph Cache["Caching"]
        EMBED_CACHE["Embedding Cache"]
        ROUTER_CACHE["Router Cache"]
        RETRIEVAL_CACHE["Retrieval Cache"]
    end

    %% Flow
    USER --> STREAMLIT
    STREAMLIT --> CHAT_SERVICE
    CHAT_SERVICE --> PREFLIGHT
    
    PREFLIGHT --> INPUT_GR
    PREFLIGHT --> ROUTER
    
    INPUT_GR -->|blocked| MEMORY
    INPUT_GR -->|pass| ROUTER
    
    ROUTER --> OUT_OF_SCOPE
    ROUTER --> CLARIFY
    ROUTER --> TOOLS_PATH
    
    OUT_OF_SCOPE --> MEMORY
    CLARIFY --> MEMORY
    TOOLS_PATH --> TOOL_EXEC
    
    TOOL_EXEC --> COV
    TOOL_EXEC --> RAG
    TOOL_EXEC --> FNOL
    TOOL_EXEC --> UM
    TOOL_EXEC --> RENTAL
    TOOL_EXEC --> ROAD
    TOOL_EXEC --> REPAIR
    TOOL_EXEC --> TOTAL
    
    RAG --> RETRIEVER
    RETRIEVER --> FAISS
    RETRIEVER --> RERANKER
    
    REPAIR --> REPAIR_CSV
    TOTAL --> TOTAL_CSV
    
    EMBED --> OPENAI
    ROUTER --> OPENAI
    SYNTH --> OPENAI
    INPUT_GR --> OPENAI
    OUTPUT_GR --> OPENAI
    
    RETRIEVER --> EMBED_CACHE
    RETRIEVER --> RETRIEVAL_CACHE
    ROUTER --> ROUTER_CACHE
    
    TOOL_EXEC --> SYNTH
    SYNTH --> OUTPUT_GR
    OUTPUT_GR -->|blocked| MEMORY
    OUTPUT_GR -->|pass| MEMORY
    MEMORY --> STREAMLIT
    
    PREFLIGHT -.-> LANGSMITH
    TOOL_EXEC -.-> LANGSMITH
    SYNTH -.-> LANGSMITH
```

## Latency by Component

| Component | Latency | Notes |
|-----------|---------|-------|
| Input Guardrail | ~200ms | GPT-4o-mini |
| Intent Router | ~300ms | GPT-4o-mini, parallel with guardrail |
| Tool Execution | ~1200ms | Parallel FAISS + Reranker + CSV |
| LLM Synthesis | **~3000ms** | **BOTTLENECK** |
| Output Guardrail | ~200ms | Rule-based + LLM checks |
| **Total** | **~5000ms** | |

## Key Features

- **Parallel preflight**: Guardrail + Router run together
- **Conditional flow**: Block / Tools / Clarify paths
- **8 Tools**: Coverage, RAG, FNOL, UM/UIM, Rental, Roadside, Repair, Total Loss
- **RAG Pipeline**: FAISS → Retriever → Reranker → Citations
- **3 Caches**: Embeddings, Router, Retrieval (LRU + TTL)
- **Streaming**: SSE tokens during synthesis

## Optimization Status

| Optimization | Status |
|-------------|--------|
| Parallel guardrail+router | ✅ |
| Reduce max_tokens 900→500 | ✅ |
| Top-3 chunks only | ✅ |
| Shorter history 10→3 | ✅ |
| Response caching | 🔄 Pending |
| Rule-based templates | 🔄 Pending |
