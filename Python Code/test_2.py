# test_2.py

import os
import json
import time
from dotenv import load_dotenv
import tiktoken
from openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import sys
sys.path.insert(0, "../lab7")
from chunking_strategies import POLICY_MARKDOWN

load_dotenv(dotenv_path="/home/ubuntu/insurance_lab/.env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

enc = tiktoken.encoding_for_model("gpt-4o-mini")

def tok(t):
    return len(enc.encode(t))

CHUNK_SIZES = [256, 512]

OVERLAP_RATIO = 0.15

def build_index(text: str, chunk_size: int):

    overlap = int(chunk_size * OVERLAP_RATIO)

    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )

    raw_chunks = splitter.create_documents([text])

    docs = [
        Document(page_content=c.page_content)
        for c in raw_chunks
        if tok(c.page_content) > 10
    ]

    token_counts = [tok(d.page_content) for d in docs]

    stats = {
        "chunk_size_target": chunk_size,
        "total_chunks": len(docs),
        "avg_tokens": round(sum(token_counts) / len(token_counts), 1),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
    }

    print(
        f"[{chunk_size} token target] "
        f"{stats['total_chunks']} chunks, "
        f"avg {stats['avg_tokens']:.0f} tokens"
    )

    vs = FAISS.from_documents(docs, embeddings)

    return vs, stats

def retrieve_and_answer(vectorstore, query: str, top_k: int = 3):

    retrieved = vectorstore.similarity_search(query, k=top_k)

    context = "\n\n---\n\n".join(
        [d.page_content for d in retrieved]
    )

    context_tokens = tok(context)

    start = time.time()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer based ONLY on the provided context. "
                    "Be specific. Say NOT FOUND if answer is absent."
                )
            },
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n{context}\n\nQUESTION: {query}"
                )
            },
        ],
        temperature=0,
        max_tokens=200
    )

    latency = round(time.time() - start, 2)

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "context_tokens": context_tokens,
        "latency_s": latency,
        "found": "NOT FOUND" not in answer.upper(),
    }


TEST_QUERIES = [
    {
        "id": "Q1",
        "question": "What is the waiting period for pre-existing diseases?",
        "location": "Section 3",
        "expected": "36 months"
    },
    {
        "id": "Q2",
        "question": "What is the maternity benefit amount for normal delivery?",
        "location": "Section 2.3",
        "expected": "Rs 50,000"
    },
    {
        "id": "Q3",
        "question": (
            "How many days after discharge must I submit "
            "reimbursement documents?"
        ),
        "location": "Section 5.2",
        "expected": "30 days"
    },
    {
        "id": "Q4",
        "question": (
            "What no-claim bonus does the policy offer "
            "and what is the maximum NCB?"
        ),
        "location": "Section 4",
        "expected": "5% per year, max 50%"
    },
]


def _print_matrix(results, indexes):

    print("\n" + "=" * 72)
    print("CHUNK SIZE EXPERIMENT — FINAL MATRIX")
    print("=" * 72)

    print("                 256 tokens    512 tokens")

    print(f"Total Chunks     ", end="")

    for s in CHUNK_SIZES:
        print(f"{indexes[s]['stats']['total_chunks']:>15}", end="")

    print()

    print(f"Avg Tokens/Chunk ", end="")

    for s in CHUNK_SIZES:
        print(f"{indexes[s]['stats']['avg_tokens']:>15.0f}", end="")

    print()

    print("-" * 72)

    for r in results:

        q_label = f"{r['query_id']} ({r['expected']})"

        print(f"{q_label:20}", end="")

        for s in CHUNK_SIZES:

            sc = r["results_by_size"][s]["score"]

            icon = "PASS" if sc == 1 else "FAIL"

            print(f"{icon:>15}", end="")

        print()

    print("-" * 72)

    print(f"TOTAL SCORE /4   ", end="")

    for s in CHUNK_SIZES:

        total = sum(
            r["results_by_size"][s]["score"]
            for r in results
        )

        print(f"{str(total) + '/4':>15}", end="")

    print("\n")


def main():

    all_results = []

    print("=== BUILDING VECTOR INDEXES ===")

    indexes = {}

    for size in CHUNK_SIZES:

        vs, stats = build_index(POLICY_MARKDOWN, size)

        indexes[size] = {
            "vs": vs,
            "stats": stats
        }

    print("\n=== RUNNING QUERIES ===")

    for q_info in TEST_QUERIES:

        q_result = {
            "query_id": q_info["id"],
            "question": q_info["question"],
            "expected": q_info["expected"],
            "results_by_size": {}
        }

        for size in CHUNK_SIZES:

            vs = indexes[size]["vs"]

            result = retrieve_and_answer(
                vs,
                q_info["question"]
            )

            score = (
                1 if q_info["expected"].lower()
                in result["answer"].lower()
                else 0
            )

            result["score"] = score

            q_result["results_by_size"][size] = result

            print(
                f"{q_info['id']} @ {size} tokens: "
                f"score={score}, "
                f"latency={result['latency_s']}s"
            )

            time.sleep(0.5)

        all_results.append(q_result)

    _print_matrix(all_results, indexes)

    with open("test_2.json", "w") as f:
        json.dump(
            {"results": all_results},
            f,
            indent=2,
            default=str
        )


if __name__ == "__main__":
    main()