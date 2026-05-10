# chunking_strategies.py

import os
import json
from dotenv import load_dotenv

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
)

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import tiktoken

load_dotenv(dotenv_path="/home/ubuntu/insurance_lab/.env")

enc = tiktoken.encoding_for_model("gpt-4o-mini")

embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

POLICY_MARKDOWN = """
# InsureSafe HealthGuard Gold — Policy Wording

## Section 1: Definitions

**Insured Person** means any person named in the Schedule.

**Sum Insured** is the maximum payable amount during a Policy Year.

**Pre-Existing Disease** means any condition diagnosed within 48 months prior
to first policy issuance, including diabetes, hypertension, and cardiac conditions.

## Section 2: Coverage and Benefits

### 2.1 In-Patient Hospitalisation

Covered for minimum 24-hour admission.

### 2.2 Pre and Post Hospitalisation

Pre-hospitalisation: 60 days.
Post-hospitalisation: 90 days.

## Section 3: Exclusions

### 3.1 Pre-Existing Diseases

36-month waiting period from policy inception date.

## Section 5: Claims Procedure

### 5.2 Reimbursement Claims

Submit documents within 30 days of discharge.

### 5.3 Settlement Timeline

Settle or reject within 30 days of complete documents.
"""


def strategy_character(text: str):

    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )

    chunks = splitter.create_documents([text])

    print(f"CharacterSplitter: {len(chunks)} chunks")

    return chunks


def strategy_recursive(text: str):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.create_documents([text])

    print(f"RecursiveSplitter: {len(chunks)} chunks")

    return chunks


def strategy_token(text: str):

    splitter = TokenTextSplitter(
        chunk_size=128,
        chunk_overlap=16
    )

    chunks = splitter.create_documents([text])

    print(f"TokenSplitter: {len(chunks)} chunks")

    for i, c in enumerate(chunks[:2]):
        tok = len(enc.encode(c.page_content))
        print(f"Chunk {i}: {tok} tokens")

    return chunks


def strategy_markdown(markdown_text: str):

    headers_to_split = [
        ("#", "document_title"),
        ("##", "section"),
        ("###", "subsection"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split,
        strip_headers=False
    )

    chunks = splitter.split_text(markdown_text)

    print(f"MarkdownSplitter: {len(chunks)} chunks")

    for i, c in enumerate(chunks[:3]):
        print(f"Chunk {i} metadata: {c.metadata}")

    return chunks


def evaluate_strategy(strategy_name: str, chunks: list, query: str):

    print(f"\n--- Evaluating: {strategy_name} ---")

    docs = [
        Document(
            page_content=c if isinstance(c, str) else c.page_content,
            metadata=c.metadata if hasattr(c, "metadata") else {}
        )
        for c in chunks
    ]

    vectorstore = FAISS.from_documents(docs, embeddings)

    retrieved = vectorstore.similarity_search(query, k=3)

    chunk_tokens = [
        len(enc.encode(d.page_content))
        for d in docs
    ]

    avg_tokens = sum(chunk_tokens) / len(chunk_tokens)

    print(f"Total chunks: {len(docs)}")
    print(f"Avg tokens/chunk: {avg_tokens:.0f}")

    print(
        f"Top retrieved chunk preview: "
        f"{retrieved[0].page_content[:120]}..."
    )

    return {
        "strategy": strategy_name,
        "total_chunks": len(docs),
        "avg_tokens": round(avg_tokens, 1),
        "top_chunk_preview": retrieved[0].page_content[:200],
        "top_chunk_metadata": retrieved[0].metadata,
    }


def main():

    query = (
        "What is the claims settlement timeline "
        "and document submission deadline?"
    )

    print(f"Test Query: {query}\n")

    strategies = [
        ("CharacterSplitter", strategy_character(POLICY_MARKDOWN)),
        ("RecursiveSplitter", strategy_recursive(POLICY_MARKDOWN)),
        ("TokenSplitter", strategy_token(POLICY_MARKDOWN)),
        ("MarkdownSplitter", strategy_markdown(POLICY_MARKDOWN)),
    ]

    all_results = []

    for name, chunks in strategies:

        result = evaluate_strategy(name, chunks, query)

        all_results.append(result)

    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 60)

    print("Strategy Chunks Avg Tokens")
    print("-" * 45)

    for r in all_results:
        print(
            f"{r['strategy']:<22} "
            f"{r['total_chunks']:>8} "
            f"{r['avg_tokens']:>12.0f}"
        )

    with open("strategy_results.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()