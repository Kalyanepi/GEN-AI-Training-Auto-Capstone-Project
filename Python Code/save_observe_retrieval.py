
# save_observe_retrieval.py

import json
from datetime import datetime, UTC
from config import *

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


OUTPUT_FILE = "retrieval_observations.json"


def build_qa_chain():

    loader = DirectoryLoader(
        "documents",
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )

    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    prompt = PromptTemplate(
        template="""
Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 20,
                "lambda_mult": 0.7
            }
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return chain, vectorstore


def observe_query(chain, vectorstore, query, label):

    raw_hits = vectorstore.similarity_search_with_score(query, k=3)

    retrieved_chunks = []
    for i, (doc, score) in enumerate(raw_hits, start=1):
        retrieved_chunks.append({
            "rank": i,
            "score": float(score),
            "content": doc.page_content
        })

    result = chain.invoke({"query": query})

    return {
        "label": label,
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "llm_answer": result["result"]
    }


if __name__ == "__main__":

    chain, vectorstore = build_qa_chain()

    test_cases = [
        ("How many sick leave days do employees get?", "RELEVANT"),
        ("What is the capital of France?", "OUT-OF-DOMAIN"),
        ("How many annual leave days?", "VOCABULARY MISMATCH"),
        ("Tell me about leave policies and expense policies", "MULTI-TOPIC"),
        ("What happens if I miss the 90 day deadline?", "AMBIGUOUS")
    ]

    observations = {
        "run_timestamp": datetime.now(UTC).isoformat(),
        "results": []
    }

    for query, label in test_cases:
        observations["results"].append(
            observe_query(chain, vectorstore, query, label)
        )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(observations, f, indent=2, ensure_ascii=False)
