# observe_retrieval.py

import json
from config import *

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader
)

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import (
    OpenAIEmbeddings,
    ChatOpenAI
)

from langchain_community.vectorstores import FAISS

from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate


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

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

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
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        #retriever=vectorstore.as_retriever(
         #   search_kwargs={"k": 3}
        #),
        #Maximum Marginal Relevance (MMR)
        retriever=vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":3,
            "fetch_k": 20,
            "lambda_mult": 0.7 # 0 is more diverse and 1 is less diverse from context
        }    
    ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return chain, vectorstore


def observe_query(chain, vectorstore, query, label):

    print("\n" + "=" * 60)

    print(f"TEST: {label}")
    print(f"Query: {query}")

    raw_hits = vectorstore.similarity_search_with_score(
        query,
        k=3
    )

    print("\nRaw Retrieved Chunks:")

    for i, (doc, score) in enumerate(raw_hits):

        print(f"[{i+1}] Score: {score:.4f}")
        print(doc.page_content[:120])
        print()

    result = chain.invoke({"query": query})

    print(f"\nLLM Answer:\n{result['result']}")

if __name__ == "__main__":

    chain, vectorstore = build_qa_chain()

    test_cases = [
        (
            "How many sick leave days do employees get?",
            "RELEVANT"
        ),

        (
            "What is the capital of France?",
            "OUT-OF-DOMAIN"
        ),

        (
            "How many annual leave days?",
            "VOCABULARY MISMATCH"
        ),

        (
            "Tell me about leave policies and expense policies",
            "MULTI-TOPIC"
        ),

        (
            "What happens if I miss the 90 day deadline?",
            "AMBIGUOUS"
        )
    ]

    for query, label in test_cases:
        observe_query(chain, vectorstore, query, label)