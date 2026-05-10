# insurance_chain.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Load API key from .env
load_dotenv(dotenv_path="/home/ubuntu/insurance_lab/.env")

# Define system and human message prompts
prompt = ChatPromptTemplate.from_messages([
("system",
"You are a knowledgeable insurance advisor for an Indian insurance company. "
"Answer questions about health, life, motor, and property insurance. "
"Be concise, accurate, and professional. If unsure, say so."),
("human", "{query}"),
])

# Initialize the LLM (gpt-4o-mini is cost-effective for labs)
llm = ChatOpenAI(
model="gpt-4o-mini",
temperature=0.3,
openai_api_key=os.getenv("OPENAI_API_KEY")
)
# Build the chain using LCEL (LangChain Expression Language)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser
# The | operator pipes: prompt → llm → parser
# Test with insurance-specific queries
queries = [
"What is the difference between term life and whole life insurance?",
"What is not covered under a standard health insurance policy in India?",
"How is the premium for motor insurance calculated?",
"What is the claim settlement ratio and why does it matter?",
]
for i, query in enumerate(queries, 1):
    print(f"\n" + "="*60)
    print(f"Query {i}: {query}")
    print("="*60)
    response = chain.invoke({"query": query})
    print(response)