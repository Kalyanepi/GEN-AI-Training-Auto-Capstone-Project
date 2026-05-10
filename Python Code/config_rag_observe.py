# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model settings
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0

# RAG hyperparameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4

# Paths
DOCS_DIR = "documents"
INDEX_PATH = "faiss_index"