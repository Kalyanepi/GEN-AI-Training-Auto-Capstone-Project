#!/usr/bin/env bash
# Build the FAISS index from the source PDFs. Run once after PDFs change.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a; source .env; set +a
fi

echo "[ingest] Building FAISS index from $PDF_DIR ..."
python -m ingestion.build_index
echo "[ingest] Done. Index at: $FAISS_INDEX_DIR"
