#!/usr/bin/env bash
# Run the full 30-question + 4-adversarial evaluation suite.
set -euo pipefail

cd "$(dirname "$0")/.."

API_BASE="${API_BASE_URL:-http://localhost:8000}"

echo "[eval] Main suite (30 questions) against $API_BASE"
python -m evaluation.run_eval --api-base "$API_BASE" --output evaluation_results.json

echo
echo "[eval] Adversarial guardrail tests (4 cases)"
python -m evaluation.adversarial_tests --api-base "$API_BASE"
