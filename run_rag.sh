#!/usr/bin/env bash
# Demo 1 - RAG chatbot. Builds the index first if it is missing.
set -e
cd "$(dirname "$0")/01_rag_chatbot"
PY=../.venv/bin/python
[ -f index/vector_index.json ] || "$PY" ingest.py
exec "$PY" chat.py "$@"
