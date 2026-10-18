#!/usr/bin/env bash
# Demo 1, stage 1 only - rebuild the RAG index (run this on stage to show it).
set -e
cd "$(dirname "$0")/01_rag_chatbot"
exec ../.venv/bin/python ingest.py "$@"
