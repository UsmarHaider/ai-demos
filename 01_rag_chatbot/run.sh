#!/usr/bin/env bash
# Demo 1 - RAG chatbot.
#
#   ./run.sh                 ask questions (builds the index first if missing)
#   ./run.sh ingest          only rebuild the index, and show each stage
#   ./run.sh --ask "..."     one question, then exit
#   ./run.sh --k 5           retrieve 5 chunks instead of 3
#   ./run.sh --hide-prompt   cleaner screen
#
# Always use this script, or ../.venv/bin/python directly. Plain `python` will
# pick up the wrong packages.

set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python

[ -x "$PY" ] || { echo "No virtualenv yet. Run ../setup.sh first."; exit 1; }

if [ "${1:-}" = "ingest" ]; then
  shift
  exec "$PY" ingest.py "$@"
fi

[ -f index/vector_index.json ] || "$PY" ingest.py
exec "$PY" chat.py "$@"
