#!/usr/bin/env bash
# Demo 2 - LangGraph agent chatbot.
#
#   ./run.sh                 interactive, with memory
#   ./run.sh --ask "..."     one question, then exit
#   ./run.sh --graph         print the graph diagram and exit
#
# Always use this script, or ../.venv/bin/python directly. Plain `python` will
# pick up the wrong packages.

set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python

[ -x "$PY" ] || { echo "No virtualenv yet. Run ../setup.sh first."; exit 1; }

exec "$PY" agent.py "$@"
