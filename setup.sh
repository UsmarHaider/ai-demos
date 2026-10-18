#!/usr/bin/env bash
# Creates the project's own virtualenv. Run once per machine.
set -e
cd "$(dirname "$0")"
echo "Creating virtualenv in .venv ..."
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
echo
echo "Done. Next:"
echo "  ./01_rag_chatbot/run.sh ingest                build the RAG index"
echo "  ./.venv/bin/python -m shared.preflight        verify everything works"
