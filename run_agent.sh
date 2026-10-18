#!/usr/bin/env bash
# Demo 2 - LangGraph agent chatbot.
set -e
cd "$(dirname "$0")/02_agent_langgraph"
exec ../.venv/bin/python agent.py "$@"
