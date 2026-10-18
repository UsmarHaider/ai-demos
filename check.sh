#!/usr/bin/env bash
cd "$(dirname "$0")"
exec ./.venv/bin/python preflight.py
