#!/usr/bin/env bash
# First run: creates a private Python environment next to this file and installs the app.
# Later runs: just starts it. Needs Python 3.10 or newer.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
    echo "Setting up BVR Motion Viewer for the first time. This takes a few minutes..."
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -e .
fi
exec .venv/bin/python -m bvrviewer "$@"
