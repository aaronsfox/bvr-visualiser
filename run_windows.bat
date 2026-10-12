@echo off
rem First run: creates a private Python environment next to this file and installs the app.
rem Later runs: just starts it. Needs Python 3.10 or newer from python.org.
cd /d "%~dp0"
if not exist .venv (
    echo Setting up BVR Motion Viewer for the first time. This takes a few minutes...
    py -3 -m venv .venv || python -m venv .venv
    .venv\Scripts\python -m pip install --upgrade pip
    .venv\Scripts\python -m pip install -e .
)
start "" .venv\Scripts\pythonw -m bvrviewer %*
