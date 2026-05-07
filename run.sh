#!/usr/bin/env bash
set -e

PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python not found. Install Python 3.10+ first."
  exit 1
fi

if [ ! -d ".venv" ]; then
  "$PYTHON" -m venv .venv
fi
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python app.py
