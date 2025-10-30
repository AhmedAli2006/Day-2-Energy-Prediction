#!/usr/bin/env bash
set -e

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip & install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# Register Jupyter kernel
python -m ipykernel install --user --name "$(basename "$PWD")-env" --display-name "Python ($(basename "$PWD"))"

echo "✅ Done. Activate with: source .venv/bin/activate"
