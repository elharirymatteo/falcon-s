#!/usr/bin/env bash
# From a clean clone to a green `falcons check`. Requires python3.10 and an NVIDIA driver for CUDA 12.6.
set -euo pipefail
cd "$(dirname "$0")/.."
python3.10 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e ".[test]" --no-deps
.venv/bin/falcons check
