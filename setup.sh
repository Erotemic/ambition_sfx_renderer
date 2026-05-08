#!/usr/bin/env bash
set -euo pipefail

# Local developer setup. Assumes uv is installed.
uv venv
uv pip install -e .

echo "OK: ambition_sfx_renderer environment is ready"
echo "Try: uv run python -m ambition_sfx_renderer list"
echo "Then: uv run python -m ambition_sfx_renderer render jump"
