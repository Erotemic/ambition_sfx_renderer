#!/usr/bin/env bash
set -euo pipefail


if command -v apt-get >/dev/null 2>&1; then
    cat <<'EOF'
Recommended Ubuntu/Debian native deps:

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg
EOF
fi


# Local developer setup. Assumes uv is installed.
PYTHON_VERSION=3.12
uv venv --python "$PYTHON_VERSION"
source .venv/bin/activate
uv pip install -e .

echo "OK: ambition_sfx_renderer environment is ready"
echo "Try: uv run python -m ambition_sfx_renderer list"
echo "Then: uv run python -m ambition_sfx_renderer render jump"
