#!/usr/bin/env zsh
# comfyui.sh - Launch ComfyUI using the shared workspace .venv
# Usage: ./comfyui.sh [extra ComfyUI args]

set -e

WORKSPACE_DIR="$(cd "$(dirname "$0")" && pwd)"
COMFYUI_DIR="$WORKSPACE_DIR/vendor/ComfyUI"
VENV_PYTHON="$WORKSPACE_DIR/.venv/bin/python"

if [[ ! -d "$COMFYUI_DIR" ]]; then
  echo "❌ ComfyUI not found at $COMFYUI_DIR"
  echo "   Run: git clone https://github.com/comfyanonymous/ComfyUI.git vendor/ComfyUI"
  exit 1
fi

if [[ ! -f "$VENV_PYTHON" ]]; then
  echo "❌ .venv not found — run: uv sync --all-packages"
  exit 1
fi

echo "🚀 Starting ComfyUI from shared .venv"
echo "   Python : $VENV_PYTHON"
echo "   Dir    : $COMFYUI_DIR"
echo "   UI     : http://localhost:8188"
echo ""

# Load .env from workspace root
if [[ -f "$WORKSPACE_DIR/.env" ]]; then
  set -a
  source "$WORKSPACE_DIR/.env"
  set +a
fi

# Suppress macOS symlink warning and authenticate with HF Hub
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
[[ -n "$HF_TOKEN" ]] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# Link project's models/ and outputs/ into ComfyUI so they share the same dirs
mkdir -p "$WORKSPACE_DIR/models"
[[ -L "$COMFYUI_DIR/models" ]] || ln -sf "$WORKSPACE_DIR/models" "$COMFYUI_DIR/models"
[[ -L "$COMFYUI_DIR/output" ]] || ln -sf "$WORKSPACE_DIR/outputs" "$COMFYUI_DIR/output"

cd "$COMFYUI_DIR"
exec "$VENV_PYTHON" main.py \
  --force-fp16 \
  --preview-method auto \
  "$@"
