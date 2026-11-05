#!/bin/bash
# Installs the Python environment for THIS computer (macOS / Linux) into env_<computer name>.
# Helper: run mac_setup.command instead. Used by mac_setup.command and automatically by mac_app.command (via _env.sh).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
HOST_NAME="$(scutil --get LocalHostName 2>/dev/null || hostname -s)"
ENVDIR="$ROOT/env_$HOST_NAME"
PY="$ENVDIR/bin/python"
fail() { echo "ERROR: $1"; echo "Please send the messages above to Claude."; exit 1; }
echo "============================================================"
echo "  Sign trainer - installing Python environment for this computer"
echo "  folder: env_$HOST_NAME"
echo "============================================================"

# ---------- 1) uv (Python installer) ----------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "[1/5] Installing uv from astral.sh ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh || fail "could not install uv. Check the internet connection and try again."
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || fail "uv was installed but cannot be found."
fi
echo "[1/5] uv ready."

# ---------- 2) Python 3.12 environment ----------
if "$PY" -c "import sys" >/dev/null 2>&1; then
  echo "[2/5] Environment exists - installing missing packages only."
else
  echo "[2/5] Creating Python 3.12 environment ..."
  uv venv "$ENVDIR" --python 3.12 --clear || fail "could not create the environment."
fi

# ---------- 3) PyTorch ----------
if [ "$(uname)" = "Darwin" ]; then
  echo "[3/5] Installing PyTorch (Apple GPU / MPS support is built in) ..."
  uv pip install --python "$PY" torch torchvision || fail "PyTorch install failed."
else
  echo "[3/5] Checking the graphics card ..."
  "$PY" app/setup_helper.py info
  TORCH_INDEX="$("$PY" app/setup_helper.py index)"
  echo "      PyTorch source: $TORCH_INDEX"
  uv pip install --python "$PY" torch torchvision --index-url "$TORCH_INDEX" || fail "PyTorch install failed."
fi

# ---------- 4) YOLO, MediaPipe, OpenCV ----------
echo "[4/5] Installing packages from requirements.txt ..."
uv pip install --python "$PY" -r requirements.txt || fail "package install failed."

# ---------- 5) check ----------
echo "[5/5] Checking the installation ..."
"$PY" app/setup_helper.py check || fail "check failed."
echo "============================================================"
echo "  Setup finished."
echo "============================================================"
exit 0
