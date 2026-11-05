# Sourced by mac_app.command (macOS / Linux). Each computer gets its own environment folder: env_<computer name>
# (so the project folder can be copied between computers without conflicts). Sets ROOT, ENVDIR, PY.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_NAME="$(scutil --get LocalHostName 2>/dev/null || hostname -s)"
ENVDIR="$ROOT/env_$HOST_NAME"
PY="$ENVDIR/bin/python"
export PYTORCH_ENABLE_MPS_FALLBACK=1          # Apple GPU: run the few unsupported operations on the CPU
_env_ok() { "$PY" -c "import torch, ultralytics, mediapipe, cv2, PIL" >/dev/null 2>&1; }
if ! _env_ok; then
  echo
  echo "No working Python environment for this computer ($HOST_NAME) yet."
  echo "Installing it now - first run on this computer only, about 10-20 minutes."
  echo
  bash "$ROOT/scripts/_setup.sh" called
  if ! _env_ok; then
    echo
    echo "Setup did not finish. Run mac_setup.command and send the messages to Claude."
    return 1
  fi
fi
return 0
