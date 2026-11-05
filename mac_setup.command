#!/bin/bash
# 수어 학습기 by Langpeu - install (or repair) the Python environment for this Mac (env_<computer name>).
# Normally not needed: mac_app.command installs it automatically on first run.
cd "$(dirname "$0")" || exit 1
bash scripts/_setup.sh called
rc=$?
echo
if [ "$rc" = "0" ]; then echo "Done. Run mac_app.command to open the sign trainer."; else echo "Setup failed."; fi
read -n 1 -s -r -p "Press any key to close."
echo
