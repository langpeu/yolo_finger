#!/bin/bash
# 수어 학습기 by Langpeu - macOS launcher: double-click in Finder (Windows: win_app.bat)
cd "$(dirname "$0")" || exit 1
source scripts/_env.sh || { echo; read -n 1 -s -r -p "Press any key to close."; echo; exit 1; }
echo "Sign trainer web page: http://127.0.0.1:8000 (opens in your browser). Use the 종료 button on the page (or close this window) to stop."
"$PY" app/web_app.py
rc=$?
# [Quit rule - do not change] exit code 99 from the web 종료 button closes this window. See README.md.
if [ "$rc" = "99" ]; then
  (sleep 0.5; osascript -e 'tell application "Terminal" to close (every window whose name contains "mac_app")' >/dev/null 2>&1) &
  exit 0
fi
echo
read -n 1 -s -r -p "Press any key to close."
echo
