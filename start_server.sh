#!/bin/bash
# macOS: double-click start_server.command, or: bash start_server.sh
cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "[ERROR] Need Python 3 on PATH."
  echo "Install: brew install python   or  https://www.python.org/downloads/"
  read -r -p "Press Enter to close..."
  exit 1
fi

if [ ! -f "serve.py" ]; then
  echo "[ERROR] serve.py not found in $(pwd)"
  read -r -p "Press Enter to close..."
  exit 1
fi

if [ ! -f "vendor/ort/ort.webgpu.min.mjs" ]; then
  echo "[ERROR] Missing vendor/ort/ — copy the full minimal folder including vendor/ort/"
  read -r -p "Press Enter to close..."
  exit 1
fi

echo ""
echo " Leaf Fossil YOLOE Minimal (WebGPU)"
echo " ------------------------------------"
echo " Port auto-picks (default 7681); browser opens automatically."
echo " Python:  $PY ($(command -v "$PY"))"
echo " Keep this terminal OPEN. Press Ctrl+C to stop."
echo " Do NOT open index.html via file://"
echo " Prefer Chrome or Edge (WebGPU)."
echo ""

exec "$PY" -u serve.py
