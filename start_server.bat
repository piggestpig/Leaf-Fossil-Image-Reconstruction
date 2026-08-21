@echo off
setlocal
cd /d "%~dp0"

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY where python >nul 2>&1 && set "PY=python"
if not defined PY (
  echo [ERROR] Need Python 3 on PATH ^(python or py^).
  echo Install from https://www.python.org/downloads/ and re-run.
  pause
  exit /b 1
)

if not exist "vendor\ort\ort.webgpu.min.mjs" (
  echo [ERROR] Missing vendor\ort\ - copy the full minimal folder including vendor\ort\
  pause
  exit /b 1
)

echo.
echo  Leaf Fossil YOLOE Minimal ^(WebGPU^)
echo  ------------------------------------
echo  Do NOT open index.html via file://
echo  Need Chrome/Edge + WebGPU
echo  Port auto-picks ^(default 7681^); browser opens automatically.
echo  Press Ctrl+C to stop.
echo.

%PY% -u serve.py
if errorlevel 1 (
  echo.
  echo [ERROR] Server failed to bind a port.
  echo   Tip: netsh interface ipv4 show excludedportrange protocol=tcp
  echo   Or:  set LEAF_FOSSIL_PORT=9400 ^&^& start_server.bat
  pause
  exit /b 1
)
pause
