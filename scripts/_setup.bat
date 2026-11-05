@echo off
rem Installs the Python environment for THIS PC into env_<COMPUTERNAME>.
rem Helper: run win_setup.bat instead. Used by win_setup.bat and automatically by win_app.bat (via _env.bat).
setlocal
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
cd /d "%ROOT%"
set "ENVDIR=%ROOT%\env_%COMPUTERNAME%"
set "PY=%ENVDIR%\Scripts\python.exe"
echo ============================================================
echo   Sign trainer - installing Python environment for this PC
echo   folder: env_%COMPUTERNAME%
echo ============================================================

rem ---------- 1) uv (Python installer) ----------
where uv >nul 2>nul
if not errorlevel 1 goto uv_ok
if exist "%USERPROFILE%\.local\bin\uv.exe" goto uv_path
echo [1/5] Installing uv from astral.sh ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
:uv_path
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 goto uv_fail
:uv_ok
echo [1/5] uv ready.

rem ---------- 2) Python 3.12 environment ----------
if not exist "%PY%" goto make_env
"%PY%" -c "import sys" >nul 2>nul
if not errorlevel 1 goto env_exists
:make_env
echo [2/5] Creating Python 3.12 environment ...
uv venv "%ENVDIR%" --python 3.12 --clear
if errorlevel 1 goto fail
goto env_ready
:env_exists
echo [2/5] Environment exists - installing missing packages only.
:env_ready

rem ---------- 3) PyTorch build for this GPU ----------
echo [3/5] Checking the graphics card ...
"%PY%" app\setup_helper.py info
set "TORCH_INDEX="
for /f "usebackq delims=" %%u in (`call "%PY%" app\setup_helper.py index`) do set "TORCH_INDEX=%%u"
if "%TORCH_INDEX%"=="" set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
echo       PyTorch source: %TORCH_INDEX%
echo [3/5] Installing PyTorch (1-3 GB download, this takes a while) ...
uv pip install --python "%PY%" torch torchvision --index-url %TORCH_INDEX%
if errorlevel 1 goto fail

rem ---------- 4) YOLO, MediaPipe, OpenCV ----------
echo [4/5] Installing packages from requirements.txt ...
uv pip install --python "%PY%" -r requirements.txt
if errorlevel 1 goto fail

rem ---------- 5) check ----------
echo [5/5] Checking the installation ...
"%PY%" app\setup_helper.py check
if errorlevel 1 goto fail
echo ============================================================
echo   Setup finished.
echo ============================================================
set "RC=0"
goto done

:uv_fail
echo ERROR: could not install uv. Check the internet connection and try again.
set "RC=1"
goto done
:fail
echo ERROR: setup stopped. Please send the messages above to Claude.
set "RC=1"
goto done

:done
rem started by double-click (not from win_setup.bat / win_app.bat): keep the window open
if /i not "%~1"=="called" (
  echo.
  echo Press any key to close.
  pause > nul
)
endlocal & exit /b %RC%
