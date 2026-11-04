@echo off
chcp 65001 > nul
cd /d "%~dp0"
call scripts\_env.bat
if errorlevel 1 goto end
echo Sign trainer web page: http://127.0.0.1:8000 (opens in your browser). Use the 종료 button on the page (or close this window) to stop.
"%PY%" app\web_app.py
rem [Quit rule - do not change] exit code 99 from the web 종료 button closes this window. See README.md.
if "%errorlevel%"=="99" exit
:end
echo.
echo Press any key to close.
pause > nul
