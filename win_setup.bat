@echo off
chcp 65001 > nul
cd /d "%~dp0"
rem Install (or repair) the Python environment for this PC (env_<COMPUTERNAME>). Launchers also do this automatically on first run.
call scripts\_setup.bat called
set RC=%errorlevel%
echo.
if not "%RC%"=="0" (echo Setup failed.) else (echo Done. Run win_app.bat to open the sign trainer.)
echo Press any key to close.
pause > nul
