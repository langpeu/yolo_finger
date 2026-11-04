@echo off
rem Called by 1_web.bat. Each PC gets its own environment folder in the project folder: env_<COMPUTERNAME>
rem (so the project folder can be copied between PCs without conflicts).
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "ENVDIR=%ROOT%\env_%COMPUTERNAME%"
set "PY=%ENVDIR%\Scripts\python.exe"
set "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0"
"%PY%" -c "import torch, ultralytics, mediapipe, cv2, PIL" >nul 2>nul
if not errorlevel 1 exit /b 0
echo.
echo No working Python environment for this PC (%COMPUTERNAME%) yet.
echo Installing it now - first run on this PC only, about 10-20 minutes.
echo.
call "%~dp0_setup.bat" called
"%PY%" -c "import torch, ultralytics, mediapipe, cv2, PIL" >nul 2>nul
if errorlevel 1 (
  echo.
  echo Setup did not finish. Run 0_setup.bat and send the messages to Claude.
  exit /b 1
)
exit /b 0
