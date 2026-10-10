@echo off
REM Shortcut this file onto the Desktop. It starts the YOLO service for the trainer.
REM The trainer itself does not load torch. Leave this window open while teaching.
set "VENV=%USERPROFILE%\Desktop\FlyWire-Go2\yolo\venv\Scripts\python.exe"
set "WEIGHTS=%USERPROFILE%\Desktop\FlyWire-Go2\yolo\runs\go2_mix_v2\weights\best.pt"
if not exist "%VENV%" (
  echo No YOLO venv at %VENV%
  pause
  exit /b 1
)
if not exist "%WEIGHTS%" (
  echo No weights at %WEIGHTS%
  pause
  exit /b 1
)
"%VENV%" "%~dp0serve.py" --weights "%WEIGHTS%" --host 127.0.0.1 --port 8091
pause
