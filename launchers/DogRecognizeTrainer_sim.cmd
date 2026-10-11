@echo off
REM Debug launcher. A console stays open. The desktop shortcut uses the .vbs.
REM Copy this folder next to DogRecognizeTrainer.exe (or recognize_trainer.exe).
cd /d "%~dp0"
set "EXE=%~dp0DogRecognizeTrainer.exe"
if not exist "%EXE%" set "EXE=%~dp0recognize_trainer.exe"
if not exist "%EXE%" (
  echo No trainer exe next to this launcher.
  pause
  exit /b 1
)
"%EXE%" --sim %*
pause
