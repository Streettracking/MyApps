@echo off
rem Rebuilds both exes: FlyWireGo2Sim (arena simulator) and DogRecognizeTrainer (dog + MB recognition training).
rem Lives in Desktop\FlyWire-Go2 (repo is MyApps\ beside it). Onedir builds from MyApps\.venv.
rem Trainer is --windowed (no console). Stdout goes to %APPDATA%\recognize_trainer\trainer.log, crash.log beside it.
rem Launchers (.vbs .lnk .cmd) beside DogRecognizeTrainer.exe survive the mirror; the .vbs are refreshed from MyApps\launchers.
cd /d "%~dp0MyApps"
set "PY=%~dp0MyApps\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo Missing venv python: %PY%
  exit /b 1
)

"%PY%" -m PyInstaller --noconfirm --clean --onedir --name FlyWireGo2Sim --console --paths . --workpath build\sim --distpath dist --add-data "artifacts;artifacts" --collect-all pygame --hidden-import moderngl --collect-submodules sim tools\gui_exe_entry.py
if errorlevel 1 exit /b 1
robocopy dist\FlyWireGo2Sim "%~dp0FlyWireGo2Sim" /MIR /XD logs /NFL /NDL /NJH /NJS
if errorlevel 8 exit /b 1

"%PY%" -m PyInstaller --noconfirm --clean --onedir --name DogRecognizeTrainer --windowed --paths . --workpath build\trainer --distpath dist --add-data "artifacts;artifacts" --collect-all pygame --hidden-import moderngl --collect-submodules sim tools\recognize_trainer_entry.py
if errorlevel 1 exit /b 1

set "DEST=%~dp0DogRecognizeTrainer"
if not exist "%DEST%" mkdir "%DEST%"
robocopy dist\DogRecognizeTrainer "%DEST%" /MIR /XD logs /XF *.vbs *.lnk *.cmd /NFL /NDL /NJH /NJS
if errorlevel 8 exit /b 1

copy /Y "%~dp0MyApps\launchers\*.vbs" "%DEST%\" >nul

set "YOLO=%~dp0tools\yolo_teacher"
if not exist "%YOLO%" mkdir "%YOLO%"
copy /Y "%~dp0MyApps\tools\yolo_teacher\serve.py" "%YOLO%\" >nul
copy /Y "%~dp0MyApps\tools\yolo_teacher\start_teacher.vbs" "%YOLO%\" >nul
copy /Y "%~dp0MyApps\tools\yolo_teacher\start_teacher.cmd" "%YOLO%\" >nul

echo Rebuilt: %~dp0FlyWireGo2Sim\FlyWireGo2Sim.exe
echo Rebuilt: %~dp0DogRecognizeTrainer\DogRecognizeTrainer.exe
exit /b 0
