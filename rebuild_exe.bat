@echo off
REM One-file trainer, no console (--windowed). The 3D brain is the same exe with --mb-view3d.
REM Stdout goes to %APPDATA%\recognize_trainer\trainer.log. crash.log stays beside it.
REM FlyWireGo2Sim stays a console exe. Windows add-data separator is ;
cd /d "%~dp0"

pyinstaller --onefile --windowed --noconfirm --name recognize_trainer --collect-all pygame --add-data "artifacts\connectome_mb_v1.npz;artifacts" --add-data "artifacts\connectome_mb_v1_np1.npz;artifacts" --add-data "artifacts\connectome_mb_v1_neurons.csv.gz;artifacts" --add-data "artifacts\connectome_mb_v1_edges.csv.gz;artifacts" --add-data "artifacts\fly_brain_view.npz;artifacts" --hidden-import sim.recognize --hidden-import sim.recognize_train --hidden-import sim.recognize_train_live --hidden-import sim.frame_sense --hidden-import sim.train_monitor --hidden-import sim.go2_udp --hidden-import sim.raw_sense --hidden-import sim.world --hidden-import sim.mb_runtime --hidden-import sim.mb_train --hidden-import sim.mb_confidence --hidden-import sim.map_marks --hidden-import sim.lidar_fresh --hidden-import sim.learn_flash --hidden-import sim.pilot --hidden-import sim.onboard_link --hidden-import sim.npz_compat --hidden-import sim.hemifield --hidden-import sim.frame_record --hidden-import sim.mb_view3d --hidden-import sim.mb_flywire --hidden-import sim.yolo_teacher --hidden-import numpy tools\recognize_trainer_entry.py
if errorlevel 1 exit /b 1

pyinstaller --onefile --noconfirm --name FlyWireGo2Sim --collect-all pygame --collect-submodules sim --add-data "artifacts\connectome_mb_v1.npz;artifacts" --hidden-import numpy sim\run_sim.py
if errorlevel 1 exit /b 1

set "DEST=%USERPROFILE%\Desktop\DogRecognizeTrainer"
if not exist "%DEST%" mkdir "%DEST%"
REM /MIR refreshes the exes. /XD leaves logs and кадры in place. /XF leaves launchers in place; the copy below refreshes the .vbs.
robocopy "%~dp0dist" "%DEST%" /MIR /XD logs кадры /XF *.vbs *.lnk *.cmd install_shortcuts.vbs
if errorlevel 8 exit /b 1

copy /Y "%DEST%\recognize_trainer.exe" "%DEST%\DogRecognizeTrainer.exe" >nul
copy /Y "%~dp0launchers\*.vbs" "%DEST%\" >nul
copy /Y "%~dp0launchers\install_shortcuts.vbs" "%DEST%\" >nul

set "YOLO=%USERPROFILE%\Desktop\FlyWire-Go2\tools\yolo_teacher"
if not exist "%YOLO%" mkdir "%YOLO%"
copy /Y "%~dp0tools\yolo_teacher\serve.py" "%YOLO%\" >nul
copy /Y "%~dp0tools\yolo_teacher\start_teacher.vbs" "%YOLO%\" >nul
copy /Y "%~dp0tools\yolo_teacher\start_teacher.cmd" "%YOLO%\" >nul
exit /b 0
