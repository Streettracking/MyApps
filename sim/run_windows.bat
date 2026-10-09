@echo off
REM Launch FlyWire MB x Go2 simulator on Windows
cd /d %~dp0\..
python -m pip install -r requirements-sim.txt
python -m sim.run_sim --seconds 0
pause
