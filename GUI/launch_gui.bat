@echo off
setlocal
cd /d "%~dp0"
call "F:\ANACONDA\Scripts\activate.bat" "F:\ANACONDA\envs\HEME"
if errorlevel 1 goto :error
set "LOKY_MAX_CPU_COUNT=1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
start "" "F:\ANACONDA\envs\HEME\pythonw.exe" "%CD%\timeseries_predictor_gui.py"
exit /b 0
:error
echo Failed to activate the HEME Conda environment.
pause
