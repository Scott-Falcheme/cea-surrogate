@echo off
setlocal

cd /d "%~dp0"

set "CONDA_ENV=C:\Users\ss3340_admin\.conda\envs\cea-surrogate"
set "GUI_SCRIPT=%CD%\timeseries_predictor_gui.py"

if not exist "%CONDA_ENV%\pythonw.exe" goto :missing_env
if not exist "%GUI_SCRIPT%" goto :missing_gui

set "PATH=%CONDA_ENV%;%CONDA_ENV%\Library\bin;%CONDA_ENV%\Scripts;%CONDA_ENV%\bin;%PATH%"
set "PYTHONPATH=%CD%\..\Scripts;%PYTHONPATH%"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "LOKY_MAX_CPU_COUNT=1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"

start "" "%CONDA_ENV%\pythonw.exe" -c "import torch, runpy; runpy.run_path(r'%GUI_SCRIPT%', run_name='__main__')"
exit /b 0

:missing_env
echo Could not find the cea-surrogate Conda environment at "%CONDA_ENV%".
pause
exit /b 1

:missing_gui
echo Could not find the GUI script at "%GUI_SCRIPT%".
pause
exit /b 1
