@echo off
REM run_demo.bat - create venv, install requirements, fetch dataset, and run PCA demo
REM Place this file in the project root: c:\Projects\PCA_Analysis\Wine-Quality-Analysis-PCA

nREM Switch to script directory
cd /d "%~dp0"

necho [1/6] Creating virtual environment (.venv) if missing...
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto error
) else (
    echo ".venv already exists"
)

necho [2/6] Activating virtual environment...
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto error

necho [3/6] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 goto error

necho [4/6] Installing requirements from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 goto error

necho [5/6] Fetching dataset (main.py)...
python main.py
if errorlevel 1 goto error

necho [6/6] Running PCA demo (pca_demo.py)...
python pca_demo.py
if errorlevel 1 goto error

necho All steps completed. Outputs (plots, summary) are saved to pca_outputs\ if the demo runs successfully.
pause
exit /b 0

n:error
echo.
echo Error encountered during run. Check the output above for details.
pause
exit /b 1
