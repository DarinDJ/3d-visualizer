@echo off
rem ---------------------------------------------------------------------------
rem  Double-click this file to open the viewer with the default model.
rem  Drag a .obj / .stl / .ply file onto this file to open that model instead.
rem ---------------------------------------------------------------------------
cd /d "%~dp0"

python --version >nul 2>nul
if errorlevel 1 (
    echo Python was not found.
    echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
    pause
    exit /b 1
)

python -c "import OpenGL, glfw, numpy" >nul 2>nul
if errorlevel 1 (
    echo Installing the required packages, this happens only once...
    python -m pip install -r requirements.txt
)

python main.py %*
if errorlevel 1 (
    echo.
    echo The viewer stopped with an error, see the message above.
    pause
)
