@echo off
cd /d "%~dp0"
set OLLAMA_VULKAN=1
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python virtualenv .venv not found.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" main.py --gui
