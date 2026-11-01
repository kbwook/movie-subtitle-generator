@echo off
chcp 65001 > nul
cd /d "%~dp0"
set OLLAMA_VULKAN=1
if not exist ".venv\Scripts\python.exe" (
    echo [오류] 가상환경(.venv)이 없습니다. 먼저 패키지를 설치해 주세요.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" main.py --gui
