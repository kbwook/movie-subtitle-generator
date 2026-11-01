# PowerShell용 GUI 실행 스크립트
$env:OLLAMA_VULKAN = "1"
Set-Location -Path $PSScriptRoot
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\main.py" --gui
