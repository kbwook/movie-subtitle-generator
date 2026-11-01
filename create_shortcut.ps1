# 바탕화면 및 Windows 시작 메뉴에 '로컬 AI 영화 자막 생성기' 바로가기(.lnk)를 생성하는 스크립트
# pythonw.exe를 사용하여 검은색 콘솔 창 없이 깔끔하게 GUI 창만 단독 실행됩니다.

$ErrorActionPreference = "Stop"
$projectDir = $PSScriptRoot
$pythonwPath = Join-Path $projectDir ".venv\Scripts\pythonw.exe"
$mainPyPath = Join-Path $projectDir "main.py"

if (-not (Test-Path $pythonwPath)) {
    $pythonwPath = Join-Path $projectDir ".venv\Scripts\python.exe"
}

$WshShell = New-Object -ComObject WScript.Shell

# 1. 바탕화면 바로가기 생성
$desktopDir = [Environment]::GetFolderPath("Desktop")
$desktopShortcutPath = Join-Path $desktopDir "로컬 AI 영화 자막 생성기.lnk"
$sc1 = $WshShell.CreateShortcut($desktopShortcutPath)
$sc1.TargetPath = $pythonwPath
$sc1.Arguments = "`"$mainPyPath`" --gui"
$sc1.WorkingDirectory = $projectDir
$sc1.Description = "로컬 AI 다국어 영화 자막 생성기 (Faster-Whisper + Ollama Qwen2.5)"
$sc1.IconLocation = "$env:SystemRoot\System32\shell32.dll,115"
$sc1.Save()
Write-Host "✅ 바탕화면 바로가기 생성 완료: $desktopShortcutPath" -ForegroundColor Green

# 2. Windows 시작 메뉴(모든 앱) 바로가기 생성
$startMenuDir = [Environment]::GetFolderPath("Programs")
$startMenuShortcutPath = Join-Path $startMenuDir "로컬 AI 영화 자막 생성기.lnk"
$sc2 = $WshShell.CreateShortcut($startMenuShortcutPath)
$sc2.TargetPath = $pythonwPath
$sc2.Arguments = "`"$mainPyPath`" --gui"
$sc2.WorkingDirectory = $projectDir
$sc2.Description = "로컬 AI 다국어 영화 자막 생성기 (Faster-Whisper + Ollama Qwen2.5)"
$sc2.IconLocation = "$env:SystemRoot\System32\shell32.dll,115"
$sc2.Save()
Write-Host "✅ 시작 메뉴 바로가기 생성 완료: $startMenuShortcutPath" -ForegroundColor Green
