$ErrorActionPreference = "Stop"
$projectDir = $PSScriptRoot
$pythonwPath = Join-Path $projectDir ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pythonwPath)) {
    $pythonwPath = Join-Path $projectDir ".venv\Scripts\python.exe"
}
$mainPyPath = Join-Path $projectDir "main.py"

# "로컬 AI 영화 자막 생성기.lnk" in Unicode codepoints (ASCII-safe for PowerShell 5.1)
$shortcutName = "$([char]0xB85C)$([char]0xCEEC) AI $([char]0xC601)$([char]0xD654) $([char]0xC790)$([char]0xB9C9) $([char]0xC0DD)$([char]0xC131)$([char]0xAE30).lnk"

$WshShell = New-Object -ComObject WScript.Shell

foreach ($folder in @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs"))) {
    $lnkPath = Join-Path $folder $shortcutName
    $sc = $WshShell.CreateShortcut($lnkPath)
    $sc.TargetPath = $pythonwPath
    $sc.Arguments = "`"$mainPyPath`" --gui"
    $sc.WorkingDirectory = $projectDir
    $sc.IconLocation = "$env:SystemRoot\System32\shell32.dll,115"
    $sc.Save()
    Write-Host "Shortcut created: $lnkPath" -ForegroundColor Green
}
