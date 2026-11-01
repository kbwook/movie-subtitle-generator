# PowerShell 5.1 호환 바탕화면 및 시작 메뉴 바로가기(.lnk) 생성 스크립트
$ErrorActionPreference = "Stop"
$projectDir = $PSScriptRoot
& "$projectDir\.venv\Scripts\python.exe" -c "
import subprocess, os
from pathlib import Path

project_dir = Path(r'$projectDir').resolve()
pythonw = project_dir / '.venv' / 'Scripts' / 'pythonw.exe'
if not pythonw.exists():
    pythonw = project_dir / '.venv' / 'Scripts' / 'python.exe'
main_py = project_dir / 'main.py'

ps_code = f'''
`$WshShell = New-Object -ComObject WScript.Shell
foreach (`$folder in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {{
    `$lnk = Join-Path `$folder '로컬 AI 영화 자막 생성기.lnk'
    `$sc = `$WshShell.CreateShortcut(`$lnk)
    `$sc.TargetPath = '{pythonw}'
    `$sc.Arguments = '\"{main_py}\" --gui'
    `$sc.WorkingDirectory = '{project_dir}'
    `$sc.Description = '로컬 AI 다국어 영화 자막 생성기'
    `$sc.IconLocation = \"`$env:SystemRoot\\System32\\shell32.dll,115\"
    `$sc.Save()
    Write-Output \"Created: `$lnk\"
}}
'''
subprocess.run(['powershell', '-NoProfile', '-Command', ps_code], check=True)
"
