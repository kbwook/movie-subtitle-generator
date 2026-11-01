# Intel Core Ultra 7 155H + Intel Arc Graphics 로컬 LLM 환경 자동 구축 스크립트
# 역할:
# 1. Intel Arc 내장 GPU의 Vulkan 가속(OLLAMA_VULKAN=1) 환경변수 등록
# 2. Ollama 설치 여부 확인 및 미설치 시 winget으로 자동 설치
# 3. Ollama 서버 구동 및 중·일·영 -> 한국어 번역용 Qwen2.5 모델 다운로드

$ErrorActionPreference = "Continue"

Write-Host "=== [1/3] Intel Arc GPU Vulkan 가속 환경변수 설정 ===" -ForegroundColor Cyan
[Environment]::SetEnvironmentVariable("OLLAMA_VULKAN", "1", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_IGPU_ENABLE", "1", "User")
$env:OLLAMA_VULKAN = "1"
$env:OLLAMA_IGPU_ENABLE = "1"
Write-Host "-> OLLAMA_VULKAN=1 및 OLLAMA_IGPU_ENABLE=1 설정 완료 (Intel Arc 내장 GPU 가속 활성화)" -ForegroundColor Green

Write-Host "`n=== [2/3] Ollama 설치 상태 확인 ===" -ForegroundColor Cyan
$ollamaPath = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
if (-not (Get-Command ollama -ErrorAction SilentlyContinue) -and -not (Test-Path $ollamaPath)) {
    Write-Host "-> Ollama가 설치되어 있지 않습니다. winget을 통해 설치를 진행합니다..." -ForegroundColor Yellow
    winget install --id Ollama.Ollama -e --accept-package-agreements --accept-source-agreements
}

if (Test-Path $ollamaPath) {
    $env:Path += ";$env:LOCALAPPDATA\Programs\Ollama"
}

Write-Host "`n=== [3/3] Qwen2.5-7B 다국어 번역 모델 다운로드 ===" -ForegroundColor Cyan
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    # 백그라운드 서버가 떠 있지 않으면 시작
    $proc = Get-Process -Name "ollama*" -ErrorAction SilentlyContinue
    if (-not $proc) {
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 3
    }
    Write-Host "-> qwen2.5:7b 모델을 다운로드합니다 (약 4.7GB)..." -ForegroundColor Yellow
    ollama pull qwen2.5:7b
    Write-Host "`n[완료] 모든 로컬 LLM 준비가 끝났습니다! 이제 자막 생성기를 실행할 수 있습니다." -ForegroundColor Green
} else {
    Write-Host "-> Ollama 실행 파일을 찾을 수 없습니다. 터미널을 재시작하거나 https://ollama.com 에서 설치해 주세요." -ForegroundColor Red
}
