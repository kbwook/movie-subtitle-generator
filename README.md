# 로컬 AI 다국어 영화 자막 생성기 (Intel Core Ultra 7 + Arc 최적화)

다양한 코덱(`.mkv`, `.mp4`, `.avi`, `.ts`, `.mov`, `.wmv`, `.webm` 등)의 **중국어·일본어·영어 영화 파일**에서 음성을 추출하여 자연스러운 **한국어 자막(`.srt`, `.ass`)**을 자동 생성하는 로컬 AI 파이프라인 및 데스크톱 GUI 프로그램입니다.

---

## 1. 시스템 구성 내역 (Architecture & Tech Stack)

현재 노트북(**Intel® Core™ Ultra 7 155H / Intel® Arc™ Graphics / 32GB LPDDR5X RAM**)에 맞춰 메모리 점유율과 처리 속도를 최적화한 4단계 순차 파이프라인으로 구성되어 있습니다.

| 단계 | 담당 모듈 | 적용 기술 / 모델 | 상세 구성 및 최적화 내역 |
| :--- | :--- | :--- | :--- |
| **1단계: 코덱 분석 & 오디오 추출** | `pipeline/extractor.py` | **PyAV** (내장 FFmpeg C 바인딩) | 별도 외부 코덱 설치 없이 H.264/HEVC/AV1 비디오 및 AAC, AC3, E-AC3, DTS, FLAC, Opus 등 모든 오디오 트랙을 탐색하고 **16kHz Mono 16-bit PCM WAV**로 고속 디코딩 |
| **2단계: 다국어 음성 인식 (STT)** | `pipeline/transcriber.py` | **Faster-Whisper** (`large-v3-turbo`) + **Silero VAD** | Ultra 7 155H 멀티코어(`int8`, 12스레드) 추론 및 `Silero VAD` 음성 구간 검출로 영화 배경음악(BGM) 구간의 자막 환각(Hallucination) 차단. **STT 완료 즉시 Whisper 모델 메모리 해제(Sequential Offloading)** |
| **3단계: 문맥 기반 로컬 LLM 번역** | `pipeline/translator.py` | **Ollama** (`qwen2.5:7b` 기본 / `14b` 선택) | Intel Arc 내장 GPU의 **Vulkan 가속(`OLLAMA_VULKAN=1`)** 및 백그라운드 서버 자동 구동. 직전 5문장(문맥 참조) + 현재 10문장 배치를 JSON 스키마로 묶어 번역하여 화자의 반말/존댓말과 고유명사 일관성 유지 |
| **4단계: 자막 포맷팅 및 싱크 보정** | `pipeline/formatter.py` | 자체 SRT / ASS 렌더러 | 타임스탬프 겹침 및 최소 노출 시간 자동 보정. **한국어 단독 자막**, **한글+외국어 원문 이중 자막**, **원문 자막(`.orig.srt`)** 동시 출력 지원 |

---

## 2. 프로젝트 파일 구조

```text
자막생성기/
├── run_gui.bat              # [원클릭 실행] 더블클릭으로 데스크톱 GUI 즉시 실행
├── main.py                  # CLI 및 GUI 통합 진입점
├── gui.py                   # 멀티 파일 대기열(Queue) 관리 및 상세 옵션 데스크톱 GUI
├── config.yaml              # STT / 로컬 LLM / 자막 출력 기본 설정 파일
├── setup_env.ps1            # Ollama 설치, Vulkan 가속 설정, Qwen2.5-7B 자동 다운로드 스크립트
├── requirements.txt         # 가상환경(.venv)에 설치된 Python 패키지 목록
├── pipeline/
│   ├── models.py            # 자막 세그먼트, 오디오 트랙, 비디오 메타데이터 모델
│   ├── extractor.py         # 다중 코덱 오디오 트랙 분석 및 16kHz WAV 추출기
│   ├── transcriber.py       # Faster-Whisper + Silero VAD 음성 인식기
│   ├── translator.py        # Ollama 로컬 LLM 슬라이딩 윈도우 문맥 번역기
│   ├── formatter.py         # .srt / .ass(스타일 자막) 생성 및 싱크 보정기
│   └── runner.py            # 1~4단계 통합 실행 오케스트레이터
└── tests/
    └── test_pipeline.py     # 다중 코덱 추출, JSON 파싱, GUI 대기열 단위 테스트
```

---

## 3. 실행 방법

### 3.1 데스크톱 GUI 실행 (가장 간편한 방법)

1. 프로젝트 폴더 내 **`run_gui.bat`** 파일을 더블클릭하거나, 터미널에서 아래 명령어를 실행합니다.
   ```powershell
   .\.venv\Scripts\python.exe main.py --gui
   ```
2. **영화 리스트(대기열) 추가 및 관리**:
   * **`➕ 영화 파일 추가`** 또는 **`📁 폴더 통째로 추가`** 버튼을 눌러 자막을 만들 영화들을 등록합니다.
   * 리스트에 등록되는 즉시 각 영화의 **비디오 코덱 / 재생 시간 / 용량**과 **내장된 오디오 트랙 목록(언어·코덱)**이 자동으로 표시됩니다.
   * 특정 영화에 다중 오디오(예: 일본어/중국어/영어 더빙)가 들어있다면, 해당 항목을 클릭한 뒤 하단 **`▶ 선택 영화 개별 설정`**에서 원하는 오디오 트랙, 개별 언어, 작품 배경 힌트(예: `무협 시대극`, `SF 영화`)를 지정하고 **`선택 항목에 적용`**을 누릅니다.
3. **옵션 지정**:
   * **음성 인식(STT) 및 자막 출력 옵션**: 원본 언어(`auto`, `zh`, `ja`, `en`), Whisper 모델(`large-v3-turbo` 권장), `.srt` / `.ass` 출력 여부, **`한글 + 외국어 원문 동시 표기 (이중 자막)`** 여부를 체크합니다.
   * **로컬 LLM 번역 설정**: 기본 설치된 `qwen2.5:7b`(속도·품질 밸런스) 또는 `qwen2.5:14b`(고품질 번역)를 선택할 수 있습니다. 새로운 모델을 선택한 경우 **`⬇ 모델 준비/점검`** 버튼을 누르면 GUI에서 바로 모델을 다운로드합니다.
   * 자주 쓰는 옵션 조합은 **`💾 현재 옵션 기본값 저장`** 버튼을 눌러 `config.yaml`에 영구 저장할 수 있습니다.
4. **자막 생성 시작 및 중지**:
   * 우측 하단 **`▶ 대기열 자막 생성 시작`** 버튼을 누르면 등록된 영화들이 순차적으로 처리됩니다.
   * 작업 도중 멈추고 싶을 때 **`⏹ 현재 파일 완료 후 중지`**를 누르면 현재 처리 중인 영화의 자막까지 안전하게 저장한 뒤 멈춥니다.

---

### 3.2 명령줄(CLI) 실행 방법

터미널(PowerShell)에서 자동화나 배치 스크립트로 실행할 때 사용할 수 있습니다.

```powershell
# 1. 환경 및 로컬 LLM(Ollama) 연결 상태 점검
.\.venv\Scripts\python.exe main.py --check-env

# 2. 단일 영화 파일 자막 생성 (언어 자동 감지 -> 한국어 번역)
.\.venv\Scripts\python.exe main.py "D:\Movies\sample_movie.mkv"

# 3. 원본 언어 지정 (zh: 중국어, ja: 일본어, en: 영어) + 한글/원문 이중 자막 생성
.\.venv\Scripts\python.exe main.py "D:\Movies\jp_movie.mp4" --lang ja --bilingual

# 4. 고품질 14B 모델로 번역 실행
.\.venv\Scripts\python.exe main.py "D:\Movies\cn_movie.mkv" --lang zh --llm-model qwen2.5:14b

# 5. 폴더 내 모든 영화 파일 일괄 자막 생성
.\.venv\Scripts\python.exe main.py "D:\Movies\"

# 6. 영화 파일 내 오디오 트랙(코덱/언어) 목록 확인 및 특정 트랙 번호 지정 추출
.\.venv\Scripts\python.exe main.py "D:\Movies\multi_audio.mkv" --list-tracks
.\.venv\Scripts\python.exe main.py "D:\Movies\multi_audio.mkv" --track 1

# 7. LLM 번역 없이 원문 자막(STT)만 빠르게 추출
.\.venv\Scripts\python.exe main.py "D:\Movies\en_movie.mp4" --lang en --no-translate
```

---

## 4. 생성되는 자막 파일 규칙

원본 영화 파일이 `movie.mkv`이고 감지된 언어가 일본어(`ja`)인 경우, 기본적으로 같은 폴더(또는 GUI에서 지정한 폴더)에 다음 파일들이 생성됩니다.

* **`movie.ko.srt`**: 한국어 번역 표준 SRT 자막 (이중 자막 체크 시 한국어+원문 동시 수록)
* **`movie.ko.ass`**: 가독성 높은 외곽선·그림자 스타일이 적용된 고화질 ASS 자막 (이중 자막 체크 시 한국어 하단에 하늘색 작은 글씨로 원문 표시)
* **`movie.ja.orig.srt`**: Whisper가 인식한 외국어 원문 전용 SRT 자막 (`원문 보관` 체크 시 생성)

---

## 5. 환경 재설정 및 테스트 방법

* **Ollama 및 기본 모델(`qwen2.5:7b`) 재설치/점검 스크립트**:
  ```powershell
  powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
  ```
* **파이프라인 및 GUI 자동화 테스트 실행**:
  ```powershell
  .\.venv\Scripts\python.exe -m unittest discover -s tests -v
  ```
