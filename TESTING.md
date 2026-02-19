# 테스트 동작을 위해 필요한 것

## 1. 환경 준비

### 1.1 Python
- Python 3.10 이상 권장.

### 1.2 의존성 설치
프로젝트 루트(`c:\agentic_sample`)에서:

```bash
pip install -r requirements.txt
```

### 1.3 API 키 (Gemini / Google ADK)
- 메인 에이전트·플래너·paper/sns 에이전트는 Google ADK(Gemini)를 사용합니다.
- 다음 중 하나를 설정하세요.
  - **GOOGLE_API_KEY** (권장): [Google AI Studio](https://aistudio.google.com/apikey)에서 발급
  - 또는 Google ADK 문서에 맞는 다른 키 환경 변수

```powershell
# PowerShell 예시
$env:GOOGLE_API_KEY = "your-api-key"
```

### 1.4 (선택) Slack MCP
- paper_agent / sns_agent의 Slack 전송 기능을 쓰려면 **Slack MCP 서버**가 필요합니다.
- 환경 변수 `SLACK_MCP_SERVER_PATH`에 해당 서버 스크립트 경로를 넣습니다.
- 미설정 시 `slack_post_message` 호출 시 안내 메시지만 반환됩니다.

---

## 2. 데이터 디렉터리

- **db/paper/**  
  - 논문 MCP 도구(`search_papers`)는 여기 아래의 `*.pdf`만 검색합니다.  
  - 테스트 시 빈 폴더여도 되고, 파일명에 키워드가 들어간 PDF를 넣으면 해당 키워드로 검색 가능합니다.

- **db/sns/**  
  - SNS MCP 도구(`search_sns_posts`)는 여기 아래의 `*.json`만 검색합니다.  
  - 예시: `db/sns/sample_posts.json` (이미 포함).  
  - `"data"[].message`에 키워드가 포함된 게시글만 반환됩니다.

---

## 3. 실행 방법

**반드시 프로젝트 루트를 현재 디렉터리로 두고 실행하세요.**  
(MCP 클라이언트가 서버를 `python -m mcp.paper_server` 형태로 띄우기 때문입니다.)

```powershell
cd c:\agentic_sample
```

### 3.1 전체 진입점 (메인 에이전트 + 플래너 + event_manager)

```powershell
python user_entry_point.py
```

- 사용자 입력 → planner가 계획 수립 → event_manager가 실행.
- 현재 event_manager는 **로컬 에이전트(paper/sns)를 직접 실행하지 않고**, A2A 에이전트가 1개일 때만 A2A 호출을 합니다.  
  따라서 로컬 에이전트만 등록된 지금은 **계획 텍스트만 출력**됩니다.  
  로컬 에이전트 실행 로직을 event_manager에 추가하면 여기서부터 paper/sns가 동작합니다.

### 3.2 MCP + 에이전트만 빠르게 테스트 (권장)

paper_agent 또는 sns_agent를 직접 호출해 **MCP 스크래핑 도구**가 잘 동작하는지 확인할 수 있습니다.

```powershell
python scripts/run_agent_test.py sns   "AI"
python scripts/run_agent_test.py paper "machine learning"
```

- `sns`: sns_agent에 "AI" 질의 → `scrape_sns_with_mcp("AI")` 호출 → db/sns JSON에서 "AI" 포함 게시글 반환.
- `paper`: paper_agent에 "machine learning" 질의 → `scrape_papers_with_mcp("machine learning")` 호출 → db/paper 아래에서 파일명에 "machine learning"이 들어간 PDF 목록 반환.

---

## 4. 체크리스트 요약

| 항목 | 필요 여부 |
|------|------------|
| `pip install -r requirements.txt` | 필수 |
| `GOOGLE_API_KEY` (또는 동등한 Gemini 키) | 메인/에이전트 LLM 사용 시 필수 |
| 프로젝트 루트에서 실행 | 필수 |
| db/paper, db/sns 디렉터리 | MCP 도구 테스트 시 필요 (이미 생성됨) |
| db/sns 샘플 JSON | SNS 테스트 시 필요 (sample_posts.json 포함) |
| SLACK_MCP_SERVER_PATH | Slack 전송 테스트 시만 필요 |
| agent_cards/agent_card.json | 플래너가 paper/sns 에이전트를 계획에 넣으려면 필요 (이미 포함) |

위를 만족하면 **3.2**로 MCP + 에이전트 테스트를 먼저 해보고, 이어서 **3.1**로 전체 플로우를 실행하면 됩니다.
