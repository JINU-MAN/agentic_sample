# agentic_sample_ad

## 0) Quick Start From Scratch (Windows PowerShell)

Use this section when you start from an empty environment.

1. Check Python installation (recommended: 3.11+)

```powershell
python --version
```

2. Move to project root

```powershell
cd C:\agentic_sample_api\agentic_sample_ad
```

3. Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

4. Install dependencies

```powershell
pip install -r requirements.txt
```

5. Create env file

```powershell
Copy-Item .env.example .env
```

6. Set minimum `.env` values

```env
GOOGLE_API_KEY=your_google_api_key
SLACK_MCP_SERVER_PATH=./mcp_local/slack_server.py
SLACK_BOT_TOKEN=your_slack_bot_token
```

- `GOOGLE_API_KEY` is required.
- `SLACK_BOT_TOKEN` is required only when you use Slack posting.
- You can run the app without Slack posting even if `SLACK_BOT_TOKEN` is empty.

7. Run

```powershell
python start_agentic.py
```

8. Exit

- In CLI, type `exit` or `quit`.

---

`agentic_sample_ad`는 `MainAgent`가 여러 specialist 에이전트를 오케스트레이션하는 멀티 에이전트 시스템입니다.

- `PaperAnalyst`
- `WebSearchAnalyst`
- `SocialMediaAnalyst`

이 문서는 현재 코드 구현 기준으로 다음 내용을 설명합니다.

1. 전체 아키텍처와 실행 흐름
2. 에이전트 역할과 도구 소유권
3. MCP, A2A 상호작용 구조
4. 로그/세션 아카이브 동작
5. 운영, 설정, 확장 방법

---

## 1) 핵심 구조

- 사용자 입력은 `MainAgent`가 처리합니다.
- `planner.py`가 아래 3가지를 생성합니다.
  - `raw_plan`
  - `routing_hint`
  - `collaboration_plan`
- `event_manager.py`가 단계 실행, 재계획, 실패 복구, 일시중단(pause)을 담당합니다.
- specialist 실행은 **A2A HTTP JSON-RPC** 직접 호출로 이루어집니다.
- 외부 기능(웹/논문/SNS/Slack)은 MCP 서버를 통해 실행됩니다.
- Slack 전송 기능은 정책상 `MainAgent` 전용입니다.

---

## 2) 저장소 구조

```text
agentic_sample_ad/
|-- start_agentic.py
|-- planner.py
|-- event_manager.py
|-- system_logger.py
|-- model_settings.py
|-- .env / .env.example
|
|-- main_agent/
|   |-- start_agentic.py
|   |-- agent.py
|   |-- card_registry.py
|   |-- user_entry_point.py
|   |-- session_memory.py
|   `-- system_logger.py
|
|-- agent/
|   |-- paper_agent.py
|   |-- web_search_agent.py
|   |-- sns_agent.py
|   `-- tool/slack_mcp_tool.py
|
|-- paper_agent/
|-- web_search_agent/
|-- sns_agent/
|   |-- agent.py
|   |-- a2a_server.py
|   `-- well_known/agent_card.json
|
|-- common/a2a_agent_server.py
|-- scripts/start_a2a_agents.py
|-- agent_cards/agent_card.json
|
|-- mcp_local/
|   |-- client.py
|   |-- web_search_server.py
|   |-- paper_server.py
|   |-- sns_server.py
|   |-- slack_server.py
|   `-- a2a_bridge_server.py   # 레거시 브리지 구현 (기본 경로 아님)
|
|-- db/paper
|-- db/sns
`-- log/
    |-- session_log.jsonl
    |-- system_events.jsonl
    |-- components/*.jsonl
    `-- session_log_ex_XXXXXXXXXX/
```

---

## 3) 실행 흐름

## 3.1 시작 (`python start_agentic.py`)

`main_agent/start_agentic.py::main()` 동작:

1. 로그 세션을 새로 시작하고(reset) 이전 세션을 아카이브합니다.
2. `.env`를 로드합니다.
3. 에이전트 카드를 탐색합니다.
4. specialist A2A 서버를 실행합니다.
5. 런타임 엔드포인트를 `AGENTIC_RUNTIME_AGENT_CARDS`에 게시합니다.
6. CLI 입력 루프에 진입합니다.

## 3.2 사용자 요청 1턴 처리

1. `MainAgent` 생성
2. planner 실행으로 계획 생성
3. event_manager 실행으로 협업 워크플로우 진행
4. 필요 시 A2A specialist 호출 + MCP 도구 호출
5. 결과 합성 후 사용자에게 반환

## 3.3 종료

- `exit` 또는 `quit` 입력 시 종료
- 시작 시 띄운 A2A specialist 서버 종료
- 로깅 finalize

---

## 4) Planner + EventManager 동작

## 4.1 Planner 산출물

- `raw_plan`: 사람이 읽는 계획 텍스트
- `routing_hint`: 선택 에이전트, 키워드, 이유
- `collaboration_plan`: 실제 실행 step 목록

Planner는 하드코딩보다 카드/런타임 메타데이터(capability/tool)를 우선 사용합니다.

## 4.2 Event Manager 실행

`event_manager.execute_plan()`:

1. 실행 가능한 에이전트(local/a2a) 선별
2. collaboration step 정규화
3. step 루프 실행
4. step 결과를 기반으로 재계획 여부 판단
5. 필요 시 실패 복구 시도
6. 최종 결과 합성

## 4.3 Additional Needs 위임

specialist는 다른 에이전트를 직접 호출하지 않고 handoff 요청을 발행합니다.

JSON 예시:

```json
{"needs":[{"target":"WebSearchAnalyst","request":"..."}]}
```

텍스트 예시:

```text
Additional Needs:
- [WebSearchAnalyst] ...
- [MainAgent] ...
```

`event_manager.py`가 이를 파싱해 후속 step으로 반영합니다.

## 4.4 사용자 확인 필요 시 pause

사용자 확인이 필요한 상황에서는 워크플로우를 일시중단합니다.

- `workflow_paused = true`
- `pause_reason = awaiting_user_clarification`

사용자 응답이 들어오기 전까지 실행을 계속하지 않습니다.

## 4.5 Slack 도구 소유권

Slack 전달 capability는 `MainAgent` 소유입니다.

- `comm.slack.post`
- `slack_post_message`

다른 에이전트 단계에서 Slack 전송이 필요하더라도 `MainAgent`로 재라우팅됩니다.

---

## 5) 에이전트 구성

| Agent | 역할 | 기본 모델 | 핵심 capability | 도구 |
|---|---|---|---|---|
| MainAgent | 조정자/재계획/최종 전달 | `gemini-2.5-flash-lite` | coordination, workflow_replanning, user_clarification_routing, comm.slack.post | `slack_post_message` |
| PaperAnalyst | 로컬 PDF 연구 분석 | `gemini-2.5-flash-lite` | paper_search, paper_summary, paper_memory_query | `scrape_papers_with_mcp`, `load_paper_memory_with_mcp`, `expand_paper_memory_with_mcp`, `query_paper_memory` |
| WebSearchAnalyst | 웹 탐색/검증 | `gemini-2.5-flash-lite` | web_search, web_summary, fact_check | `search_web_candidates_with_mcp`, `fetch_web_page_with_mcp` |
| SocialMediaAnalyst | SNS 수집/요약 | `gemini-2.5-flash-lite` | sns_search, sns_summary | `scrape_sns_with_mcp` |

모델 해석은 `model_settings.py`의 `resolve_agent_model()`이 담당합니다.

---

## 6) A2A 통신 구조 (현재 기본 경로)

현재 기본 실행은 브리지 경유가 아니라 specialist 서버 직접 호출입니다.

```text
event_manager
  -> A2AClient (HTTP JSON-RPC)
  -> http://127.0.0.1:{dynamic_port}/
  -> specialist a2a_server (FastAPI)
  -> InMemoryRunner(local LlmAgent)
  -> text response
```

- 카드 endpoint: `GET /.well-known/agent-card.json`
- 메시지 endpoint: `POST /` (`message/send`)
- 런처가 카드 health를 확인해 launch/skip 결정
- 기본값은 동적 포트 사용 (`A2A_DYNAMIC_PORTS=true`)

`mcp_local/a2a_bridge_server.py`는 레거시 코드이며 `start_agentic` 기본 경로에서 사용되지 않습니다.

---

## 7) MCP 도구 호출 구조

도구 호출 경로:

```text
Agent tool function
  -> mcp_local/client.py::call_mcp_tool
  -> MCP server process (stdio)
  -> JSON result
```

주요 MCP 서버:

| 서버 | 주요 tool | 역할 |
|---|---|---|
| `mcp_local/web_search_server.py` | `search_web`, `fetch_page` | 웹 후보 검색 + 본문 추출 |
| `mcp_local/paper_server.py` | `search_papers`, `get_paper_head`, `get_paper_content` | 로컬 PDF 검색/본문 추출 |
| `mcp_local/sns_server.py` | `search_sns_posts` | 로컬 SNS JSON 검색 |
| `mcp_local/slack_server.py` | `post_message` | Slack `chat.postMessage` |

---

## 8) 실행 중 모델 변경 명령

CLI 명령:

```text
/setting {AgentName} -m {ModelName}
/setting {AgentName} -model {ModelName}
```

동작:

- `MainAgent`: 다음 턴부터 즉시 적용 (턴마다 재생성)
- specialist: 대상 A2A 서버 프로세스 재기동 후 적용

관련 환경변수:

- `AGENTIC_DEFAULT_MODEL`
- `AGENTIC_AGENT_MODEL_OVERRIDES` (JSON map)

---

## 9) 로그와 세션 아카이브

`system_logger.py`는 이벤트를 다음 파일에 동시에 기록합니다.

- `log/system_events.jsonl`
- `log/components/<component>.jsonl`
- `log/session_log.jsonl`

모든 이벤트는 증가하는 `session_seq`를 포함합니다.

## 9.1 시작 시 세션 아카이브

`start_new_logging_session(reset_files=True)` 실행 시:

1. 기존 `session_log.jsonl` 읽기
2. `log/session_log_ex_{num}` 폴더 생성
3. JSONL 한 줄을 JSON 파일 1개로 분해 저장
4. active `session_log.jsonl` 및 sequence 상태 초기화

파일명 규칙:

```text
{seq:010d}_{component}_{action}_{YYYYMMDDTHHMMSS_microZ}.json
```

## 9.2 function trace

`function_trace`는 현재 의도적으로 비활성화되어 있습니다.

## 9.3 ExceptionGroup 기록

`log_exception()`은 traceback뿐 아니라 `sub_exceptions`도 함께 기록합니다.

## 9.4 Git 로그 추적 정책

Git에는 `log/session_log.jsonl`만 추적하고, 나머지 `log/*`는 `.gitignore`로 제외합니다.

---

## 10) 에이전트 간 메시지 로그

컴포넌트: `event_manager.agent_message`

action 종류:

- `sent`
- `received`
- `need_requested`

주요 필드:

- `from_agent`, `to_agent`
- `channel` (예: `collaboration`, `direct_a2a`)
- `workflow_id`, `workflow_step`
- `message_preview`, `message_length`

---

## 11) 실행 명령

## 11.1 메인 시스템 실행

```bash
pip install -r requirements.txt
python start_agentic.py
```

또는:

```bash
python -m agentic_sample_ad.start_agentic
```

## 11.2 specialist 단독 실행

```bash
python -m agentic_sample_ad.paper_agent.user_entry_point
python -m agentic_sample_ad.web_search_agent.user_entry_point
python -m agentic_sample_ad.sns_agent.user_entry_point
```

---

## 12) 환경변수

필수:

- `GOOGLE_API_KEY`

모델:

- `AGENTIC_DEFAULT_MODEL` (기본 `gemini-2.5-flash-lite`)
- `AGENTIC_AGENT_MODEL_OVERRIDES` (JSON)

Slack:

- `SLACK_MCP_SERVER_PATH` (예: `./mcp_local/slack_server.py`)
- `SLACK_BOT_TOKEN`

A2A/협업 제어:

- `A2A_DYNAMIC_PORTS`
- `A2A_AGENT_SERVER_READY_TIMEOUT_SEC`
- `A2A_REQUEST_TIMEOUT_SEC`
- `A2A_CARD_TIMEOUT_SEC`
- `A2A_CONNECT_TIMEOUT_SEC`
- `COLLAB_MAX_STEPS`

---

## 13) 확장 가이드

## 13.1 새 에이전트 추가

1. `new_agent/agent.py`에 agent 정의
2. `new_agent/well_known/agent_card.json` 작성
3. `new_agent/a2a_server.py`에서 `common/a2a_agent_server.py::run_server` 사용
4. `start_agentic.py` 실행 후 카드 탐색/서버 실행 확인

## 13.2 새 도구 추가

1. 에이전트에 tool 함수 추가
2. MCP server tool 구현(또는 기존 재사용)
3. agent card capability 갱신
4. 런타임 메타데이터와 planner 라우팅 검증

capability 소유권 정책은 `event_manager.py`의 `CAPABILITY_POLICIES`에서 관리합니다.

---

## 14) 트러블슈팅

## Slack 실패

1. `SLACK_MCP_SERVER_PATH` 경로 확인
2. `SLACK_BOT_TOKEN` 유효성 확인
3. 채널명 및 봇 권한 확인

## A2A 실패

1. `/.well-known/agent-card.json` health 확인
2. 포트 충돌 확인
3. timeout 조정 (`A2A_REQUEST_TIMEOUT_SEC`, `A2A_CARD_TIMEOUT_SEC`)

## 워크플로우 pause

`awaiting_user_clarification` 상태는 사용자 확인이 필요한 정상 동작일 수 있습니다.

## 로그 중심 디버깅 순서

1. `log/session_log.jsonl`
2. `log/components/event_manager.collaboration.jsonl`
3. `log/components/event_manager.agent_message.jsonl`

---

## 15) 한 줄 요약

`MainAgent`가 계획/정책/최종 전달을 책임지고, specialist는 직접 A2A 통신으로 실행되며, 도구는 MCP로 호출되고, 전체 상호작용은 `session_seq` 기반 JSONL 로그로 추적되는 구조입니다.
