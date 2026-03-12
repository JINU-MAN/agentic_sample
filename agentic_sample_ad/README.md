# agentic_sample_ad

`agentic_sample_ad`는 `MainAgent`가 작업을 계획하고, specialist 에이전트를 A2A로 호출하며, 실제 외부 기능은 MCP 도구로 수행하는 멀티에이전트 샘플입니다.

현재 기본 구조는 다음과 같습니다.

- 공개 계약: `agent_cards/*.json`, `*/well_known/agent_card.json`
- coordinator: `main_agent/`
- worker: `paper_agent/`, `web_search_agent/`, `sns_agent/`
- 계획과 실행: `planner.py`, `event_manager.py`
- A2A 서버 런처: `scripts/start_a2a_agents.py`
- MCP 서버: `mcp_local/`
- 정적 agent memory: `*/memory/session_memory.json`
- 동적 workflow memory: `workflow_memory_runtime.py`

## 빠른 시작

### 1. 환경 준비

```powershell
cd C:\agentic_sample_api\agentic_sample_ad
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

최소 `.env` 값:

```env
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
OPENAI_API_KEY=your_openai_api_key
SLACK_BOT_TOKEN=your_slack_bot_token
```

설명:

- `GOOGLE_API_KEY`: 필수
- `TAVILY_API_KEY`: 웹 검색 사용 시 필요
- `SLACK_BOT_TOKEN`: Slack 전송 사용 시 필요
- `OPENAI_API_KEY` : openai 모델 사용 시 필요, 모델명 앞에 openai/ 붙여줘야함 (ex. openai/gpt_5.2)
### 2. 실행

```powershell
python start_agentic.py
```

CLI 명령:

- `exit`, `quit`: CLI 종료
- `reset`, `/reset`: 현재 세션 메모리 초기화
- `/setting {AgentName|all} -m {ModelName}`: 특정 agent 또는 전체 agent 모델 변경

예시:

```text
/setting WebSearchAnalyst -m gemini-2.5-flash
/setting all -m gemini-2.5-flash-lite
```

## 에이전트 구조

### Coordinator와 worker

| Agent | 역할 | 주요 callable tool |
|---|---|---|
| `MainAgent` | 계획, 재계획, 사용자 확인, 최종 전달 | `slack_post_message`, `read_workflow_memory`, `load_session_memory` |
| `PaperAnalyst` | 로컬 PDF 검색, paper memory, 외부 논문 참조 처리 | `scrape_papers_with_mcp`, `load_paper_memory_with_mcp`, `query_paper_memory`, `expand_paper_memory_with_mcp`, `fetch_external_paper_with_mcp`, `load_session_memory` |
| `WebSearchAnalyst` | 웹 검색과 citation-grounded evidence synthesis | `search_web_with_mcp`, `load_session_memory` |
| `SocialMediaAnalyst` | SNS 검색과 social signal 요약 | `scrape_sns_with_mcp`, `load_session_memory` |

중요한 원칙:

- planner는 raw tool 이름이 아니라 `role`, `capabilities`, `ownership` 기준으로 step을 고릅니다.
- Slack 전달은 `MainAgent`만 직접 소유합니다.
- specialist는 다른 agent를 직접 호출하지 않고, 구조화된 `needs`와 `artifacts`로 handoff를 요청합니다.

### 공개 계약과 내부 구현 분리

- 공개 계약: agent card
  - `planner.py`와 `MainAgent`는 여기 있는 `description`, `capabilities`, `ownership`을 기준으로 판단합니다.
- 내부 구현: 각 agent 내부의 tool과 skill
  - 각 agent는 내부적으로 `load_skill`, `load_skill_resource`, `load_session_memory`를 사용합니다.

## 디렉터리 구조

```text
agentic_sample_ad/
|-- start_agentic.py
|-- planner.py
|-- event_manager.py
|-- network_retry.py
|-- skill_runtime.py
|-- agent_session_memory_runtime.py
|-- workflow_memory_runtime.py
|-- system_logger.py
|-- model_settings.py
|-- .env.example
|
|-- main_agent/
|   |-- agent.py
|   |-- start_agentic.py
|   |-- user_entry_point.py
|   |-- card_registry.py
|   |-- slack_mcp_tool.py
|   |-- workflow_memory_tool.py
|   |-- session_memory.py
|   |-- memory/session_memory.json
|   `-- skills/coordinator-operations/
|
|-- paper_agent/
|   |-- agent.py
|   |-- a2a_server.py
|   |-- event_manager.py
|   |-- session_memory.py
|   |-- user_entry_point.py
|   |-- tool/
|   |-- memory/session_memory.json
|   |-- skills/paper-research-operations/
|   `-- well_known/agent_card.json
|
|-- web_search_agent/
|   |-- agent.py
|   |-- a2a_server.py
|   |-- event_manager.py
|   |-- session_memory.py
|   |-- user_entry_point.py
|   |-- tool/
|   |-- memory/session_memory.json
|   |-- skills/web-research-operations/
|   `-- well_known/agent_card.json
|
|-- sns_agent/
|   |-- agent.py
|   |-- a2a_server.py
|   |-- event_manager.py
|   |-- session_memory.py
|   |-- user_entry_point.py
|   |-- tool/
|   |-- memory/session_memory.json
|   |-- skills/social-media-research-operations/
|   `-- well_known/agent_card.json
|
|-- common/
|   `-- a2a_agent_server.py
|
|-- scripts/
|   |-- start_a2a_agents.py
|   `-- refresh_agent_session_memory.py
|
|-- agent_cards/agent_card.json
|
|-- mcp_local/
|   |-- client.py
|   |-- web_search_server.py
|   |-- paper_server.py
|   |-- sns_server.py
|   |-- slack_server.py
|   `-- a2a_bridge_server.py
|
`-- log/
    |-- session_log.jsonl
    |-- system_events.jsonl
    `-- components/
```

보조 디렉터리:

- `agent/`: 레거시 문서와 템플릿
- `db/`: paper와 SNS 데이터 저장소

## 실행 흐름

### 부팅

`python start_agentic.py`는 내부적으로 `main_agent/start_agentic.py`를 실행합니다.

부팅 순서:

1. `.env` 로드
2. sub-agent card 수집
3. specialist A2A 서버 자동 기동
4. 런타임 endpoint를 `AGENTIC_RUNTIME_AGENT_CARDS`에 게시
5. CLI 루프 진입

### 요청 처리

`run_main_agent()`의 동작:

1. 세션 로드
2. `MainAgent` 생성
3. unified agent registry 구성
4. `planner.py`에서 다음 생성
   - `raw_plan`
   - `routing_hint`
   - `collaboration_plan`
5. `event_manager.py`가 collaboration workflow 실행
6. 계획과 결과를 세션과 로그에 저장

### Step 실행

각 workflow step은 다음 정보를 포함한 context packet으로 실행됩니다.

- 사용자 요청 요약
- 이전 step 결과
- input artifacts
- open needs
- 남은 step 힌트
- delegation target 요약

worker는 이 context를 바탕으로:

- 자신의 tool로 바로 처리하거나
- 구조화된 `artifacts`를 남기거나
- 구조화된 `needs`를 `MainAgent`로 올립니다

## Memory 구조

현재 memory는 3계층으로 분리돼 있습니다.

### 1. Skill

용도:

- 정적 작업 지침
- tool 사용 원칙
- handoff 스타일

사용 도구:

- `load_skill`
- `load_skill_resource`

파일 위치:

- `*/skills/*/SKILL.md`
- `*/skills/*/references/`

### 2. Session memory

용도:

- 현재 agent의 정적 tool inventory
- ownership, capabilities, handoff contract
- coordinator가 아는 sub-agent 계약

사용 도구:

- `load_session_memory(section="", query="", max_items=6)`

파일 위치:

- `main_agent/memory/session_memory.json`
- `paper_agent/memory/session_memory.json`
- `web_search_agent/memory/session_memory.json`
- `sns_agent/memory/session_memory.json`

다음 변경 후 재생성해야 합니다.

- agent card 변경
- tool 추가 또는 삭제
- ownership 문구 변경

재생성 명령:

```powershell
python scripts/refresh_agent_session_memory.py
```

### 3. Workflow memory

용도:

- 이전 step 출력
- 현재 artifacts
- open needs
- pending steps
- activated agent snapshots

사용 도구:

- `read_workflow_memory(query="", max_items=6)`

특징:

- 정적 파일이 아니라 실행 중 유지되는 동적 상태입니다.
- `MainAgent`가 막힌 specialist를 도와주거나 현재 workflow 상태를 다시 읽을 때 사용합니다.

## 재계획과 점검

현재 구조는 매 step마다 무조건 재계획하지 않습니다. `event_manager.py`가 필요한 경우에만 coordinator review를 호출합니다.

대표 트리거:

- `open_needs` 존재
- pending step 소진
- 최신 step이 빈 응답 반환
- blocker 문구 감지
- worker가 유효한 `artifacts`나 `needs` 없이 종료
- 같은 작업이 반복 임계값 초과

### 같은 작업 반복 임계값

- 기준 키: `agent|goal`
- 환경 변수: `AGENTIC_SAME_TASK_REVIEW_THRESHOLD`
- 기본값: `2`

의미:

- 동일 작업 2회까지 허용
- 3회째부터 이상 징후로 보고 coordinator review 트리거

## A2A와 MCP

### A2A

기본 실행 경로:

```text
MainAgent / event_manager
  -> A2AClient
  -> http://127.0.0.1:{port}/
  -> specialist a2a_server
  -> specialist LlmAgent
  -> text response
```

관련 파일:

- `scripts/start_a2a_agents.py`
- `common/a2a_agent_server.py`
- `paper_agent/a2a_server.py`
- `web_search_agent/a2a_server.py`
- `sns_agent/a2a_server.py`

### MCP

실제 외부 기능은 MCP를 통해 수행됩니다.

```text
Agent tool wrapper
  -> mcp_local/client.py
  -> MCP server process
  -> JSON result
```

주요 MCP 서버:

| Server | 역할 |
|---|---|
| `mcp_local/web_search_server.py` | Tavily 기반 웹 검색 |
| `mcp_local/paper_server.py` | 로컬 PDF 검색과 paper memory |
| `mcp_local/sns_server.py` | SNS 검색 |
| `mcp_local/slack_server.py` | Slack 전송 |

## 환경 변수

필수:

- `GOOGLE_API_KEY`

주요 선택 항목:

- `TAVILY_API_KEY`
- `SLACK_BOT_TOKEN`
- `SLACK_MCP_SERVER_PATH`
- `AGENTIC_DEFAULT_MODEL`
- `AGENTIC_AGENT_MODEL_OVERRIDES`
- `A2A_DYNAMIC_PORTS`
- `A2A_AGENT_SERVER_READY_TIMEOUT_SEC`
- `A2A_CONNECT_TIMEOUT_SEC`
- `A2A_CARD_TIMEOUT_SEC`
- `A2A_REQUEST_TIMEOUT_SEC`
- `A2A_WRITE_TIMEOUT_SEC`
- `A2A_POOL_TIMEOUT_SEC`
- `A2A_CARD_RETRY_COUNT`
- `A2A_CARD_RETRY_DELAY_SEC`
- `COLLAB_MAX_STEPS`
- `AGENTIC_SAME_TASK_REVIEW_THRESHOLD`
- `AGENTIC_NETWORK_RETRY_ATTEMPTS`
- `AGENTIC_NETWORK_RETRY_BASE_DELAY_SEC`
- `AGENTIC_NETWORK_RETRY_MAX_DELAY_SEC`

## 로그

주요 로그 파일:

- `log/session_log.jsonl`
- `log/system_events.jsonl`
- `log/components/*.jsonl`

특징:

- 모든 이벤트에 `session_seq`가 붙습니다.
- agent 간 메시지는 `event_manager.agent_message`로 기록됩니다.
- step 시작, 완료, pause, review, timeout control, recovery가 개별 이벤트로 남습니다.

문제 분석 시 우선 볼 파일:

1. `log/session_log.jsonl`
2. `log/components/event_manager.collaboration.jsonl`
3. `log/components/event_manager.agent_message.jsonl`

## 운영 명령

### A2A 서버만 별도 실행

```powershell
python scripts/start_a2a_agents.py
python scripts/start_a2a_agents.py --only WebSearchAnalyst
```

### 정적 session memory 재생성

```powershell
python scripts/refresh_agent_session_memory.py
```

### 동작이 바뀌었을 때 같이 봐야 하는 파일

- planning 규칙: `planner.py`
- workflow 실행과 복구 규칙: `event_manager.py`
- 공개 계약: `agent_cards/agent_card.json`, `*/well_known/agent_card.json`
- skill 로딩: `skill_runtime.py`
- session memory 로딩: `agent_session_memory_runtime.py`
- workflow memory 로딩: `workflow_memory_runtime.py`

## 트러블슈팅

### `GOOGLE_API_KEY`가 없음

CLI가 키 입력을 요청하고, 입력된 값을 `.env`에 저장하려고 시도합니다.

### 웹 검색이 안 되는 것처럼 보임

다음 순서로 확인합니다.

1. `TAVILY_API_KEY`
2. `WebSearchAnalyst` A2A 서버 기동 여부
3. `log/session_log.jsonl`에 `tool.search_web_with_mcp` 호출이 찍혔는지
4. worker가 `artifacts`나 `needs`를 남겼는지

### Slack 전송 실패

다음 항목을 확인합니다.

1. `SLACK_BOT_TOKEN`
2. 채널 이름
3. `MainAgent`가 실제로 `slack_post_message`를 호출했는지

### 재계획이 너무 자주 도는 경우

다음 값을 조정합니다.

- `AGENTIC_SAME_TASK_REVIEW_THRESHOLD`
- `COLLAB_MAX_STEPS`
- A2A timeout 관련 변수

### Session memory가 오래된 경우

다음 명령으로 다시 생성합니다.

```powershell
python scripts/refresh_agent_session_memory.py
```

## 요약

현재 시스템의 핵심 규칙:

- planner는 capability와 ownership 중심으로 step을 만든다.
- worker는 specialist 작업만 수행하고 peer agent를 직접 호출하지 않는다.
- 정적 지침은 skill, 정적 계약은 session memory, 동적 상태는 workflow memory로 분리한다.
- `MainAgent`가 orchestration과 Slack delivery를 소유한다.
- A2A는 agent 간 계약을, MCP는 실제 외부 기능 실행을 담당한다.
