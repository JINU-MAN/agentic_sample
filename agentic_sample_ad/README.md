# agentic_sample_ad

이 저장소는 `MainAgent`가 여러 전문 에이전트(`PaperAnalyst`, `WebSearchAnalyst`, `SocialMediaAnalyst`)를 오케스트레이션해서 하나의 사용자 요청을 협업 처리하는 멀티 에이전트 시스템입니다.

이 문서는 코드 없이도 시스템을 파악할 수 있도록, 실제 구현 기준으로 아래를 모두 설명합니다.

1. 전체 아키텍처와 실행 경로
2. 에이전트별 역할, 모델, 도구
3. MCP 도구 호출 방식과 서버 상호작용
4. 에이전트 간 협업/위임(Additional Needs) 방식
5. 세션/로그 저장 구조와 `session_log_example` 해석법

---

## 1) 핵심 개념 요약

- 사용자와 직접 대화하는 주체는 `MainAgent`입니다.
- `MainAgent`는 `planner.py`로 계획을 만들고, `event_manager.py`로 실행합니다.
- 전문 작업은 A2A 브리지 뒤에 있는 서브 에이전트들이 담당합니다.
- 외부 도구(웹 검색/Slack/SNS/논문 DB)는 MCP 서버를 통해 호출합니다.
- 서브 에이전트는 다른 에이전트를 직접 호출하지 않고, `Additional Needs`로 간접 요청합니다.
- Slack 전송 능력(`comm.slack.post`, `slack_post_message`)은 정책상 `MainAgent` 전용입니다.

---

## 2) 디렉터리 구조

```text
agentic_sample_ad/
├─ start_agentic.py                  # 진입점 (main_agent/start_agentic.main 위임)
├─ planner.py                        # 계획 수립: raw_plan + routing_hint + collaboration_plan
├─ event_manager.py                  # 실행 엔진: 협업 step 실행, 재계획, pause 처리
├─ system_logger.py                  # 공통 JSONL 로깅 + 세션 아카이브
├─ requirements.txt
├─ .env / .env.example
│
├─ main_agent/
│  ├─ start_agentic.py               # 프로세스 시작/종료, 브리지 런처 연동
│  ├─ agent.py                       # MainAgent 생성 및 전체 워크플로우 진입
│  ├─ event_manager.py               # 루트 event_manager 포워딩 래퍼
│  ├─ card_registry.py               # 서브 에이전트 카드 수집/로딩
│  ├─ session_memory.py              # MainAgent 세션 메모리
│  ├─ user_entry_point.py            # CLI 루프
│  └─ system_logger.py               # main 전용 로거 래퍼
│
├─ agent/                            # 실제 specialist LlmAgent 정의
│  ├─ paper_agent.py
│  ├─ web_search_agent.py
│  ├─ sns_agent.py
│  └─ tool/
│     └─ slack_mcp_tool.py           # Slack MCP 도구 래퍼
│
├─ paper_agent/                      # paper 에이전트 패키지 래퍼 + 카드
├─ web_search_agent/                 # web 에이전트 패키지 래퍼 + 카드
├─ sns_agent/                        # sns 에이전트 패키지 래퍼 + 카드
│  ├─ agent.py                       # agent/* 재-export
│  ├─ event_manager.py               # 단독 실행용 로컬 이벤트 매니저
│  ├─ session_memory.py
│  ├─ user_entry_point.py
│  └─ well_known/agent_card.json
│
├─ agent_cards/agent_card.json       # 통합 카드(3개 서브 에이전트)
│
├─ mcp_local/
│  ├─ client.py                      # MCP stdio 클라이언트 공통 래퍼
│  ├─ a2a_bridge_server.py           # 로컬 LlmAgent를 A2A(JSON-RPC) 서버로 노출
│  ├─ web_search_server.py           # MCP: search_web, fetch_page
│  ├─ paper_server.py                # MCP: search_papers, get_paper_head/content
│  ├─ sns_server.py                  # MCP: search_sns_posts
│  └─ slack_server.py                # MCP: post_message
│
├─ scripts/start_a2a_agents.py       # 카드 기반 브리지 자동 기동/정지
├─ common/                           # 세션메모리/로컬 agent event manager 공용
├─ db/
│  ├─ paper/                         # 로컬 PDF 코퍼스
│  └─ sns/                           # 로컬 SNS JSON
└─ log/
   ├─ system_events.jsonl
   ├─ session_log.jsonl
   ├─ components/*.jsonl
   ├─ session_log_example/*.json
   └─ session_log_ex_0000000001/*.json  # 이전 세션 아카이브
```

---

## 3) 런타임 아키텍처

```text
User
  -> main_agent/user_entry_point.py (CLI)
  -> main_agent/agent.py::run_main_agent
  -> planner.py::plan_with_main_agent
       - raw_plan
       - routing_hint
       - collaboration_plan
  -> event_manager.py::execute_plan
       - step 실행
       - additional needs 수집/해결
       - 재계획/실패복구/일시중단
  -> MainAgent 최종 응답

도구 호출 경로:
Agent Tool Function
  -> mcp_local/client.py::call_mcp_tool
  -> MCP Server (stdio)
  -> tool 결과 JSON
```

A2A 경로(서브 에이전트):

```text
event_manager
  -> A2AClient (HTTP JSON-RPC)
  -> mcp_local/a2a_bridge_server.py
  -> InMemoryRunner(local LlmAgent)
  -> 응답 텍스트 반환
```

---

## 4) 실행 시작부터 종료까지

### 4.1 시작 (`python start_agentic.py`)

`main_agent/start_agentic.py::main()` 기준:

1. `start_main_logging_session(reset_files=True)` 호출
2. `.env` 로드
3. `card_registry`로 서브 에이전트 카드 탐색
4. `scripts/start_a2a_agents.py::launch_bridges()`로 브리지 기동
5. `input_loop()`에서 사용자 입력 대기

### 4.2 사용자 요청 1턴 처리

`main_agent/agent.py::run_main_agent()` 기준:

1. 세션 메모리 업데이트(user turn 저장)
2. `MainAgent` 생성 (모델: `gemini-3-flash-preview`)
3. 카드 + 런타임 메타데이터로 통합 에이전트 레지스트리 구성
4. `planner.plan_with_main_agent(...)`
5. `event_manager.execute_plan(...)`
6. 실행 결과를 세션 메모리에 저장 후 사용자에게 반환

### 4.3 종료

- 사용자 `exit`/`quit` 입력 시 루프 종료
- 시작 시 띄운 브리지 프로세스 종료
- 로깅 finalize 수행

---

## 5) Planner 설계 (`planner.py`)

Planner는 한 번에 3개 산출물을 만듭니다.

1. `raw_plan`
- 사람 읽기용 계획 텍스트

2. `routing_hint`
- `selected_agents`: 어떤 에이전트를 사용할지
- `keywords`: 라우팅 힌트 키워드
- `reason`: 선택 이유

3. `collaboration_plan`
- 실제 실행 가능한 step 배열
- step 스키마: `agent`, `goal`, `deliverable`, `tool_hints`

Planner 주요 특징:

- 에이전트 이름 하드코딩 대신 카드/런타임 capability/tool 메타데이터 기반으로 선택
- 품질 우선(정확도/커버리지) 정책
- 필요 시 specialist coverage 보강(step 자동 추가)
- JSON 파싱 실패 시 fallback collaboration plan 생성

---

## 6) 실행 엔진 설계 (`event_manager.py`)

`execute_plan()`는 아래 순서로 동작합니다.

1. 실행 가능 에이전트 선별(local + a2a)
2. planner의 `collaboration_plan`을 실제 step으로 정규화
3. step 루프 실행 (`_run_collaboration_workflow`)
4. step마다 MainAgent 리뷰 기반 재계획 여부 판단
5. 실패 시 실패분석/복구 플로우
6. 최종 요약 생성

### 6.1 step 입력 컨텍스트 패킷

각 step 실행 시 대상 에이전트에 아래를 전달합니다.

- workflow 메타(`workflow_id`, 현재 step, total hint)
- 현재 task(goal, deliverable, tool hints)
- 사용자 요청 + 대화 히스토리
- 이전 step 결과(prior_results)
- 미해결 needs(open_needs)
- 남은 step 목록
- 전체 에이전트 카탈로그(이름/역할/capability/tools)
- capability 정책

즉, 프롬프트로 규칙을 강제하는 대신, 협업에 필요한 시스템 상태를 컨텍스트로 전달하는 구조입니다.

### 6.2 Additional Needs 기반 간접 위임

서브 에이전트는 직접 다른 에이전트를 호출하지 않습니다.
필요 작업은 응답 끝에 `Additional Needs`로 표준화해 요청합니다.

지원 포맷:

1. JSON
```json
{"needs": [{"target": "WebSearchAnalyst", "request": "..."}]}
```

2. 텍스트
```text
Additional Needs:
- [WebSearchAnalyst] 최근 외부 근거 수집
- [MainAgent] 사용자에게 기간/범위 재질문
```

`event_manager`가 이 needs를 파싱하고 open_needs에 적재한 뒤, 재계획 또는 fallback step 증강으로 실제 실행합니다.

### 6.3 사용자 확인 필요 시 워크플로우 일시중단

`Additional Needs`에 사용자 확인 요청이 감지되면:

- `workflow_paused = True`
- `pause_reason = awaiting_user_clarification`
- 사용자에게 질문을 반환하고 해당 turn 처리를 중단

즉, "MainAgent가 사용자 응답을 받아야 하는 상황이면 작업 프로세스를 멈춘다"는 정책이 반영되어 있습니다.

### 6.4 Capability 소유권 정책 (중요)

현재 정책:

- `comm.slack.post` capability 및 `slack_post_message` tool owner = `MainAgent`

그래서 협업 step이 다른 에이전트에 Slack 관련으로 잡혀도, owner policy로 `MainAgent`로 재라우팅됩니다.

---

## 7) 에이전트 정의

| 에이전트 | 역할 | 모델 | 핵심 capability | 직접 도구 |
|---|---|---|---|---|
| MainAgent | 코디네이터/재계획/최종전달 | `gemini-3-flash-preview` | coordination, workflow_replanning, user_clarification_routing, comm.slack.post | `slack_post_message` |
| PaperAnalyst | 로컬 PDF 기반 연구 분석 | `gemini-3-flash-preview` | paper_search, paper_summary, paper_memory_query 등 | `scrape_papers_with_mcp`, `load_paper_memory_with_mcp`, `expand_paper_memory_with_mcp`, `query_paper_memory` |
| WebSearchAnalyst | 웹 검색/검증 | `gemini-3-flash-preview` | web_search, web_summary, fact_check | `search_web_candidates_with_mcp`, `fetch_web_page_with_mcp` |
| SocialMediaAnalyst | SNS 신호 수집/요약 | `gemini-3-flash-preview` | sns_search, sns_summary | `scrape_sns_with_mcp` |

---

## 8) MCP 도구 호출 구조

### 8.1 공통 클라이언트

`mcp_local/client.py::call_mcp_tool()`:

1. `server_script_path`, `tool_name`, `arguments` 수신
2. stdio MCP 세션 생성 (`ClientSession`)
3. `session.call_tool(...)` 실행
4. 결과 model dump 후 반환

추가 구현 포인트:

- 경로가 `mcp_local/*_server.py`면 스크립트 실행이 아니라 패키지 모듈 실행(`python -m ...`) 모드로 전환
- standalone 폴더 이동 시 import 경로 안정성을 위해 `PYTHONPATH`를 보정

### 8.2 MCP 서버별 역할

| 서버 | tool | 기능 |
|---|---|---|
| `mcp_local/web_search_server.py` | `search_web`, `fetch_page` | DuckDuckGo Instant Answer + Bing HTML fallback, URL 본문 추출 |
| `mcp_local/paper_server.py` | `search_papers`, `get_paper_head`, `get_paper_content` | 로컬 PDF 검색/본문 추출 (`pypdf` 없으면 제한 모드) |
| `mcp_local/sns_server.py` | `search_sns_posts` | `db/sns` JSON에서 키워드 매칭 게시글 수집 |
| `mcp_local/slack_server.py` | `post_message` | Slack `chat.postMessage` 호출 |

---

## 9) A2A 브리지 상호작용

### 9.1 브리지 자동 기동

`scripts/start_a2a_agents.py`는 각 카드의 `base_url`, `module`, `attr`를 읽어 브리지 프로세스를 띄웁니다.

- 포트 열림 + 카드 endpoint 정상 -> launch skip
- 포트 열림 + 카드 비정상 -> 충돌로 판단
- 신규 기동 시 `/.well-known/agent-card.json` 준비될 때까지 대기

### 9.2 브리지 서버 동작

`mcp_local/a2a_bridge_server.py`:

1. 카드 요청: `GET /.well-known/agent-card.json`
2. 메시지 요청: `POST /` (JSON-RPC `message/send`)
3. 내부에서 로컬 `LlmAgent` 실행 후 텍스트 응답 반환

---

## 10) 로그 시스템 설계

### 10.1 기본 출력 파일

`system_logger.py`는 이벤트를 동시에 3곳에 기록합니다.

- `log/system_events.jsonl` (전체 이벤트)
- `log/components/<component>.jsonl` (컴포넌트별 분리)
- `log/session_log.jsonl` (현재 active 세션 스트림)

각 이벤트는 `session_seq`를 포함하며 증가 순서대로 기록됩니다.

### 10.2 세션 아카이브 정책

`start_new_logging_session(reset_files=True)` 호출 시:

1. 기존 `log/session_log.jsonl` 내용을 읽음
2. `log/session_log_ex_{num}` 폴더 생성
3. JSONL 한 줄당 JSON 파일 1개 생성
4. 새 `session_log.jsonl`과 시퀀스 파일 초기화

생성 파일명 규칙:

```text
{seq:010d}_{component}_{action}_{YYYYMMDDTHHMMSS_microZ}.json
```

예:

```text
0000000396_mcp_server.slack_tool_called_20260224T062818_629956Z.json
```

### 10.3 함수 콜 트레이싱

함수 단위 call trace(`function_trace`)는 현재 코드에서 의도적으로 비활성화되어 있습니다.

### 10.4 TaskGroup/ExceptionGroup 로깅

`log_exception()`은 traceback뿐 아니라 `sub_exceptions`를 포함한 ExceptionGroup 스냅샷도 기록합니다.

---

## 11) `session_log_example`로 보는 실제 실행 흐름

`log/session_log_example`는 실제 한 세션의 이벤트를 순번 파일로 분해한 예시입니다.

### 11.1 부트스트랩

- `0000000001_ad.main_agent_bridge_launch_started_...`
- `0000000002_a2a.bridge.launcher_launch_started_...`
- `0000000014_ad.main_agent_bridge_launch_completed_...`

### 11.2 사용자 요청 진입

- `0000000017_ad.main_agent_run_started_...`
- `0000000023_ad.main_agent_available_agents_finalized_...`

### 11.3 Planner 단계

- `0000000024_planner_planning_started_...`
- `0000000030_planner_prompt_completed_...`
- `0000000033_planner_routing_hint_coverage_augmented_...`
- `0000000040_planner_collaboration_plan_derived_...`
- `0000000041_planner_planning_completed_...`

### 11.4 실행 오케스트레이션 시작

- `0000000043_event_manager_execute_plan_started_...`
- `0000000047_event_manager.collaboration_workflow_selected_...`
- `0000000048_event_manager.collaboration_step_started_...`
- `0000000049_event_manager.agent_message_sent_...`

### 11.5 WebSearch 도구 호출 체인

- `0000000056_tool.search_web_candidates_with_mcp_call_started_...`
- `0000000058_mcp_client_tool_call_started_...`
- `0000000060_mcp_server.web_search_tool_called_...`
- `0000000061_mcp_server.web_search_tool_completed_...`
- `0000000063_mcp_client_tool_call_returned_...`

### 11.6 Slack 전송 단계(MainAgent owner)

- `0000000392_tool.slack_post_message_call_started_...`
- `0000000396_mcp_server.slack_tool_called_...`
- `0000000397_mcp_server.slack_tool_completed_...`
- `0000000400_tool.slack_post_message_call_completed_...`

### 11.7 종료

- `0000000408_event_manager.main_synthesis_synthesis_started_...`
- `0000000409_event_manager.main_synthesis_synthesis_completed_...`
- `0000000411_ad.main_agent_run_completed_...`
- `0000000412~0414_a2a.bridge.launcher_bridge_process_stopped_...`

이 순서를 보면 "계획 -> step 실행 -> 도구 호출 -> 재검토 -> 최종 합성"이 로그에서 그대로 추적됩니다.

---

## 12) 에이전트 간 메시지 로깅

에이전트 간 상호작용은 `component = event_manager.agent_message`로 기록됩니다.

`action` 값:

- `sent`: MainAgent -> 대상 에이전트
- `received`: 대상 에이전트 -> MainAgent
- `need_requested`: Additional Needs로 간접 요청 발생

`details`에는 `from_agent`, `to_agent`, `channel`, `workflow_id`, `workflow_step`, `message_preview`가 포함됩니다.

---

## 13) 실행 방법

### 13.1 메인 시스템 실행

```bash
pip install -r requirements.txt
python start_agentic.py
```

패키지 모드 실행(상위 폴더 기준)은 아래도 가능합니다.

```bash
python -m agentic_sample_ad.start_agentic
```

### 13.2 서브 에이전트 단독 실행

```bash
python -m agentic_sample_ad.paper_agent.user_entry_point
python -m agentic_sample_ad.web_search_agent.user_entry_point
python -m agentic_sample_ad.sns_agent.user_entry_point
```

---

## 14) 환경 변수

최소 필수:

- `GOOGLE_API_KEY`

Slack 전송 사용 시:

- `SLACK_MCP_SERVER_PATH` (예: `./mcp_local/slack_server.py`)
- `SLACK_BOT_TOKEN`

A2A/MCP 타임아웃 관련 변수는 코드에서 기본값이 있으므로 필요 시에만 오버라이드하면 됩니다.

---

## 15) 확장 방법 (에이전트/도구 추가 전제)

### 15.1 새 에이전트 추가

1. `new_agent/agent.py`에 `LlmAgent` 정의
2. `new_agent/well_known/agent_card.json` 작성 (`name`, `base_url`, `module`, `attr`, `capabilities`)
3. 필요시 `new_agent/event_manager.py`, `user_entry_point.py` 추가
4. `python start_agentic.py` 실행 시 카드 자동 탐색/브리지 기동

### 15.2 새 도구 추가

1. 에이전트 코드에 tool 함수 추가
2. MCP 서버 tool 구현 또는 기존 MCP 재사용
3. 카드 capability/description 갱신
4. 런타임 메타데이터 enrichment로 MainAgent 컨텍스트에 자동 반영

중요: 특정 capability 소유권(예: Slack)은 `event_manager.py`의 policy에서 통제합니다.

---

## 16) 자주 발생하는 문제 체크리스트

### Slack 호출 실패

1. `.env`에 `SLACK_MCP_SERVER_PATH`가 실제 파일 경로와 일치하는지
2. `SLACK_BOT_TOKEN`이 유효한지
3. Slack 채널명이 올바른지(봇 접근 권한 포함)

### 브리지 호출 실패(A2A card fetch timeout 등)

1. 대상 포트가 이미 점유되어 있지 않은지
2. `/.well-known/agent-card.json`이 실제 응답하는지
3. 필요 시 `A2A_REQUEST_TIMEOUT_SEC`/`A2A_CARD_TIMEOUT_SEC` 조정

### 로그 해석이 어려울 때

1. `log/session_log.jsonl`에서 현재 세션 연속 흐름 확인
2. `log/components/event_manager.collaboration.jsonl`로 협업 단계만 필터링
3. `event_manager.agent_message`로 에이전트 간 메시지 왕복 확인

---

## 17) 구현 기준 요약

이 시스템은 아래 한 줄로 요약됩니다.

- `MainAgent`가 계획/정책/최종응답을 책임지고, specialist는 capability 기반 step 실행과 `Additional Needs` 기반 간접 협업을 수행하며, 모든 상호작용은 JSONL 로그로 추적 가능한 구조입니다.

