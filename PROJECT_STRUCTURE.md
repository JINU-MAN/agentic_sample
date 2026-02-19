# 프로젝트 구조 (`c:\agentic_sample`)

이 문서는 현재 프로젝트의 디렉터리 구조와 실행 흐름을 코드 기준으로 정리한 스냅샷입니다.

## 1) 최상위 구조

```text
c:\agentic_sample
+-- start_agentic.py
+-- user_entry_point.py
+-- agentic.py
+-- planner.py
+-- event_manager.py
+-- session_store.py
+-- system_logger.py
+-- requirements.txt
+-- .env
+-- PROJECT_STRUCTURE.md
+-- AGENTIC_TEST_GUIDE.md
+-- TESTING.md
+-- 아이디어.txt
+-- agent/
+-- agent_cards/
+-- mcp_local/
+-- db/
+-- log/
`-- scripts/
```

## 2) 진입점(Entry Points)

- `start_agentic.py`
  - 메인 실행 파일.
  - 입력 루프 시작 전에 A2A 브리지 서버를 자동 기동.
  - 종료 시 기동한 브리지 프로세스를 정리.
  - `user_entry_point.input_loop()` 호출.

- `user_entry_point.py`
  - 콘솔 입력 루프.
  - 사용자 입력을 `run_main_agent(...)`로 전달.
  - 같은 세션 ID를 재사용.
  - `reset`, `/reset`으로 메모리 컨텍스트 초기화.

- `scripts/start_a2a_agents.py`
  - `agent_cards/agent_card.json` 기반으로 로컬 A2A 브리지 실행.
  - 카드의 `module`/`attr`/`base_url` 사용.
  - 포트 점유/헬스(에이전트 카드) 체크 후 실행.

- `scripts/run_agent_test.py`
  - 단일 에이전트 스모크 테스트.
  - 대상: `sns`, `paper`, `web`.

## 3) 오케스트레이션 레이어

- `agentic.py`
  - 프로세스 로깅 초기화.
  - `.env` 로드.
  - `a2a` 패키지 로그 브리지 활성화.
  - 메인 에이전트(`create_main_agent`) 생성.
  - `agent_cards/*.json` 로드.
  - `agent/*.py`에서 `LlmAgent` 런타임 탐색.
  - 카드 메타데이터 + 런타임 메타데이터 병합/보강.
  - `planner.plan_with_main_agent(...)` 호출.
  - `event_manager.execute_plan(...)` 호출.
  - `session_store`에 대화 이력 저장.

- `planner.py`
  - 메인 LLM으로 아래를 생성:
    - `raw_plan`
    - `routing_hint` (`selected_agents`, `keywords`, `reason`)
    - `collaboration_plan` (`steps`, `tool_hints`, `notes`)
  - 정확도/커버리지 우선 정책:
    - 전문 에이전트 우선 선택.
    - 필요 시 `selected_agents`를 확장해 커버리지 보강.
    - 선택된 전문 에이전트 스텝 누락 시 자동 보정.

- `event_manager.py`
  - 플랜 메타데이터 기반으로 로컬/A2A 실행 경로 결정.
  - 협업 스텝 실행 시 공유 컨텍스트 제공:
    - 이전 출력, open needs, 남은 스텝, 에이전트 카드 요약.
  - 간접 위임 모델:
    - 하위 에이전트는 `Additional Needs`로 요청.
    - 메인 에이전트가 재계획(`updated_steps`)으로 라우팅.
  - 재계획 실패 시 폴백:
    - `[AgentName] request` 형태의 need를 실제 스텝으로 자동 추가.
  - 오류 처리:
    - 실패 원인 분석 -> 재계획 또는 중단 메시지 결정.

- `session_store.py`
  - 세션별 인메모리 대화 컨텍스트 관리.

## 4) 로깅 레이어 (`system_logger.py`, `log/`)

- `system_logger.py`
  - JSONL 구조 로그 기록.
  - 각 이벤트를 다음에 동시 기록:
    - `log/system_events.jsonl`
    - `log/components/<component>.jsonl`
    - `log/test_log.jsonl`
  - 프로세스 시작 시 `log/test_log.jsonl` 초기화.
  - Python `a2a` 패키지 로그를 `a2a.package` 이벤트로 브리지.

- `log/`
  - `system_events.jsonl`: 전체 이벤트 스트림.
  - `test_log.jsonl`: 실행 단위 테스트 로그(실행마다 초기화).
  - `components/`: 컴포넌트별 분리 로그.

## 5) 에이전트/툴 레이어 (`agent/`)

- `agent/paper_agent.py`
  - 논문 분석 에이전트(`PaperAnalyst`).
  - 핵심 도구:
    - `scrape_papers_with_mcp`
    - `load_paper_memory_with_mcp`
    - `expand_paper_memory_with_mcp`
    - `query_paper_memory`
    - `slack_post_message`
  - 메모리 정책:
    - 기본은 `overview`(제목/초반부 기반 요약) 우선 적재.
    - 상세 질문 시 필요한 논문만 `full` 본문 지연 로딩.

- `agent/sns_agent.py`
  - SNS 분석 에이전트(`SocialMediaAnalyst`).
  - 도구: `scrape_sns_with_mcp`, `slack_post_message`.

- `agent/web_search_agent.py`
  - 웹 리서치 에이전트(`WebSearchAnalyst`).
  - 도구: `search_web_candidates_with_mcp`, `fetch_web_page_with_mcp`, `slack_post_message`.

- `agent/tool/slack_mcp_tool.py`
  - Slack MCP 도구 래퍼.
  - MCP `post_message` 호출 담당.

- `agent/tool/a2a_delegate_tool.py`
  - A2A 위임용 공통 도구 모듈.
  - 현재는 재사용 가능 모듈로 `agent/tool/`에 유지.

- `agent/NEW_AGENT_TEMPLATE.md`
  - 새 에이전트 추가 템플릿.
  - `Additional Needs` 협업 출력 규약 포함.

## 6) 에이전트 등록 (`agent_cards/`)

- `agent_cards/agent_card.json`
  - 실행 가능한 에이전트 카드 메타데이터.
  - 주요 필드:
    - `type` (`local` 또는 `a2a`)
    - `base_url` (`a2a` 타입에서 사용)
    - `module`, `attr`
    - `capabilities`

참고:
- 카드가 없어도 `agent/*.py` 런타임 탐색으로 로컬 에이전트 발견 가능.
- 카드 메타데이터와 런타임 메타데이터는 병합 후 계획/실행에 사용.

## 7) MCP 레이어 (`mcp_local/`)

- `mcp_local/client.py`
  - 로컬 MCP 서버 스크립트를 stdio로 실행하고 도구 호출.

- `mcp_local/paper_server.py`
  - `db/paper/**/*.pdf` 검색.
  - 파일명 + 추출 텍스트 기반 점수화.
  - `get_paper_head`, `get_paper_content` 제공.

- `mcp_local/sns_server.py`
  - `db/sns/**/*.json` 게시물 검색.

- `mcp_local/web_search_server.py`
  - 웹 검색 및 페이지 텍스트 추출.

- `mcp_local/slack_server.py`
  - Slack Web API(`chat.postMessage`) 호출.

- `mcp_local/a2a_bridge_server.py`
  - 로컬 `LlmAgent`를 A2A JSON-RPC HTTP 서버로 노출.
  - 엔드포인트:
    - `GET /.well-known/agent-card.json`
    - `POST /` (`message/send`)

## 8) 데이터 레이어 (`db/`)

- `db/paper/`: 논문 PDF 데이터.
- `db/sns/`: SNS 샘플 데이터(`sample_posts.json`).

## 9) 설정/문서

- `.env`
  - 주요 키 예시:
    - `GOOGLE_API_KEY`
    - `SLACK_MCP_SERVER_PATH`
    - `SLACK_BOT_TOKEN`
    - `A2A_PACKAGE_LOG_LEVEL`
    - `A2A_*_TIMEOUT*` 계열 옵션

- `requirements.txt`
  - 핵심 라이브러리: `google-adk`, `a2a-sdk`, `httpx`, `mcp` 등.

- `AGENTIC_TEST_GUIDE.md`, `TESTING.md`
  - 테스트 가이드/체크리스트.

## 10) 권장 실행 흐름

1. `python start_agentic.py`
2. 콘솔에서 사용자 요청 입력
3. 필요 시 `reset` 또는 `/reset`
4. 종료: `exit` 또는 `quit`

단일 에이전트 테스트:

- `python scripts/run_agent_test.py sns "AI"`
- `python scripts/run_agent_test.py paper "machine learning"`
- `python scripts/run_agent_test.py web "latest AI policy update"`
