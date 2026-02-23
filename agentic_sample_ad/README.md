# agentic_sample_ad

이 폴더는 단독 실행 가능한 멀티 에이전트 패키지입니다.  
상위 `agentic_sample_api`의 모듈에 의존하지 않도록, 런타임에 필요한 코어 모듈과 도구가 이 폴더 내부에 포함되어 있습니다.

## 1. 디렉터리 구조

```text
agentic_sample_ad/
├─ start_agentic.py
├─ planner.py
├─ event_manager.py
├─ system_logger.py
├─ requirements.txt
├─ .env.example
├─ scripts/
│  └─ start_a2a_agents.py
├─ agent/
│  ├─ paper_agent.py
│  ├─ sns_agent.py
│  ├─ web_search_agent.py
│  └─ tool/
├─ mcp_local/
│  ├─ a2a_bridge_server.py
│  ├─ client.py
│  ├─ paper_server.py
│  ├─ sns_server.py
│  ├─ web_search_server.py
│  └─ slack_server.py
├─ db/
│  ├─ paper/
│  └─ sns/
├─ agent_cards/
│  └─ agent_card.json
├─ main_agent/
├─ paper_agent/
├─ sns_agent/
├─ web_search_agent/
└─ common/
```

## 2. 핵심 모듈 역할

- `main_agent/start_agentic.py`
  - `.env` 로드, 서브 에이전트 card 수집, A2A bridge 실행/종료 관리.
- `main_agent/agent.py`
  - 메인 계획 수립(`planner.py`) + 실행(`event_manager.py`) 오케스트레이션.
- `scripts/start_a2a_agents.py`
  - card 기반 A2A bridge 프로세스 실행.
- `agent/*.py`
  - 도메인 전문 에이전트(Paper/SNS/Web) 정의.
- `mcp_local/*.py`
  - 로컬 MCP 서버/클라이언트(논문, SNS, 웹검색, Slack).

## 3. 실행 방법

의존성 설치:

```bash
pip install -r requirements.txt
```

메인 실행(권장):

```bash
python -m agentic_sample_ad.start_agentic
```

직접 실행(폴더 내부에서):

```bash
python start_agentic.py
```

서브 에이전트 단독 테스트:

```bash
python -m agentic_sample_ad.paper_agent.user_entry_point
python -m agentic_sample_ad.sns_agent.user_entry_point
python -m agentic_sample_ad.web_search_agent.user_entry_point
```

## 4. 동작 예시 흐름 (메인 멀티 에이전트)

예시 요청:  
`최근 AI 반도체 동향을 웹/SNS/논문 관점으로 요약해줘`

1. `start_agentic`가 서브 에이전트 card를 읽고 A2A bridge를 실행합니다.
2. `input_loop`가 사용자 입력을 받아 `run_main_agent`로 전달합니다.
3. `run_main_agent`가 세션 히스토리를 반영하고 사용 가능한 에이전트 메타데이터를 보강합니다.
4. `planner.plan_with_main_agent`가 `raw_plan`, `routing_hint`, `collaboration_plan`을 생성합니다.
5. `event_manager.execute_plan`이 step 단위 협업 워크플로를 실행합니다.
6. step 결과의 `Additional Needs`를 파싱해 필요 시 재계획(replan)합니다.
7. 결과를 `=== Execution Results ===` + `=== Final Summary ===`로 합쳐 반환합니다.
8. 종료 시 bridge 프로세스를 정리합니다.

## 5. 단독 실행 기준

- 이 폴더는 자체 `planner.py`, `event_manager.py`, `system_logger.py`, `scripts/`, `agent/`, `mcp_local/`, `db/`를 포함합니다.
- 주요 import는 `agentic_sample_ad.*` 경로로 고정되어 상위 폴더 파일을 참조하지 않습니다.
