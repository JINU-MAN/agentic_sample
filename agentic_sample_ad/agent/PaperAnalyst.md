# PaperAnalyst

## 개요
`PaperAnalyst`는 로컬 PDF DB를 기반으로 논문 검색/요약을 수행하고, 워크플로우 메모리를 활용해 후속 질의까지 처리하는 에이전트입니다.

## 구현 위치
- `agent/paper_agent.py`

## 주요 기능
- 논문 검색(`search_papers`) 및 후보 선별
- 워크플로우 단위 메모리 적재(overview/full)
- 상세 질문 시 필요한 논문만 지연 확장 로딩
- 메모리 기반 질의응답 및 스니펫 추출

## 사용 도구
- `scrape_papers_with_mcp`
  - MCP 서버: `mcp_local/paper_server.py`
  - 호출 도구: `search_papers`
- `load_paper_memory_with_mcp`
  - MCP 서버: `mcp_local/paper_server.py`
  - 호출 도구: `search_papers`, `get_paper_head`, `get_paper_content`
- `expand_paper_memory_with_mcp`
  - MCP 서버: `mcp_local/paper_server.py`
  - 호출 도구: `get_paper_content`
- `query_paper_memory`
  - 로컬 워크플로우 메모리를 질의

## 협업 규칙
- 입력에 포함된 `workflow_id`를 메모리 도구 호출에 동일하게 사용합니다.
- Slack 전송은 `MainAgent`에 위임합니다.
- 다른 에이전트를 직접 호출하지 않고 `Additional Needs` 형식으로 요청합니다.
- 로컬 DB 커버리지가 부족하면 `WebSearchAnalyst` 또는 `MainAgent` 후속 요청을 생성합니다.

## 모델
- `resolve_agent_model("PaperAnalyst")`로 모델을 결정합니다.
