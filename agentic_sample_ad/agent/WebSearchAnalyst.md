# WebSearchAnalyst

## 개요
`WebSearchAnalyst`는 웹/뉴스 소스를 탐색하고, 신뢰 가능한 URL 근거를 정리해 전달하는 에이전트입니다.

## 구현 위치
- `agent/web_search_agent.py`

## 주요 기능
- 질의 후보 다중 생성 후 웹 검색
- URL 중복 제거 및 결과 랭킹
- 상위 URL 본문 수집 후 사실 검증
- 결과와 함께 출처 URL 제공

## 사용 도구
- `search_web_candidates_with_mcp`
  - MCP 서버: `mcp_local/web_search_server.py`
  - 호출 도구: `search_web`
- `fetch_web_page_with_mcp`
  - MCP 서버: `mcp_local/web_search_server.py`
  - 호출 도구: `fetch_page`

## 협업 규칙
- Slack 전송은 `MainAgent`에 위임합니다.
- 다른 에이전트를 직접 호출하지 않고 `Additional Needs`로 간접 위임합니다.
- 근거가 약하거나 충돌하면 `PaperAnalyst`/`MainAgent` 후속 요청을 생성합니다.
- 도구 사용 상한 정책이 있습니다.
  - 검색 라운드 최대 2회
  - 페이지 fetch 최대 8개

## 모델
- `resolve_agent_model("WebSearchAnalyst")`로 모델을 결정합니다.
