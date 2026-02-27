# SocialMediaAnalyst

## 개요
`SocialMediaAnalyst`는 SNS/커뮤니티 신호를 수집하고, 사용자 요청과의 관련성을 기준으로 요약하는 에이전트입니다.

## 구현 위치
- `agent/sns_agent.py`

## 주요 기능
- SNS 데이터 검색
- 검색 결과의 의미 기반 선별/요약
- 증거가 부족할 때 후속 작업 요청(`Additional Needs`) 생성

## 사용 도구
- `scrape_sns_with_mcp`
  - MCP 서버: `mcp_local/sns_server.py`
  - 호출 도구: `search_sns_posts`

## 협업 규칙
- Slack 전송은 직접 수행하지 않고 `MainAgent`에 위임합니다.
- 다른 에이전트를 직접 호출하지 않고 `Additional Needs` 형식으로 요청합니다.
- 출력 마지막에 아래 둘 중 하나를 반드시 포함합니다.
  - `Additional Needs: none`
  - `Additional Needs:` + `[TargetAgentName] request` 목록

## 모델
- `resolve_agent_model("SocialMediaAnalyst")`로 모델을 결정합니다.
