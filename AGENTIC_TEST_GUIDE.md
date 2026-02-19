# Agentic 테스트 가이드

이 문서는 `c:\agentic_sample` 프로젝트에서 로컬 에이전트를 직접 테스트하는 방법을 정리한 문서입니다.

## 1. 사전 준비

- Python 3.10 이상 권장
- 프로젝트 루트에서 실행

```powershell
cd c:\agentic_sample
```

의존성 설치:

```powershell
pip install -r requirements.txt
```

## 2. .env 설정

프로젝트 루트 `.env`에 아래 항목이 있어야 합니다.

```env
GOOGLE_API_KEY=your_real_google_api_key
SLACK_MCP_SERVER_PATH=./mcp_local/slack_server.py
SLACK_BOT_TOKEN=your_slack_token
```

설명:
- `GOOGLE_API_KEY`: Gemini 호출에 필수
- `SLACK_MCP_SERVER_PATH`: Slack MCP 서버 파일 경로
- `SLACK_BOT_TOKEN`: Slack 전송 기능 사용 시 필요

주의:
- 키/토큰은 절대 외부에 공유하지 마세요.
- 이미 노출된 토큰은 Slack/Google 콘솔에서 재발급(rotate) 권장

## 3. 빠른 에이전트 테스트 (권장)

### 3.1 SNS 에이전트 테스트

```powershell
python scripts/run_agent_test.py sns "AI"
```

### 3.2 논문 에이전트 테스트

```powershell
python scripts/run_agent_test.py paper "machine learning"
```

성공 기준:
- 에러 없이 분석/요약 텍스트가 출력됨
- 출력 중간에 MCP 도구 호출 로그(`Processing request of type ...`)가 보일 수 있음

## 4. Slack 전송 테스트

Slack 전송을 명시적으로 요청하는 질의로 테스트할 수 있습니다.

```powershell
python scripts/run_agent_test.py sns "AI 게시글 요약해서 #general 채널에 올려줘"
```

```powershell
python scripts/run_agent_test.py paper "machine learning 논문 요약을 #general에 보내줘"
```

전송이 실패하면 보통 아래를 확인하세요:
- `SLACK_BOT_TOKEN` 유효성
- 봇이 해당 채널에 초대되어 있는지
- 채널 이름/ID 형식이 Slack 앱 권한과 맞는지

## 5. 메인 진입 실행

```powershell
python start_agentic.py
```

참고:
- `start_agentic.py`는 `user_entry_point.input_loop()`를 호출하는 메인 진입점입니다.
- `start_agentic.py` 실행 시 A2A 브리지 서버도 함께 자동 기동됩니다 (`agent_cards/agent_card.json` 기준).
- 프로그램 종료 시 해당 브리지 서버도 함께 종료됩니다.
- 세션 중 컨텍스트를 초기화하려면 `reset` 또는 `/reset` 입력
- 현재 구조상 실제 MCP 도구 검증은 `run_agent_test.py`가 가장 확실합니다.

## 6. 자주 발생하는 오류

### 6.1 `API_KEY_INVALID`
- 원인: `GOOGLE_API_KEY`가 잘못됨
- 조치: 유효한 키로 교체 후 재실행

### 6.2 `429 RESOURCE_EXHAUSTED`
- 원인: Gemini API 쿼터 초과 또는 프로젝트 한도 0
- 조치: 쿼터/결제/플랜 상태 확인 후 재시도

### 6.3 `No module named 'mcp.server.fastmcp'`
- 원인: 의존성 미설치
- 조치: `pip install -r requirements.txt`

### 6.4 Slack 관련 오류 (`missing_slack_token`, `invalid_auth` 등)
- 원인: 토큰 누락/만료/권한 부족
- 조치: `.env` 값 확인 + Slack 앱 권한 점검

### 6.5 A2A 관련 `HTTP Error 503`
- 원인: A2A 클라이언트가 아래 카드 엔드포인트에 접속하지 못함
  - `http://127.0.0.1:9101/.well-known/agent-card.json` (또는 각 에이전트 base_url)
- 조치:
  1) 브리지 서버 실행
     ```powershell
     python scripts/start_a2a_agents.py
     ```
  2) 카드 URL 헬스체크
     ```powershell
     curl http://127.0.0.1:9101/.well-known/agent-card.json
     ```
  3) 포트 충돌 시 `agent_cards/agent_card.json`의 `base_url` 포트 변경

## 7. 테스트 체크리스트

- [ ] `pip install -r requirements.txt` 완료
- [ ] `.env`에 `GOOGLE_API_KEY` 설정 완료
- [ ] (선택) `.env`에 `SLACK_BOT_TOKEN` 설정 완료
- [ ] `python scripts/run_agent_test.py sns "AI"` 성공
- [ ] `python scripts/run_agent_test.py paper "machine learning"` 성공
