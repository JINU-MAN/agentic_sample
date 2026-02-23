from pathlib import Path
from typing import Any, Dict, List
import json

from mcp.server.fastmcp import FastMCP
from agentic_sample_ad.system_logger import log_event, log_exception


mcp = FastMCP("sns-mcp-server", json_response=True)

DB_ROOT = Path(__file__).parent.parent / "db" / "sns"


@mcp.tool()
def search_sns_posts(keyword: str) -> List[Dict[str, Any]]:
    """
    db/sns 아래의 JSON 파일들에서 keyword와 관련된 게시글만 모아 반환합니다.

    JSON 예시 스키마:
    {
      "data": [
        {
          "id": "123456789_987654321",
          "message": "이번 주말에 열리는 AI 컨퍼런스 소식입니다! #AI #Tech",
          "created_time": "2026-02-12T10:30:00+0000",
          "full_picture": "https://...",
          "permalink_url": "https://...",
          "from": { "name": "Tech News Page", "id": "123456789" }
        },
        ...
      ],
      "paging": { ... }
    }
    """
    log_event(
        "mcp_server.sns",
        "tool_called",
        {"tool": "search_sns_posts", "keyword": keyword, "db_root": str(DB_ROOT)},
        direction="inbound",
    )
    if not DB_ROOT.exists():
        log_event(
            "mcp_server.sns",
            "tool_completed",
            {
                "tool": "search_sns_posts",
                "keyword": keyword,
                "result_count": 0,
                "reason": "db_root_missing",
            },
            direction="outbound",
        )
        return []

    try:
        keyword_lower = keyword.lower()
        collected: List[Dict[str, Any]] = []

        for path in DB_ROOT.rglob("*.json"):
            try:
                with path.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception:
                continue

            data_list = obj.get("data", [])
            if not isinstance(data_list, list):
                continue

            matched_posts: List[Dict[str, Any]] = []
            for item in data_list:
                message = str(item.get("message", ""))
                if keyword_lower in message.lower():
                    matched_posts.append(item)

            if matched_posts:
                collected.append(
                    {
                        "file": str(path),
                        "matched_posts": matched_posts,
                    }
                )

        log_event(
            "mcp_server.sns",
            "tool_completed",
            {
                "tool": "search_sns_posts",
                "keyword": keyword,
                "result_count": len(collected),
                "matched_files": [str(item.get("file", "")) for item in collected],
            },
            direction="outbound",
        )
        return collected
    except Exception as e:
        log_exception(
            "mcp_server.sns",
            "tool_failed",
            e,
            {"tool": "search_sns_posts", "keyword": keyword},
        )
        raise


if __name__ == "__main__":
    # 기본적으로 stdio MCP 서버로 실행
    # 예: `uv run mcp/sns_server.py` 혹은 `python -m mcp.sns_server`
    mcp.run(transport="stdio")


