from __future__ import annotations

import json
from typing import Any, Dict, List


_VALID_STATUSES = {"completed", "partial", "blocked", "failed"}


def _compact(value: str, max_chars: int = 320) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _parse_json_array(raw: str, field_name: str) -> tuple[List[Any] | None, str]:
    stripped = str(raw or "").strip()
    if not stripped:
        return [], ""
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return None, f"{field_name}: invalid JSON — {exc}"
    if not isinstance(parsed, list):
        return None, f"{field_name}: must be a JSON array, got {type(parsed).__name__}"
    return parsed, ""


def _validate_artifact(item: Any, idx: int) -> List[str]:
    errors: List[str] = []
    if not isinstance(item, dict):
        errors.append(f"artifacts[{idx}]: must be an object, got {type(item).__name__}")
        return errors
    title = str(item.get("title", item.get("name", ""))).strip()
    if not title:
        errors.append(f'artifacts[{idx}]: missing required field "title"')
    summary = str(item.get("summary", item.get("description", ""))).strip()
    if not summary:
        errors.append(f'artifacts[{idx}]: missing required field "summary"')
    return errors


def _validate_need(item: Any, idx: int) -> List[str]:
    errors: List[str] = []
    if not isinstance(item, dict):
        errors.append(f"needs[{idx}]: must be an object, got {type(item).__name__}")
        return errors
    request = str(item.get("request", "")).strip()
    if not request:
        errors.append(f'needs[{idx}]: missing required field "request"')
    return errors


def _normalize_artifact(item: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    title = str(item.get("title", item.get("name", ""))).strip()
    if title:
        normalized["title"] = _compact(title, max_chars=180)
    summary = str(item.get("summary", item.get("description", ""))).strip()
    if summary:
        normalized["summary"] = _compact(summary, max_chars=320)
    for key in ("url", "doi", "arxiv_id", "content_type", "type", "source_url"):
        val = str(item.get(key, "")).strip()
        if val:
            normalized[key] = _compact(val, max_chars=220)
    identifiers = item.get("identifiers")
    if isinstance(identifiers, dict) and identifiers:
        normalized["identifiers"] = {
            str(k): _compact(str(v), max_chars=120)
            for k, v in identifiers.items()
            if str(v).strip()
        }
    tags = item.get("tags")
    if isinstance(tags, list):
        compact_tags = [_compact(str(t), max_chars=40) for t in tags if str(t).strip()][:6]
        if compact_tags:
            normalized["tags"] = compact_tags
    return normalized


def _normalize_need(item: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {
        "request": _compact(str(item.get("request", "")).strip(), max_chars=280),
    }
    for key in ("kind", "reason"):
        val = str(item.get(key, "")).strip()
        if val:
            normalized[key] = _compact(val, max_chars=180)
    caps = item.get("required_capabilities")
    if isinstance(caps, list) and caps:
        normalized["required_capabilities"] = [str(c).strip() for c in caps if str(c).strip()]
    blocking = item.get("blocking")
    if isinstance(blocking, bool):
        normalized["blocking"] = blocking
    return normalized


def format_handoff_contract(
    status: str,
    summary: str,
    text_response: str = "",
    artifacts_json: str = "",
    needs_json: str = "",
) -> str:
    """
    Build and validate the handoff contract JSON that must be used as the agent's complete final response.

    Call this tool as the LAST step before finishing.
    Output the value returned by this tool verbatim as your entire final response.
    Do not add any prose, explanation, or markdown before or after it.

    Args:
        status: Outcome of the step. Must be one of: "completed", "partial", "blocked", "failed".
                Use "completed" when the task is fully done.
                Use "partial" when done but downstream work is still needed.
                Use "blocked" when you cannot proceed without more information.
                Use "failed" when the task could not be completed due to an error.
        summary: Required one-to-three sentence summary of what was accomplished or found.
                 Must not be empty.
        text_response: Optional detailed narrative with evidence, citations, or analysis.
                       Leave empty if the summary alone is sufficient.
        artifacts_json: Optional JSON array of reusable items for downstream agents.
                        Each object must have "title" (str) and "summary" (str).
                        Optional fields: "url", "doi", "arxiv_id", "content_type", "identifiers" (dict), "tags" (list).
                        Example: '[{"title": "Attention Is All You Need", "summary": "Transformer architecture paper.", "arxiv_id": "1706.03762"}]'
                        Leave empty string if there are no artifacts.
        needs_json: Optional JSON array of handoff requests for other agents or coordinator.
                    Each object must have "request" (str).
                    Optional fields: "kind" (str), "reason" (str), "required_capabilities" (list[str]), "blocking" (bool).
                    Example: '[{"request": "Fetch full text for arXiv:1706.03762", "required_capabilities": ["PaperAnalyst"], "blocking": true}]'
                    Leave empty string if there are no needs.

    Returns:
        Validated handoff contract JSON string. Use this as your verbatim final response.
        If validation fails, returns an error description — fix the issues and call again.
    """
    violations: List[str] = []

    # Validate status
    safe_status = str(status or "").strip().lower()
    if safe_status not in _VALID_STATUSES:
        violations.append(
            f'status: "{safe_status}" is not valid — must be one of: {", ".join(sorted(_VALID_STATUSES))}'
        )

    # Validate summary
    safe_summary = _compact(str(summary or "").strip(), max_chars=320)
    if not safe_summary:
        violations.append('summary: must not be empty')

    # Parse and validate artifacts
    artifacts_list, artifacts_parse_err = _parse_json_array(artifacts_json, "artifacts_json")
    if artifacts_parse_err:
        violations.append(artifacts_parse_err)
        artifacts_list = []
    else:
        for i, item in enumerate(artifacts_list or []):
            violations.extend(_validate_artifact(item, i))

    # Parse and validate needs
    needs_list, needs_parse_err = _parse_json_array(needs_json, "needs_json")
    if needs_parse_err:
        violations.append(needs_parse_err)
        needs_list = []
    else:
        for i, item in enumerate(needs_list or []):
            violations.extend(_validate_need(item, i))

    if violations:
        error_lines = "\n".join(f"  - {v}" for v in violations)
        return (
            f"format_handoff_contract failed validation — fix these issues and call again:\n"
            f"{error_lines}"
        )

    # Build contract
    contract: Dict[str, Any] = {
        "status": safe_status,
        "summary": safe_summary,
    }

    safe_text_response = str(text_response or "").strip()
    if safe_text_response:
        contract["text_response"] = safe_text_response

    normalized_artifacts = [
        _normalize_artifact(item)
        for item in (artifacts_list or [])
        if isinstance(item, dict)
    ]
    if normalized_artifacts:
        contract["artifacts"] = normalized_artifacts

    normalized_needs = [
        _normalize_need(item)
        for item in (needs_list or [])
        if isinstance(item, dict)
    ]
    if normalized_needs:
        contract["needs"] = normalized_needs

    return json.dumps(contract, ensure_ascii=False, indent=2)


def recover_handoff_contract_from_parts(normalized_parts: list) -> str | None:
    """
    Recover the handoff contract JSON from normalized_parts when the agent called
    format_handoff_contract but did not echo the result as a text message.

    Scans normalized_parts for a function_response whose name is
    "format_handoff_contract", extracts response.result, and returns it if it
    parses as a valid handoff contract dict (has both "status" and "summary").
    Returns None if no valid contract is found.
    """
    for part in normalized_parts:
        if not isinstance(part, dict):
            continue
        if part.get("kind") != "function_response":
            continue
        if part.get("name") != "format_handoff_contract":
            continue
        response = part.get("response", {})
        if not isinstance(response, dict):
            continue
        result = response.get("result")
        if not isinstance(result, str):
            continue
        result = result.strip()
        if not result:
            continue
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if parsed.get("status") and parsed.get("summary"):
            return result
    return None


__all__ = ["format_handoff_contract", "recover_handoff_contract_from_parts"]
