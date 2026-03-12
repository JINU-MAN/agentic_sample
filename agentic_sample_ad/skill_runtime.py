from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

from google.adk.skills import Frontmatter, Resources, Script, Skill
from google.adk.tools.skill_toolset import SkillToolset


_FRONTMATTER_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n(.*))?$", re.DOTALL)


def _strip_yaml_scalar(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _parse_frontmatter_block(block: str) -> Dict[str, object]:
    data: Dict[str, object] = {}
    current_map_key: str | None = None

    for raw_line in block.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if indent and current_map_key:
            nested = data.setdefault(current_map_key, {})
            if not isinstance(nested, dict) or ":" not in line:
                continue
            nested_key, nested_value = line.split(":", 1)
            nested[str(nested_key).strip()] = _strip_yaml_scalar(nested_value)
            continue

        current_map_key = None
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key.strip()
        normalized_value = value.strip()
        if not normalized_value:
            current_map_key = normalized_key
            data[normalized_key] = {}
            continue
        data[normalized_key] = _strip_yaml_scalar(normalized_value)
    return data


def _parse_skill_markdown(path: Path) -> Tuple[Frontmatter, str]:
    raw_text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(raw_text)
    if match is None:
        raise ValueError(f"Skill file is missing YAML frontmatter: {path}")

    frontmatter_block = match.group(1)
    body = (match.group(2) or "").strip()
    payload = _parse_frontmatter_block(frontmatter_block)

    name = str(payload.get("name", "")).strip()
    description = str(payload.get("description", "")).strip()
    if not name or not description:
        raise ValueError(f"Skill file must define name and description: {path}")

    metadata = payload.get("metadata", {})
    normalized_metadata = metadata if isinstance(metadata, dict) else {}

    frontmatter = Frontmatter(
        name=name,
        description=description,
        license=str(payload.get("license", "")).strip() or None,
        compatibility=str(payload.get("compatibility", "")).strip() or None,
        allowed_tools=str(payload.get("allowed_tools", "")).strip() or None,
        metadata={str(key): str(value) for key, value in normalized_metadata.items()},
    )
    return frontmatter, body


def _load_text_resources(directory: Path) -> Dict[str, str]:
    if not directory.exists() or not directory.is_dir():
        return {}

    resources: Dict[str, str] = {}
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix().lower()):
        if not path.is_file():
            continue
        rel_path = path.relative_to(directory).as_posix()
        resources[rel_path] = path.read_text(encoding="utf-8", errors="replace")
    return resources


def _load_script_resources(directory: Path) -> Dict[str, Script]:
    if not directory.exists() or not directory.is_dir():
        return {}

    resources: Dict[str, Script] = {}
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix().lower()):
        if not path.is_file():
            continue
        rel_path = path.relative_to(directory).as_posix()
        resources[rel_path] = Script(src=path.read_text(encoding="utf-8", errors="replace"))
    return resources


@lru_cache(maxsize=None)
def load_skills_from_dir(skills_dir: str) -> Tuple[Skill, ...]:
    root = Path(skills_dir).resolve()
    if not root.exists() or not root.is_dir():
        return ()

    loaded: list[Skill] = []
    for directory in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if not directory.is_dir():
            continue
        skill_md = directory / "SKILL.md"
        if not skill_md.exists() or not skill_md.is_file():
            continue

        frontmatter, instructions = _parse_skill_markdown(skill_md)
        resources = Resources(
            references=_load_text_resources(directory / "references"),
            assets=_load_text_resources(directory / "assets"),
            scripts=_load_script_resources(directory / "scripts"),
        )
        loaded.append(
            Skill(
                frontmatter=frontmatter,
                instructions=instructions,
                resources=resources,
            )
        )
    return tuple(loaded)


def build_skill_toolset(skills_dir: str | Path) -> SkillToolset | None:
    resolved = str(Path(skills_dir).resolve())
    skills = list(load_skills_from_dir(resolved))
    if not skills:
        return None
    return SkillToolset(skills=skills)


__all__ = ["build_skill_toolset", "load_skills_from_dir"]
