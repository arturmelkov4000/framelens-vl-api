"""Small helpers that do not depend on heavy ML packages."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def as_file_uri(path: str | Path) -> str:
    """Return a file:// URI for a local path."""
    return Path(path).expanduser().resolve().as_uri()


def extract_json_object(text: str) -> Any:
    """Best-effort JSON extraction from model output."""
    stripped = text.strip()
    if not stripped:
        return None

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    details_start = stripped.find("<details>")
    if details_start > 0:
        try:
            return json.loads(stripped[:details_start].strip())
        except json.JSONDecodeError:
            pass

    start_candidates = [idx for idx in (stripped.find("{"), stripped.find("[")) if idx >= 0]
    if not start_candidates:
        return None

    start = min(start_candidates)
    for end in range(len(stripped), start, -1):
        candidate = stripped[start:end]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def merge_prompt(base_prompt: str, user_prompt: str | None) -> str:
    """Append an optional user objective without weakening the JSON contract."""
    if not user_prompt:
        return base_prompt
    return f"{base_prompt}\n\nAdditional user objective:\n{user_prompt.strip()}\n"
