"""Helpers for reading agent-CLI transcripts, for the bench adapters.

  result_text   the text of a tool result or MCP result block (a string,
                or a list of typed text blocks), capped at `cap` chars
  sniff_codex   whether a transcript is a codex event stream (its first
                event is thread.started) rather than claude stream-json

Parsing the streams themselves, scope decisions and observation
textualization stay in each bench's adapter — they are bench semantics.
"""
from __future__ import annotations

import json


def result_text(content, cap: int) -> str:
    """tool_result / MCP content: string, or list of typed text blocks."""
    if isinstance(content, dict) and isinstance(content.get("content"), list):
        content = content["content"]          # MCP envelope
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                parts.append(str(b.get("text") or b.get("content") or "")[:cap])
            else:
                parts.append(str(b)[:cap])
        return "\n".join(p for p in parts if p)[:cap]
    if content is None:
        return ""
    return (content if isinstance(content, str) else str(content))[:cap]


def sniff_codex(lines) -> bool:
    first = next((ln for ln in lines if ln.strip()), "{}")
    try:
        return json.loads(first).get("type") == "thread.started"
    except json.JSONDecodeError:
        return False

