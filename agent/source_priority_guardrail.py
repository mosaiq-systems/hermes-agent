"""Internal-knowledge-first guardrail for external lookup tools.

This module is intentionally small and dependency-free. It prevents a common
failure mode where the agent jumps to public web/docs for user- or deployment-
specific questions that should be answered from local memory, skills, Keep,
Honcho, Obsidian, sessions, or the filesystem first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

EXTERNAL_LOOKUP_TOOL_NAMES = frozenset(
    {
        "web_search",
        "web_extract",
        "browser_navigate",
    }
)

INTERNAL_LOOKUP_TOOL_NAMES = frozenset(
    {
        "skill_view",
        "skills_list",
        "session_search",
        "read_file",
        "search_files",
        "mcp_keep_keep_flow",
        "mcp_keep_keep_prompt",
        "mcp_keep_get_prompt",
        "mcp_keep_read_resource",
        "honcho_profile",
        "honcho_search",
        "honcho_reasoning",
        "honcho_context",
    }
)

_INTERNAL_TOPIC_TERMS = (
    "mosaiq",
    "mosaiq.chat",
    "this deployment",
    "local deployment",
    "this instance",
    "our deployment",
    "our architecture",
    "our system",
    "our workflow",
    "our memory",
    "memory infrastructure",
    "endpoint auth",
    "endpoint routing",
    "local proxy",
    "proxy db",
    "profile fleet",
    "kanban profile",
    "kanban worker",
    "kanban board",
    "honcho",
    "obsidian",
    "plur",
    "keep chromadb",
    "keep memory",
    "hermes memory infrastructure",
    "hermes local",
)

_INTERNAL_CONTEXT_TERMS = (
    "what did we decide",
    "what did we do",
    "where did we leave",
    "do we have knowledge",
    "don't we have",
    "existing knowledge",
    "internal knowledge",
)

_BLOCK_MESSAGE = (
    "Internal-first source policy blocked this external lookup: the current "
    "request appears to concern Mosaiq/local Hermes deployment knowledge. "
    "Check internal knowledge first (memory, skills, Obsidian/Keep, Honcho, "
    "session_search, or local files/config). External web/docs are a last "
    "resort after internal sources are checked or when the user explicitly "
    "asks for public/current upstream information."
)


@dataclass(frozen=True)
class SourcePriorityDecision:
    blocked: bool = False
    message: str = ""
    policy: str = "internal_knowledge_first"


_ALLOW = SourcePriorityDecision()


def evaluate_source_priority(
    tool_name: str,
    tool_args: Mapping[str, Any] | None,
    *,
    messages: Sequence[Mapping[str, Any]] | None = None,
) -> SourcePriorityDecision:
    """Return whether an external lookup violates internal-first policy."""
    if tool_name not in EXTERNAL_LOOKUP_TOOL_NAMES:
        return _ALLOW

    combined_text = _combined_context(tool_args or {}, messages or [])
    if not _looks_like_internal_topic(combined_text):
        return _ALLOW

    if _has_completed_internal_lookup(messages or []):
        return _ALLOW

    return SourcePriorityDecision(blocked=True, message=_BLOCK_MESSAGE)


def source_priority_block_result(decision: SourcePriorityDecision) -> str:
    """Build the synthetic tool error payload for a blocked external lookup."""
    return json.dumps(
        {
            "error": decision.message,
            "status": "blocked",
            "policy": decision.policy,
        },
        ensure_ascii=False,
    )


def _combined_context(
    tool_args: Mapping[str, Any], messages: Sequence[Mapping[str, Any]]
) -> str:
    chunks: list[str] = []
    chunks.append(_stringify(tool_args))
    for msg in messages[-8:]:
        role = str(msg.get("role") or "")
        if role in {"user", "system"}:
            chunks.append(_stringify(msg.get("content")))
    return "\n".join(chunk for chunk in chunks if chunk).lower()


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def _looks_like_internal_topic(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if any(term in lowered for term in _INTERNAL_CONTEXT_TERMS):
        return True
    return any(term in lowered for term in _INTERNAL_TOPIC_TERMS)


def _has_completed_internal_lookup(messages: Sequence[Mapping[str, Any]]) -> bool:
    # Keep this turn-local-ish: a skill or file read from dozens of messages ago
    # should not satisfy the policy for a new internal topic.
    for msg in messages[-12:]:
        if msg.get("role") != "tool":
            continue
        name = str(msg.get("name") or "")
        if name in INTERNAL_LOOKUP_TOOL_NAMES:
            return True
        if name.startswith("mcp_keep_") or name.startswith("honcho_"):
            return True
    return False
