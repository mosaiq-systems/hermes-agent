"""Tests for internal-knowledge-first source priority guardrails."""

import json

from agent.source_priority_guardrail import (
    INTERNAL_LOOKUP_TOOL_NAMES,
    evaluate_source_priority,
    source_priority_block_result,
)


def test_blocks_web_search_for_mosaiq_internal_topic_without_internal_lookup():
    decision = evaluate_source_priority(
        "web_search",
        {"query": "analyze the Mosaiq memory infrastructure"},
        messages=[],
    )

    assert decision.blocked is True
    assert "internal knowledge" in decision.message.lower()
    assert "skills" in decision.message
    assert "Honcho" in decision.message


def test_blocks_web_extract_when_recent_user_request_is_internal_even_if_url_is_public_docs():
    messages = [
        {
            "role": "user",
            "content": "Analyze the memory infrastructure for this Mosaiq/Hermes deployment.",
        }
    ]

    decision = evaluate_source_priority(
        "web_extract",
        {"urls": ["https://hermes-agent.nousresearch.com/docs/user-guide/features/memory"]},
        messages=messages,
    )

    assert decision.blocked is True
    assert "internal-first" in decision.message.lower()


def test_allows_internal_topic_after_completed_internal_lookup():
    messages = [
        {"role": "user", "content": "Analyze the Mosaiq memory infrastructure."},
        {
            "role": "tool",
            "name": "skill_view",
            "content": "hermes-memory-architecture skill content",
            "tool_call_id": "call_1",
        },
    ]

    decision = evaluate_source_priority(
        "web_extract",
        {"urls": ["https://hermes-agent.nousresearch.com/docs/user-guide/features/memory"]},
        messages=messages,
    )

    assert decision.blocked is False


def test_allows_public_current_web_search_without_internal_lookup():
    decision = evaluate_source_priority(
        "web_search",
        {"query": "latest Python release notes"},
        messages=[],
    )

    assert decision.blocked is False


def test_block_result_is_json_tool_error():
    decision = evaluate_source_priority(
        "web_search",
        {"query": "Mosaiq endpoint auth details"},
        messages=[],
    )

    payload = json.loads(source_priority_block_result(decision))

    assert payload["status"] == "blocked"
    assert payload["policy"] == "internal_knowledge_first"
    assert payload["error"] == decision.message


def test_internal_lookup_tool_set_includes_core_memory_sources():
    assert "skill_view" in INTERNAL_LOOKUP_TOOL_NAMES
    assert "session_search" in INTERNAL_LOOKUP_TOOL_NAMES
    assert "read_file" in INTERNAL_LOOKUP_TOOL_NAMES
    assert "search_files" in INTERNAL_LOOKUP_TOOL_NAMES
