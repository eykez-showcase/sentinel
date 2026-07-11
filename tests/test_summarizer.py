"""
Tests for EventSummarizer — verifies prompt structure and output handling
without making real API calls.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from sentinel.ai.summarizer import AnthropicSummarizer, OpenAISummarizer, _build_prompt
from sentinel.schemas.event import EventRead


def make_event(
    event_type="zone_entry",
    zone_id="front_porch",
    class_name="person",
    severity="high",
    minutes_ago=0,
) -> EventRead:
    ts = datetime.now(timezone.utc).replace(
        minute=max(0, datetime.now(timezone.utc).minute - minutes_ago)
    )
    return EventRead(
        id="test-id",
        camera_id="cam-1",
        track_id=None,
        event_type=event_type,
        zone_id=zone_id,
        class_name=class_name,
        severity=severity,
        timestamp=ts,
        metadata_json=None,
        created_at=ts,
        ai_summary=None,
    )


# ── Shared prompt logic ───────────────────────────────────────────────────────

def test_build_prompt_contains_event_details():
    events = [
        make_event("zone_entry", "front_porch", "person", "high"),
        make_event("zone_exit", "front_porch", "person", "low"),
    ]
    prompt = _build_prompt(events, "My Home")

    assert "My Home" in prompt
    assert "Zone Entry" in prompt
    assert "Zone Exit" in prompt
    assert "front_porch" in prompt
    assert "person" in prompt


def test_max_events_truncation():
    from sentinel.ai.summarizer import BaseEventSummarizer

    class DummySummarizer(BaseEventSummarizer):
        async def _call(self, user_text, event_count):
            return ("", {}, {})

    summarizer = DummySummarizer(max_events=3)
    events = [make_event(minutes_ago=i) for i in range(10)]
    window = sorted(events, key=lambda e: e.timestamp)[-3:]
    assert len(window) == 3


# ── Anthropic provider ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_anthropic_summarize_returns_text():
    mock_response = MagicMock()
    mock_response.id = "msg-1"
    mock_response.model = "claude-test"
    mock_response.stop_reason = "end_turn"
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50
    mock_response.content = [MagicMock(text="A person entered the front porch.")]

    mock_client = MagicMock()
    mock_client.messages = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    summarizer = AnthropicSummarizer.__new__(AnthropicSummarizer)
    summarizer._client = mock_client
    summarizer._model = "claude-test"
    summarizer._max_events = 10

    text, raw_req, raw_resp = await summarizer.summarize([make_event()], "My Home")

    assert text == "A person entered the front porch."
    assert raw_req["model"] == "claude-test"
    assert "system" in raw_req
    assert raw_resp["usage"]["input_tokens"] == 100


# ── OpenAI provider ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_openai_summarize_returns_text():
    mock_choice = MagicMock()
    mock_choice.message.content = "Motion detected near the driveway."
    mock_choice.finish_reason = "stop"

    mock_response = MagicMock()
    mock_response.id = "chatcmpl-1"
    mock_response.model = "gpt-4o"
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 80
    mock_response.usage.completion_tokens = 40

    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    summarizer = OpenAISummarizer.__new__(OpenAISummarizer)
    summarizer._client = mock_client
    summarizer._model = "gpt-4o"
    summarizer._max_events = 10

    text, raw_req, raw_resp = await summarizer.summarize([make_event()], "My Home")

    assert text == "Motion detected near the driveway."
    assert raw_req["model"] == "gpt-4o"
    # OpenAI uses messages array with system role (no top-level system param)
    assert raw_req["messages"][0]["role"] == "system"
    assert raw_resp["usage"]["input_tokens"] == 80


# ── make_summarizer factory ───────────────────────────────────────────────────

def test_make_summarizer_unknown_provider(monkeypatch):
    import sentinel.core.config as cfg_module

    monkeypatch.setattr(cfg_module.settings, "ai_provider", "gemini")
    monkeypatch.setattr(cfg_module.settings, "anthropic_api_key", "x")
    monkeypatch.setattr(cfg_module.settings, "openai_api_key", "x")

    from sentinel.ai.summarizer import make_summarizer

    with pytest.raises(RuntimeError, match="Unknown AI_PROVIDER"):
        make_summarizer()
