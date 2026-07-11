"""
EventSummarizer: produces natural-language summaries of detected event sequences.

Supports two providers, selected via AI_PROVIDER in .env:
  - "anthropic"  (default) — uses claude-opus-4-5 or any claude-* model
  - "openai"               — uses gpt-4o or any OpenAI chat model

Both implementations share the same prompt logic and return the same tuple:
  (summary_text, raw_request_dict, raw_response_dict)

Raw request/response are stored in AISummary for auditability.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from sentinel.schemas.event import EventRead

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a home security analyst reviewing activity detected by an AI camera system.
Your job is to produce brief, factual summaries of what happened based on a list of
detected events. Be concise (2–3 sentences). Do not speculate beyond what the data
shows. Flag anything that warrants the homeowner's attention with a clear statement.
"""


def _build_prompt(events: list[EventRead], property_name: str) -> str:
    lines = [
        f"Property: {property_name}",
        f"Event window: {len(events)} events",
        "",
        "Detected events (chronological):",
    ]
    for evt in events:
        ts = evt.timestamp.strftime("%H:%M:%S")
        zone = evt.zone_id or "unknown location"
        obj = evt.class_name or "unknown object"
        lines.append(
            f"  [{ts}] {evt.event_type.replace('_', ' ').title()} — "
            f"{obj} in {zone} (severity: {evt.severity})"
        )
    lines += [
        "",
        "Provide a 2–3 sentence plain-English summary of what happened "
        "and whether any activity warrants attention.",
    ]
    return "\n".join(lines)


class BaseEventSummarizer(ABC):
    def __init__(self, max_events: int) -> None:
        self._max_events = max_events

    async def summarize(
        self,
        events: list[EventRead],
        property_name: str,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        window = sorted(events, key=lambda e: e.timestamp)[-self._max_events :]
        user_text = _build_prompt(window, property_name)
        return await self._call(user_text, len(window))

    @abstractmethod
    async def _call(
        self, user_text: str, event_count: int
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Provider-specific API call. Returns (summary_text, raw_req, raw_resp)."""


class AnthropicSummarizer(BaseEventSummarizer):
    def __init__(self, api_key: str, model: str, max_events: int) -> None:
        super().__init__(max_events)
        import anthropic
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    async def _call(
        self, user_text: str, event_count: int
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        logger.debug("Calling Anthropic for event summary (%d events)", event_count)

        raw_request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 512,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_text}],
        }
        response = await self._client.messages.create(**raw_request)

        summary_text = response.content[0].text.strip()
        raw_response: dict[str, Any] = {
            "id": response.id,
            "model": response.model,
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            "content": summary_text,
        }
        return summary_text, raw_request, raw_response


class OpenAISummarizer(BaseEventSummarizer):
    def __init__(self, api_key: str, model: str, max_events: int) -> None:
        super().__init__(max_events)
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def _call(
        self, user_text: str, event_count: int
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        logger.debug("Calling OpenAI for event summary (%d events)", event_count)

        raw_request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 512,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
        }
        response = await self._client.chat.completions.create(**raw_request)

        summary_text = response.choices[0].message.content.strip()
        raw_response: dict[str, Any] = {
            "id": response.id,
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason,
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
            "content": summary_text,
        }
        return summary_text, raw_request, raw_response


def make_summarizer() -> BaseEventSummarizer:
    """Factory — reads AI_PROVIDER from settings and returns the right implementation."""
    from sentinel.core.config import settings

    provider = settings.ai_provider.lower()

    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in .env")
        return OpenAISummarizer(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            max_events=settings.ai_summary_max_events,
        )

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set in .env")
        return AnthropicSummarizer(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            max_events=settings.ai_summary_max_events,
        )

    raise RuntimeError(
        f"Unknown AI_PROVIDER '{provider}'. Must be 'anthropic' or 'openai'."
    )
