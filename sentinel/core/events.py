"""
In-process async event bus.

The CV pipeline publishes detection and event payloads here.
The WebSocket manager subscribes and relays them to connected browsers.

Topics use dot notation:
  detection.new
  event.created
  track.updated
  ai_summary.created

This works when CV and API share the same process (Phase 1).
Replace with Redis Pub/Sub if the CV pipeline moves to a separate worker.
"""

import asyncio
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue] = []
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=512)
        async with self._lock:
            self._subscribers.append(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        async with self._lock:
            try:
                self._subscribers.remove(queue)
            except ValueError:
                pass

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        message = {"topic": topic, "payload": payload}
        async with self._lock:
            dead: list[asyncio.Queue] = []
            for q in self._subscribers:
                try:
                    q.put_nowait(message)
                except asyncio.QueueFull:
                    # Slow consumer — drop the message rather than block.
                    dead.append(q)
            for q in dead:
                self._subscribers.remove(q)


bus = EventBus()
