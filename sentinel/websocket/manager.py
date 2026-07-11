"""
WebSocket connection manager.

Each connected browser client gets its own relay task that drains its
private event bus queue and forwards messages as JSON.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import WebSocket, WebSocketDisconnect

from sentinel.core.events import EventBus

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self, bus: EventBus, heartbeat_interval: int = 30) -> None:
        self._bus = bus
        self._heartbeat_interval = heartbeat_interval
        self._connections: dict[WebSocket, asyncio.Task] = {}

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        queue = await self._bus.subscribe()
        relay_task = asyncio.create_task(
            self._relay(ws, queue), name=f"ws_relay:{id(ws)}"
        )
        self._connections[ws] = relay_task
        logger.info(
            "WebSocket connected. Total connections: %d", len(self._connections)
        )

    async def disconnect(self, ws: WebSocket) -> None:
        task = self._connections.pop(ws, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        logger.info(
            "WebSocket disconnected. Total connections: %d", len(self._connections)
        )

    async def broadcast(self, message: dict) -> None:
        """Directly broadcast a message to all clients (used for system messages)."""
        dead: list[WebSocket] = []
        for ws in list(self._connections):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)

    async def _relay(self, ws: WebSocket, queue: asyncio.Queue) -> None:
        """Drain the bus queue for this client and forward messages."""
        try:
            while True:
                try:
                    message = await asyncio.wait_for(
                        queue.get(), timeout=self._heartbeat_interval
                    )
                    message["ts"] = datetime.now(timezone.utc).isoformat()
                    await ws.send_text(json.dumps(message, default=str))
                except asyncio.TimeoutError:
                    # Send a heartbeat ping so the browser knows we're alive.
                    await ws.send_json({"topic": "heartbeat", "ts": datetime.now(timezone.utc).isoformat()})
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except Exception:
            logger.exception("WebSocket relay error")
        finally:
            await self._bus.unsubscribe(queue)


async def websocket_endpoint(ws: WebSocket, manager: "ConnectionManager") -> None:
    await manager.connect(ws)
    try:
        # Keep the connection open; client can send pings or filter commands.
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(ws)
