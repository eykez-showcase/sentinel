from datetime import datetime, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.models.camera import Camera
from sentinel.models.event import Event


@pytest.mark.asyncio
async def test_list_events_empty(client: AsyncClient):
    response = await client.get("/api/v1/events/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_list_events_filter_by_camera(
    client: AsyncClient,
    db_session: AsyncSession,
    sample_camera: Camera,
):
    event = Event(
        camera_id=sample_camera.id,
        event_type="zone_entry",
        zone_id="front_porch",
        class_name="person",
        severity="high",
        timestamp=datetime.now(timezone.utc),
    )
    db_session.add(event)
    await db_session.commit()

    response = await client.get(
        "/api/v1/events/", params={"camera_id": sample_camera.id}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert all(e["camera_id"] == sample_camera.id for e in data)


@pytest.mark.asyncio
async def test_get_event_not_found(client: AsyncClient):
    response = await client.get("/api/v1/events/nonexistent-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_event_stats(client: AsyncClient):
    response = await client.get("/api/v1/events/stats")
    assert response.status_code == 200
    assert "stats" in response.json()
