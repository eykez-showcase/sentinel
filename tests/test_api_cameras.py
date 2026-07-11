import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_camera(client: AsyncClient):
    response = await client.post(
        "/api/v1/cameras/",
        json={
            "name": "Front Door",
            "stream_url": "rtsp://192.168.1.1/stream",
            "is_active": True,
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Front Door"
    assert "id" in data


@pytest.mark.asyncio
async def test_list_cameras(client: AsyncClient):
    await client.post(
        "/api/v1/cameras/",
        json={"name": "Cam 1", "stream_url": "rtsp://a/1"},
    )
    response = await client.get("/api/v1/cameras/")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert data["total"] >= 1


@pytest.mark.asyncio
async def test_get_camera_not_found(client: AsyncClient):
    response = await client.get("/api/v1/cameras/nonexistent-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_update_camera(client: AsyncClient):
    create = await client.post(
        "/api/v1/cameras/",
        json={"name": "Old Name", "stream_url": "rtsp://b/1"},
    )
    camera_id = create.json()["id"]

    response = await client.patch(
        f"/api/v1/cameras/{camera_id}",
        json={"name": "New Name"},
    )
    assert response.status_code == 200
    assert response.json()["name"] == "New Name"


@pytest.mark.asyncio
async def test_delete_camera_soft(client: AsyncClient):
    create = await client.post(
        "/api/v1/cameras/",
        json={"name": "To Delete", "stream_url": "rtsp://c/1"},
    )
    camera_id = create.json()["id"]

    delete = await client.delete(f"/api/v1/cameras/{camera_id}")
    assert delete.status_code == 204

    # Camera still exists but is_active=False.
    get = await client.get(f"/api/v1/cameras/{camera_id}")
    assert get.status_code == 200
    assert get.json()["is_active"] is False
