import asyncio
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from sentinel.core.database import Base, get_db
from sentinel.models import Camera
from sentinel.zones.loader import PropertyConfig, CameraConfig, ZoneConfig


TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Use a single event loop for all tests in the session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    import sentinel.models  # noqa: F401 — ensure models are registered
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with factory() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def client(test_engine) -> AsyncGenerator[AsyncClient, None]:
    from sentinel.api.main import app

    factory = async_sessionmaker(test_engine, expire_on_commit=False)

    async def override_get_db():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    # Skip lifespan (CV pipelines) during tests.
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def no_lifespan(app):
        yield

    app.router.lifespan_context = no_lifespan

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def sample_camera(db_session: AsyncSession) -> Camera:
    camera = Camera(
        name="Test Camera",
        stream_url="rtsp://localhost/test",
        config_id="test_cam",
        is_active=True,
    )
    db_session.add(camera)
    await db_session.commit()
    await db_session.refresh(camera)
    return camera


@pytest.fixture
def property_config() -> PropertyConfig:
    return PropertyConfig(
        name="Test Property",
        address="1 Test St",
        cameras=[
            CameraConfig(
                id="test_cam",
                name="Test Camera",
                stream_url="rtsp://localhost/test",
                position={},
                fov_degrees=90,
            )
        ],
        zones=[
            ZoneConfig(
                id="front_porch",
                name="Front Porch",
                camera_id="test_cam",
                polygon=[(0.3, 0.5), (0.7, 0.5), (0.7, 1.0), (0.3, 1.0)],
                alert_classes=["person"],
            ),
            ZoneConfig(
                id="driveway",
                name="Driveway",
                camera_id="test_cam",
                polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)],
                alert_classes=["person", "car"],
            ),
        ],
    )
