from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from sentinel.core.config import settings


class Base(DeclarativeBase):
    pass


def _make_engine():
    connect_args = {}
    # Enable WAL mode for SQLite to support concurrent reads during writes.
    if settings.database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    engine = create_async_engine(
        settings.database_url,
        echo=settings.app_env == "development",
        connect_args=connect_args,
    )

    if settings.database_url.startswith("sqlite"):
        from sqlalchemy import event as sa_event

        @sa_event.listens_for(engine.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

    return engine


engine = _make_engine()

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """Create all tables. Used in development; production uses Alembic."""
    # Import models so they register with Base.metadata before create_all.
    import sentinel.models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """FastAPI dependency that yields an async database session."""
    async with AsyncSessionLocal() as session:
        yield session
