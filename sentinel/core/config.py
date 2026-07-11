from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"

    # Database
    database_url: str = "sqlite+aiosqlite:///./sentinel.db"

    # Camera
    default_frame_rate: int = 10

    # YOLO
    yolo_model_path: str = "yolov8n.pt"
    yolo_confidence_threshold: float = 0.45
    yolo_device: str = "cpu"

    # Property / zone config
    property_config_path: str = "config/property.yaml"

    # Event thresholds
    loitering_threshold_seconds: int = 30

    # AI reasoning
    ai_provider: str = "anthropic"  # "anthropic" | "openai"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-4-5"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    ai_summary_max_events: int = 20

    # WebSocket
    ws_heartbeat_interval: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
