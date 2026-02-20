from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment/.env."""

    environment: str = Field(default="dev")
    api_base_url: str = Field(default="https://clearinghouse.net/api/v2p1")
    api_key: str | None = Field(default=None, alias="api_token")
    user_agent: str = Field(default="CivilRightsSummarizedAI/0.1")
    api_timeout: float = Field(default=30.0)
    api_max_retries: int = Field(default=4)
    api_backoff_seconds: float = Field(default=0.5)
    api_max_backoff_seconds: float = Field(default=8.0)
    database_url: str = Field(default="sqlite:///data/dev.db")
    fixture_path: Path = Field(default=Path("data/fixtures/mock_dataset.json"))
    live_checkpoint_key: str = Field(default="live-default")
    live_resume_from_checkpoint: bool = Field(default=True)
    archive_raw_payloads: bool = Field(default=True)
    continue_on_error: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    default_summary_style: str = Field(default="bullet")

    model_config = SettingsConfigDict(
        env_prefix="CLEARINGHOUSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


settings = Settings()
