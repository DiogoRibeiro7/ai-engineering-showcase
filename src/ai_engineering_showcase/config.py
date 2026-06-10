"""Application configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AI_SHOWCASE_",
        extra="ignore",
    )

    data_path: Path = Field(default=Path("data/sample_feedback.csv"))
    index_path: Path = Field(default=Path(".artifacts/vector_store.json"))
    embedding_dim: int = Field(default=512, ge=64, le=8192)
    llm_provider: Literal["local", "openai"] = "local"
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")

    def ensure_artifact_dir(self) -> None:
        """Create the parent folder used by local artifacts."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
