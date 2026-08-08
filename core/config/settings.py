"""
AFE — Centralized Application Settings

Uses Pydantic Settings for type-safe, environment-aware configuration.
All platform settings are loaded from environment variables or a `.env` file
in the project root.  Secrets like `SECRET_KEY` and `DATABASE_URL` default
to safe development values but MUST be overridden in production.
"""

from __future__ import annotations

import os
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ──────────────────────────────────────────────
# Path constants
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # e:\Softwares\IVX


class Settings(BaseSettings):
    """Platform-wide configuration."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        env_prefix="AFE_",
        case_sensitive=False,
        extra="ignore",
    )

    # ── General ──────────────────────────────────
    APP_NAME: str = "AI Foundry Enterprise"
    APP_VERSION: str = "0.1.0"
    ENV: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = True
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(64))

    # ── Server ───────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 5000

    # ── Database ─────────────────────────────────
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'data' / 'afe.db'}"

    # ── Storage ──────────────────────────────────
    STORAGE_BACKEND: Literal["local", "s3"] = "local"
    STORAGE_ROOT: str = str(BASE_DIR / "storage")

    # S3 / MinIO (only when STORAGE_BACKEND == "s3")
    S3_ENDPOINT: str = ""
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""
    S3_BUCKET: str = "afe"
    S3_REGION: str = "us-east-1"

    # ── Security ─────────────────────────────────
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRES_MINUTES: int = 60
    JWT_REFRESH_TOKEN_EXPIRES_DAYS: int = 30
    BCRYPT_ROUNDS: int = 12
    ALLOW_SELF_REGISTRATION: bool = False

    # ── Queue / Workers ──────────────────────────
    QUEUE_BACKEND: Literal["memory", "redis"] = "memory"
    REDIS_URL: str = "redis://localhost:6379/0"
    MAX_WORKERS: int = 4
    TRAINING_BACKEND: Literal["auto", "transformers", "simulated"] = "auto"

    # ── Resource Management ──────────────────────
    GPU_MEMORY_FRACTION: float = 0.9
    MAX_GPU_TEMP_C: int = 85

    # ── Logging ──────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    LOG_FILE: str = str(BASE_DIR / "logs" / "afe.log")

    # ── Plugin System ────────────────────────────
    PLUGINS_DIR: str = str(BASE_DIR / "plugins")
    PLUGINS_ENABLED: bool = True
    PLUGIN_SIGNING_KEY: str = ""

    # ── Inference ────────────────────────────────
    DEFAULT_INFERENCE_BACKEND: Literal["pytorch", "vllm", "llamacpp"] = "pytorch"
    MAX_MODEL_CONCURRENCY: int = 2

    @model_validator(mode="after")
    def validate_runtime_configuration(self) -> "Settings":
        """Reject unsafe production defaults before the app starts."""
        if self.is_production:
            if "SECRET_KEY" not in self.model_fields_set:
                raise ValueError("AFE_SECRET_KEY must be explicitly configured in production")
            if not self.SECRET_KEY or len(self.SECRET_KEY) < 32:
                raise ValueError("AFE_SECRET_KEY must be at least 32 characters in production")
            if self.DEBUG:
                raise ValueError("AFE_DEBUG must be false in production")
        if self.STORAGE_BACKEND == "s3" and not self.S3_BUCKET:
            raise ValueError("AFE_S3_BUCKET is required when AFE_STORAGE_BACKEND=s3")
        return self

    @property
    def is_production(self) -> bool:
        return self.ENV == "production"

    @property
    def base_dir(self) -> Path:
        return BASE_DIR


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
