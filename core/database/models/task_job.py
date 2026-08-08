"""
AFE — TaskJob Model

Represents any asynchronous background job: training runs, dataset
preprocessing, evaluation benchmarks, model conversions, etc.
The background worker queue picks up PENDING jobs and transitions
them through RUNNING → COMPLETED / FAILED.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from core.database.base import Base, TimestampMixin, UUIDMixin

import enum


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, enum.Enum):
    TRAINING = "training"
    FINETUNING = "finetuning"
    EVALUATION = "evaluation"
    DATASET_PROCESSING = "dataset_processing"
    TOKENIZER_TRAINING = "tokenizer_training"
    MODEL_CONVERSION = "model_conversion"
    INFERENCE = "inference"
    EXPORT = "export"
    CUSTOM = "custom"


class TaskJob(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "task_jobs"

    name: Mapped[str] = mapped_column(String(256), nullable=False)
    job_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(32), default=JobStatus.PENDING.value, nullable=False, index=True
    )
    progress: Mapped[float] = mapped_column(Float, default=0.0, comment="0.0 to 1.0")

    # ownership
    user_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    project_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )

    # configuration & results
    config_json: Mapped[Optional[str]] = mapped_column(
        Text, default="{}",
        comment="JSON-encoded task configuration / hyperparameters"
    )
    result_json: Mapped[Optional[str]] = mapped_column(
        Text, default="{}",
        comment="JSON-encoded results, metrics, or error details"
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # resource tracking
    gpu_ids: Mapped[Optional[str]] = mapped_column(String(256), nullable=True, comment="Comma-separated GPU indices")
    peak_vram_mb: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    peak_ram_mb: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # priority & retry
    priority: Mapped[int] = mapped_column(Integer, default=5, comment="1=highest, 10=lowest")
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)

    def __repr__(self) -> str:
        return f"<TaskJob {self.name!r} status={self.status}>"

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
