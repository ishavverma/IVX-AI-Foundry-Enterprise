"""
AFE — Experiment & Checkpoint Models

Tracks distributed pretraining, fine-tuning runs (LoRA, QLoRA, DPO),
live metrics history, loss curves, and saved weight checkpoints.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import BigInteger, Float, ForeignKey, Integer, String, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from core.database.base import Base, TimestampMixin, UUIDMixin


class Experiment(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "experiments"

    name: Mapped[str] = mapped_column(String(256), nullable=False)
    slug: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, default="")
    
    # Run type & Model/Dataset linking
    run_type: Mapped[str] = mapped_column(String(32), default="pretraining", comment="pretraining, sft, lora, qlora, dpo, rlhf")
    model_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("model_definitions.id", ondelete="SET NULL"), nullable=True)
    dataset_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)
    
    # Status & Progress
    status: Mapped[str] = mapped_column(String(32), default="queued", comment="queued, running, completed, failed, stopped")
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    max_steps: Mapped[int] = mapped_column(Integer, default=1000)
    current_epoch: Mapped[int] = mapped_column(Integer, default=1)
    
    # Metrics
    current_loss: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    best_loss: Mapped[Optional[float]] = mapped_column(Float, default=999.0)
    total_tokens_trained: Mapped[int] = mapped_column(BigInteger, default=0)
    tokens_per_second: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    
    # Storage & Telemetry
    hyperparams_json: Mapped[Text] = mapped_column(Text, default="{}", comment="LR, batch size, warmup, precision, LoRA rank")
    metrics_history_json: Mapped[Text] = mapped_column(Text, default="[]", comment="JSON array of step/loss/tokens records")
    log_file_key: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    
    # Ownership
    project_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    def __repr__(self) -> str:
        return f"<Experiment {self.name} type={self.run_type} step={self.current_step}/{self.max_steps}>"


class Checkpoint(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "checkpoints"

    name: Mapped[str] = mapped_column(String(256), nullable=False)
    step: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    loss: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Format & Storage
    format: Mapped[str] = mapped_column(String(32), default="safetensors", comment="safetensors, gguf, onnx, bin")
    quantization: Mapped[str] = mapped_column(String(32), default="fp16", comment="fp16, bf16, fp8, int8, int4")
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    is_best: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Linking
    experiment_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=True, index=True)
    model_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("model_definitions.id", ondelete="SET NULL"), nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    def __repr__(self) -> str:
        return f"<Checkpoint {self.name} step={self.step} loss={self.loss:.4f} format={self.format}>"
