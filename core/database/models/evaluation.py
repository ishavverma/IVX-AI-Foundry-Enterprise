"""
AFE — Evaluation, Registry & Inference Endpoint Models

Defines the database schema for automated benchmark evaluation suites,
the centralized asset registry, and active inference serving endpoints.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import BigInteger, Float, ForeignKey, Integer, String, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from core.database.base import Base, TimestampMixin, UUIDMixin


class EvaluationRun(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "evaluation_runs"

    benchmark_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True, comment="mmlu, humaneval, gsm8k, math, bigbench, latency")
    score: Mapped[float] = mapped_column(Float, default=0.0, comment="Overall score percentage 0.0 to 100.0")
    pass_at_1: Mapped[Optional[float]] = mapped_column(Float, nullable=True, comment="Pass@1 for coding benchmarks")
    accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_latency_ms: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    tokens_per_sec: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    
    # Status & Linking
    status: Mapped[str] = mapped_column(String(32), default="completed", comment="running, completed, failed")
    detailed_results_json: Mapped[Text] = mapped_column(Text, default="{}", comment="Per-category sub-scores and question traces")
    
    model_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("model_definitions.id", ondelete="SET NULL"), nullable=True)
    checkpoint_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("checkpoints.id", ondelete="SET NULL"), nullable=True)
    project_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    def __repr__(self) -> str:
        return f"<EvaluationRun {self.benchmark_name} score={self.score:.1f}%>"


class RegistryAsset(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "registry_assets"

    name: Mapped[str] = mapped_column(String(256), nullable=False)
    slug: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    asset_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True, comment="model, dataset, tokenizer, checkpoint")
    version: Mapped[str] = mapped_column(String(32), default="v1.0.0")
    license: Mapped[str] = mapped_column(String(64), default="Apache-2.0")
    tags: Mapped[Optional[str]] = mapped_column(String(512), default="hybrid, mamba, gqa")
    description: Mapped[Optional[str]] = mapped_column(Text, default="")
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    metadata_json: Mapped[Text] = mapped_column(Text, default="{}")
    is_public: Mapped[bool] = mapped_column(Boolean, default=True)
    created_by: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    def __repr__(self) -> str:
        return f"<RegistryAsset {self.name} ({self.asset_type}) {self.version}>"


class InferenceEndpoint(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "inference_endpoints"

    name: Mapped[str] = mapped_column(String(256), nullable=False)
    slug: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    backend: Mapped[str] = mapped_column(String(32), default="pytorch", comment="pytorch, vllm, llamacpp, tensorrt")
    port: Mapped[int] = mapped_column(Integer, default=8000)
    is_running: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Serving parameters
    max_concurrency: Mapped[int] = mapped_column(Integer, default=4)
    temperature: Mapped[float] = mapped_column(Float, default=0.7)
    top_p: Mapped[float] = mapped_column(Float, default=0.9)
    gpu_indices: Mapped[Optional[str]] = mapped_column(String(64), default="0")
    
    # Telemetry
    total_requests: Mapped[int] = mapped_column(BigInteger, default=0)
    avg_latency_ms: Mapped[float] = mapped_column(Float, default=42.5)
    
    model_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("model_definitions.id", ondelete="SET NULL"), nullable=True)
    checkpoint_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("checkpoints.id", ondelete="SET NULL"), nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    def __repr__(self) -> str:
        return f"<InferenceEndpoint {self.name} port={self.port} running={self.is_running}>"
