"""
AFE — Model Definition Model

Stores compiled neural architecture definitions, hyperparameter configurations,
analytical parameter counts, FLOP estimates, and generated PyTorch code.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import BigInteger, Float, ForeignKey, Integer, String, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from core.database.base import Base, TimestampMixin, UUIDMixin


class ModelDefinition(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "model_definitions"

    name: Mapped[str] = mapped_column(String(256), nullable=False)
    slug: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, default="")

    # Architecture Type & Hierarchy
    architecture_type: Mapped[str] = mapped_column(
        String(64), default="hybrid", comment="hybrid, transformer, ssm, moe"
    )
    preset_tier: Mapped[str] = mapped_column(
        String(32), default="base", comment="nano, micro, small, base, large, xl, xxl, moe_100b, moe_1t"
    )

    # Core Hyperparameters
    d_model: Mapped[int] = mapped_column(Integer, default=4096)
    num_layers: Mapped[int] = mapped_column(Integer, default=32)
    num_heads: Mapped[int] = mapped_column(Integer, default=32)
    num_kv_heads: Mapped[int] = mapped_column(Integer, default=8, comment="GQA key-value heads")
    vocab_size: Mapped[int] = mapped_column(Integer, default=64000)
    max_seq_len: Mapped[int] = mapped_column(Integer, default=32768)
    
    # Hybrid / SSM / MoE ratios
    ssm_ratio: Mapped[str] = mapped_column(String(32), default="1:7", comment="Ratio of attention to SSM layers")
    use_moe: Mapped[bool] = mapped_column(Boolean, default=False)
    num_experts: Mapped[int] = mapped_column(Integer, default=16)
    top_k_experts: Mapped[int] = mapped_column(Integer, default=2)

    # Analytical Estimates
    param_count: Mapped[int] = mapped_column(BigInteger, default=0)
    active_param_count: Mapped[int] = mapped_column(BigInteger, default=0)
    gflops_per_token: Mapped[float] = mapped_column(Float, default=0.0)
    vram_fp16_mb: Mapped[int] = mapped_column(Integer, default=0)
    vram_fp8_mb: Mapped[int] = mapped_column(Integer, default=0)
    vram_int4_mb: Mapped[int] = mapped_column(Integer, default=0)

    # Storage & Exports
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False, comment="Path to config.json in Storage SDK")
    code_storage_key: Mapped[Optional[str]] = mapped_column(String(512), nullable=True, comment="Path to model.py")
    config_json: Mapped[Text] = mapped_column(Text, default="{}", comment="Full hyperparameter JSON manifest")
    
    # Status
    status: Mapped[str] = mapped_column(String(32), default="ready", comment="draft, ready, compiled")

    # Ownership
    project_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_by: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    def __repr__(self) -> str:
        return f"<ModelDefinition {self.name} params={self.param_count} arch={self.architecture_type}>"
