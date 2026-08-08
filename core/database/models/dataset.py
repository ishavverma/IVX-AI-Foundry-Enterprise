"""
AFE — Dataset Model

Represents a managed dataset in the platform with metadata, format details,
quality metrics, token counts, and pipeline lineage.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import BigInteger, Float, ForeignKey, Integer, String, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database.base import Base, TimestampMixin, UUIDMixin


class Dataset(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "datasets"

    name: Mapped[str] = mapped_column(String(256), nullable=False)
    slug: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, default="")
    
    # Format & Storage
    format: Mapped[str] = mapped_column(String(32), default="jsonl", comment="jsonl, csv, parquet, txt, huggingface")
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False, comment="Relative path in Storage SDK")
    processed_key: Mapped[Optional[str]] = mapped_column(String(512), nullable=True, comment="Path to cleaned/tokenized output")
    
    # Metrics
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    row_count: Mapped[int] = mapped_column(BigInteger, default=0)
    token_count: Mapped[int] = mapped_column(BigInteger, default=0)
    quality_score: Mapped[Optional[float]] = mapped_column(Float, default=1.0, comment="Quality metric score 0.0 to 1.0")
    
    # Status & Pipeline
    status: Mapped[str] = mapped_column(String(32), default="raw", comment="raw, processing, ready, failed")
    is_tokenized: Mapped[bool] = mapped_column(Boolean, default=False)
    lineage_json: Mapped[Optional[str]] = mapped_column(Text, default="{}", comment="JSON record of pipeline steps and source transforms")
    sample_preview_json: Mapped[Optional[str]] = mapped_column(Text, default="[]", comment="JSON array of head samples for instant UI preview")
    
    # Ownership
    project_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_by: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    def __repr__(self) -> str:
        return f"<Dataset {self.name} rows={self.row_count} status={self.status}>"
