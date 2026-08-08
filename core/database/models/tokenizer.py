"""
AFE — Tokenizer Model

Represents a trained or imported tokenizer definition in the platform,
storing vocabulary parameters, algorithm type, special tokens, and compression stats.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import BigInteger, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from core.database.base import Base, TimestampMixin, UUIDMixin


class TokenizerModel(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "tokenizers"

    name: Mapped[str] = mapped_column(String(256), nullable=False)
    slug: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, default="")
    
    # Algorithm & Config
    algorithm: Mapped[str] = mapped_column(String(64), default="bpe", comment="bpe, byte_level, unigram, wordpiece")
    vocab_size: Mapped[int] = mapped_column(Integer, default=32000)
    special_tokens_json: Mapped[str] = mapped_column(Text, default="{}", comment="JSON mapping of special tokens (bos, eos, unk, pad, mask, etc.)")
    
    # Storage & Exports
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False, comment="Path to tokenizer model files in Storage SDK")
    
    # Analytics
    compression_ratio: Mapped[Optional[float]] = mapped_column(Float, default=0.0, comment="Avg bytes per token")
    cjk_coverage_pct: Mapped[Optional[float]] = mapped_column(Float, default=0.0, comment="Percentage of CJK Unicode coverage")
    status: Mapped[str] = mapped_column(String(32), default="ready", comment="training, ready, failed")
    
    # Ownership
    project_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_by: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    def __repr__(self) -> str:
        return f"<TokenizerModel {self.name} vocab={self.vocab_size} algo={self.algorithm}>"
