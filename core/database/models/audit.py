"""
AFE — Audit Log Model

Immutable append-only table recording every significant action
across the platform: who did what, when, and what changed.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database.base import Base, TimestampMixin, UUIDMixin


class AuditLog(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "audit_logs"

    user_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    action: Mapped[str] = mapped_column(
        String(128), nullable=False, index=True,
        comment="e.g. user.login, project.create, model.train.start"
    )
    resource_type: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True,
        comment="e.g. user, project, model, dataset"
    )
    resource_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    detail: Mapped[Optional[str]] = mapped_column(
        Text, default="",
        comment="JSON-encoded diff or free-text description"
    )
    ip_address: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    # relationships
    user: Mapped[Optional["User"]] = relationship(back_populates="audit_logs")  # type: ignore[name-defined]

    def __repr__(self) -> str:
        return f"<AuditLog {self.action} by={self.user_id}>"
