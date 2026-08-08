"""
AFE — Notification Model

In-app notification system.  Notifications are created by platform
events (training complete, evaluation finished, approval requested)
and delivered to users via the UI and optionally email/webhook.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import Boolean, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database.base import Base, TimestampMixin, UUIDMixin

import enum


class NotificationType(str, enum.Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class Notification(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "notifications"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    message: Mapped[Optional[str]] = mapped_column(Text, default="")
    notification_type: Mapped[str] = mapped_column(
        String(32), default=NotificationType.INFO.value
    )
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    link: Mapped[Optional[str]] = mapped_column(
        String(512), nullable=True,
        comment="Optional URL to navigate to when clicked"
    )

    # relationships
    user: Mapped["User"] = relationship(back_populates="notifications")  # type: ignore[name-defined]

    def __repr__(self) -> str:
        return f"<Notification {self.title!r} for={self.user_id}>"
