"""
AFE — Organization Model

An Organization is the top-level tenant.  All projects, datasets, models,
and experiments belong to exactly one Organization.
"""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import Boolean, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database.base import Base, TimestampMixin, UUIDMixin


class Organization(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "organizations"

    name: Mapped[str] = mapped_column(String(256), unique=True, nullable=False)
    slug: Mapped[str] = mapped_column(String(256), unique=True, nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, default="")
    logo_url: Mapped[Optional[str]] = mapped_column(String(512), default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # quotas
    max_projects: Mapped[int] = mapped_column(default=50)
    max_storage_gb: Mapped[int] = mapped_column(default=100)
    max_gpu_hours: Mapped[int] = mapped_column(default=1000)

    # relationships
    projects: Mapped[List["Project"]] = relationship(back_populates="organization")  # type: ignore[name-defined]

    def __repr__(self) -> str:
        return f"<Organization {self.slug}>"
