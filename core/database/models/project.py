"""
AFE — Project Model

A Project groups related experiments, datasets, models, and runs
under a single organizational umbrella.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import ForeignKey, String, Text, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database.base import Base, TimestampMixin, UUIDMixin

import enum


class ProjectStatus(str, enum.Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class Project(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "projects"

    name: Mapped[str] = mapped_column(String(256), nullable=False)
    slug: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(
        SAEnum(ProjectStatus, values_callable=lambda e: [x.value for x in e]),
        default=ProjectStatus.ACTIVE.value,
    )
    organization_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    created_by: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    # metadata
    tags: Mapped[Optional[str]] = mapped_column(Text, default="", comment="Comma-separated tags")

    # relationships
    organization: Mapped["Organization"] = relationship(back_populates="projects")  # type: ignore[name-defined]

    def __repr__(self) -> str:
        return f"<Project {self.slug}>"
