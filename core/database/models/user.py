"""
AFE — User, Role & UserRole Models

Implements a flexible RBAC system where:
  • Roles are named permission bundles (admin, manager, member, viewer).
  • Users can hold multiple roles scoped to an Organization.
  • The `UserRole` association table links them.
"""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import Boolean, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database.base import Base, TimestampMixin, UUIDMixin


class Role(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "roles"

    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, default="")
    permissions: Mapped[str] = mapped_column(
        Text, default="", nullable=False,
        comment="Comma-separated permission slugs, e.g. 'models:read,models:write,training:run'"
    )
    is_system: Mapped[bool] = mapped_column(Boolean, default=False, comment="Built-in role; cannot be deleted")

    # relationships
    user_roles: Mapped[List["UserRole"]] = relationship(back_populates="role", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Role {self.name}>"

    @property
    def permission_set(self) -> set[str]:
        return {p.strip() for p in self.permissions.split(",") if p.strip()}


class User(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(256), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    display_name: Mapped[Optional[str]] = mapped_column(String(256), default="")
    avatar_url: Mapped[Optional[str]] = mapped_column(String(512), default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superadmin: Mapped[bool] = mapped_column(Boolean, default=False)

    # relationships
    user_roles: Mapped[List["UserRole"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    audit_logs: Mapped[List["AuditLog"]] = relationship(back_populates="user")  # type: ignore[name-defined]
    notifications: Mapped[List["Notification"]] = relationship(back_populates="user")  # type: ignore[name-defined]

    def __repr__(self) -> str:
        return f"<User {self.username}>"

    # ── Flask-Login compatibility ──
    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def is_anonymous(self) -> bool:
        return False

    def get_id(self) -> str:
        return self.id


class UserRole(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "user_roles"
    __table_args__ = (
        UniqueConstraint("user_id", "role_id", "organization_id", name="uq_user_role_org"),
    )

    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role_id: Mapped[str] = mapped_column(String(36), ForeignKey("roles.id", ondelete="CASCADE"), nullable=False)
    organization_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True,
        comment="NULL = global role"
    )

    # relationships
    user: Mapped["User"] = relationship(back_populates="user_roles")
    role: Mapped["Role"] = relationship(back_populates="user_roles")

    def __repr__(self) -> str:
        return f"<UserRole user={self.user_id} role={self.role_id}>"
