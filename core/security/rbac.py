"""
AFE — Role-Based Access Control Service

Checks whether a user holds a permission slug (e.g. "models:write")
either globally or within a specific organization.  Superadmins bypass
all checks.
"""

from __future__ import annotations

import logging
from typing import Optional

from core.database.engine import get_session
from core.database.models.user import User, UserRole, Role

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Default permission slugs
# ──────────────────────────────────────────────
PERMISSIONS = {
    # Organization
    "org:read", "org:write", "org:delete", "org:manage_members",
    # Projects
    "project:read", "project:write", "project:delete",
    # Datasets
    "dataset:read", "dataset:write", "dataset:delete",
    # Models
    "model:read", "model:write", "model:delete",
    # Training
    "training:read", "training:run", "training:cancel",
    # Evaluation
    "eval:read", "eval:run",
    # Inference / Deployment
    "inference:read", "inference:deploy", "inference:delete",
    # Admin
    "admin:users", "admin:roles", "admin:settings", "admin:audit",
}

# Default role templates
DEFAULT_ROLES = {
    "admin": PERMISSIONS,  # everything
    "manager": PERMISSIONS - {"admin:users", "admin:roles", "admin:settings", "admin:audit"},
    "member": {
        "org:read", "project:read", "project:write",
        "dataset:read", "dataset:write",
        "model:read", "model:write",
        "training:read", "training:run",
        "eval:read", "eval:run",
        "inference:read",
    },
    "viewer": {
        "org:read", "project:read", "dataset:read", "model:read",
        "training:read", "eval:read", "inference:read",
    },
}


class RBACService:
    """Permission checking utilities."""

    @staticmethod
    def user_has_permission(
        user: User,
        permission: str,
        organization_id: Optional[str] = None,
    ) -> bool:
        """Return True if *user* holds *permission*, optionally scoped to an org."""
        if user.is_superadmin:
            return True

        with get_session() as session:
            query = (
                session.query(UserRole)
                .join(Role, UserRole.role_id == Role.id)
                .filter(UserRole.user_id == user.id)
            )
            if organization_id:
                query = query.filter(
                    (UserRole.organization_id == organization_id)
                    | (UserRole.organization_id.is_(None))
                )

            for user_role in query.all():
                if permission in user_role.role.permission_set:
                    return True
        return False

    @staticmethod
    def assign_role(user_id: str, role_name: str, organization_id: Optional[str] = None) -> None:
        """Assign a named role to a user (creates the role if necessary)."""
        with get_session() as session:
            role = session.query(Role).filter(Role.name == role_name).first()
            if not role:
                perms = DEFAULT_ROLES.get(role_name, set())
                role = Role(
                    name=role_name,
                    description=f"Auto-created {role_name} role",
                    permissions=",".join(sorted(perms)),
                    is_system=role_name in DEFAULT_ROLES,
                )
                session.add(role)
                session.flush()

            existing = (
                session.query(UserRole)
                .filter(
                    UserRole.user_id == user_id,
                    UserRole.role_id == role.id,
                    UserRole.organization_id == organization_id,
                )
                .first()
            )
            if not existing:
                session.add(UserRole(user_id=user_id, role_id=role.id, organization_id=organization_id))
                logger.info("Assigned role %s to user %s (org=%s)", role_name, user_id, organization_id)

    @staticmethod
    def seed_default_roles() -> None:
        """Ensure all default roles exist in the database."""
        with get_session() as session:
            for name, perms in DEFAULT_ROLES.items():
                existing = session.query(Role).filter(Role.name == name).first()
                if not existing:
                    session.add(Role(
                        name=name,
                        description=f"System {name} role",
                        permissions=",".join(sorted(perms)),
                        is_system=True,
                    ))
            logger.info("Default RBAC roles seeded.")
