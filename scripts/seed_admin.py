"""
AFE — Seed Default Admin User

Creates a superadmin account with the following credentials:
    Email:    admin@ivx.local
    Username: admin
    Password: Admin@12345

Run:
    python scripts/seed_admin.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.database.engine import init_db, get_session
from core.database.models.user import User
from core.security.auth import AuthService
from core.security.rbac import RBACService


ADMIN_EMAIL = "admin@ivx.local"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "Admin@12345"


def seed_admin() -> None:
    """Create the default superadmin user if it does not already exist."""
    init_db()
    RBACService.seed_default_roles()

    with get_session() as session:
        existing = (
            session.query(User)
            .filter((User.email == ADMIN_EMAIL) | (User.username == ADMIN_USERNAME))
            .first()
        )
        if existing:
            print(f"[seed_admin] Admin user already exists (id={existing.id}).")
            return

        user = User(
            username=ADMIN_USERNAME,
            email=ADMIN_EMAIL,
            password_hash=AuthService.hash_password(ADMIN_PASSWORD),
            display_name="Administrator",
            is_active=True,
            is_superadmin=True,
        )
        session.add(user)
        session.flush()
        print(f"[seed_admin] Created superadmin user (id={user.id}).")
        print(f"             Email:    {ADMIN_EMAIL}")
        print(f"             Username: {ADMIN_USERNAME}")
        print(f"             Password: {ADMIN_PASSWORD}")


if __name__ == "__main__":
    seed_admin()
