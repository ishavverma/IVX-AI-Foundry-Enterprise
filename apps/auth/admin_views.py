"""Enterprise administration views for user provisioning."""

from __future__ import annotations

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user

from core.database.engine import get_session
from core.database.models.user import Role, User
from core.security.auth import AuthService
from core.security.rbac import RBACService

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


def _require_admin() -> None:
    if not current_user.is_authenticated or not RBACService.user_has_permission(current_user, "admin:users"):
        abort(403)


@admin_bp.get("/")
def dashboard():
    """Land administrators on the first enterprise control-plane surface."""
    _require_admin()
    return redirect(url_for("admin.users"))


@admin_bp.get("/users")
def users():
    """List provisioned users for administrators."""
    _require_admin()
    search = request.args.get("q", "").strip()
    with get_session() as session:
        query = session.query(User).order_by(User.created_at.desc())
        if search:
            query = query.filter((User.username.ilike(f"%{search}%")) | (User.email.ilike(f"%{search}%")))
        user_rows = query.all()
        roles = session.query(Role).order_by(Role.name).all()
        rows = []
        for user in user_rows:
            role_names = [assignment.role.name for assignment in user.user_roles]
            rows.append({"user": user, "roles": role_names})
    return render_template("admin/users.html", users=rows, roles=roles, search=search)


@admin_bp.post("/users")
def create_user():
    """Provision a user; self-service registration is intentionally unavailable."""
    _require_admin()
    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    role_name = request.form.get("role", "member").strip()
    if len(username) < 3 or "@" not in email or len(password) < 12:
        flash("Provide a valid username, email, and temporary password of at least 12 characters.", "danger")
        return redirect(url_for("admin.users"))

    with get_session() as session:
        if session.query(User).filter((User.username == username) | (User.email == email)).first():
            flash("Username or email is already provisioned.", "danger")
            return redirect(url_for("admin.users"))

    user = AuthService.register_user(username, email, password, display_name=username)
    RBACService.assign_role(user.id, role_name if role_name in {"admin", "manager", "member", "viewer"} else "member")
    flash(f"User {username} provisioned successfully.", "success")
    return redirect(url_for("admin.users"))


@admin_bp.post("/users/<user_id>/deactivate")
def deactivate_user(user_id: str):
    """Disable an account without deleting its audit history."""
    _require_admin()
    if user_id == current_user.get_id():
        flash("You cannot deactivate your own account.", "danger")
        return redirect(url_for("admin.users"))
    with get_session() as session:
        user = session.get(User, user_id)
        if not user:
            abort(404)
        user.is_active = False
    flash("User access deactivated.", "success")
    return redirect(url_for("admin.users"))
