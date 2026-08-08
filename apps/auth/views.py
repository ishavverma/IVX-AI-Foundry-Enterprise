"""Authentication views for browser sessions."""

from __future__ import annotations

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from core.config.settings import get_settings
from flask_login import current_user, login_user, logout_user

from core.database.engine import get_session
from core.database.models.user import User
from core.database.models.organization import Organization
from core.security.auth import AuthService
from core.security.rbac import RBACService
from core.security.limiter import limiter


auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/login", methods=["GET", "POST"])
@limiter.limit("5 per minute")
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        identity = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = AuthService.authenticate(identity, password)
        if user:
            login_user(user)
            flash("Signed in successfully.", "success")
            return redirect(request.args.get("next") or url_for("dashboard"))
        flash("Invalid credentials.", "danger")
    return render_template("auth/login.html")


@auth_bp.route("/register", methods=["GET", "POST"])
@limiter.limit("3 per hour")
def register():
    if not get_settings().ALLOW_SELF_REGISTRATION:
        abort(404)
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if len(username) < 3 or "@" not in email or len(password) < 12:
            flash("Use a valid username, email, and password of at least 12 characters.", "danger")
            return render_template("auth/register.html"), 400

        with get_session() as session:
            if session.query(User).filter((User.username == username) | (User.email == email)).first():
                flash("Username or email is already registered.", "danger")
                return render_template("auth/register.html"), 409
            is_first_user = session.query(User).count() == 0

        user = AuthService.register_user(username, email, password, display_name=username, is_superadmin=is_first_user)
        organization = Organization(
            name=f"{username}'s Workspace",
            slug=f"{username.lower().replace('_', '-')}-{user.id[:8]}",
        )
        with get_session() as session:
            session.add(organization)
            session.flush()
            organization_id = organization.id
        RBACService.assign_role(user.id, "admin" if is_first_user else "member", organization_id)
        login_user(user)
        flash("Account created successfully.", "success")
        return redirect(url_for("dashboard"))
    return render_template("auth/register.html")


@auth_bp.post("/logout")
def logout():
    logout_user()
    flash("Signed out.", "info")
    return redirect(url_for("auth.login"))
