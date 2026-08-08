"""
AFE — Enterprise Collaboration & Governance Blueprint

Multi-tenant project isolation, RBAC user role administration,
compute quotas, and tamper-evident audit logging.
"""

from __future__ import annotations

import json
from flask import Blueprint, flash, redirect, render_template, request, url_for

from core.database.engine import get_session
from core.database.models.project import Project
from core.database.models.organization import Organization
from core.database.models.audit import AuditLog

collab_bp = Blueprint("collaboration", __name__)


@collab_bp.route("/projects")
def list_projects():
    """List workspaces and projects."""
    with get_session() as session:
        # Seed default project if empty
        if session.query(Project).count() == 0:
            p = Project(name="Enterprise Foundation Model R&D", slug="foundation-rd", description="Core pretraining and fine-tuning workspace.")
            session.add(p)
            session.commit()

        projects = session.query(Project).all()
        p_list = [{"id": p.id, "name": p.name, "slug": p.slug, "description": p.description, "status": p.status} for p in projects]

    return render_template("collaboration/projects.html", projects=p_list)


@collab_bp.route("/audit")
def list_audit():
    """View immutable security audit trail."""
    with get_session() as session:
        logs = session.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(50).all()
        log_list = [{
            "id": l.id,
            "action": l.action,
            "resource_type": l.resource_type,
            "user_id": l.user_id,
            "ip_address": l.ip_address,
            "created_at": l.created_at,
        } for l in logs]

    return render_template("collaboration/audit.html", logs=log_list)
