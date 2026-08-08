"""Tenant context, quotas, and usage accounting."""

from __future__ import annotations

import json
from typing import Any

from flask_login import current_user
from sqlalchemy import or_
from sqlalchemy.orm import Query, Session

from core.database.engine import get_session
from core.database.models.organization import Organization
from core.database.models.project import Project
from core.database.models.usage import UsageRecord
from core.database.models.user import UserRole


class TenantService:
    @staticmethod
    def scope(query: Query, model: Any, session: Session, user_id: str | None = None) -> Query:
        """Scope an artifact query by creator or its project's organization."""
        identifier = user_id or (current_user.get_id() if current_user.is_authenticated else None)
        if not identifier or getattr(current_user, "is_superadmin", False):
            return query
        organizations = TenantService.organization_ids(identifier)
        clauses = []
        if hasattr(model, "created_by"):
            clauses.append(model.created_by == identifier)
        if hasattr(model, "project_id"):
            query = query.outerjoin(Project, model.project_id == Project.id)
            clauses.append(Project.organization_id.in_(organizations or ["__none__"]))
        return query.filter(or_(*clauses)) if clauses else query.filter(False)

    @staticmethod
    def organization_for_user(user_id: str) -> str | None:
        return next(iter(TenantService.organization_ids(user_id)), None)

    @staticmethod
    def organization_ids(user_id: str | None = None) -> list[str]:
        identifier = user_id or (current_user.get_id() if current_user.is_authenticated else None)
        if not identifier:
            return []
        with get_session() as session:
            return [row.organization_id for row in session.query(UserRole).filter(
                UserRole.user_id == identifier, UserRole.organization_id.is_not(None)
            ).all()]

    @staticmethod
    def require_quota(organization_id: str, metric: str, requested: float) -> None:
        with get_session() as session:
            organization = session.get(Organization, organization_id)
            if not organization:
                raise ValueError("Organization not found")
            limits = {
                "gpu_hours": organization.max_gpu_hours,
                "storage_gb": organization.max_storage_gb,
                "projects": organization.max_projects,
            }
            if metric not in limits:
                return
            used = sum(row.amount for row in session.query(UsageRecord).filter(
                UsageRecord.organization_id == organization_id, UsageRecord.metric == metric
            ).all())
            if used + requested > limits[metric]:
                raise PermissionError(f"{metric} quota exceeded ({used:g}/{limits[metric]:g})")

    @staticmethod
    def record(organization_id: str, metric: str, amount: float, unit: str, resource_id: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        with get_session() as session:
            session.add(UsageRecord(
                organization_id=organization_id,
                metric=metric,
                amount=amount,
                unit=unit,
                resource_id=resource_id,
                metadata_json=json.dumps(metadata or {}, default=str),
            ))
