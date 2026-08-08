"""
AFE — Monitoring & Observability Blueprint

Hardware telemetry streaming, GPU memory bandwidth utilization,
queue worker states, and operational logs.
"""

from __future__ import annotations

import json
import psutil
from flask import Blueprint, render_template, jsonify

from core.database.engine import get_session
from core.database.models.task_job import TaskJob
from core.resources.hardware import hardware_monitor
from core.resources.allocator import resource_allocator

monitoring_bp = Blueprint("monitoring", __name__, url_prefix="/monitoring")


@monitoring_bp.route("/")
def index():
    """System health, GPU cluster telemetry, and queue metrics."""
    snap = hardware_monitor.snapshot()
    alloc = resource_allocator.summary()

    with get_session() as session:
        recent_jobs = session.query(TaskJob).order_by(TaskJob.created_at.desc()).limit(15).all()
        job_list = [{
            "id": j.id,
            "name": j.name,
            "type": j.job_type,
            "status": j.status,
            "progress": j.progress,
            "created_at": j.created_at,
        } for j in recent_jobs]

    return render_template(
        "monitoring/index.html",
        snapshot=snap.to_dict(),
        allocation=alloc,
        jobs=job_list,
    )


@monitoring_bp.route("/api/live")
def live_metrics():
    """Real-time JSON polling endpoint for live charts."""
    snap = hardware_monitor.snapshot()
    return jsonify({
        "cpu_percent": snap.cpu.usage_percent,
        "ram_percent": snap.memory.usage_percent,
        "disk_percent": snap.disk.usage_percent,
        "gpus": [g.to_dict() if hasattr(g, 'to_dict') else g.__dict__ for g in snap.gpus],
    })
