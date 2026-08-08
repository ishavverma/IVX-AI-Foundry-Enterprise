"""
AFE — Application Entry Point

Flask application factory with blueprint registration, database
initialization, RBAC seeding, and plugin loading.

Run:
    python app.py
    # or
    flask --app app run --debug
"""

from __future__ import annotations

import logging
import json
import sys
from pathlib import Path

import psutil
from flask import Flask, Response, abort, jsonify, render_template, request, stream_with_context
from flask_login import LoginManager, current_user, login_user
from flask_wtf.csrf import CSRFProtect
from sqlalchemy import text

from core.config.settings import get_settings, BASE_DIR
from core.database.engine import init_db, get_session
from core.database.models.task_job import TaskJob, JobStatus
from core.database.models.project import Project
from core.security.rbac import RBACService
from core.plugins.loader import plugin_loader
from core.resources.hardware import hardware_monitor
from core.resources.allocator import resource_allocator
from core.events.bus import event_bus
from core.storage.paths import storage_paths
from core.database.models.dataset import Dataset
from core.database.models.tokenizer import TokenizerModel
from core.database.models.model_def import ModelDefinition
from core.database.models.audit import AuditLog
from core.database.models.user import User
from core.security.auth import AuthService
from core.tenancy import TenantService
from core.storage import get_storage_backend
from core.observability import JsonFormatter, metrics
from core.security.limiter import limiter
from apps.datasets.views import datasets_bp
from apps.tokenizer.views import tokenizer_bp
from apps.model_factory.views import models_bp
from apps.training.views import training_bp
from apps.finetuning.views import finetuning_bp
from apps.checkpoints.views import checkpoints_bp
from apps.registry.views import registry_bp
from apps.evaluation.views import evaluation_bp
from apps.deployment.views import deployment_bp
from apps.chat.views import chat_bp
from apps.prompts.views import prompts_bp
from apps.agents.views import agents_bp
from apps.mcp.views import mcp_bp
from apps.monitoring.views import monitoring_bp
from apps.marketplace.views import marketplace_bp
from apps.collaboration.views import collab_bp
from apps.auth.views import auth_bp
from apps.auth.admin_views import admin_bp
from apps.api.views import api_bp

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
settings = get_settings()

log_dir = Path(settings.LOG_FILE).parent
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format=settings.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("afe")
for handler in logging.getLogger().handlers:
    handler.setFormatter(JsonFormatter())
login_manager = LoginManager()
login_manager.login_view = "auth.login"
csrf = CSRFProtect()


WRITE_PERMISSIONS = {
    "/datasets": "dataset:write",
    "/tokenizer": "model:write",
    "/models": "model:write",
    "/training": "training:run",
    "/finetuning": "training:run",
    "/evaluation": "eval:run",
    "/inference": "inference:deploy",
    "/api/v1/training": "training:run",
}


# ──────────────────────────────────────────────
# Application Factory
# ──────────────────────────────────────────────
def create_app() -> Flask:
    """Build and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "ui" / "templates"),
        static_folder=str(BASE_DIR / "ui" / "static"),
    )
    app.secret_key = settings.SECRET_KEY
    app.config["DEBUG"] = settings.DEBUG
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=settings.is_production,
        WTF_CSRF_ENABLED=True,
    )
    login_manager.init_app(app)
    csrf.init_app(app)
    limiter.init_app(app)

    @login_manager.user_loader
    def load_user(user_id: str) -> User | None:
        return AuthService.get_user_by_id(user_id)

    @app.context_processor
    def inject_navigation_permissions():
        can_manage_users = current_user.is_authenticated and RBACService.user_has_permission(current_user, "admin:users")
        return {"can_manage_users": can_manage_users}

    # ── Initialize database ──
    with app.app_context():
        init_db()
        RBACService.seed_default_roles()
        logger.info("Database initialized and RBAC roles seeded.")

    # ── Initialise storage directories ──
    storage_paths.init_all()

    # ── Load plugins ──
    if settings.PLUGINS_ENABLED:
        plugin_loader.load_all(app)

    # ── Register blueprints ──
    app.register_blueprint(datasets_bp)
    app.register_blueprint(tokenizer_bp)
    app.register_blueprint(models_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(finetuning_bp)
    app.register_blueprint(checkpoints_bp)
    app.register_blueprint(registry_bp)
    app.register_blueprint(evaluation_bp)
    app.register_blueprint(deployment_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(prompts_bp)
    app.register_blueprint(agents_bp)
    app.register_blueprint(mcp_bp)
    app.register_blueprint(monitoring_bp)
    app.register_blueprint(marketplace_bp)
    app.register_blueprint(collab_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(api_bp)

    # ── Register routes ──
    _register_routes(app)
    _register_websocket(app)

    @app.before_request
    def enforce_access_control():
        """Require a session for application routes and RBAC for writes."""
        metrics.inc("afe_http_requests_total", {"method": request.method, "path": request.path})
        if request.endpoint in {
            "static",
            "health",
            "metrics_endpoint",
            "openapi_spec",
            "auth.login",
            "auth.register",
        }:
            return None
        if request.path.startswith("/api/") and not current_user.is_authenticated:
            authorization = request.headers.get("Authorization", "")
            if authorization.lower().startswith("bearer "):
                token_data = AuthService.decode_token(authorization[7:].strip())
                if token_data and token_data.get("type") == "access":
                    user = AuthService.get_user_by_id(str(token_data.get("sub", "")))
                    if user and user.is_active:
                        login_user(user)
            if not current_user.is_authenticated:
                return jsonify({"error": "authentication_required"}), 401
        if not current_user.is_authenticated:
            return login_manager.unauthorized()

        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            permission = next(
                (value for prefix, value in WRITE_PERMISSIONS.items() if request.path.startswith(prefix)),
                None,
            )
            if permission and not RBACService.user_has_permission(current_user, permission):
                abort(403)
        return None

    @app.after_request
    def record_mutation(response):
        """Record successful state-changing requests for operational auditability."""
        if (
            current_user.is_authenticated
            and request.method in {"POST", "PUT", "PATCH", "DELETE"}
            and response.status_code < 400
            and request.endpoint not in {"auth.logout"}
        ):
            try:
                with get_session() as session:
                    session.add(AuditLog(
                        user_id=current_user.get_id(),
                        action=f"http.{request.method.lower()}",
                        resource_type=request.endpoint or request.path,
                        ip_address=request.remote_addr,
                        user_agent=request.user_agent.string[:512],
                    ))
            except Exception:
                logger.exception("Unable to write audit event for %s", request.path)
        return response

    @app.route("/metrics")
    def metrics_endpoint():
        """Expose Prometheus-compatible application counters."""
        return app.response_class(metrics.render(), mimetype="text/plain; version=0.0.4")

    @app.route("/api/openapi.json")
    def openapi_spec():
        """Expose a minimal OpenAPI contract for the stable platform endpoints."""
        return jsonify({
            "openapi": "3.0.3",
            "info": {"title": settings.APP_NAME, "version": settings.APP_VERSION},
            "paths": {
                "/health": {"get": {"responses": {"200": {"description": "Health status"}}}},
                "/metrics": {"get": {"responses": {"200": {"description": "Prometheus metrics"}}}},
                "/v1/chat/completions": {"post": {"responses": {"200": {"description": "Chat completion"}}}},
            },
        })

    @app.route("/api/jobs/<job_id>/events")
    def job_events(job_id: str):
        """Stream recent job events as Server-Sent Events for dashboard clients."""
        def stream():
            for event in event_bus.history_for(job_id):
                yield f"data: {json.dumps(event, default=str)}\n\n"
            yield "event: ready\ndata: {}\n\n"

        import json
        return Response(stream_with_context(stream()), mimetype="text/event-stream")

    logger.info(
        "%s v%s started [env=%s, db=%s]",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.ENV,
        settings.DATABASE_URL[:60],
    )
    return app


def _register_websocket(app: Flask) -> None:
    """Attach WebSocket transport when the optional realtime extra is installed."""
    try:
        from flask_sock import Sock
    except ImportError:
        return

    sock = Sock(app)

    @sock.route("/ws/jobs/<job_id>")
    def job_websocket(connection, job_id: str):
        for event in event_bus.history_for(job_id):
            connection.send(json.dumps(event, default=str))
        connection.send(json.dumps({"event": "ready", "job_id": job_id}))


# ──────────────────────────────────────────────
# Routes (will be extracted to blueprints later)
# ──────────────────────────────────────────────
def _register_routes(app: Flask) -> None:
    """Register core application routes."""

    @app.route("/")
    def dashboard():
        """Main dashboard page."""
        with get_session() as session:
            organization_ids = TenantService.organization_ids()
            project_query = session.query(Project)
            if not current_user.is_superadmin:
                project_query = project_query.filter(Project.organization_id.in_(organization_ids or ["__none__"]))
            project_count = project_query.count()
            project_ids = [project.id for project in project_query.all()]
            model_query = session.query(ModelDefinition).filter(ModelDefinition.project_id.in_(project_ids or ["__none__"]))
            dataset_query = session.query(Dataset).filter(Dataset.project_id.in_(project_ids or ["__none__"]))
            tokenizer_query = session.query(TokenizerModel).filter(TokenizerModel.project_id.in_(project_ids or ["__none__"]))
            model_count = model_query.count()
            dataset_count = dataset_query.count()
            tokenizer_count = tokenizer_query.count()
            job_query = session.query(TaskJob).filter(TaskJob.project_id.in_(project_ids or ["__none__"]))
            active_jobs = (
                job_query
                .filter(TaskJob.status.in_([JobStatus.RUNNING.value, JobStatus.QUEUED.value]))
                .count()
            )
            recent_jobs = (
                job_query
                .order_by(TaskJob.created_at.desc())
                .limit(10)
                .all()
            )

        # System resource snapshot via HardwareMonitor
        snap = hardware_monitor.snapshot()

        system_resources = [
            {"icon": "bi-cpu", "label": f"CPU ({snap.cpu.logical_cores} cores)", "value": f"{snap.cpu.usage_percent:.0f}%", "percent": snap.cpu.usage_percent},
            {"icon": "bi-memory", "label": "RAM", "value": f"{snap.memory.used_gb:.1f} / {snap.memory.total_gb:.1f} GB", "percent": snap.memory.usage_percent},
            {"icon": "bi-hdd", "label": "Disk", "value": f"{snap.disk.used_gb:.1f} / {snap.disk.total_gb:.1f} GB", "percent": snap.disk.usage_percent},
        ]
        for gpu in snap.gpus:
            system_resources.append({
                "icon": "bi-gpu-card",
                "label": f"GPU {gpu.index} ({gpu.name})",
                "value": f"{gpu.vram_used_mb / 1024:.1f} / {gpu.vram_total_mb / 1024:.1f} GB  •  {gpu.temperature_c}°C",
                "percent": gpu.vram_usage_percent,
            })

        alloc_summary = resource_allocator.summary()

        return render_template(
            "dashboard.html",
            stats={
                "projects": project_count,
                "models": model_count,
                "datasets": dataset_count,
                "tokenizers": tokenizer_count,
                "active_jobs": active_jobs,
            },
            recent_jobs=recent_jobs,
            system_resources=system_resources,
            allocation=alloc_summary,
        )

    @app.route("/health")
    def health():
        """Health check endpoint."""
        checks = {"database": "ok", "storage": "ok"}
        status_code = 200
        try:
            with get_session() as session:
                session.execute(text("SELECT 1"))
        except Exception:
            logger.exception("Database health check failed")
            checks["database"] = "error"
            status_code = 503
        try:
            backend = get_storage_backend()
            if not storage_paths.root.exists() and settings.STORAGE_BACKEND == "local":
                raise RuntimeError("storage root is missing")
            backend.exists(".healthcheck")
        except Exception:
            logger.exception("Storage health check failed")
            checks["storage"] = "error"
            status_code = 503
        return jsonify({
            "status": "ok" if status_code == 200 else "degraded",
            "version": settings.APP_VERSION,
            "checks": checks,
        }), status_code

    # ── Resource & Storage API ────────────────────

    @app.route("/api/resources")
    def api_resources():
        """Return a full hardware snapshot as JSON."""
        snap = hardware_monitor.snapshot()
        return jsonify(snap.to_dict())

    @app.route("/api/resources/allocator")
    def api_allocator():
        """Return resource allocation summary as JSON."""
        return jsonify(resource_allocator.summary())

    @app.route("/api/storage")
    def api_storage():
        """Return storage directory info."""
        from core.storage import get_storage_backend
        backend = get_storage_backend()
        return jsonify({
            "backend": settings.STORAGE_BACKEND,
            "root": str(storage_paths.root),
            "directories": [
                "datasets", "models", "checkpoints", "experiments",
                "tokenizers", "artifacts", "logs", "exports",
                "uploads", "registry",
            ],
        })


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
app = create_app()

if __name__ == "__main__":
    app.run(
        host=settings.HOST,
        port=settings.PORT,
        debug=settings.DEBUG,
    )
