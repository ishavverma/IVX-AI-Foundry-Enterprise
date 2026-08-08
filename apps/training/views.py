"""
AFE — Training App Blueprint

Orchestrates distributed pretraining experiments, live loss curve visualization,
hyperparameter configuration, and checkpoint tracking.
"""

from __future__ import annotations

import json
import uuid
from flask import Blueprint, flash, redirect, render_template, request, url_for, jsonify
from flask_login import current_user

from core.database.engine import get_session
from core.database.models.experiment import Experiment, Checkpoint
from core.database.models.model_def import ModelDefinition
from core.database.models.dataset import Dataset
from core.database.models.task_job import TaskJob, JobType, JobStatus
from core.queue.worker import task_queue
from engines.training.trainer import TrainingEngine, TrainingHyperparams
from core.tenancy import TenantService

training_bp = Blueprint("training", __name__, url_prefix="/training")


@training_bp.route("/")
def list_experiments():
    """List all distributed pretraining runs."""
    with get_session() as session:
        exps = (
            TenantService.scope(session.query(Experiment), Experiment, session)
            .filter(Experiment.run_type == "pretraining")
            .order_by(Experiment.created_at.desc())
            .all()
        )
        exp_list = []
        for e in exps:
            exp_list.append({
                "id": e.id,
                "name": e.name,
                "slug": e.slug,
                "status": e.status,
                "current_step": e.current_step,
                "max_steps": e.max_steps,
                "current_loss": e.current_loss,
                "best_loss": e.best_loss,
                "total_tokens": e.total_tokens_trained,
                "tokens_per_sec": e.tokens_per_second,
                "created_at": e.created_at,
            })
    return render_template("training/list.html", experiments=exp_list)


@training_bp.route("/new", methods=["GET", "POST"])
def new_training():
    """Launch a new distributed pretraining experiment."""
    if request.method == "POST":
        name = request.form.get("name", "Pretraining Run 1").strip()
        model_id = request.form.get("model_id")
        dataset_id = request.form.get("dataset_id")
        max_steps = int(request.form.get("max_steps", 1000))
        learning_rate = float(request.form.get("learning_rate", 1e-4))
        batch_size = int(request.form.get("batch_size", 16))
        precision = request.form.get("precision", "fp16")

        hp = TrainingHyperparams(
            learning_rate=learning_rate,
            max_steps=max_steps,
            batch_size=batch_size,
            precision=precision,
        )
        organization_id = TenantService.organization_for_user(current_user.get_id())
        estimated_gpu_hours = max(0.1, max_steps / 3600)
        if organization_id:
            try:
                TenantService.require_quota(organization_id, "gpu_hours", estimated_gpu_hours)
            except PermissionError as exc:
                flash(str(exc), "danger")
                return redirect(url_for("training.new_training")), 429

        exp_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())

        with get_session() as session:
            exp = Experiment(
                id=exp_id,
                name=name,
                slug=name.lower().replace(" ", "-")[:64],
                run_type="pretraining",
                model_id=model_id,
                dataset_id=dataset_id,
                status="queued",
                max_steps=max_steps,
                hyperparams_json=json.dumps(hp.to_dict()),
                created_by=current_user.get_id(),
            )
            session.add(exp)

            job = TaskJob(
                id=job_id,
                name=f"Pretrain: {name}",
                job_type=JobType.TRAINING.value,
                status=JobStatus.QUEUED.value,
                config_json=json.dumps(hp.to_dict()),
                user_id=current_user.get_id(),
            )
            session.add(job)

        # Launch in background worker queue
        task_queue.submit(TrainingEngine.run_training_loop, job_id, exp_id, hp.to_dict())

        flash(f"Pretraining run '{name}' queued in background!", "success")
        return redirect(url_for("training.detail_experiment", experiment_id=exp_id))

    with get_session() as session:
        models = TenantService.scope(session.query(ModelDefinition), ModelDefinition, session).all()
        datasets = TenantService.scope(session.query(Dataset).filter(Dataset.status == "ready"), Dataset, session).all()
        model_list = [{"id": m.id, "name": m.name} for m in models]
        dataset_list = [{"id": d.id, "name": d.name} for d in datasets]

    return render_template("training/new.html", models=model_list, datasets=dataset_list)


@training_bp.route("/<experiment_id>")
def detail_experiment(experiment_id: str):
    """Detailed view with live loss curves and checkpoints."""
    with get_session() as session:
        exp = TenantService.scope(session.query(Experiment).filter(Experiment.id == experiment_id), Experiment, session).first()
        if not exp:
            flash("Experiment not found", "danger")
            return redirect(url_for("training.list_experiments"))

        checkpoints = (
            session.query(Checkpoint)
            .filter(Checkpoint.experiment_id == experiment_id)
            .order_by(Checkpoint.step.desc())
            .all()
        )

        history = json.loads(exp.metrics_history_json or "[]")
        hp = json.loads(exp.hyperparams_json or "{}")

        exp_data = {
            "id": exp.id,
            "name": exp.name,
            "slug": exp.slug,
            "status": exp.status,
            "current_step": exp.current_step,
            "max_steps": exp.max_steps,
            "current_loss": exp.current_loss,
            "best_loss": exp.best_loss,
            "total_tokens": exp.total_tokens_trained,
            "tokens_per_sec": exp.tokens_per_second,
            "history": history,
            "hyperparams": hp,
            "checkpoints": [{
                "id": c.id, "name": c.name, "step": c.step, "loss": c.loss,
                "format": c.format, "quantization": c.quantization, "size_mb": round(c.size_bytes / (1024**2), 1),
                "is_best": c.is_best,
            } for c in checkpoints],
            "created_at": exp.created_at,
        }

    return render_template("training/detail.html", experiment=exp_data)
