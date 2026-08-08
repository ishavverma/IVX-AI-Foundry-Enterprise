"""Versioned JSON API for core AFE platform resources."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from flask import Blueprint, jsonify, request
from flask_login import current_user

from core.database.engine import get_session
from core.database.models.dataset import Dataset
from core.database.models.experiment import Experiment
from core.database.models.model_def import ModelDefinition
from core.database.models.task_job import JobStatus, JobType, TaskJob
from core.queue.worker import task_queue
from core.storage import get_storage_backend
from core.tenancy import TenantService
from engines.dataset.connectors import DatasetConnector
from engines.dataset.ingestion import DatasetIngestion
from engines.training.trainer import TrainingEngine, TrainingHyperparams


api_bp = Blueprint("api_v1", __name__, url_prefix="/api/v1")


def _json_date(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _dataset_dict(dataset: Dataset) -> dict[str, Any]:
    return {
        "id": dataset.id,
        "name": dataset.name,
        "slug": dataset.slug,
        "description": dataset.description,
        "format": dataset.format,
        "status": dataset.status,
        "size_bytes": dataset.size_bytes,
        "row_count": dataset.row_count,
        "token_count": dataset.token_count,
        "quality_score": dataset.quality_score,
        "project_id": dataset.project_id,
        "created_by": dataset.created_by,
        "created_at": _json_date(dataset.created_at),
        "updated_at": _json_date(dataset.updated_at),
    }


def _model_dict(model: ModelDefinition) -> dict[str, Any]:
    return {
        "id": model.id,
        "name": model.name,
        "slug": model.slug,
        "description": model.description,
        "architecture_type": model.architecture_type,
        "preset_tier": model.preset_tier,
        "d_model": model.d_model,
        "num_layers": model.num_layers,
        "num_heads": model.num_heads,
        "num_kv_heads": model.num_kv_heads,
        "vocab_size": model.vocab_size,
        "max_seq_len": model.max_seq_len,
        "param_count": model.param_count,
        "vram_fp16_mb": model.vram_fp16_mb,
        "status": model.status,
        "project_id": model.project_id,
        "created_by": model.created_by,
        "created_at": _json_date(model.created_at),
    }


def _experiment_dict(experiment: Experiment) -> dict[str, Any]:
    return {
        "id": experiment.id,
        "name": experiment.name,
        "slug": experiment.slug,
        "run_type": experiment.run_type,
        "status": experiment.status,
        "model_id": experiment.model_id,
        "dataset_id": experiment.dataset_id,
        "project_id": experiment.project_id,
        "current_step": experiment.current_step,
        "max_steps": experiment.max_steps,
        "current_loss": experiment.current_loss,
        "best_loss": experiment.best_loss,
        "total_tokens_trained": experiment.total_tokens_trained,
        "tokens_per_second": experiment.tokens_per_second,
        "metrics_history": json.loads(experiment.metrics_history_json or "[]"),
        "created_by": experiment.created_by,
        "created_at": _json_date(experiment.created_at),
    }


def _job_dict(job: TaskJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "name": job.name,
        "job_type": job.job_type,
        "status": job.status,
        "progress": job.progress,
        "result": json.loads(job.result_json or "{}"),
        "error": job.error_message,
        "started_at": _json_date(job.started_at),
        "completed_at": _json_date(job.completed_at),
        "created_at": _json_date(job.created_at),
    }


def _page(query: Any) -> tuple[list[Any], dict[str, int]]:
    try:
        page = max(1, int(request.args.get("page", 1)))
        per_page = min(100, max(1, int(request.args.get("per_page", 25))))
    except ValueError:
        page, per_page = 1, 25
    total = query.count()
    rows = query.offset((page - 1) * per_page).limit(per_page).all()
    return rows, {"page": page, "per_page": per_page, "total": total}


@api_bp.get("/datasets")
def datasets() -> Any:
    with get_session() as session:
        query = TenantService.scope(session.query(Dataset), Dataset, session).order_by(Dataset.created_at.desc())
        rows, pagination = _page(query)
        return jsonify({"data": [_dataset_dict(row) for row in rows], "pagination": pagination})


@api_bp.post("/datasets/import")
def import_dataset() -> Any:
    payload = request.get_json(silent=True) or {}
    source_type = str(payload.get("source_type", "")).lower()
    try:
        if source_type == "github":
            source = DatasetConnector.github(
                str(payload["owner"]), str(payload["repo"]), str(payload["path"]), str(payload.get("ref", "main"))
            )
        elif source_type == "huggingface":
            source = DatasetConnector.huggingface(str(payload["repo_id"]), str(payload["filename"]), str(payload.get("revision", "main")))
        elif source_type == "kaggle":
            source = DatasetConnector.kaggle(str(payload["dataset"]), str(payload["filename"]))
        elif source_type == "http":
            source = DatasetConnector.http(str(payload["url"]), payload.get("filename"))
        else:
            return jsonify({"error": "unsupported_source_type"}), 400
    except KeyError as exc:
        return jsonify({"error": "missing_source_field", "field": str(exc)}), 400

    dataset_id = str(uuid.uuid4())
    try:
        storage = get_storage_backend()
        storage_key = DatasetConnector.download_to_storage(source, storage, f"datasets/{dataset_id}")
        samples, row_count, size_bytes = DatasetIngestion.sample_head(storage.get_url(storage_key), n=5)
    except (OSError, ValueError, requests.RequestException) as exc:
        return jsonify({"error": "dataset_import_failed", "detail": str(exc)}), 502

    with get_session() as session:
        row = Dataset(
            id=dataset_id,
            name=str(payload.get("name", source.filename))[:256],
            slug=f"{source.filename.rsplit('.', 1)[0].lower().replace(' ', '-')[:48]}-{dataset_id[:8]}",
            description=str(payload.get("description", "")),
            format=Path(source.filename).suffix.lstrip(".") or "txt",
            storage_key=storage_key,
            size_bytes=size_bytes,
            row_count=row_count,
            sample_preview_json=json.dumps(samples, default=str),
            status="raw",
            project_id=payload.get("project_id"),
            created_by=current_user.get_id(),
        )
        session.add(row)
        return jsonify({"data": _dataset_dict(row)}), 201


@api_bp.get("/datasets/<dataset_id>")
def dataset(dataset_id: str) -> Any:
    with get_session() as session:
        row = TenantService.scope(session.query(Dataset).filter(Dataset.id == dataset_id), Dataset, session).first()
        if not row:
            return jsonify({"error": "dataset_not_found"}), 404
        return jsonify({"data": _dataset_dict(row)})


@api_bp.get("/models")
def models() -> Any:
    with get_session() as session:
        query = TenantService.scope(session.query(ModelDefinition), ModelDefinition, session).order_by(ModelDefinition.created_at.desc())
        rows, pagination = _page(query)
        return jsonify({"data": [_model_dict(row) for row in rows], "pagination": pagination})


@api_bp.get("/models/<model_id>")
def model(model_id: str) -> Any:
    with get_session() as session:
        row = TenantService.scope(session.query(ModelDefinition).filter(ModelDefinition.id == model_id), ModelDefinition, session).first()
        if not row:
            return jsonify({"error": "model_not_found"}), 404
        return jsonify({"data": _model_dict(row)})


@api_bp.get("/experiments")
def experiments() -> Any:
    with get_session() as session:
        query = TenantService.scope(session.query(Experiment), Experiment, session).order_by(Experiment.created_at.desc())
        rows, pagination = _page(query)
        return jsonify({"data": [_experiment_dict(row) for row in rows], "pagination": pagination})


@api_bp.get("/experiments/<experiment_id>")
def experiment(experiment_id: str) -> Any:
    with get_session() as session:
        row = TenantService.scope(session.query(Experiment).filter(Experiment.id == experiment_id), Experiment, session).first()
        if not row:
            return jsonify({"error": "experiment_not_found"}), 404
        return jsonify({"data": _experiment_dict(row)})


@api_bp.post("/training")
def launch_training() -> Any:
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "API Training Run")).strip()[:256]
    if not name:
        return jsonify({"error": "name_required"}), 400
    try:
        hp = TrainingHyperparams(**payload.get("hyperparams", {}))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": "invalid_hyperparams", "detail": str(exc)}), 400

    experiment_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    with get_session() as session:
        experiment = Experiment(
            id=experiment_id,
            name=name,
            slug=f"{name.lower().replace(' ', '-')[:48]}-{experiment_id[:8]}",
            run_type=str(payload.get("run_type", "pretraining")),
            model_id=payload.get("model_id"),
            dataset_id=payload.get("dataset_id"),
            project_id=payload.get("project_id"),
            status="queued",
            max_steps=hp.max_steps,
            hyperparams_json=json.dumps(hp.to_dict()),
            created_by=current_user.get_id(),
        )
        job = TaskJob(
            id=job_id,
            name=f"Training: {name}",
            job_type=JobType.TRAINING.value,
            status=JobStatus.QUEUED.value,
            config_json=json.dumps(hp.to_dict()),
            user_id=current_user.get_id(),
            project_id=payload.get("project_id"),
        )
        session.add_all([experiment, job])

    task_queue.submit(TrainingEngine.run_training_loop, job_id, experiment_id, hp.to_dict())
    return jsonify({"data": {"experiment_id": experiment_id, "job_id": job_id}}), 202


@api_bp.get("/jobs/<job_id>")
def job(job_id: str) -> Any:
    with get_session() as session:
        row = session.query(TaskJob).filter(TaskJob.id == job_id, TaskJob.user_id == current_user.get_id()).first()
        if not row:
            return jsonify({"error": "job_not_found"}), 404
        return jsonify({"data": _job_dict(row)})