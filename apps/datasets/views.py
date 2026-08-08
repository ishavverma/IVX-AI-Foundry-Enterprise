"""
AFE — Dataset Studio App Blueprint

Web views for managing dataset ingestion, pipeline execution,
quality metrics inspection, and sample previews.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from flask import Blueprint, flash, redirect, render_template, request, url_for, jsonify
from flask_login import current_user

from core.database.engine import get_session
from core.database.models.dataset import Dataset
from core.database.models.task_job import TaskJob, JobType, JobStatus
from core.queue.worker import task_queue
from core.storage.sdk import get_storage_backend
from core.storage.paths import storage_paths
from engines.dataset.ingestion import DatasetIngestion
from engines.dataset.pipeline import DatasetPipeline
from core.tenancy import TenantService

datasets_bp = Blueprint("datasets", __name__, url_prefix="/datasets")
ALLOWED_FORMATS = {"jsonl", "json", "csv", "txt", "parquet", "huggingface"}


@datasets_bp.route("/")
def list_datasets():
    """List all managed datasets."""
    with get_session() as session:
        datasets = TenantService.scope(session.query(Dataset), Dataset, session).order_by(Dataset.created_at.desc()).all()
        # Detach attributes for rendering
        dataset_list = []
        for d in datasets:
            dataset_list.append({
                "id": d.id,
                "name": d.name,
                "slug": d.slug,
                "description": d.description,
                "format": d.format,
                "row_count": d.row_count,
                "size_bytes": d.size_bytes,
                "quality_score": d.quality_score,
                "status": d.status,
                "created_at": d.created_at,
            })
    return render_template("datasets/list.html", datasets=dataset_list)


# ---------------------------------------------------------------------------
# Pre‑training dataset recommendations (4‑6 GB range)
# ---------------------------------------------------------------------------

PRETRAIN_DATASETS = [
    {
        "id": "openwebtext2",
        "name": "OpenWebText2 (≈5 GB)",
        "size_gb": 5,
        "description": "Curated English web text scraped from Reddit links, similar to OpenAI's WebText.",
        "source": "https://huggingface.co/datasets/openwebtext",
        "huggingface": "openwebtext",
    },
    {
        "id": "the_pile_small",
        "name": "The Pile – Small Slice (≈4.5 GB)",
        "size_gb": 4.5,
        "description": "Mixed‑domain web crawl (books, academic, code, etc.) – a balanced subset.",
        "source": "https://huggingface.co/datasets/EleutherAI/the-pile",
        "huggingface": "EleutherAI/the-pile",
    },
    {
        "id": "c4_filtered",
        "name": "C4 Filtered (≈5 GB)",
        "size_gb": 5,
        "description": "Cleaned English web crawl from Common Crawl, low‑duplication.",
        "source": "https://huggingface.co/datasets/allenai/c4",
        "huggingface": "allenai/c4",
    },
]


@datasets_bp.route("/pretrain")
def pretrain_datasets():
    """Show a curated list of ~4‑6 GB datasets suitable for LLM pre‑training."""
    return render_template("datasets/pretrain.html", datasets=PRETRAIN_DATASETS)


@datasets_bp.route("/new", methods=["GET", "POST"])
def new_dataset():
    """Create or upload a new dataset."""
    if request.method == "POST":
        name = request.form.get("name", "Untitled Dataset").strip()
        description = request.form.get("description", "").strip()
        fmt = request.form.get("format", "jsonl").strip().lower()
        if fmt not in ALLOWED_FORMATS:
            flash("Unsupported dataset format.", "danger")
            return render_template("datasets/new.html"), 400
        slug = name.lower().replace(" ", "-")[:64]
        
        file = request.files.get("file")
        raw_text = request.form.get("raw_text", "").strip()

        dataset_id = str(uuid.uuid4())
        filename = f"raw_{dataset_id}.{fmt if fmt != 'huggingface' else 'jsonl'}"
        storage_key = f"datasets/{dataset_id}/{filename}"
        storage = get_storage_backend()

        if file and file.filename:
            storage.write(storage_key, file.stream)
        elif raw_text:
            storage.write(storage_key, raw_text.encode("utf-8"))
        else:
            # Fallback starter demo dataset
            demo_lines = [
                json.dumps({"text": f"Sample document {i}: AI Foundry Enterprise provides unified MLOps and LLMOps foundation tooling for high performance systems."})
                for i in range(1, 21)
            ]
            storage.write(storage_key, "\n".join(demo_lines).encode("utf-8"))

        raw_path = storage.get_url(storage_key)
        samples, row_count, size_bytes = DatasetIngestion.sample_head(raw_path, n=5)

        with get_session() as session:
            ds = Dataset(
                id=dataset_id,
                name=name,
                slug=slug,
                description=description,
                format=fmt,
                storage_key=storage_key,
                size_bytes=size_bytes,
                row_count=row_count,
                status="raw",
                sample_preview_json=json.dumps(samples, default=str),
                created_by=current_user.get_id(),
            )
            session.add(ds)

        flash(f"Dataset '{name}' created successfully.", "success")
        return redirect(url_for("datasets.detail_dataset", dataset_id=dataset_id))

    return render_template("datasets/new.html")


@datasets_bp.route("/<dataset_id>")
def detail_dataset(dataset_id: str):
    """Detailed view for a specific dataset."""
    with get_session() as session:
        ds = TenantService.scope(session.query(Dataset).filter(Dataset.id == dataset_id), Dataset, session).first()
        if not ds:
            flash("Dataset not found", "danger")
            return redirect(url_for("datasets.list_datasets"))

        samples = json.loads(ds.sample_preview_json or "[]")
        lineage = json.loads(ds.lineage_json or "{}")

        dataset_data = {
            "id": ds.id,
            "name": ds.name,
            "slug": ds.slug,
            "description": ds.description,
            "format": ds.format,
            "storage_key": ds.storage_key,
            "processed_key": ds.processed_key,
            "row_count": ds.row_count,
            "size_bytes": ds.size_bytes,
            "quality_score": ds.quality_score,
            "status": ds.status,
            "samples": samples,
            "lineage": lineage,
            "created_at": ds.created_at,
        }

    return render_template("datasets/detail.html", dataset=dataset_data)


@datasets_bp.route("/<dataset_id>/process", methods=["POST"])
def process_dataset(dataset_id: str):
    """Trigger the data preparation pipeline for a dataset."""
    clean = request.form.get("clean") == "on"
    deduplicate = request.form.get("deduplicate") == "on"
    chunk = request.form.get("chunk") == "on"
    try:
        chunk_size = int(request.form.get("chunk_size", 512))
        min_quality_score = float(request.form.get("min_quality", 0.3))
    except (TypeError, ValueError):
        flash("Chunk size and minimum quality must be numeric.", "danger")
        return redirect(url_for("datasets.detail_dataset", dataset_id=dataset_id)), 400
    if chunk_size < 1 or not 0 <= min_quality_score <= 1:
        flash("Chunk size must be positive and minimum quality must be between 0 and 1.", "danger")
        return redirect(url_for("datasets.detail_dataset", dataset_id=dataset_id)), 400

    config = {
        "clean": clean,
        "deduplicate": deduplicate,
        "chunk": chunk,
        "chunk_size": chunk_size,
        "min_quality_score": min_quality_score,
    }

    job_id = str(uuid.uuid4())
    with get_session() as session:
        job = TaskJob(
            id=job_id,
            name=f"Process Dataset {dataset_id[:8]}",
            job_type=JobType.DATASET_PROCESSING.value,
            status=JobStatus.QUEUED.value,
            config_json=json.dumps(config),
        )
        session.add(job)

        ds = TenantService.scope(session.query(Dataset).filter(Dataset.id == dataset_id), Dataset, session).first()
        if ds:
            ds.status = "processing"

    # Submit to worker queue
    task_queue.submit(DatasetPipeline.run_pipeline, job_id, dataset_id, config)

    flash("Dataset processing pipeline queued in background.", "info")
    return redirect(url_for("datasets.detail_dataset", dataset_id=dataset_id))
