"""
AFE — Fine-Tuning App Blueprint

Wizards for LoRA, QLoRA (4-bit NF4), and DPO human preference alignment.
"""

from __future__ import annotations

import json
import uuid
from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user

from core.database.engine import get_session
from core.database.models.experiment import Experiment, Checkpoint
from core.database.models.model_def import ModelDefinition
from core.database.models.dataset import Dataset
from core.database.models.task_job import TaskJob, JobType, JobStatus
from core.queue.worker import task_queue
from engines.finetuning.peft import PEFTEngine, LoRAConfig, DPOConfig
from core.tenancy import TenantService

finetuning_bp = Blueprint("finetuning", __name__, url_prefix="/finetuning")


@finetuning_bp.route("/")
def list_finetuning():
    """List all PEFT / alignment fine-tuning runs."""
    with get_session() as session:
        exps = (
            TenantService.scope(session.query(Experiment), Experiment, session)
            .filter(Experiment.run_type.in_(["sft", "lora", "qlora", "dpo", "rlhf"]))
            .order_by(Experiment.created_at.desc())
            .all()
        )
        exp_list = []
        for e in exps:
            exp_list.append({
                "id": e.id,
                "name": e.name,
                "slug": e.slug,
                "run_type": e.run_type,
                "status": e.status,
                "current_step": e.current_step,
                "max_steps": e.max_steps,
                "best_loss": e.best_loss,
                "created_at": e.created_at,
            })
    return render_template("finetuning/list.html", runs=exp_list)


@finetuning_bp.route("/new", methods=["GET", "POST"])
def new_finetuning():
    """Configure and launch a LoRA, QLoRA, or DPO alignment task."""
    if request.method == "POST":
        name = request.form.get("name", "LoRA Alignment v1").strip()
        run_type = request.form.get("run_type", "lora").strip()
        model_id = request.form.get("model_id")
        dataset_id = request.form.get("dataset_id")
        lora_rank = int(request.form.get("lora_rank", 16))
        lora_alpha = int(request.form.get("lora_alpha", 32))
        max_steps = int(request.form.get("max_steps", 400))
        learning_rate = float(request.form.get("learning_rate", 2e-4))

        cfg_dict = {
            "run_type": run_type,
            "r": lora_rank,
            "lora_alpha": lora_alpha,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
        }
        organization_id = TenantService.organization_for_user(current_user.get_id())
        if organization_id:
            try:
                TenantService.require_quota(organization_id, "gpu_hours", max(0.1, max_steps / 3600))
            except PermissionError as exc:
                flash(str(exc), "danger")
                return redirect(url_for("finetuning.new_finetuning")), 429

        exp_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())

        with get_session() as session:
            exp = Experiment(
                id=exp_id,
                name=name,
                slug=name.lower().replace(" ", "-")[:64],
                run_type=run_type,
                model_id=model_id,
                dataset_id=dataset_id,
                status="queued",
                max_steps=max_steps,
                hyperparams_json=json.dumps(cfg_dict),
                created_by=current_user.get_id(),
            )
            session.add(exp)

            job = TaskJob(
                id=job_id,
                name=f"{run_type.upper()}: {name}",
                job_type=JobType.FINETUNING.value,
                status=JobStatus.QUEUED.value,
                config_json=json.dumps(cfg_dict),
                user_id=current_user.get_id(),
            )
            session.add(job)

        task_queue.submit(PEFTEngine.run_finetuning_loop, job_id, exp_id, cfg_dict)

        flash(f"{run_type.upper()} alignment task '{name}' queued in background!", "success")
        return redirect(url_for("finetuning.list_finetuning"))

    with get_session() as session:
        models = TenantService.scope(session.query(ModelDefinition), ModelDefinition, session).all()
        datasets = TenantService.scope(session.query(Dataset).filter(Dataset.status == "ready"), Dataset, session).all()
        model_list = [{"id": m.id, "name": m.name} for m in models]
        dataset_list = [{"id": d.id, "name": d.name} for d in datasets]

    return render_template("finetuning/new.html", models=model_list, datasets=dataset_list)
