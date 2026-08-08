"""
AFE — Checkpoint Management App Blueprint

Weights explorer, quantization converter (FP16 → INT8 / INT4), and GGUF exporter.
"""

from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, request, url_for, jsonify
from core.tenancy import TenantService

from core.database.engine import get_session
from core.database.models.experiment import Checkpoint, Experiment
from engines.checkpoints.manager import CheckpointManager

checkpoints_bp = Blueprint("checkpoints", __name__, url_prefix="/checkpoints")


@checkpoints_bp.route("/")
def list_checkpoints():
    """List all saved weight checkpoints."""
    with get_session() as session:
        ckpts = TenantService.scope(session.query(Checkpoint), Checkpoint, session).order_by(Checkpoint.created_at.desc()).all()
        ckpt_list = []
        for c in ckpts:
            ckpt_list.append({
                "id": c.id,
                "name": c.name,
                "step": c.step,
                "loss": c.loss,
                "format": c.format,
                "quantization": c.quantization,
                "size_mb": round(c.size_bytes / (1024**2), 1),
                "is_best": c.is_best,
                "storage_key": c.storage_key,
                "created_at": c.created_at,
            })
    return render_template("checkpoints/list.html", checkpoints=ckpt_list)


@checkpoints_bp.route("/<checkpoint_id>/quantize", methods=["POST"])
def quantize_checkpoint(checkpoint_id: str):
    """Trigger GPTQ / AWQ post-training quantization."""
    quant_type = request.form.get("quantization_type", "int4")
    try:
        res = CheckpointManager.quantize_weights(checkpoint_id, quant_type)
        flash(f"Model quantized to {quant_type.upper()} ({res.compression_ratio}x compression ratio)!", "success")
    except Exception as e:
        flash(f"Quantization failed: {e}", "danger")
    return redirect(url_for("checkpoints.list_checkpoints"))


@checkpoints_bp.route("/<checkpoint_id>/export/gguf", methods=["POST"])
def export_gguf(checkpoint_id: str):
    """Export checkpoint to GGUF format."""
    try:
        gguf_key = CheckpointManager.export_to_gguf(checkpoint_id)
        flash(f"Exported to GGUF at '{gguf_key}'", "success")
    except Exception as e:
        flash(f"GGUF export failed: {e}", "danger")
    return redirect(url_for("checkpoints.list_checkpoints"))
