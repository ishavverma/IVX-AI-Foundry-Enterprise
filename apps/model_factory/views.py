"""
AFE — Model Factory & Architecture Compiler App Blueprint

Interactive neural architecture composer with real-time FLOP/VRAM calculation,
standard tier presets (Nano to MoE-1T), and PyTorch code export.
"""

from __future__ import annotations

import json
import uuid
from flask import Blueprint, flash, redirect, render_template, request, url_for, jsonify, Response
from flask_login import current_user

from core.database.engine import get_session
from core.database.models.model_def import ModelDefinition
from core.storage.sdk import get_storage_backend
from engines.model.config import ArchitectureConfig, MODEL_PRESETS
from engines.model.calculator import ArchitectureCalculator
from engines.model.exporter import ModelExporter
from core.tenancy import TenantService

models_bp = Blueprint("models", __name__, url_prefix="/models")


@models_bp.route("/")
def list_models():
    """Render the Model Factory dashboard with model cards and preset data.

    The UI now lives in ``dashboard.html`` which provides a responsive grid of
    model cards, a "New Model" wizard modal, and live validation.  The route
    supplies ``models`` (compiled definitions) and ``presets`` (tier presets) to
    the template.
    """
    with get_session() as session:
        defs = (
            TenantService.scope(
                session.query(ModelDefinition), ModelDefinition, session
            )
            .order_by(ModelDefinition.created_at.desc())
            .all()
        )
        model_list = []
        for m in defs:
            model_list.append(
                {
                    "id": m.id,
                    "name": m.name,
                    "slug": m.slug,
                    "architecture_type": m.architecture_type,
                    "preset_tier": m.preset_tier,
                    "d_model": m.d_model,
                    "num_layers": m.num_layers,
                    "param_count": m.param_count,
                    "vram_fp16_mb": m.vram_fp16_mb,
                    "gflops_per_token": m.gflops_per_token,
                    "status": m.status,
                    "created_at": m.created_at,
                }
            )

    presets_data = []
    for tier, cfg in MODEL_PRESETS.items():
        metrics = ArchitectureCalculator.compute_metrics(cfg)
        presets_data.append(
            {
                "tier": tier,
                "name": cfg.name,
                "d_model": cfg.d_model,
                "num_layers": cfg.num_layers,
                "params": metrics.total_params,
                "vram_fp16": metrics.weights_fp16_mb,
                "gflops": metrics.gflops_forward_per_token,
                "use_ssm": cfg.use_ssm,
                "use_moe": cfg.use_moe,
            }
        )

    # Render the new dashboard template that includes the modal wizard.
    return render_template("model_factory/dashboard.html", models=model_list, presets=presets_data)


@models_bp.route("/new", methods=["GET", "POST"])
def new_model():
    """Interactive visual architecture builder and compiler."""
    if request.method == "POST":
        name = request.form.get("name", "Custom Hybrid Model").strip()
        tier = request.form.get("tier", "base").strip()
        d_model = int(request.form.get("d_model", 4096))
        num_layers = int(request.form.get("num_layers", 32))
        num_heads = int(request.form.get("num_heads", 32))
        num_kv_heads = int(request.form.get("num_kv_heads", 8))
        vocab_size = int(request.form.get("vocab_size", 64000))
        max_seq_len = int(request.form.get("max_seq_len", 32768))
        use_ssm = request.form.get("use_ssm") == "on"
        ssm_ratio = request.form.get("ssm_attn_ratio", "1:7")
        use_moe = request.form.get("use_moe") == "on"
        num_experts = int(request.form.get("num_experts", 16))
        top_k = int(request.form.get("top_k_experts", 2))

        cfg = ArchitectureConfig(
            name=name,
            tier=tier,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            use_ssm=use_ssm,
            ssm_attn_ratio=ssm_ratio,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k_experts=top_k,
        )

        metrics = ArchitectureCalculator.compute_metrics(cfg)
        pytorch_code = ModelExporter.generate_pytorch_code(cfg)
        config_json = ModelExporter.generate_config_json(cfg)

        model_id = str(uuid.uuid4())
        config_key = f"models/{model_id}/config.json"
        code_key = f"models/{model_id}/model.py"

        storage = get_storage_backend()
        storage.write(config_key, config_json.encode("utf-8"))
        storage.write(code_key, pytorch_code.encode("utf-8"))

        with get_session() as session:
            m = ModelDefinition(
                id=model_id,
                name=name,
                slug=name.lower().replace(" ", "-")[:64],
                architecture_type="hybrid" if use_ssm else ("moe" if use_moe else "dense"),
                preset_tier=tier,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                vocab_size=vocab_size,
                max_seq_len=max_seq_len,
                ssm_ratio=ssm_ratio,
                use_moe=use_moe,
                num_experts=num_experts,
                top_k_experts=top_k,
                param_count=metrics.total_params,
                active_param_count=metrics.active_params,
                gflops_per_token=metrics.gflops_forward_per_token,
                vram_fp16_mb=metrics.weights_fp16_mb,
                vram_fp8_mb=metrics.weights_fp8_mb,
                vram_int4_mb=metrics.weights_int4_mb,
                storage_key=config_key,
                code_storage_key=code_key,
                config_json=config_json,
                status="compiled",
                created_by=current_user.get_id(),
            )
            session.add(m)

        flash(f"Architecture '{name}' compiled ({metrics.total_params / 1e9:.2f}B parameters)!", "success")
        return redirect(url_for("models.detail_model", model_id=model_id))

    return render_template("model_factory/new.html", presets=MODEL_PRESETS)


@models_bp.route("/<model_id>")
def detail_model(model_id: str):
    """Detailed view for a compiled neural architecture."""
    storage = get_storage_backend()
    with get_session() as session:
        m = TenantService.scope(session.query(ModelDefinition).filter(ModelDefinition.id == model_id), ModelDefinition, session).first()
        if not m:
            flash("Model definition not found", "danger")
            return redirect(url_for("models.list_models"))

        code_text = storage.read(m.code_storage_key).decode("utf-8") if m.code_storage_key else ""
        cfg_dict = json.loads(m.config_json or "{}")

        model_data = {
            "id": m.id,
            "name": m.name,
            "slug": m.slug,
            "architecture_type": m.architecture_type,
            "preset_tier": m.preset_tier,
            "d_model": m.d_model,
            "num_layers": m.num_layers,
            "num_heads": m.num_heads,
            "num_kv_heads": m.num_kv_heads,
            "vocab_size": m.vocab_size,
            "max_seq_len": m.max_seq_len,
            "ssm_ratio": m.ssm_ratio,
            "use_moe": m.use_moe,
            "num_experts": m.num_experts,
            "top_k_experts": m.top_k_experts,
            "param_count": m.param_count,
            "active_param_count": m.active_param_count,
            "gflops_per_token": m.gflops_per_token,
            "vram_fp16_mb": m.vram_fp16_mb,
            "vram_fp8_mb": m.vram_fp8_mb,
            "vram_int4_mb": m.vram_int4_mb,
            "status": m.status,
            "code": code_text,
            "config": cfg_dict,
            "created_at": m.created_at,
        }

    return render_template("model_factory/detail.html", model=model_data)


@models_bp.route("/<model_id>/download/<code>")
def download_file(model_id: str, code: str):
    """Download generated PyTorch model.py or config.json."""
    storage = get_storage_backend()
    with get_session() as session:
        m = session.get(ModelDefinition, model_id)
        if not m:
            return "Not found", 404
        if code == "code" and m.code_storage_key:
            data = storage.read(m.code_storage_key).decode("utf-8")
            return Response(data, mimetype="text/x-python", headers={"Content-Disposition": f"attachment; filename={m.slug}_model.py"})
        elif code == "config":
            return Response(m.config_json, mimetype="application/json", headers={"Content-Disposition": f"attachment; filename={m.slug}_config.json"})
    return "Invalid request", 400


@models_bp.route("/calculate", methods=["POST"])
def calculate_metrics():
    """Live API for visual builder calculating params/VRAM/FLOPs dynamically."""
    data = request.json or {}
    cfg = ArchitectureConfig(
        d_model=int(data.get("d_model", 4096)),
        num_layers=int(data.get("num_layers", 32)),
        num_heads=int(data.get("num_heads", 32)),
        num_kv_heads=int(data.get("num_kv_heads", 8)),
        vocab_size=int(data.get("vocab_size", 64000)),
        max_seq_len=int(data.get("max_seq_len", 32768)),
        use_ssm=bool(data.get("use_ssm", True)),
        ssm_attn_ratio=str(data.get("ssm_attn_ratio", "1:7")),
        use_moe=bool(data.get("use_moe", False)),
        num_experts=int(data.get("num_experts", 16)),
        top_k_experts=int(data.get("top_k_experts", 2)),
    )
    metrics = ArchitectureCalculator.compute_metrics(cfg)
    return jsonify(metrics.to_dict())
