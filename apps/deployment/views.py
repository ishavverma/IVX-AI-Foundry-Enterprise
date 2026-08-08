"""
AFE — Inference Deployment & Serving App Blueprint

Manages live OpenAI-compatible endpoints (/v1/chat/completions), GPU allocations,
continuous batching simulation, and latency telemetry.
"""

from __future__ import annotations

import json
import time
import uuid
from flask import Blueprint, flash, redirect, render_template, request, url_for, jsonify, Response
from flask_login import current_user

from core.database.engine import get_session
from core.database.models.evaluation import InferenceEndpoint
from core.database.models.model_def import ModelDefinition
from core.database.models.experiment import Checkpoint
from engines.inference.server import InferenceServer
from core.tenancy import TenantService
from core.security.limiter import limiter

deployment_bp = Blueprint("deployment", __name__, url_prefix="/inference")


@deployment_bp.route("/")
def list_endpoints():
    """List all configured inference serving endpoints."""
    with get_session() as session:
        # Seed default endpoint if empty
        if session.query(InferenceEndpoint).count() == 0:
            default_ep = InferenceEndpoint(
                name="AFE-Base-7B Production Serving Endpoint",
                slug="afe-base-7b-prod",
                backend="pytorch",
                port=8000,
                is_running=True,
                max_concurrency=8,
                temperature=0.7,
                top_p=0.9,
                gpu_indices="0",
                total_requests=1420,
                avg_latency_ms=34.8,
            )
            session.add(default_ep)
            session.commit()

        endpoints = TenantService.scope(session.query(InferenceEndpoint), InferenceEndpoint, session).order_by(InferenceEndpoint.created_at.desc()).all()
        ep_list = []
        for ep in endpoints:
            ep_list.append({
                "id": ep.id,
                "name": ep.name,
                "slug": ep.slug,
                "backend": ep.backend,
                "port": ep.port,
                "is_running": ep.is_running,
                "max_concurrency": ep.max_concurrency,
                "gpu_indices": ep.gpu_indices,
                "total_requests": ep.total_requests,
                "avg_latency_ms": ep.avg_latency_ms,
                "created_at": ep.created_at,
            })

    return render_template("deployment/list.html", endpoints=ep_list)


# ──────────────────────────────────────────────
# OpenAI-Compatible Serving APIs
# ──────────────────────────────────────────────

@deployment_bp.route("/v1/models", methods=["GET"])
def openai_models():
    """OpenAI-compatible models catalog endpoint."""
    with get_session() as session:
        models = TenantService.scope(session.query(ModelDefinition), ModelDefinition, session).all()
        data = [{
            "id": m.slug or m.id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "afe-enterprise",
        } for m in models]
        if not data:
            data.append({"id": "afe-hybrid-7b", "object": "model", "created": int(time.time()), "owned_by": "afe"})

    return jsonify({"object": "list", "data": data})


@deployment_bp.route("/v1/chat/completions", methods=["POST"])
@limiter.limit("60 per minute")
def openai_chat_completions():
    """OpenAI-compatible chat completions endpoint (streaming + non-streaming)."""
    body = request.json or {}
    messages = body.get("messages", [])
    model_name = body.get("model", "afe-hybrid-7b")
    stream = body.get("stream", False)
    max_tokens = int(body.get("max_tokens", 128))
    temperature = float(body.get("temperature", 0.7))

    if stream:
        def stream_gen():
            for chunk in InferenceServer.generate_stream(
                prompt=messages[-1].get("content", "") if messages else "",
                max_tokens=max_tokens,
                temperature=temperature,
                model_name=model_name,
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return Response(stream_gen(), mimetype="text/event-stream")

    resp = InferenceServer.generate_complete(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        model_name=model_name,
    )
    return jsonify(resp)
