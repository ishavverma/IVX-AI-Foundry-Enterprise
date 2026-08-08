"""
AFE — Tokenizer Studio App Blueprint

Interactive views for training BPE / ByteLevel / Unigram tokenizers,
exploring vocabularies, viewing live token colorization, and exporting to HuggingFace / GGUF.
"""

from __future__ import annotations

import json
import uuid
from flask import Blueprint, flash, redirect, render_template, request, url_for, jsonify, Response

from core.database.engine import get_session
from core.database.models.tokenizer import TokenizerModel
from core.database.models.dataset import Dataset
from core.storage.sdk import get_storage_backend
from engines.tokenizer.engine import BaseTokenizer
from engines.tokenizer.analytics import TokenizerAnalytics
from engines.tokenizer.export import TokenizerExporter
from engines.dataset.ingestion import DatasetIngestion

tokenizer_bp = Blueprint("tokenizer", __name__, url_prefix="/tokenizer")


@tokenizer_bp.route("/")
def list_tokenizers():
    """List all available tokenizers."""
    with get_session() as session:
        tokenizers = session.query(TokenizerModel).order_by(TokenizerModel.created_at.desc()).all()
        tok_list = []
        for t in tokenizers:
            tok_list.append({
                "id": t.id,
                "name": t.name,
                "slug": t.slug,
                "algorithm": t.algorithm,
                "vocab_size": t.vocab_size,
                "compression_ratio": t.compression_ratio,
                "cjk_coverage_pct": t.cjk_coverage_pct,
                "status": t.status,
                "created_at": t.created_at,
            })
    return render_template("tokenizer/list.html", tokenizers=tok_list)


@tokenizer_bp.route("/new", methods=["GET", "POST"])
def new_tokenizer():
    """Train or create a new tokenizer."""
    if request.method == "POST":
        name = request.form.get("name", "Custom BPE Tokenizer").strip()
        algo = request.form.get("algorithm", "bpe").strip()
        vocab_size = int(request.form.get("vocab_size", 4000))
        dataset_id = request.form.get("dataset_id")
        training_text = request.form.get("training_text", "").strip()

        storage = get_storage_backend()
        texts: list[str] = []

        if dataset_id:
            with get_session() as session:
                ds = session.get(Dataset, dataset_id)
                if ds:
                    path = storage.get_url(ds.processed_key or ds.storage_key)
                    for rec in DatasetIngestion.read_records(path, max_rows=500):
                        texts.append(rec.get("text", ""))

        if not texts:
            if training_text:
                texts = training_text.splitlines()
            else:
                texts = [
                    "AI Foundry Enterprise provides high-throughput distributed foundation model pipelines.",
                    "Hybrid transformer and state space architectures enable linear scaling long-context inference.",
                    "Mixture of Experts routes tokens dynamically across sparse feedforward sub-networks.",
                    "Grouped Query Attention reduces key-value cache memory footprints during high concurrency serving.",
                ] * 20

        # Train tokenizer
        tok = BaseTokenizer(vocab_size=vocab_size, algorithm=algo)
        tok.train_from_texts(texts, target_vocab_size=vocab_size)

        # Analytics
        metrics = TokenizerAnalytics.evaluate_corpus(tok, texts[:20])

        tok_id = str(uuid.uuid4())
        storage_key = f"tokenizers/{tok_id}/tokenizer.json"
        
        # Save serialized model
        storage.write(storage_key, json.dumps(tok.to_dict(), indent=2).encode("utf-8"))

        with get_session() as session:
            model_rec = TokenizerModel(
                id=tok_id,
                name=name,
                slug=name.lower().replace(" ", "-")[:64],
                algorithm=algo,
                vocab_size=len(tok.vocab),
                storage_key=storage_key,
                compression_ratio=metrics.bytes_per_token,
                cjk_coverage_pct=metrics.cjk_coverage_pct,
                status="ready",
                special_tokens_json=json.dumps(tok.special_tokens),
            )
            session.add(model_rec)

        flash(f"Tokenizer '{name}' trained successfully with {len(tok.vocab)} tokens!", "success")
        return redirect(url_for("tokenizer.detail_tokenizer", tokenizer_id=tok_id))

    with get_session() as session:
        datasets = session.query(Dataset).filter(Dataset.status == "ready").all()
        ds_list = [{"id": d.id, "name": d.name} for d in datasets]

    return render_template("tokenizer/new.html", datasets=ds_list)


@tokenizer_bp.route("/<tokenizer_id>")
def detail_tokenizer(tokenizer_id: str):
    """Interactive Tokenizer Studio workspace."""
    storage = get_storage_backend()
    
    with get_session() as session:
        tok_rec = session.get(TokenizerModel, tokenizer_id)
        if not tok_rec:
            flash("Tokenizer not found", "danger")
            return redirect(url_for("tokenizer.list_tokenizers"))

        raw_json = storage.read(tok_rec.storage_key).decode("utf-8")
        tok_dict = json.loads(raw_json)
        tok = BaseTokenizer.from_dict(tok_dict)

        tok_data = {
            "id": tok_rec.id,
            "name": tok_rec.name,
            "slug": tok_rec.slug,
            "algorithm": tok_rec.algorithm,
            "vocab_size": tok_rec.vocab_size,
            "compression_ratio": tok_rec.compression_ratio,
            "cjk_coverage_pct": tok_rec.cjk_coverage_pct,
            "status": tok_rec.status,
            "created_at": tok_rec.created_at,
            "merges_count": len(tok.merges),
            "sample_vocab": list(tok.vocab.items())[:60],
        }

    return render_template("tokenizer/detail.html", tokenizer=tok_data)


@tokenizer_bp.route("/<tokenizer_id>/tokenize", methods=["POST"])
def live_tokenize(tokenizer_id: str):
    """JSON API for live interactive tokenization visualizer."""
    storage = get_storage_backend()
    text = request.json.get("text", "") if request.is_json else request.form.get("text", "")

    with get_session() as session:
        tok_rec = session.get(TokenizerModel, tokenizer_id)
        if not tok_rec:
            return jsonify({"error": "Tokenizer not found"}), 404

        raw_json = storage.read(tok_rec.storage_key).decode("utf-8")
        tok = BaseTokenizer.from_dict(json.loads(raw_json))
        encoded = tok.encode(text)

        # Palette colors for visualization
        colors = [
            "rgba(108, 92, 231, 0.25)", "rgba(0, 210, 211, 0.25)",
            "rgba(0, 184, 148, 0.25)", "rgba(253, 203, 110, 0.25)",
            "rgba(225, 112, 85, 0.25)", "rgba(116, 185, 255, 0.25)",
        ]

        token_spans = []
        for idx, (token_str, tid) in enumerate(zip(encoded.tokens, encoded.token_ids)):
            color = colors[idx % len(colors)]
            token_spans.append({
                "index": idx,
                "token": token_str,
                "id": tid,
                "color": color,
            })

        return jsonify({
            "token_ids": encoded.token_ids,
            "token_count": encoded.token_count,
            "char_count": encoded.char_count,
            "bytes_per_token": encoded.bytes_per_token,
            "spans": token_spans,
        })


@tokenizer_bp.route("/<tokenizer_id>/export/<fmt>")
def export_tokenizer(tokenizer_id: str, fmt: str):
    """Download exported tokenizer parameters."""
    storage = get_storage_backend()
    with get_session() as session:
        tok_rec = session.get(TokenizerModel, tokenizer_id)
        if not tok_rec:
            return "Not found", 404
        raw_json = storage.read(tok_rec.storage_key).decode("utf-8")
        tok = BaseTokenizer.from_dict(json.loads(raw_json))

    if fmt == "huggingface":
        content = TokenizerExporter.to_huggingface_json(tok)
        return Response(content, mimetype="application/json", headers={"Content-Disposition": f"attachment; filename={tok_rec.slug}_tokenizer.json"})
    elif fmt == "gguf":
        meta = TokenizerExporter.to_gguf_metadata(tok)
        return jsonify(meta)
    else:
        return Response(json.dumps(tok.vocab, indent=2), mimetype="application/json", headers={"Content-Disposition": f"attachment; filename={tok_rec.slug}_vocab.json"})
