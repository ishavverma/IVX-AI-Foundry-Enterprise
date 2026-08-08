"""
AFE — Central Asset Registry Blueprint

Catalog for foundation models, datasets, tokenizers, and checkpoint artifacts
with semantic versioning (v1.0.0, v1.1.0) and access controls.
"""

from __future__ import annotations

import json
import uuid
from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user

from core.database.engine import get_session
from core.database.models.evaluation import RegistryAsset
from core.database.models.model_def import ModelDefinition
from core.database.models.dataset import Dataset
from core.database.models.tokenizer import TokenizerModel
from core.tenancy import TenantService

registry_bp = Blueprint("registry", __name__, url_prefix="/registry")


@registry_bp.route("/")
def list_registry():
    """List all registered platform assets."""
    with get_session() as session:
        # Seed default catalog assets if empty
        if session.query(RegistryAsset).count() == 0:
            default_assets = [
                RegistryAsset(
                    name="AFE-Base-7B Hybrid Foundation Model",
                    slug="afe-base-7b",
                    asset_type="model",
                    version="v1.0.0",
                    license="Apache-2.0",
                    tags="hybrid, mamba, gqa, 32k-context",
                    description="Official 7B base foundation model combining 1:7 Attention-to-Mamba ratio with SwiGLU.",
                    storage_key="registry/models/afe-base-7b/",
                ),
                RegistryAsset(
                    name="Enterprise Global Pretraining Corpus 100M",
                    slug="enterprise-corpus-100m",
                    asset_type="dataset",
                    version="v1.2.0",
                    license="CC-BY-4.0",
                    tags="curated, deduplicated, quality-filtered",
                    description="Cleaned, deduplicated instruction and reasoning dataset.",
                    storage_key="registry/datasets/corpus-100m/",
                ),
                RegistryAsset(
                    name="AFE Universal Multilingual 64K BPE",
                    slug="afe-universal-64k",
                    asset_type="tokenizer",
                    version="v1.0.0",
                    license="MIT",
                    tags="bpe, byte-level, cjk-coverage-100",
                    description="64,000 token vocabulary optimized for code and multilingual text.",
                    storage_key="registry/tokenizers/universal-64k/",
                ),
            ]
            session.add_all(default_assets)
            session.commit()

        assets = TenantService.scope(session.query(RegistryAsset), RegistryAsset, session).order_by(RegistryAsset.created_at.desc()).all()
        asset_list = []
        for a in assets:
            asset_list.append({
                "id": a.id,
                "name": a.name,
                "slug": a.slug,
                "asset_type": a.asset_type,
                "version": a.version,
                "license": a.license,
                "tags": a.tags.split(",") if a.tags else [],
                "description": a.description,
                "storage_key": a.storage_key,
                "created_at": a.created_at,
            })

    return render_template("registry/list.html", assets=asset_list)
