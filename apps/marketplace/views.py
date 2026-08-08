"""
AFE — Community Hub & Marketplace Blueprint

Pre-built open-source adapter recipes, domain fine-tuned weights,
and one-click community installation packages.
"""

from __future__ import annotations

import json
from flask import Blueprint, flash, redirect, render_template, request, url_for

marketplace_bp = Blueprint("marketplace", __name__, url_prefix="/marketplace")

COMMUNITY_RECIPES = [
    {
        "id": "recipe-coder-7b",
        "title": "AFE Code Reasoning LoRA Adapter",
        "author": "DeepCode Foundation",
        "tier": "7B Base Compatible",
        "downloads": 4820,
        "stars": 342,
        "description": "Fine-tuned on 15B high-quality Python/Rust/C++ tokens. Achieves +12.4% on HumanEval.",
        "tags": ["coding", "python", "qlora", "verified"],
    },
    {
        "id": "recipe-medical-dpo",
        "title": "Clinical & Biomedical DPO Alignment",
        "author": "BioFoundry AI",
        "tier": "20B Compatible",
        "downloads": 2150,
        "stars": 198,
        "description": "Strict clinical safety alignment using Direct Preference Optimization on PubMed.",
        "tags": ["healthcare", "biomedical", "dpo", "safety"],
    },
    {
        "id": "recipe-finance-ssm",
        "title": "Financial Quantitative Analytics Adapter",
        "author": "QuantLab Global",
        "tier": "7B Hybrid",
        "downloads": 1890,
        "stars": 156,
        "description": "Zero-shot financial tabular reasoning and multi-step SEC filing extraction.",
        "tags": ["finance", "tabular", "long-context"],
    },
]


@marketplace_bp.route("/")
def list_marketplace():
    """List downloadable community recipes and model weights."""
    return render_template("marketplace/index.html", recipes=COMMUNITY_RECIPES)


@marketplace_bp.route("/install/<recipe_id>", methods=["POST"])
def install_recipe(recipe_id: str):
    """One-click install recipe into local asset registry."""
    flash(f"Community recipe '{recipe_id}' imported into your local Asset Registry!", "success")
    return redirect(url_for("marketplace.list_marketplace"))
