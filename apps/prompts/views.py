"""
AFE — Prompt Engineering Studio Blueprint

Template management, variable interpolation ({{var}}), and prompt versioning.
"""

from __future__ import annotations

import json
import uuid
from flask import Blueprint, flash, redirect, render_template, request, url_for, jsonify
from flask_login import current_user

from core.database.engine import get_session
from core.database.models.studio import PromptTemplate
from core.tenancy import TenantService

prompts_bp = Blueprint("prompts", __name__, url_prefix="/prompts")


@prompts_bp.route("/")
def list_prompts():
    """List all prompt templates."""
    with get_session() as session:
        # Seed default prompt templates if empty
        if session.query(PromptTemplate).count() == 0:
            default_prompts = [
                PromptTemplate(
                    name="Chain-of-Thought Reasoning",
                    slug="chain-of-thought",
                    description="Decomposes complex multi-step math and coding problems.",
                    system_prompt="You are a rigorous analytical AI. Always show your step-by-step reasoning before concluding.",
                    user_template="Problem: {{problem}}\nLet's think step by step:\n",
                    version="v1.0.0",
                    variables_json=json.dumps(["problem"]),
                ),
                PromptTemplate(
                    name="Structured JSON Extraction",
                    slug="json-extraction",
                    description="Extracts strongly-typed schema fields from unstructured documents.",
                    system_prompt="You output strictly valid JSON matching the requested schema without markdown backticks.",
                    user_template="Document:\n{{document}}\n\nExtract JSON matching schema: {{schema}}",
                    version="v1.1.0",
                    variables_json=json.dumps(["document", "schema"]),
                ),
            ]
            session.add_all(default_prompts)
            session.commit()

        templates = TenantService.scope(session.query(PromptTemplate), PromptTemplate, session).order_by(PromptTemplate.created_at.desc()).all()
        tpl_list = []
        for t in templates:
            tpl_list.append({
                "id": t.id,
                "name": t.name,
                "slug": t.slug,
                "description": t.description,
                "version": t.version,
                "system_prompt": t.system_prompt,
                "user_template": t.user_template,
                "variables": json.loads(t.variables_json or "[]"),
            })

    return render_template("prompts/list.html", prompts=tpl_list)


@prompts_bp.route("/new", methods=["GET", "POST"])
def new_prompt():
    """Create a new prompt template."""
    if request.method == "POST":
        name = request.form.get("name", "New Prompt Template").strip()
        system_prompt = request.form.get("system_prompt", "").strip()
        user_template = request.form.get("user_template", "").strip()
        description = request.form.get("description", "").strip()

        # Parse variables like {{var}}
        import re
        vars_found = re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", user_template)

        with get_session() as session:
            pt = PromptTemplate(
                name=name,
                slug=name.lower().replace(" ", "-")[:64],
                description=description,
                system_prompt=system_prompt,
                user_template=user_template,
                version="v1.0.0",
                variables_json=json.dumps(list(set(vars_found))),
                created_by=current_user.get_id(),
            )
            session.add(pt)

        flash(f"Prompt template '{name}' saved successfully!", "success")
        return redirect(url_for("prompts.list_prompts"))

    return render_template("prompts/new.html")
