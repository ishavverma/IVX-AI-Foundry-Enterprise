"""
AFE — Agent Studio Blueprint

Autonomous ReAct agents with tool calling, MCP connections, and multi-step reasoning.
"""

from __future__ import annotations

import json
import uuid
from flask import Blueprint, flash, redirect, render_template, request, url_for, jsonify
from flask_login import current_user

from core.database.engine import get_session
from core.database.models.studio import AgentDefinition, MCPServerConfig
from engines.agent.runtime import AgentRuntime
from core.tenancy import TenantService

agents_bp = Blueprint("agents", __name__, url_prefix="/agents")


@agents_bp.route("/")
def list_agents():
    """List all configured autonomous agents."""
    with get_session() as session:
        # Seed default agent if empty
        if session.query(AgentDefinition).count() == 0:
            default_agent = AgentDefinition(
                name="AFE Research & Code Assistant",
                slug="research-code-assistant",
                role_description="Autonomous coding, math verification, and documentation research agent.",
                system_instructions="You are an autonomous engineering assistant with access to python execution and database queries.",
                model_name="afe-hybrid-7b",
                tools_json=json.dumps(["calculator", "code_eval", "search_docs"]),
            )
            session.add(default_agent)
            session.commit()

        agents = TenantService.scope(session.query(AgentDefinition), AgentDefinition, session).order_by(AgentDefinition.created_at.desc()).all()
        agent_list = []
        for a in agents:
            agent_list.append({
                "id": a.id,
                "name": a.name,
                "slug": a.slug,
                "role": a.role_description,
                "model": a.model_name,
                "tools": json.loads(a.tools_json or "[]"),
                "is_active": a.is_active,
            })

    return render_template("agents/list.html", agents=agent_list)


@agents_bp.route("/new", methods=["GET", "POST"])
def new_agent():
    """Create a new autonomous agent definition."""
    if request.method == "POST":
        name = request.form.get("name", "Custom Agent").strip()
        role = request.form.get("role_description", "").strip()
        instructions = request.form.get("system_instructions", "").strip()
        model_name = request.form.get("model_name", "afe-hybrid-7b")
        tools = request.form.getlist("tools")

        with get_session() as session:
            agent = AgentDefinition(
                name=name,
                slug=name.lower().replace(" ", "-")[:64],
                role_description=role,
                system_instructions=instructions,
                model_name=model_name,
                tools_json=json.dumps(tools),
                created_by=current_user.get_id(),
            )
            session.add(agent)

        flash(f"Agent '{name}' created successfully!", "success")
        return redirect(url_for("agents.list_agents"))

    return render_template("agents/new.html")


@agents_bp.route("/run", methods=["POST"])
def run_agent():
    """Execute an interactive multi-step ReAct agent task."""
    data = request.json or {}
    prompt = data.get("prompt", "")
    instructions = data.get("instructions", "You are an autonomous agent.")
    
    result = AgentRuntime.execute_react_loop(prompt, system_instructions=instructions)
    return jsonify({
        "final_answer": result.final_answer,
        "steps": [{"thought": s.thought, "action": s.action, "observation": s.observation} for s in result.steps],
        "total_tokens": result.total_tokens,
    })
