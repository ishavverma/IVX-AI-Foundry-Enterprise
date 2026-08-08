"""
AFE — Model Context Protocol (MCP) Blueprint

Manages MCP tool server connectors (stdio / SSE) and dynamic capability discovery.
"""

from __future__ import annotations

import json
from flask import Blueprint, flash, redirect, render_template, request, url_for, jsonify
from core.tenancy import TenantService

from core.database.engine import get_session
from core.database.models.studio import MCPServerConfig
from engines.agent.mcp_client import MCPClient

mcp_bp = Blueprint("mcp", __name__, url_prefix="/mcp")


@mcp_bp.route("/")
def list_mcp():
    """List all registered MCP tool server connectors."""
    with get_session() as session:
        # Seed default MCP server if empty
        if session.query(MCPServerConfig).count() == 0:
            default_mcp = MCPServerConfig(
                name="Local Development Tools MCP",
                slug="local-dev-tools",
                transport="stdio",
                command_or_url="npx -y @modelcontextprotocol/server-filesystem",
                tools_json=json.dumps(["read_file", "write_file", "list_dir"]),
                is_active=True,
            )
            session.add(default_mcp)
            session.commit()

        servers = TenantService.scope(session.query(MCPServerConfig), MCPServerConfig, session).all()
        server_list = []
        for s in servers:
            server_list.append({
                "id": s.id,
                "name": s.name,
                "slug": s.slug,
                "transport": s.transport,
                "command": s.command_or_url,
                "tools": json.loads(s.tools_json or "[]"),
                "is_active": s.is_active,
            })

    return render_template("mcp/list.html", servers=server_list)


@mcp_bp.route("/discover", methods=["POST"])
def discover_tools():
    """Discover tools exposed by a local command or remote SSE endpoint."""
    command = request.form.get("command", "")
    transport = request.form.get("transport", "stdio")
    tools = MCPClient.discover_tools(command, transport)
    return jsonify([{"name": t.name, "description": t.description} for t in tools])
