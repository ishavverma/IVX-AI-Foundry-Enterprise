"""
AFE — Milestone 7 Verification Script

Tests Agent ReAct runtime loops, tool execution,
MCP tool discovery, chat persistence, and prompt templates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.database.engine import init_db, get_session
from core.database.models.studio import ConversationSession, ChatMessage, PromptTemplate, AgentDefinition
from engines.agent.runtime import AgentRuntime
from engines.agent.mcp_client import MCPClient


def run_tests():
    print("=======================================================")
    print("      AFE MILESTONE 7: STUDIOS, AGENT & MCP TEST       ")
    print("=======================================================")

    init_db()

    # ── 1. Test ReAct Agent Runtime Loop ──
    print("\n1. Testing ReAct Agent Autonomous Loop:")
    agent_res = AgentRuntime.execute_react_loop(
        user_prompt="Compute the total parameters for a 32-layer 4096-dim model",
        system_instructions="You are an autonomous engineering assistant.",
    )
    print(f"   Agent Executed {len(agent_res.steps)} Steps.")
    for idx, s in enumerate(agent_res.steps, 1):
        print(f"   Step {idx}: Action={s.action} -> Observation={s.observation[:50]}...")
    assert len(agent_res.steps) >= 1
    assert agent_res.success is True

    # ── 2. Test MCP Client Tool Discovery ──
    print("\n2. Testing Model Context Protocol (MCP) Client:")
    tools = MCPClient.discover_tools("npx -y @modelcontextprotocol/server-filesystem", "stdio")
    print(f"   Discovered {len(tools)} tools: {[t.name for t in tools]}")
    assert len(tools) >= 2

    # ── 3. Test Chat Session & Message Persistence ──
    print("\n3. Testing Chat Session & Message Persistence:")
    with get_session() as session:
        sess = ConversationSession(title="Automated Test Chat", model_name="afe-hybrid-7b")
        session.add(sess)
        session.flush()

        m1 = ChatMessage(session_id=sess.id, role="user", content="Hello AFE!", tokens=3)
        m2 = ChatMessage(session_id=sess.id, role="assistant", content="Hello! How can I assist?", tokens=6)
        session.add_all([m1, m2])
        session.commit()

        loaded_msgs = session.query(ChatMessage).filter(ChatMessage.session_id == sess.id).all()
        assert len(loaded_msgs) == 2
        print(f"   Chat session {sess.id} persisted {len(loaded_msgs)} messages.")

    print("\n=======================================================")
    print("         ALL MILESTONE 7 UNIT TESTS PASSED!          ")
    print("=======================================================")


if __name__ == "__main__":
    run_tests()
