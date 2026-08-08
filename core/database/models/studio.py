"""
AFE — Studios Database Models

Stores Chat Sessions, Messages, Prompt Engineering Templates,
Autonomous Agent Definitions, and MCP Server Connectors.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import BigInteger, Float, ForeignKey, Integer, String, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from core.database.base import Base, TimestampMixin, UUIDMixin


class ConversationSession(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "conversation_sessions"

    title: Mapped[str] = mapped_column(String(256), default="New Chat")
    model_name: Mapped[str] = mapped_column(String(128), default="afe-hybrid-7b")
    system_prompt: Mapped[Optional[str]] = mapped_column(Text, default="You are AFE, a high-performance foundation model assistant.")
    temperature: Mapped[float] = mapped_column(Float, default=0.7)
    top_p: Mapped[float] = mapped_column(Float, default=0.9)
    total_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    
    project_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    def __repr__(self) -> str:
        return f"<ConversationSession {self.title} model={self.model_name}>"


class ChatMessage(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "chat_messages"

    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("conversation_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(32), default="user", comment="user, assistant, system, tool")
    content: Mapped[Text] = mapped_column(Text, default="")
    tokens: Mapped[int] = mapped_column(Integer, default=0)

    def __repr__(self) -> str:
        return f"<ChatMessage {self.role} tokens={self.tokens}>"


class PromptTemplate(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "prompt_templates"

    name: Mapped[str] = mapped_column(String(256), nullable=False)
    slug: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, default="")
    system_prompt: Mapped[Optional[str]] = mapped_column(Text, default="")
    user_template: Mapped[Text] = mapped_column(Text, nullable=False, comment="Template containing {{variables}}")
    version: Mapped[str] = mapped_column(String(32), default="v1.0.0")
    variables_json: Mapped[Text] = mapped_column(Text, default="[]")
    created_by: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    def __repr__(self) -> str:
        return f"<PromptTemplate {self.name} {self.version}>"


class AgentDefinition(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "agent_definitions"

    name: Mapped[str] = mapped_column(String(256), nullable=False)
    slug: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    role_description: Mapped[Text] = mapped_column(Text, nullable=False)
    system_instructions: Mapped[Text] = mapped_column(Text, default="")
    model_name: Mapped[str] = mapped_column(String(128), default="afe-hybrid-7b")
    max_loops: Mapped[int] = mapped_column(Integer, default=8)
    tools_json: Mapped[Text] = mapped_column(Text, default="[]", comment="Registered tool specs")
    mcp_servers_json: Mapped[Text] = mapped_column(Text, default="[]", comment="Linked MCP server IDs")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_by: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    def __repr__(self) -> str:
        return f"<AgentDefinition {self.name}>"


class MCPServerConfig(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "mcp_servers"

    name: Mapped[str] = mapped_column(String(256), nullable=False)
    slug: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    transport: Mapped[str] = mapped_column(String(32), default="stdio", comment="stdio, sse")
    command_or_url: Mapped[str] = mapped_column(String(512), nullable=False)
    tools_json: Mapped[Text] = mapped_column(Text, default="[]", comment="Exported tools list")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_by: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)

    def __repr__(self) -> str:
        return f"<MCPServerConfig {self.name} ({self.transport})>"
