"""
AFE — Chat Studio App Blueprint

Interactive ChatGPT-like multi-turn conversational interface with
model selection, system prompt drawers, and streaming message exchanges.
"""

from __future__ import annotations

import json
import uuid
from flask import Blueprint, flash, redirect, render_template, request, url_for, jsonify
from flask_login import current_user

from core.database.engine import get_session
from core.database.models.studio import ConversationSession, ChatMessage
from core.database.models.model_def import ModelDefinition
from engines.inference.server import InferenceServer
from core.tenancy import TenantService

chat_bp = Blueprint("chat", __name__, url_prefix="/chat")


@chat_bp.route("/")
def index():
    """Main Chat Studio interface."""
    with get_session() as session:
        sessions = TenantService.scope(session.query(ConversationSession), ConversationSession, session).order_by(ConversationSession.created_at.desc()).all()
        session_list = [{"id": s.id, "title": s.title, "model": s.model_name} for s in sessions]
        
        # Load or create active session
        active_id = request.args.get("session_id")
        active_sess = TenantService.scope(session.query(ConversationSession).filter(ConversationSession.id == active_id), ConversationSession, session).first() if active_id else None
        if not active_sess and sessions:
            active_sess = sessions[0]
        elif not active_sess:
            active_sess = ConversationSession(title="New Conversation", model_name="afe-hybrid-7b", created_by=current_user.get_id(), user_id=current_user.get_id())
            session.add(active_sess)
            session.commit()

        messages = (
            session.query(ChatMessage)
            .filter(ChatMessage.session_id == active_sess.id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )
        msg_list = [{"id": m.id, "role": m.role, "content": m.content, "tokens": m.tokens} for m in messages]

        models = TenantService.scope(session.query(ModelDefinition), ModelDefinition, session).all()
        model_names = [m.name for m in models] or ["AFE-Base-7B Hybrid", "AFE-Small-2B"]

    return render_template(
        "chat/index.html",
        sessions=session_list,
        active_session=active_sess,
        messages=msg_list,
        models=model_names,
    )


@chat_bp.route("/new_session", methods=["POST"])
def new_session():
    """Create a new conversational chat thread."""
    with get_session() as session:
        title = request.form.get("title", "").strip() or "New Conversation"
        model_name = request.form.get("model_name", "AFE-Base-7B Hybrid").strip()
        system_prompt = request.form.get("system_prompt", "You are AFE, a high-performance foundation model assistant.").strip()
        
        s = ConversationSession(
            title=title,
            model_name=model_name,
            system_prompt=system_prompt,
            created_by=current_user.get_id(),
            user_id=current_user.get_id(),
        )
        session.add(s)
        session.flush()
        return redirect(url_for("chat.index", session_id=s.id))


@chat_bp.route("/delete/<session_id>", methods=["POST"])
def delete_session(session_id: str):
    """Delete a conversation session and all its messages."""
    with get_session() as session:
        sess = TenantService.scope(session.query(ConversationSession).filter(ConversationSession.id == session_id), ConversationSession, session).first()
        if sess:
            session.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
            session.delete(sess)
    return redirect(url_for("chat.index"))


@chat_bp.route("/clear/<session_id>", methods=["POST"])
def clear_session(session_id: str):
    """Clear all chat messages for a specific session."""
    with get_session() as session:
        sess = TenantService.scope(session.query(ConversationSession).filter(ConversationSession.id == session_id), ConversationSession, session).first()
        if sess:
            session.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
            sess.total_tokens = 0
    return redirect(url_for("chat.index", session_id=session_id))


@chat_bp.route("/send", methods=["POST"])
def send_message():
    """Post a message and return the generated assistant response."""
    data = request.json or {}
    session_id = data.get("session_id")
    user_text = data.get("message", "").strip()

    if not user_text or not session_id:
        return jsonify({"error": "Missing message or session_id"}), 400

    with get_session() as session:
        owned_session = TenantService.scope(session.query(ConversationSession).filter(ConversationSession.id == session_id), ConversationSession, session).first()
        if not owned_session:
            return jsonify({"error": "Conversation not found"}), 404

        # Save user message
        u_tokens = len(user_text.split())
        u_msg = ChatMessage(session_id=session_id, role="user", content=user_text, tokens=u_tokens)
        session.add(u_msg)

        # Build message context
        messages_payload = []
        if owned_session.system_prompt:
            messages_payload.append({"role": "system", "content": owned_session.system_prompt})

        prior_messages = (
            session.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )
        for msg in prior_messages:
            messages_payload.append({"role": msg.role, "content": msg.content})
        messages_payload.append({"role": "user", "content": user_text})

        # Generate response
        resp = InferenceServer.generate_complete(
            messages=messages_payload,
            temperature=owned_session.temperature or 0.7,
            model_name=owned_session.model_name or "afe-hybrid-7b",
        )
        assistant_content = resp["choices"][0]["message"]["content"]
        a_tokens = resp.get("usage", {}).get("completion_tokens", len(assistant_content.split()))
        
        a_msg = ChatMessage(session_id=session_id, role="assistant", content=assistant_content, tokens=a_tokens)
        session.add(a_msg)

        owned_session.total_tokens = (owned_session.total_tokens or 0) + u_tokens + a_tokens

    return jsonify({
        "role": "assistant",
        "content": assistant_content,
        "tokens": a_tokens,
    })
