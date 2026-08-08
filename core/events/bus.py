"""
AFE — In-Process Event Bus

A lightweight publish/subscribe event bus that decouples platform
components.  Handlers are registered by event name and invoked
synchronously in the order they were registered.

Usage:
    from core.events import event_bus

    @event_bus.on("training.completed")
    def handle_training_done(payload):
        print(f"Training finished: {payload}")

    event_bus.emit("training.completed", {"run_id": "abc", "loss": 0.42})
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable

logger = logging.getLogger(__name__)

EventHandler = Callable[..., Any]


class EventBus:
    """Simple synchronous publish/subscribe event bus."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._history: list[dict[str, Any]] = []
        self._max_history = 500

    # ── Subscribe ────────────────────────────────

    def on(self, event_name: str) -> Callable:
        """Decorator to register a handler for *event_name*."""
        def decorator(fn: EventHandler) -> EventHandler:
            self._handlers[event_name].append(fn)
            logger.debug("Registered handler %s for event '%s'", fn.__name__, event_name)
            return fn
        return decorator

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        """Imperatively register a handler for *event_name*."""
        self._handlers[event_name].append(handler)

    def unsubscribe(self, event_name: str, handler: EventHandler) -> None:
        """Remove a previously registered handler."""
        try:
            self._handlers[event_name].remove(handler)
        except ValueError:
            pass

    # ── Publish ──────────────────────────────────

    def emit(self, event_name: str, payload: Any = None) -> None:
        """Fire *event_name* and call all registered handlers with *payload*."""
        handlers = self._handlers.get(event_name, [])
        logger.info("Event '%s' → %d handler(s)", event_name, len(handlers))

        # Keep a bounded history
        self._history.append({"event": event_name, "payload": payload})
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        for handler in handlers:
            try:
                handler(payload)
            except Exception:
                logger.exception("Handler %s failed for event '%s'", handler.__name__, event_name)

    # ── Inspection ───────────────────────────────

    @property
    def registered_events(self) -> list[str]:
        return list(self._handlers.keys())

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def clear(self) -> None:
        """Remove all handlers and history (useful for tests)."""
        self._handlers.clear()
        self._history.clear()

    def history_for(self, job_id: str) -> list[dict[str, Any]]:
        """Return recent job lifecycle events for dashboard subscribers."""
        return [event for event in self._history if event.get("payload", {}).get("job_id") == job_id]


# ── Module-level singleton ───────────────────
event_bus = EventBus()
