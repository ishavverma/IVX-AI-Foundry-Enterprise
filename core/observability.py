"""Low-dependency observability primitives used by the Flask application."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from typing import Any


class JsonFormatter(logging.Formatter):
    """Render log records as one structured JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class Metrics:
    """Small Prometheus-compatible counter registry without a runtime dependency."""

    def __init__(self) -> None:
        self._counters: Counter[tuple[str, tuple[tuple[str, str], ...]]] = Counter()

    def inc(self, name: str, labels: dict[str, str] | None = None, value: int = 1) -> None:
        key = (name, tuple(sorted((labels or {}).items())))
        self._counters[key] += value

    def render(self) -> str:
        lines: list[str] = []
        for (name, labels), value in sorted(self._counters.items()):
            label_text = ""
            if labels:
                label_text = "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}"
            lines.append(f"{name}{label_text} {value}")
        return "\n".join(lines) + "\n"


metrics = Metrics()
