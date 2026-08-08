"""Optional Flask-Limiter integration with a safe local fallback."""

from __future__ import annotations

from collections import defaultdict
from functools import wraps
from threading import Lock
import time
from typing import Any, Callable

try:
    from flask_limiter import Limiter as _Limiter
    from flask_limiter.util import get_remote_address
except ImportError:  # pragma: no cover - exercised only in minimal installs
    _Limiter = None
    get_remote_address = lambda: "local"


class _FallbackLimiter:
    def __init__(self) -> None:
        self._hits: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._lock = Lock()

    def init_app(self, app: Any) -> None:
        app.extensions["limiter"] = self

    def limit(self, limit: str) -> Callable:
        parts = limit.lower().replace(" per ", "/").split("/")
        count_text, window_text = parts[0], parts[1]
        count = int(count_text.strip())
        duration = window_text.strip()
        named_windows = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}
        if duration in named_windows or duration.rstrip("s") in named_windows:
            window_seconds = named_windows[duration.rstrip("s")]
        else:
            window = int(duration.rstrip("smhd"))
            suffix = duration[-1]
            window_seconds = window * {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(suffix, 60)

        def decorator(function: Callable) -> Callable:
            @wraps(function)
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                from flask import abort, request

                key = (request.remote_addr or "unknown", function.__name__)
                now = time.monotonic()
                with self._lock:
                    self._hits[key] = [stamp for stamp in self._hits[key] if now - stamp < window_seconds]
                    if len(self._hits[key]) >= count:
                        abort(429)
                    self._hits[key].append(now)
                return function(*args, **kwargs)

            return wrapped

        return decorator


limiter = (
    _Limiter(key_func=get_remote_address, storage_uri="memory://")
    if _Limiter is not None
    else _FallbackLimiter()
)
