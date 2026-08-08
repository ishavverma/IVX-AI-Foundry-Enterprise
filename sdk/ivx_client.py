"""Small typed client generated from the stable IVX OpenAPI surface."""

from __future__ import annotations

from typing import Any

import requests


class IVXClient:
    def __init__(self, base_url: str, token: str | None = None, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def health(self) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def chat_completion(self, messages: list[dict[str, str]], model: str = "afe-hybrid-7b", **options: Any) -> dict[str, Any]:
        payload = {"model": model, "messages": messages, **options}
        response = self.session.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
