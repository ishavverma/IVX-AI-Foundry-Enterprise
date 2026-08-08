"""
AFE — Plugin Loader

Discovers and loads plugins from the configured plugins directory.
Each plugin is a Python package with a `plugin.py` module that
exposes a `register(app)` function called during startup.

Plugin directory layout:
    plugins/
    ├── my_custom_arch/
    │   ├── __init__.py
    │   └── plugin.py          ← must expose register(app)
    └── my_dataset_loader/
        ├── __init__.py
        └── plugin.py
"""

from __future__ import annotations

import importlib
import hashlib
import hmac
import json
import logging
import sys
from pathlib import Path
from typing import Any

from core.config.settings import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


class PluginMeta:
    """Metadata about a discovered plugin."""

    def __init__(self, name: str, path: Path, module: Any = None) -> None:
        self.name = name
        self.path = path
        self.module = module
        self.loaded = False
        self.error: str | None = None

    def __repr__(self) -> str:
        status = "loaded" if self.loaded else ("error" if self.error else "pending")
        return f"<Plugin {self.name} [{status}]>"


class PluginLoader:
    """Discovers and registers plugins from the plugins directory."""

    def __init__(self) -> None:
        self._plugins: dict[str, PluginMeta] = {}
        self._plugins_dir = Path(_settings.PLUGINS_DIR)

    def discover(self) -> list[PluginMeta]:
        """Scan the plugins directory and return metadata for each plugin found."""
        if not self._plugins_dir.exists():
            self._plugins_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created plugins directory: %s", self._plugins_dir)
            return []

        for child in sorted(self._plugins_dir.iterdir()):
            if child.is_dir() and (child / "plugin.py").exists():
                meta = PluginMeta(name=child.name, path=child)
                self._plugins[child.name] = meta

        logger.info("Discovered %d plugin(s)", len(self._plugins))
        return list(self._plugins.values())

    def load_all(self, app: Any = None) -> list[PluginMeta]:
        """Load all discovered plugins by calling their `register()` function."""
        if not self._plugins:
            self.discover()

        for meta in self._plugins.values():
            try:
                if _settings.PLUGIN_SIGNING_KEY and not self._verify_signature(meta.path):
                    meta.error = "signature verification failed"
                    logger.error("Rejected unsigned or invalid plugin '%s'", meta.name)
                    continue
                # Add the plugin path to sys.path so local imports work
                plugin_parent = str(meta.path.parent)
                if plugin_parent not in sys.path:
                    sys.path.insert(0, plugin_parent)

                module = importlib.import_module(f"{meta.name}.plugin")
                meta.module = module
                meta.loaded = True

                # Call register() if it exists
                register_fn = getattr(module, "register", None)
                if callable(register_fn):
                    register_fn(app)
                    logger.info("Plugin '%s' registered successfully", meta.name)
                else:
                    logger.warning("Plugin '%s' has no register() function", meta.name)

            except Exception as exc:
                meta.error = str(exc)
                logger.exception("Failed to load plugin '%s'", meta.name)

        return list(self._plugins.values())

    @staticmethod
    def _verify_signature(plugin_path: Path) -> bool:
        manifest_path = plugin_path / "manifest.json"
        if not manifest_path.exists():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        signature = manifest.pop("signature", "")
        payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        expected = hmac.new(_settings.PLUGIN_SIGNING_KEY.encode(), payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(signature, expected)

    @property
    def plugins(self) -> dict[str, PluginMeta]:
        return dict(self._plugins)


# ── Module-level singleton ───────────────────
plugin_loader = PluginLoader()
