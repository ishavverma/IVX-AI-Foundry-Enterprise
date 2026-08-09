"""Public interface for the AFE Model Engine.

The package provides three main capabilities:

1. **Configuration & Presets** – ``ArchitectureConfig`` and ``MODEL_PRESETS``
   (see :pymod:`engines.model.config`).
2. **Analytical Calculator** – ``ArchitectureCalculator`` can compute parameter
   counts, FLOPs, VRAM estimates, etc. (see :pymod:`engines.model.calculator`).
3. **Builder / Exporter** – ``AFEHybridModel`` builds a runnable PyTorch
   model from a configuration dictionary, and ``ModelExporter`` can emit a
   self‑contained ``model.py`` and ``config.json`` pair.

In addition to re‑exporting the core symbols, this module offers a small helper
function ``load_model_from_storage`` that reads a stored ``config.json`` (using
the unified storage SDK) and returns an instantiated ``AFEHybridModel``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from engines.model.config import ArchitectureConfig, MODEL_PRESETS
from engines.model.calculator import ArchitectureCalculator, ArchitectureMetrics
from engines.model.builder import (
    RMSNorm,
    RotaryEmbedding,
    GroupedQueryAttention,
    SSMLayer,
    SwiGLUFeedForward,
    MoEFeedForward,
    HybridBlock,
    AFEHybridModel,
)
from engines.model.exporter import ModelExporter
from core.storage import get_storage_backend

logger = logging.getLogger(__name__)

__all__ = [
    "ArchitectureConfig",
    "MODEL_PRESETS",
    "ArchitectureCalculator",
    "ArchitectureMetrics",
    "RMSNorm",
    "RotaryEmbedding",
    "GroupedQueryAttention",
    "SSMLayer",
    "SwiGLUFeedForward",
    "MoEFeedForward",
    "HybridBlock",
    "AFEHybridModel",
    "ModelExporter",
    "load_model_from_storage",
]


def load_model_from_storage(storage_key: str) -> Any:
    """Load a model definition JSON from the storage backend and instantiate it.

    Parameters
    ----------
    storage_key: str
        The key (path) in the unified storage where the ``config.json`` lives.

    Returns
    -------
    Any
        An instantiated ``AFEHybridModel`` (or raises ``ValueError`` if the
        configuration is invalid).
    """
    storage = get_storage_backend()
    try:
        raw = storage.read(storage_key)
        cfg_dict: Dict[str, Any] = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # pragma: no cover – defensive
        logger.error("Failed to load model config from %s: %s", storage_key, exc)
        raise ValueError(f"Unable to read model config at {storage_key}") from exc

    # Validate against the ArchitectureConfig schema (simple field check).
    try:
        cfg = ArchitectureConfig.from_dict(cfg_dict)
    except Exception as exc:  # pragma: no cover
        logger.error("Invalid model configuration: %s", exc)
        raise ValueError("Model configuration validation failed") from exc

    model = AFEHybridModel(cfg.to_dict())
    logger.info("Model instantiated from storage key %s (config %s)", storage_key, cfg.name)
    return model

__all__ = [
    "ArchitectureConfig",
    "MODEL_PRESETS",
    "ArchitectureCalculator",
    "ArchitectureMetrics",
    "RMSNorm",
    "RotaryEmbedding",
    "GroupedQueryAttention",
    "SSMLayer",
    "SwiGLUFeedForward",
    "MoEFeedForward",
    "HybridBlock",
    "AFEHybridModel",
    "ModelExporter",
]
