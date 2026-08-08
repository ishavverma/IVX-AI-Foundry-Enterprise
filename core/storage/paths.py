"""
AFE — Storage Path Resolver

Standardised directory layout inside the storage root.  Every platform
component uses these path helpers instead of constructing paths manually,
which guarantees a consistent, predictable storage hierarchy:

    storage/
    ├── datasets/
    │   └── {project_id}/{dataset_id}/
    ├── models/
    │   └── {project_id}/{model_id}/
    ├── checkpoints/
    │   └── {project_id}/{experiment_id}/{run_id}/
    ├── experiments/
    │   └── {project_id}/{experiment_id}/
    ├── tokenizers/
    │   └── {project_id}/{tokenizer_id}/
    ├── artifacts/
    │   └── {project_id}/{artifact_type}/{artifact_id}/
    ├── logs/
    │   └── {job_type}/{job_id}/
    ├── exports/
    │   └── {project_id}/
    ├── uploads/
    │   └── {user_id}/
    └── registry/
        └── {asset_type}/{asset_id}/
"""

from __future__ import annotations

import logging
from pathlib import Path

from core.config.settings import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


class StoragePaths:
    """Resolves canonical storage paths for every asset type."""

    def __init__(self, root: str | None = None) -> None:
        self._root = Path(root or _settings.STORAGE_ROOT)

    @property
    def root(self) -> Path:
        return self._root

    # ── Top-level directories ────────────────────

    def _ensure(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def datasets_root(self) -> Path:
        return self._ensure(self._root / "datasets")

    @property
    def models_root(self) -> Path:
        return self._ensure(self._root / "models")

    @property
    def checkpoints_root(self) -> Path:
        return self._ensure(self._root / "checkpoints")

    @property
    def experiments_root(self) -> Path:
        return self._ensure(self._root / "experiments")

    @property
    def tokenizers_root(self) -> Path:
        return self._ensure(self._root / "tokenizers")

    @property
    def artifacts_root(self) -> Path:
        return self._ensure(self._root / "artifacts")

    @property
    def logs_root(self) -> Path:
        return self._ensure(self._root / "logs")

    @property
    def exports_root(self) -> Path:
        return self._ensure(self._root / "exports")

    @property
    def uploads_root(self) -> Path:
        return self._ensure(self._root / "uploads")

    @property
    def registry_root(self) -> Path:
        return self._ensure(self._root / "registry")

    # ── Scoped paths ─────────────────────────────

    def dataset_dir(self, project_id: str, dataset_id: str) -> Path:
        return self._ensure(self.datasets_root / project_id / dataset_id)

    def model_dir(self, project_id: str, model_id: str) -> Path:
        return self._ensure(self.models_root / project_id / model_id)

    def checkpoint_dir(self, project_id: str, experiment_id: str, run_id: str) -> Path:
        return self._ensure(self.checkpoints_root / project_id / experiment_id / run_id)

    def experiment_dir(self, project_id: str, experiment_id: str) -> Path:
        return self._ensure(self.experiments_root / project_id / experiment_id)

    def tokenizer_dir(self, project_id: str, tokenizer_id: str) -> Path:
        return self._ensure(self.tokenizers_root / project_id / tokenizer_id)

    def artifact_dir(self, project_id: str, artifact_type: str, artifact_id: str) -> Path:
        return self._ensure(self.artifacts_root / project_id / artifact_type / artifact_id)

    def job_log_dir(self, job_type: str, job_id: str) -> Path:
        return self._ensure(self.logs_root / job_type / job_id)

    def export_dir(self, project_id: str) -> Path:
        return self._ensure(self.exports_root / project_id)

    def upload_dir(self, user_id: str) -> Path:
        return self._ensure(self.uploads_root / user_id)

    def registry_dir(self, asset_type: str, asset_id: str) -> Path:
        return self._ensure(self.registry_root / asset_type / asset_id)

    # ── Convenience: relative key for Storage SDK ─

    def dataset_key(self, project_id: str, dataset_id: str, filename: str) -> str:
        return f"datasets/{project_id}/{dataset_id}/{filename}"

    def model_key(self, project_id: str, model_id: str, filename: str) -> str:
        return f"models/{project_id}/{model_id}/{filename}"

    def checkpoint_key(self, project_id: str, experiment_id: str, run_id: str, filename: str) -> str:
        return f"checkpoints/{project_id}/{experiment_id}/{run_id}/{filename}"

    def tokenizer_key(self, project_id: str, tokenizer_id: str, filename: str) -> str:
        return f"tokenizers/{project_id}/{tokenizer_id}/{filename}"

    # ── Initialise all top-level dirs ────────────

    def init_all(self) -> None:
        """Pre-create every top-level storage directory."""
        for prop_name in [
            "datasets_root", "models_root", "checkpoints_root",
            "experiments_root", "tokenizers_root", "artifacts_root",
            "logs_root", "exports_root", "uploads_root", "registry_root",
        ]:
            getattr(self, prop_name)
        logger.info("Storage directories initialised at %s", self._root)


# ── Module-level singleton ───────────────────
storage_paths = StoragePaths()
