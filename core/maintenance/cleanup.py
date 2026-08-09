"""Periodic maintenance tasks for the AFE platform.

Currently includes:
- ``cleanup_old_checkpoints`` – removes checkpoint records and their
  associated files that are older than a configurable retention period.

The functions are intended to be scheduled via ``core.scheduler``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from core.database.engine import get_session
from core.database.models.experiment import Checkpoint
from core.storage import get_storage_backend
from core.observability import metrics

logger = logging.getLogger(__name__)

# Retention period (days) – can be overridden via environment variable if needed.
RETENTION_DAYS = 30


def cleanup_old_checkpoints() -> None:
    """Delete checkpoint entries older than ``RETENTION_DAYS``.

    For each stale checkpoint we:
    1. Remove the file from the storage backend (if it exists).
    2. Delete the DB row.
    The operation is logged; failures on individual items do not abort the whole run.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
    storage = get_storage_backend()
    deleted = 0
    with get_session() as session:
        stale = (
            session.query(Checkpoint)
            .filter(Checkpoint.created_at < cutoff)
            .order_by(Checkpoint.created_at.asc())
            .all()
        )
        for ckpt in stale:
            try:
                if ckpt.storage_key:
                    storage.delete(ckpt.storage_key)
                session.delete(ckpt)
                deleted += 1
            except Exception as exc:  # pragma: no cover – defensive
                logger.error("Failed to delete checkpoint %s: %s", ckpt.id, exc)
        session.commit()
    logger.info("Checkpoint cleanup completed – %d records removed", deleted)
    metrics.inc("afe_checkpoint_cleanup_total", value=deleted)
