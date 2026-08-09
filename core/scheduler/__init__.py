"""AFE — Scheduler utilities.

This module provides a **singleton** :class:`apscheduler.schedulers.background.BackgroundScheduler`
instance that can be started from the Flask application factory. It is used
to run periodic maintenance jobs (e.g., checkpoint cleanup) without requiring
external cron infrastructure.

Only the ``apscheduler`` package is required – it is already listed in
``requirements.txt``. The scheduler is started lazily the first time
``get_scheduler()`` is called and can be shut down via ``shutdown()``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)

_scheduler: Optional[BackgroundScheduler] = None


def get_scheduler() -> BackgroundScheduler:
    """Return a lazily‑initialised singleton scheduler.

    The scheduler runs in a daemon thread, so it will not block the Flask
    process from exiting. It is safe to call this function multiple times –
    the same instance is returned.
    """
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler()
        logger.info("BackgroundScheduler initialised")
    return _scheduler


def add_interval_job(func: Callable[[], None], seconds: int, job_id: str) -> None:
    """Schedule ``func`` to run every ``seconds`` seconds.

    If a job with the same ``job_id`` already exists it will be replaced.
    ``func`` must be a no‑argument callable; any exception is logged and the
    job continues to be scheduled.
    """
    scheduler = get_scheduler()
    trigger = IntervalTrigger(seconds=seconds, start_date=datetime.now(timezone.utc))
    scheduler.add_job(func, trigger, id=job_id, replace_existing=True)
    logger.info("Scheduled periodic job %s every %s seconds", job_id, seconds)


def shutdown() -> None:
    """Gracefully shut down the scheduler (called on Flask teardown)."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        logger.info("BackgroundScheduler shut down")
        _scheduler = None
