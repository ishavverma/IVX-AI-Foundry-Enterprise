"""
AFE — Lightweight Background Task Queue

Uses a thread-pool executor to run background tasks without requiring
external infrastructure (Celery / Redis).  Each submitted task is
persisted to the TaskJob database table so progress and results can
be tracked from the UI.

For production deployments, this can be swapped out for a Celery
backend by implementing the same `submit()` / `cancel()` interface.
"""

from __future__ import annotations

import logging
import traceback
import importlib
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from core.config.settings import get_settings
from core.database.engine import get_session
from core.database.models.task_job import JobStatus, TaskJob
from core.events.bus import event_bus

logger = logging.getLogger(__name__)
_settings = get_settings()

try:
    from celery import Celery
except ImportError:  # pragma: no cover - optional production dependency
    Celery = None  # type: ignore[assignment,misc]


def _resolve_callable(module_name: str, qualified_name: str) -> Callable[..., Any]:
    target: Any = importlib.import_module(module_name)
    for part in qualified_name.split("."):
        target = getattr(target, part)
    return target


if Celery is not None:
    celery_app = Celery("afe", broker=_settings.REDIS_URL, backend=_settings.REDIS_URL)

    @celery_app.task(name="afe.execute_job")
    def execute_celery_job(module_name: str, qualified_name: str, job_id: str, args: tuple, kwargs: dict) -> Any:
        task_fn = _resolve_callable(module_name, qualified_name)
        try:
            result = task_fn(job_id, *args, **kwargs)
            TaskQueue._update_job_status(job_id, JobStatus.COMPLETED, result=result)
            event_bus.emit("job.completed", {"job_id": job_id, "result": result})
            return result
        except Exception as exc:
            TaskQueue._update_job_status(job_id, JobStatus.FAILED, error=str(exc), traceback=traceback.format_exc())
            event_bus.emit("job.failed", {"job_id": job_id, "error": str(exc)})
            raise


class TaskQueue:
    """In-process background task executor."""

    def __init__(self, max_workers: int | None = None) -> None:
        self._max_workers = max_workers or _settings.MAX_WORKERS
        self._executor: Optional[ThreadPoolExecutor] = None
        self._futures: dict[str, Future] = {}

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None or self._executor._shutdown:
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="afe-worker",
            )
        return self._executor

    def submit(
        self,
        task_fn: Callable[..., Any],
        job_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """
        Submit a callable for background execution.

        Parameters
        ----------
        task_fn : callable
            The function to execute.  It receives (job_id, *args, **kwargs).
        job_id : str
            The TaskJob.id that this execution corresponds to.

        Returns
        -------
        str
            The job_id for tracking.
        """
        # Mark the job as RUNNING
        self._update_job_status(job_id, JobStatus.RUNNING)

        if _settings.QUEUE_BACKEND == "redis" and Celery is not None:
            execute_celery_job.delay(
                task_fn.__module__,
                task_fn.__qualname__,
                job_id,
                args,
                kwargs,
            )
            logger.info("Submitted job %s to Celery/Redis", job_id)
            return job_id

        executor = self._ensure_executor()

        future = executor.submit(self._run_wrapper, task_fn, job_id, *args, **kwargs)
        self._futures[job_id] = future
        logger.info("Submitted job %s to worker pool", job_id)
        return job_id

    def cancel(self, job_id: str) -> bool:
        """Attempt to cancel a queued/running job."""
        future = self._futures.get(job_id)
        if future and not future.done():
            cancelled = future.cancel()
            if cancelled:
                self._update_job_status(job_id, JobStatus.CANCELLED)
            return cancelled
        return False

    def shutdown(self, wait: bool = True) -> None:
        if self._executor:
            self._executor.shutdown(wait=wait)
            logger.info("Worker pool shut down (wait=%s)", wait)

    # ── Internal ─────────────────────────────────

    def _run_wrapper(
        self,
        task_fn: Callable[..., Any],
        job_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute the task, update the DB, and emit lifecycle events."""
        try:
            result = task_fn(job_id, *args, **kwargs)
            self._update_job_status(job_id, JobStatus.COMPLETED, result=result)
            event_bus.emit("job.completed", {"job_id": job_id, "result": result})
            return result
        except Exception as exc:
            tb = traceback.format_exc()
            self._update_job_status(job_id, JobStatus.FAILED, error=str(exc), traceback=tb)
            event_bus.emit("job.failed", {"job_id": job_id, "error": str(exc)})
            logger.exception("Job %s failed", job_id)
            raise

    @staticmethod
    def _update_job_status(
        job_id: str,
        status: JobStatus,
        result: Any = None,
        error: str | None = None,
        traceback: str | None = None,
    ) -> None:
        import json

        with get_session() as session:
            job = session.get(TaskJob, job_id)
            if not job:
                logger.warning("TaskJob %s not found for status update", job_id)
                return
            job.status = status.value
            now = datetime.now(timezone.utc)
            if status == JobStatus.RUNNING:
                job.started_at = now
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                job.completed_at = now
                job.progress = 1.0 if status == JobStatus.COMPLETED else job.progress
            if result is not None:
                try:
                    job.result_json = json.dumps(result, default=str)
                except (TypeError, ValueError):
                    job.result_json = json.dumps({"raw": str(result)})
            if error:
                job.error_message = f"{error}\n\n{traceback}" if traceback else error


# ── Module-level singleton ───────────────────
task_queue = TaskQueue()
