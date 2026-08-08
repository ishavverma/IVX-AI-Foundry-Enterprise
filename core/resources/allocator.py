"""
AFE — Resource Allocator

Manages reservation and release of computational resources (GPU devices,
CPU cores, RAM budgets) for background jobs.  Every job that needs a GPU
must acquire a lease through the allocator before starting; when the job
finishes the lease is released.

The allocator is in-process and thread-safe.  It keeps a lightweight
ledger of active leases so the dashboard and scheduler can see what is
reserved at a glance.

Design decisions:
  • GPU reservation is device-level (whole GPUs, not fractional).
  • RAM/CPU reservations are advisory budgets, not enforced cgroups.
  • The allocator consults the HardwareMonitor for live availability.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from core.config.settings import get_settings
from core.resources.hardware import HardwareMonitor, hardware_monitor

logger = logging.getLogger(__name__)
_settings = get_settings()


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class ResourceLease:
    """Tracks a single resource reservation."""
    lease_id: str
    job_id: str
    gpu_indices: List[int] = field(default_factory=list)
    cpu_cores: int = 0
    ram_gb: float = 0.0
    vram_mb: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""

    def to_dict(self) -> dict:
        import dataclasses
        d = dataclasses.asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d


@dataclass
class AllocationResult:
    """Returned by acquire() — tells the caller what they got."""
    success: bool
    lease_id: str = ""
    gpu_indices: List[int] = field(default_factory=list)
    reason: str = ""


# ──────────────────────────────────────────────
# Allocator
# ──────────────────────────────────────────────

class ResourceAllocator:
    """Thread-safe resource reservation manager."""

    def __init__(self, monitor: HardwareMonitor | None = None) -> None:
        self._monitor = monitor or hardware_monitor
        self._lock = threading.Lock()
        self._leases: Dict[str, ResourceLease] = {}  # lease_id → lease

    # ── Query ────────────────────────────────────

    @property
    def active_leases(self) -> List[ResourceLease]:
        with self._lock:
            return list(self._leases.values())

    @property
    def reserved_gpu_indices(self) -> set[int]:
        with self._lock:
            indices: set[int] = set()
            for lease in self._leases.values():
                indices.update(lease.gpu_indices)
            return indices

    def available_gpu_indices(self) -> List[int]:
        """Return GPU indices that are not currently reserved."""
        all_gpus = self._monitor.gpu_devices()
        reserved = self.reserved_gpu_indices
        return [g.index for g in all_gpus if g.index not in reserved]

    def total_reserved_ram_gb(self) -> float:
        with self._lock:
            return sum(l.ram_gb for l in self._leases.values())

    def total_reserved_cpu_cores(self) -> int:
        with self._lock:
            return sum(l.cpu_cores for l in self._leases.values())

    def summary(self) -> dict:
        """Dashboard-friendly summary of resource allocation."""
        snap = self._monitor.snapshot()
        available_gpus = self.available_gpu_indices()
        return {
            "gpu_total": snap.gpu_count,
            "gpu_available": len(available_gpus),
            "gpu_reserved": snap.gpu_count - len(available_gpus),
            "gpu_available_indices": available_gpus,
            "cpu_total_cores": snap.cpu.logical_cores,
            "cpu_reserved_cores": self.total_reserved_cpu_cores(),
            "ram_total_gb": snap.memory.total_gb,
            "ram_reserved_gb": round(self.total_reserved_ram_gb(), 2),
            "active_leases": len(self._leases),
        }

    # ── Acquire ──────────────────────────────────

    def acquire(
        self,
        job_id: str,
        gpu_count: int = 0,
        gpu_indices: List[int] | None = None,
        cpu_cores: int = 0,
        ram_gb: float = 0.0,
        vram_mb: int = 0,
        description: str = "",
    ) -> AllocationResult:
        """
        Reserve resources for a job.

        Parameters
        ----------
        job_id : str
            The TaskJob.id this reservation belongs to.
        gpu_count : int
            Number of GPUs to reserve (ignored if gpu_indices is set).
        gpu_indices : list[int] | None
            Specific GPU indices to reserve.  If None, the allocator
            picks the first ``gpu_count`` available GPUs.
        cpu_cores / ram_gb / vram_mb
            Advisory resource budgets.
        """
        with self._lock:
            # Determine which GPUs to assign
            assigned_gpus: List[int] = []

            if gpu_indices is not None:
                # Specific GPUs requested
                reserved = self.reserved_gpu_indices
                for idx in gpu_indices:
                    if idx in reserved:
                        return AllocationResult(
                            success=False,
                            reason=f"GPU {idx} is already reserved",
                        )
                assigned_gpus = list(gpu_indices)

            elif gpu_count > 0:
                available = self.available_gpu_indices()
                if len(available) < gpu_count:
                    return AllocationResult(
                        success=False,
                        reason=(
                            f"Requested {gpu_count} GPU(s) but only "
                            f"{len(available)} available: {available}"
                        ),
                    )
                assigned_gpus = available[:gpu_count]

            # Check RAM budget (advisory)
            if ram_gb > 0:
                snap = self._monitor.snapshot()
                remaining = snap.memory.available_gb - self.total_reserved_ram_gb()
                if ram_gb > remaining:
                    return AllocationResult(
                        success=False,
                        reason=f"Requested {ram_gb:.1f} GB RAM but only {remaining:.1f} GB available",
                    )

            # Check GPU temperature
            if assigned_gpus and self._monitor.is_gpu_overheating():
                return AllocationResult(
                    success=False,
                    reason=f"GPU temperature exceeds {_settings.MAX_GPU_TEMP_C}°C safety limit",
                )

            # Create the lease
            lease_id = str(uuid.uuid4())
            lease = ResourceLease(
                lease_id=lease_id,
                job_id=job_id,
                gpu_indices=assigned_gpus,
                cpu_cores=cpu_cores,
                ram_gb=ram_gb,
                vram_mb=vram_mb,
                description=description,
            )
            self._leases[lease_id] = lease

            logger.info(
                "Acquired lease %s for job %s — GPUs=%s, CPU=%d, RAM=%.1fGB",
                lease_id[:8], job_id[:8], assigned_gpus, cpu_cores, ram_gb,
            )
            return AllocationResult(
                success=True,
                lease_id=lease_id,
                gpu_indices=assigned_gpus,
            )

    # ── Release ──────────────────────────────────

    def release(self, lease_id: str) -> bool:
        """Release a resource lease.  Returns True if found and removed."""
        with self._lock:
            lease = self._leases.pop(lease_id, None)
            if lease:
                logger.info(
                    "Released lease %s (job=%s, GPUs=%s)",
                    lease_id[:8], lease.job_id[:8], lease.gpu_indices,
                )
                return True
            logger.warning("Lease %s not found for release", lease_id[:8])
            return False

    def release_by_job(self, job_id: str) -> int:
        """Release all leases belonging to a job.  Returns count released."""
        with self._lock:
            to_remove = [lid for lid, l in self._leases.items() if l.job_id == job_id]
            for lid in to_remove:
                del self._leases[lid]
            if to_remove:
                logger.info("Released %d lease(s) for job %s", len(to_remove), job_id[:8])
            return len(to_remove)

    def release_all(self) -> int:
        """Release every active lease (emergency / shutdown)."""
        with self._lock:
            count = len(self._leases)
            self._leases.clear()
            if count:
                logger.warning("Emergency release: cleared %d lease(s)", count)
            return count


# ── Module-level singleton ───────────────────
resource_allocator = ResourceAllocator()
