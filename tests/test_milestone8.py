"""
AFE — Milestone 8 Verification Script

Tests Hardware Telemetry snapshot, Resource Allocator, and Enterprise Governance.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.database.engine import init_db, get_session
from core.database.models.project import Project
from core.database.models.audit import AuditLog
from core.resources.hardware import hardware_monitor
from core.resources.allocator import resource_allocator


def run_tests():
    print("=======================================================")
    print("  AFE MILESTONE 8: OBSERVABILITY & ENTERPRISE TEST     ")
    print("=======================================================")

    init_db()

    # ── 1. Test Hardware Telemetry ──
    print("\n1. Testing Hardware Telemetry Snapshot:")
    snap = hardware_monitor.snapshot()
    print(f"   CPU Usage: {snap.cpu.usage_percent}% ({snap.cpu.logical_cores} cores)")
    print(f"   RAM: {snap.memory.used_gb:.2f} / {snap.memory.total_gb:.2f} GB ({snap.memory.usage_percent}%)")
    print(f"   Disk: {snap.disk.used_gb:.2f} / {snap.disk.total_gb:.2f} GB ({snap.disk.usage_percent}%)")
    print(f"   GPUs detected: {len(snap.gpus)}")
    assert snap.cpu.logical_cores > 0
    assert snap.memory.total_gb > 0

    # ── 2. Test Resource Allocator ──
    print("\n2. Testing Resource Allocator:")
    alloc_summary = resource_allocator.summary()
    print(f"   Total GPUs: {alloc_summary['gpu_total']}, Active Leases: {alloc_summary['active_leases']}")
    assert "gpu_total" in alloc_summary

    # ── 3. Test Enterprise Projects & Audit Log ──
    print("\n3. Testing Enterprise Governance & Audit Logs:")
    with get_session() as session:
        audit = AuditLog(
            action="milestone8_verification_passed",
            resource_type="system",
            user_id=None,
            ip_address="127.0.0.1",
        )
        session.add(audit)
        session.commit()

        count = session.query(AuditLog).count()
        print(f"   Total Audit Events Logged: {count}")
        assert count > 0

    print("\n=======================================================")
    print("         ALL MILESTONE 8 UNIT TESTS PASSED!          ")
    print("=======================================================")


if __name__ == "__main__":
    run_tests()
