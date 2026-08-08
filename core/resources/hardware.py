"""
AFE — Hardware Monitor

Real-time system telemetry: CPU, RAM, disk, temperature, and GPU
(NVIDIA CUDA via pynvml, with graceful CPU-only fallback).

The monitor provides two interfaces:
  1. `snapshot()` — a single point-in-time reading of all hardware.
  2. `gpu_devices()` — detailed per-GPU information (name, VRAM, temp,
     utilization, driver version).

All values are returned as plain dataclasses so they can be trivially
serialised to JSON for the dashboard and REST API.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from typing import List, Optional

import psutil

from core.config.settings import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class CPUInfo:
    physical_cores: int
    logical_cores: int
    frequency_mhz: float
    usage_percent: float
    per_core_percent: List[float] = field(default_factory=list)


@dataclass
class MemoryInfo:
    total_gb: float
    available_gb: float
    used_gb: float
    usage_percent: float


@dataclass
class DiskInfo:
    total_gb: float
    used_gb: float
    free_gb: float
    usage_percent: float
    mount_point: str = ""


@dataclass
class GPUDevice:
    index: int
    name: str
    driver_version: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int
    vram_usage_percent: float
    temperature_c: int
    gpu_utilization_percent: int
    memory_utilization_percent: int
    power_draw_w: float
    power_limit_w: float
    compute_capability: str = ""


@dataclass
class SystemSnapshot:
    """Complete point-in-time snapshot of system hardware."""
    hostname: str
    os: str
    cpu: CPUInfo
    memory: MemoryInfo
    disk: DiskInfo
    gpus: List[GPUDevice] = field(default_factory=list)
    gpu_available: bool = False
    gpu_count: int = 0
    gpu_driver: str = ""
    cuda_version: str = ""
    temperatures: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Flat dict for JSON serialisation / template rendering."""
        import dataclasses
        return dataclasses.asdict(self)


# ──────────────────────────────────────────────
# Hardware Monitor
# ──────────────────────────────────────────────

class HardwareMonitor:
    """Reads hardware telemetry from the local machine."""

    def __init__(self) -> None:
        self._nvml_available = False
        self._init_nvml()

    # ── NVML Bootstrap ───────────────────────────

    def _init_nvml(self) -> None:
        """Try to initialise NVIDIA Management Library."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_available = True
            count = pynvml.nvmlDeviceGetCount()
            driver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode()
            logger.info("NVML initialised — %d GPU(s) detected, driver %s", count, driver)
        except Exception:
            self._nvml_available = False
            logger.info("NVML not available — running in CPU-only mode")

    # ── CPU ──────────────────────────────────────

    @staticmethod
    def cpu() -> CPUInfo:
        freq = psutil.cpu_freq()
        return CPUInfo(
            physical_cores=psutil.cpu_count(logical=False) or 1,
            logical_cores=psutil.cpu_count(logical=True) or 1,
            frequency_mhz=freq.current if freq else 0.0,
            usage_percent=psutil.cpu_percent(interval=0.1),
            per_core_percent=psutil.cpu_percent(interval=0.1, percpu=True),
        )

    # ── Memory ───────────────────────────────────

    @staticmethod
    def memory() -> MemoryInfo:
        mem = psutil.virtual_memory()
        gb = 1024 ** 3
        return MemoryInfo(
            total_gb=round(mem.total / gb, 2),
            available_gb=round(mem.available / gb, 2),
            used_gb=round(mem.used / gb, 2),
            usage_percent=mem.percent,
        )

    # ── Disk ─────────────────────────────────────

    @staticmethod
    def disk(path: str | None = None) -> DiskInfo:
        target = path or str(_settings.base_dir)
        usage = psutil.disk_usage(target)
        gb = 1024 ** 3
        return DiskInfo(
            total_gb=round(usage.total / gb, 2),
            used_gb=round(usage.used / gb, 2),
            free_gb=round(usage.free / gb, 2),
            usage_percent=usage.percent,
            mount_point=target,
        )

    # ── Temperature ──────────────────────────────

    @staticmethod
    def temperatures() -> dict:
        """Return CPU/chipset temperatures if available (Linux mainly)."""
        try:
            temps = psutil.sensors_temperatures()
            result = {}
            for chip, entries in temps.items():
                result[chip] = [
                    {"label": e.label or chip, "current": e.current, "high": e.high, "critical": e.critical}
                    for e in entries
                ]
            return result
        except (AttributeError, Exception):
            return {}

    # ── GPU ──────────────────────────────────────

    def gpu_devices(self) -> List[GPUDevice]:
        """Return per-GPU information via NVML."""
        if not self._nvml_available:
            return []

        import pynvml
        devices = []
        try:
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()

                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mb = 1024 ** 2

                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    temp = 0

                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = util.gpu
                    mem_util = util.memory
                except Exception:
                    gpu_util = 0
                    mem_util = 0

                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
                except Exception:
                    power = 0.0

                try:
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                except Exception:
                    power_limit = 0.0

                try:
                    cc_major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    compute_cap = f"{cc_major[0]}.{cc_major[1]}" if isinstance(cc_major, tuple) else str(cc_major)
                except Exception:
                    compute_cap = ""

                driver = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(driver, bytes):
                    driver = driver.decode()

                devices.append(GPUDevice(
                    index=i,
                    name=name,
                    driver_version=driver,
                    vram_total_mb=int(mem.total / mb),
                    vram_used_mb=int(mem.used / mb),
                    vram_free_mb=int(mem.free / mb),
                    vram_usage_percent=round((mem.used / mem.total) * 100, 1) if mem.total else 0,
                    temperature_c=temp,
                    gpu_utilization_percent=gpu_util,
                    memory_utilization_percent=mem_util,
                    power_draw_w=round(power, 1),
                    power_limit_w=round(power_limit, 1),
                    compute_capability=compute_cap,
                ))
        except Exception:
            logger.exception("Error reading GPU devices")

        return devices

    # ── Full Snapshot ────────────────────────────

    def snapshot(self) -> SystemSnapshot:
        """Return a complete hardware snapshot."""
        gpus = self.gpu_devices()

        cuda_ver = ""
        driver = ""
        if self._nvml_available:
            try:
                import pynvml
                driver = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(driver, bytes):
                    driver = driver.decode()
                cuda_ver = pynvml.nvmlSystemGetCudaDriverVersion_v2()
                # Convert packed int to version string (e.g. 12020 → 12.2)
                if isinstance(cuda_ver, int):
                    major = cuda_ver // 1000
                    minor = (cuda_ver % 1000) // 10
                    cuda_ver = f"{major}.{minor}"
                else:
                    cuda_ver = str(cuda_ver)
            except Exception:
                pass

        return SystemSnapshot(
            hostname=platform.node(),
            os=f"{platform.system()} {platform.release()}",
            cpu=self.cpu(),
            memory=self.memory(),
            disk=self.disk(),
            gpus=gpus,
            gpu_available=len(gpus) > 0,
            gpu_count=len(gpus),
            gpu_driver=driver,
            cuda_version=cuda_ver,
            temperatures=self.temperatures(),
        )

    # ── Overheat Check ───────────────────────────

    def is_gpu_overheating(self, threshold_c: int | None = None) -> bool:
        """Return True if any GPU exceeds the temperature threshold."""
        limit = threshold_c or _settings.MAX_GPU_TEMP_C
        for gpu in self.gpu_devices():
            if gpu.temperature_c >= limit:
                logger.warning(
                    "GPU %d (%s) temperature %d°C exceeds threshold %d°C",
                    gpu.index, gpu.name, gpu.temperature_c, limit,
                )
                return True
        return False


# ── Module-level singleton ───────────────────
hardware_monitor = HardwareMonitor()
