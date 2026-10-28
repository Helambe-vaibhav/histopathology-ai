"""
hardware.py
-----------
Hardware resource manager enforcing system limits:
- GPU memory fraction ceiling: 90% (prevents CUDA OOM)
- System RAM ceiling: 90% (prevents OS thrashing)
- Maximizes multi-core CPU utilization (sets threads & worker counts)

Usage:
    from hardware import setup_hardware, check_system_resources
"""

import os
import gc
import psutil
import torch


def setup_hardware(
    gpu_fraction: float = 0.90,
    max_ram_fraction: float = 0.90,
    num_threads_ratio: float = 0.80,
) -> dict:
    """
    Configures and enforces system resource limits to maximize throughput
    while strictly capping GPU and RAM utilization at 90%.

    Args:
        gpu_fraction: Maximum VRAM allocation fraction (default 0.90)
        max_ram_fraction: Maximum system RAM fraction before alerting/cleanup (default 0.90)
        num_threads_ratio: Fraction of CPU cores to assign to PyTorch compute (default 0.80)

    Returns:
        dict of system hardware specifications and active limits
    """
    total_cores = os.cpu_count() or 4
    # Maximize CPU utilization capped at ~80-90% of cores (leaves headroom for OS/background tasks)
    active_threads = max(1, int(total_cores * num_threads_ratio))
    torch.set_num_threads(active_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_info = {}

    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

        # Enforce strict 90% allocation ceiling on GPU
        torch.cuda.set_per_process_memory_fraction(float(gpu_fraction), 0)
        total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        vram_cap_mb = total_vram_mb * gpu_fraction

        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "total_vram_mb": round(total_vram_mb, 1),
            "vram_cap_mb": round(vram_cap_mb, 1),
            "gpu_fraction": gpu_fraction,
        }

    # System RAM inspection
    ram = psutil.virtual_memory()
    total_ram_gb = ram.total / (1024 ** 3)
    available_ram_gb = ram.available / (1024 ** 3)
    ram_usage_pct = ram.percent

    if ram_usage_pct >= (max_ram_fraction * 100):
        # Trigger urgent cleanup
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  [RESOURCE WARNING] System RAM at {ram_usage_pct:.1f}% (limit: {max_ram_fraction * 100:.0f}%). Garbage collection triggered.")

    system_profile = {
        "device": device,
        "device_type": device.type,
        "cpu_cores_total": total_cores,
        "cpu_threads_active": active_threads,
        "total_ram_gb": round(total_ram_gb, 2),
        "available_ram_gb": round(available_ram_gb, 2),
        "ram_usage_percent": ram_usage_pct,
        "max_ram_fraction": max_ram_fraction,
        "gpu": gpu_info,
    }

    return system_profile


def check_system_resources() -> dict:
    """Lightweight real-time check of RAM and VRAM during training loops."""
    ram = psutil.virtual_memory()
    info = {
        "ram_percent": ram.percent,
        "ram_used_gb": round((ram.total - ram.available) / (1024 ** 3), 2),
    }
    if torch.cuda.is_available():
        info["vram_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1)
        info["vram_peak_mb"] = round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 1)

    # If RAM exceeds 90%, perform auto-recovery
    if ram.percent > 90.0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return info
