"""
benchmark.py
------------
FP32 vs FP16/AMP inference benchmark.

Usage:
    python src/benchmark.py --config configs/config.yaml --checkpoint results/checkpoints/unet_best.pth
"""

import time
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path

from models import build_model


# ---------------------------------------------------------------------------
# Core measurement utilities
# ---------------------------------------------------------------------------

def measure_latency(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    n_warmup: int = 10,
    n_iter: int = 100,
    use_amp: bool = False,
) -> float:
    """
    Measure mean per-batch latency in milliseconds.

    Args:
        model:        eval-mode model on GPU
        input_tensor: (B, 3, H, W) float32 tensor on GPU
        n_warmup:     warmup iterations (not timed)
        n_iter:       timed iterations
        use_amp:      whether to use torch.cuda.amp.autocast

    Returns:
        mean latency in ms
    """
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            if use_amp:
                with torch.amp.autocast("cuda"):
                    _ = model(input_tensor)
            else:
                _ = model(input_tensor)
        torch.cuda.synchronize()

        # Timed
        start = time.perf_counter()
        for _ in range(n_iter):
            if use_amp:
                with torch.amp.autocast("cuda"):
                    _ = model(input_tensor)
            else:
                _ = model(input_tensor)
        torch.cuda.synchronize()
        end = time.perf_counter()

    return (end - start) * 1000 / n_iter   # ms per batch


def measure_throughput(latency_ms: float, batch_size: int) -> float:
    """Images per second from latency in ms and batch size."""
    return batch_size / (latency_ms / 1000.0)


def measure_vram() -> float:
    """Current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 2
    return 0.0


# ---------------------------------------------------------------------------
# Full benchmark run
# ---------------------------------------------------------------------------

def run_benchmark(model, device, batch_sizes, tile_size=256,
                  warmup=10, n_iter=100) -> list:
    """
    Run FP32 and FP16 benchmarks over multiple batch sizes.

    Returns list of result dicts.
    """
    results = []
    model = model.to(device).eval()

    for bs in batch_sizes:
        x = torch.randn(bs, 3, tile_size, tile_size, device=device)

        for precision, use_amp in [("FP32", False), ("FP16", True)]:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()

            try:
                lat = measure_latency(model, x, n_warmup=warmup, n_iter=n_iter, use_amp=use_amp)
                throughput = measure_throughput(lat, bs)
                vram = torch.cuda.max_memory_allocated(device) / 1024 ** 2
            except RuntimeError as e:
                print(f"  OOM at batch={bs} {precision}: {e}")
                lat, throughput, vram = float("nan"), float("nan"), float("nan")

            results.append({
                "precision": precision,
                "batch_size": bs,
                "latency_ms": round(lat, 2),
                "throughput_img_s": round(throughput, 1),
                "vram_mb": round(vram, 1),
                "dice": "— (eval on test set)",
            })

            print(f"  {precision} | batch={bs:2d} | "
                  f"lat={lat:.1f}ms | tput={throughput:.1f} img/s | VRAM={vram:.0f}MB")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model checkpoint (.pth)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available. Benchmark on CPU is not meaningful.")
    else:
        gpu_fraction = float(cfg.get("hardware", {}).get("gpu_memory_fraction", 0.90))
        torch.cuda.set_per_process_memory_fraction(gpu_fraction, 0)
        torch.cuda.empty_cache()
        print(f"CUDA benchmark on: {torch.cuda.get_device_name(0)}")
        print(f"Enforced 90% VRAM ceiling: {gpu_fraction * 100:.0f}%")

    model = build_model(cfg["model"])
    if args.checkpoint and Path(args.checkpoint).exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"Loaded checkpoint: {args.checkpoint}")

    batch_sizes  = cfg["benchmark"]["batch_sizes"]
    warmup_iters = cfg["benchmark"]["warmup_iters"]
    bench_iters  = cfg["benchmark"]["benchmark_iters"]

    print(f"\nRunning benchmark: batch_sizes={batch_sizes}\n")
    results = run_benchmark(model, device, batch_sizes,
                            warmup=warmup_iters, n_iter=bench_iters)

    # Print summary table
    import pandas as pd
    df = pd.DataFrame(results)
    print("\n=== Benchmark Results ===")
    print(df.to_string(index=False))

    out_path = Path("results/tables/benchmark.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
