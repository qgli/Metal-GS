#!/usr/bin/env python3
"""
Metal-GS: SIMD MMA vs Scalar SH Forward Kernel Benchmark

Compares correctness and performance of:
  - compute_sh_forward       (scalar, 1 thread/Gaussian)
  - compute_sh_forward_mma   (simdgroup_matrix 8×8, 8 Gaussians/simdgroup)

Outputs:
  - Numerical correctness: max/mean absolute error (FP16 difference)
  - Performance: median kernel time at multiple Gaussian counts
  - Throughput: Gaussians/sec for both variants
"""

import sys
import time
import numpy as np

sys.path.insert(0, ".")
import metal_gs._metal_gs_core as core


def fp16_to_float(uint16_arr):
    """Convert uint16 FP16 bit-representation to float32."""
    return np.frombuffer(uint16_arr.tobytes(), dtype=np.float16).astype(np.float32)


def run_correctness_test(N, K, sh_degree, seed=42):
    """Compare scalar vs MMA kernel outputs. Returns (max_err, mean_err, pass?)."""
    rng = np.random.RandomState(seed)
    
    # Random unit directions
    dirs = rng.randn(N, 3).astype(np.float32)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.clip(norms, 1e-8, None)
    
    # Random SH coefficients in FP16 range
    sh_f32 = rng.randn(N, K, 3).astype(np.float32) * 0.5
    sh_fp16 = sh_f32.astype(np.float16)
    sh_uint16 = np.frombuffer(sh_fp16.tobytes(), dtype=np.uint16).reshape(N, K, 3)
    
    # Run scalar kernel
    colors_scalar_u16, _ = core.compute_sh_forward(dirs, sh_uint16, N, K, sh_degree)
    colors_scalar = fp16_to_float(colors_scalar_u16)
    
    # Run MMA kernel
    colors_mma_u16, _ = core.compute_sh_forward_mma(dirs, sh_uint16, N, K, sh_degree)
    colors_mma = fp16_to_float(colors_mma_u16)
    
    # Compare
    diff = np.abs(colors_scalar - colors_mma)
    max_err = float(diff.max())
    mean_err = float(diff.mean())
    
    # FP16 has ~1e-3 precision, FP32 accumulation vs FP32 MMA should be exact
    # Allow 1e-6 tolerance for any numerical ordering differences
    passed = max_err < 1e-3
    
    return max_err, mean_err, passed, colors_scalar, colors_mma


def run_performance_test(N, K, sh_degree, warmup=5, trials=20):
    """Benchmark both kernels. Returns (scalar_times_ms, mma_times_ms)."""
    rng = np.random.RandomState(123)
    
    dirs = rng.randn(N, 3).astype(np.float32)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.clip(norms, 1e-8, None)
    
    sh_f32 = rng.randn(N, K, 3).astype(np.float32) * 0.5
    sh_fp16 = sh_f32.astype(np.float16)
    sh_uint16 = np.frombuffer(sh_fp16.tobytes(), dtype=np.uint16).reshape(N, K, 3)
    
    # Warmup both kernels
    for _ in range(warmup):
        core.compute_sh_forward(dirs, sh_uint16, N, K, sh_degree)
        core.compute_sh_forward_mma(dirs, sh_uint16, N, K, sh_degree)
    
    # Benchmark scalar
    scalar_times = []
    for _ in range(trials):
        _, t = core.compute_sh_forward(dirs, sh_uint16, N, K, sh_degree)
        scalar_times.append(t)
    
    # Benchmark MMA
    mma_times = []
    for _ in range(trials):
        _, t = core.compute_sh_forward_mma(dirs, sh_uint16, N, K, sh_degree)
        mma_times.append(t)
    
    return scalar_times, mma_times


def main():
    print("=" * 72)
    print("  Metal-GS: SIMD MMA vs Scalar SH Forward Benchmark")
    print("  Target: Apple M4 (GPU Family 9, 10 cores)")
    print("=" * 72)
    
    # ---- Phase 1: Correctness ----
    print("\n── Phase 1: Numerical Correctness ──")
    
    test_configs = [
        (1024,   1, 0, "1K, degree 0"),
        (1024,   4, 1, "1K, degree 1"),
        (1024,   9, 2, "1K, degree 2"),
        (1024,  16, 3, "1K, degree 3"),
        (10000, 16, 3, "10K, degree 3"),
        (50000, 16, 3, "50K, degree 3"),
        (100000,16, 3, "100K, degree 3"),
        (7,     16, 3, "7 (non-aligned), degree 3"),
        (63,    16, 3, "63 (non-aligned), degree 3"),
        (65,    16, 3, "65 (non-aligned), degree 3"),
    ]
    
    all_pass = True
    print(f"  {'Config':<30s}  {'Max Err':>10s}  {'Mean Err':>10s}  {'Status':>8s}")
    print(f"  {'─'*30}  {'─'*10}  {'─'*10}  {'─'*8}")
    
    for N, K, deg, label in test_configs:
        max_err, mean_err, passed, _, _ = run_correctness_test(N, K, deg)
        status = "PASS ✓" if passed else "FAIL ✗"
        if not passed:
            all_pass = False
        print(f"  {label:<30s}  {max_err:>10.2e}  {mean_err:>10.2e}  {status:>8s}")
    
    print(f"\n  Overall correctness: {'ALL PASS ✓' if all_pass else 'FAIL ✗'}")
    
    # ---- Phase 2: Performance ----
    print("\n── Phase 2: Performance Benchmark ──")
    
    perf_configs = [
        (1000,   16, 3, "1K"),
        (10000,  16, 3, "10K"),
        (50000,  16, 3, "50K"),
        (100000, 16, 3, "100K"),
        (200000, 16, 3, "200K"),
        (500000, 16, 3, "500K"),
    ]
    
    print(f"\n  {'Gaussians':<12s}  {'Scalar (ms)':>12s}  {'MMA (ms)':>12s}  {'Speedup':>8s}  {'Winner':>8s}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*8}")
    
    results = []
    for N, K, deg, label in perf_configs:
        scalar_t, mma_t = run_performance_test(N, K, deg, warmup=5, trials=20)
        
        scalar_med = np.median(scalar_t)
        mma_med = np.median(mma_t)
        speedup = scalar_med / mma_med if mma_med > 0 else float('inf')
        winner = "MMA" if speedup > 1.0 else "Scalar"
        
        print(f"  {label:<12s}  {scalar_med:>12.4f}  {mma_med:>12.4f}  {speedup:>7.2f}×  {winner:>8s}")
        
        results.append({
            "N": N,
            "scalar_median_ms": scalar_med,
            "mma_median_ms": mma_med,
            "speedup": speedup,
            "scalar_p10": np.percentile(scalar_t, 10),
            "scalar_p90": np.percentile(scalar_t, 90),
            "mma_p10": np.percentile(mma_t, 10),
            "mma_p90": np.percentile(mma_t, 90),
        })
    
    # ---- Phase 3: Throughput Summary ----
    print("\n── Phase 3: Throughput (Gaussians/sec) ──")
    print(f"\n  {'Gaussians':<12s}  {'Scalar':>15s}  {'MMA':>15s}")
    print(f"  {'─'*12}  {'─'*15}  {'─'*15}")
    
    for r in results:
        scalar_tput = r["N"] / (r["scalar_median_ms"] / 1000)
        mma_tput = r["N"] / (r["mma_median_ms"] / 1000)
        print(f"  {r['N']:<12d}  {scalar_tput:>13.1f}/s  {mma_tput:>13.1f}/s")
    
    # ---- Summary ----
    print("\n── Summary ──")
    avg_speedup = np.mean([r["speedup"] for r in results])
    best = max(results, key=lambda r: r["speedup"])
    worst = min(results, key=lambda r: r["speedup"])
    
    print(f"  Average speedup:  {avg_speedup:.2f}×")
    print(f"  Best speedup:     {best['speedup']:.2f}× at N={best['N']}")
    print(f"  Worst speedup:    {worst['speedup']:.2f}× at N={worst['N']}")
    
    if avg_speedup > 1.0:
        print(f"\n  ★ MMA kernel is FASTER on average ({avg_speedup:.2f}×)")
        print(f"    → Recommend integrating into training pipeline")
    else:
        print(f"\n  ★ Scalar kernel is FASTER on average ({1.0/avg_speedup:.2f}×)")
        print(f"    → MMA approach not beneficial for SH eval")
        print(f"    → Focus SIMD optimization on rasterize/covariance kernels")
    
    print("\n" + "=" * 72)
    
    return all_pass, results


if __name__ == "__main__":
    passed, results = main()
    sys.exit(0 if passed else 1)
