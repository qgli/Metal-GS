#!/usr/bin/env python3
"""
Metal-GS Phase 1 Verification: Spherical Harmonics forward pass.

Compares the Metal compute kernel against a pure-Python/NumPy reference
implementation for correctness, then benchmarks both at 1M Gaussians.

Usage:
    python verify_sh.py
"""

import sys
import time
import numpy as np

# ============================================================================
#  Reference implementation: Pure NumPy SH evaluation (FP32 for ground truth)
# ============================================================================

# SH constants (identical to the Metal kernel and original 3DGS)
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199

SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]

SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]


def sh_reference_numpy(directions: np.ndarray, sh_coeffs: np.ndarray, sh_degree: int) -> np.ndarray:
    """
    Reference SH → RGB evaluation in pure NumPy (FP32).
    
    Parameters
    ----------
    directions : [N, 3] float32 — unit view directions
    sh_coeffs  : [N, K, 3] float32 — SH coefficients (K bases, RGB)
    sh_degree  : int (0–3)
    
    Returns
    -------
    colors : [N, 3] float32 — RGB colours with +0.5 offset, clamped [0,1]
    """
    N = directions.shape[0]
    
    # Normalise directions (just in case)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    d = directions / norms
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    
    # sh_coeffs[:, i, :] gives basis i's RGB for all points → [N, 3]
    result = SH_C0 * sh_coeffs[:, 0, :]
    
    if sh_degree >= 1:
        result = result + SH_C1 * (
            -y[:, None] * sh_coeffs[:, 1, :]
            + z[:, None] * sh_coeffs[:, 2, :]
            - x[:, None] * sh_coeffs[:, 3, :]
        )
    
    if sh_degree >= 2:
        xx, yy, zz = x*x, y*y, z*z
        xy, yz, xz = x*y, y*z, x*z
        
        result = result + (
            SH_C2[0] * (xy)[:, None]               * sh_coeffs[:, 4, :]
          + SH_C2[1] * (yz)[:, None]               * sh_coeffs[:, 5, :]
          + SH_C2[2] * (2*zz - xx - yy)[:, None]   * sh_coeffs[:, 6, :]
          + SH_C2[3] * (xz)[:, None]               * sh_coeffs[:, 7, :]
          + SH_C2[4] * (xx - yy)[:, None]           * sh_coeffs[:, 8, :]
        )
    
    if sh_degree >= 3:
        xx, yy, zz = x*x, y*y, z*z
        xy, yz, xz = x*y, y*z, x*z
        
        result = result + (
            SH_C3[0] * (y * (3*xx - yy))[:, None]               * sh_coeffs[:, 9, :]
          + SH_C3[1] * (xy * z)[:, None]                         * sh_coeffs[:, 10, :]
          + SH_C3[2] * (y * (4*zz - xx - yy))[:, None]           * sh_coeffs[:, 11, :]
          + SH_C3[3] * (z * (2*zz - 3*xx - 3*yy))[:, None]      * sh_coeffs[:, 12, :]
          + SH_C3[4] * (x * (4*zz - xx - yy))[:, None]           * sh_coeffs[:, 13, :]
          + SH_C3[5] * (z * (xx - yy))[:, None]                  * sh_coeffs[:, 14, :]
          + SH_C3[6] * (x * (xx - 3*yy))[:, None]                * sh_coeffs[:, 15, :]
        )
    
    # +0.5 offset and clamp (matching the Metal kernel)
    result = result + 0.5
    result = np.clip(result, 0.0, 1.0)
    
    return result


# ============================================================================
#  PyTorch reference (CPU)
# ============================================================================

def sh_reference_torch(directions, sh_coeffs, sh_degree):
    """Same as numpy reference but using PyTorch CPU for timing comparison."""
    import torch
    
    dirs_t = torch.from_numpy(directions).float()
    sh_t   = torch.from_numpy(sh_coeffs.astype(np.float32)).float()
    
    norms = torch.norm(dirs_t, dim=1, keepdim=True).clamp(min=1e-8)
    d = dirs_t / norms
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    
    result = SH_C0 * sh_t[:, 0, :]
    
    if sh_degree >= 1:
        result = result + SH_C1 * (
            -y.unsqueeze(1) * sh_t[:, 1, :]
            + z.unsqueeze(1) * sh_t[:, 2, :]
            - x.unsqueeze(1) * sh_t[:, 3, :]
        )
    
    if sh_degree >= 2:
        xx, yy, zz = x*x, y*y, z*z
        xy, yz, xz = x*y, y*z, x*z
        result = result + (
            SH_C2[0] * xy.unsqueeze(1)             * sh_t[:, 4, :]
          + SH_C2[1] * yz.unsqueeze(1)             * sh_t[:, 5, :]
          + SH_C2[2] * (2*zz-xx-yy).unsqueeze(1)   * sh_t[:, 6, :]
          + SH_C2[3] * xz.unsqueeze(1)             * sh_t[:, 7, :]
          + SH_C2[4] * (xx-yy).unsqueeze(1)         * sh_t[:, 8, :]
        )
    
    if sh_degree >= 3:
        xx, yy, zz = x*x, y*y, z*z
        xy, yz, xz = x*y, y*z, x*z
        result = result + (
            SH_C3[0] * (y*(3*xx-yy)).unsqueeze(1)           * sh_t[:, 9, :]
          + SH_C3[1] * (xy*z).unsqueeze(1)                   * sh_t[:, 10, :]
          + SH_C3[2] * (y*(4*zz-xx-yy)).unsqueeze(1)         * sh_t[:, 11, :]
          + SH_C3[3] * (z*(2*zz-3*xx-3*yy)).unsqueeze(1)     * sh_t[:, 12, :]
          + SH_C3[4] * (x*(4*zz-xx-yy)).unsqueeze(1)         * sh_t[:, 13, :]
          + SH_C3[5] * (z*(xx-yy)).unsqueeze(1)               * sh_t[:, 14, :]
          + SH_C3[6] * (x*(xx-3*yy)).unsqueeze(1)             * sh_t[:, 15, :]
        )
    
    result = (result + 0.5).clamp(0.0, 1.0)
    return result.numpy()


# ============================================================================
#  Main: generate data, compare, benchmark
# ============================================================================

def main():
    print("=" * 72)
    print("  Metal-GS Phase 1 Verification: Spherical Harmonics Forward Pass")
    print("=" * 72)
    
    # ---- Import Metal extension ----
    try:
        from metal_gs.sh import compute_sh_colors_metal
        print("[✓] Metal-GS extension loaded successfully")
    except Exception as e:
        print(f"[✗] Failed to load Metal-GS extension: {e}")
        sys.exit(1)
    
    # ---- Test parameters ----
    SH_DEGREE = 3
    K = (SH_DEGREE + 1) ** 2  # = 16
    
    # Sweep sizes: small for correctness, large for benchmarking
    test_sizes = [1, 64, 1024, 100_000, 1_000_000]
    
    print(f"\nSH Degree: {SH_DEGREE}  (K = {K} bases)")
    print(f"Test sizes: {test_sizes}")
    
    np.random.seed(42)
    
    for N in test_sizes:
        print(f"\n{'─' * 60}")
        print(f"  N = {N:>10,d} Gaussians")
        print(f"{'─' * 60}")
        
        # ---- Generate random test data ----
        # Random unit directions
        raw_dirs = np.random.randn(N, 3).astype(np.float32)
        norms = np.linalg.norm(raw_dirs, axis=1, keepdims=True)
        directions = (raw_dirs / np.maximum(norms, 1e-8)).astype(np.float32)
        
        # Random SH coefficients in a reasonable range (small values as in real 3DGS)
        sh_coeffs_f32 = (np.random.randn(N, K, 3) * 0.5).astype(np.float32)
        sh_coeffs_f16 = sh_coeffs_f32.astype(np.float16)
        
        # ---- Reference computation (NumPy FP32) ----
        t0 = time.perf_counter()
        ref_colors = sh_reference_numpy(directions, sh_coeffs_f32, SH_DEGREE)
        t_numpy = (time.perf_counter() - t0) * 1000  # ms
        
        # ---- Metal computation ----
        metal_colors, t_metal = compute_sh_colors_metal(
            directions, sh_coeffs_f16, sh_degree=SH_DEGREE
        )
        
        # ---- Compare ----
        # The Metal kernel uses FP16 coefficients → expect some quantisation error.
        # We compare against a reference that also uses FP16 coefficients for fair comparison.
        ref_colors_f16_input = sh_reference_numpy(
            directions, sh_coeffs_f16.astype(np.float32), SH_DEGREE
        )
        ref_f16 = ref_colors_f16_input.astype(np.float16).astype(np.float32)
        metal_f32 = metal_colors.astype(np.float32)
        
        abs_err = np.abs(metal_f32 - ref_f16)
        max_err = abs_err.max()
        mean_err = abs_err.mean()
        
        # Also compare against full FP32 reference
        abs_err_fp32 = np.abs(metal_f32 - ref_colors)
        max_err_fp32 = abs_err_fp32.max()
        
        passed = max_err < 0.01  # FP16 tolerance: ~0.001 typical, 0.01 max
        status = "✓ PASS" if passed else "✗ FAIL"
        
        print(f"  [{status}] Max error (vs FP16 ref): {max_err:.6f}")
        print(f"           Mean error (vs FP16 ref): {mean_err:.6f}")
        print(f"           Max error (vs FP32 ref):  {max_err_fp32:.6f}")
        print(f"           NumPy CPU time:  {t_numpy:>10.3f} ms")
        print(f"           Metal GPU time:  {t_metal:>10.3f} ms")
        if t_metal > 0 and t_numpy > 0:
            speedup = t_numpy / t_metal
            print(f"           Speedup:         {speedup:>10.2f}×")
        
        # Show sample outputs
        if N <= 64:
            print(f"\n  Sample output (first {min(N, 5)} Gaussians):")
            for i in range(min(N, 5)):
                m = metal_colors[i].astype(np.float32)
                r = ref_colors[i]
                print(f"    [{i}] Metal: ({m[0]:.4f}, {m[1]:.4f}, {m[2]:.4f})  "
                      f"Ref: ({r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f})")
    
    # ---- Benchmark: multiple runs at 1M ----
    print(f"\n{'=' * 60}")
    print(f"  Benchmark: 5 runs at 1,000,000 Gaussians")
    print(f"{'=' * 60}")
    
    N = 1_000_000
    directions = np.random.randn(N, 3).astype(np.float32)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = (directions / np.maximum(norms, 1e-8)).astype(np.float32)
    sh_coeffs_f16 = (np.random.randn(N, K, 3) * 0.5).astype(np.float16)
    
    metal_times = []
    numpy_times = []
    
    for run in range(5):
        _, t_m = compute_sh_colors_metal(directions, sh_coeffs_f16, sh_degree=SH_DEGREE)
        metal_times.append(t_m)
        
        t0 = time.perf_counter()
        _ = sh_reference_numpy(directions, sh_coeffs_f16.astype(np.float32), SH_DEGREE)
        numpy_times.append((time.perf_counter() - t0) * 1000)
    
    print(f"  Metal GPU:  {np.mean(metal_times):>8.3f} ms  (±{np.std(metal_times):.3f})")
    print(f"  NumPy CPU:  {np.mean(numpy_times):>8.3f} ms  (±{np.std(numpy_times):.3f})")
    print(f"  Speedup:    {np.mean(numpy_times)/np.mean(metal_times):>8.2f}×")
    
    print(f"\n{'=' * 60}")
    print(f"  Phase 1 verification complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
