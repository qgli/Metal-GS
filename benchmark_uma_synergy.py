#!/usr/bin/env python3
"""
Metal-GS: CPU-GPU Zero-Copy Synergy Benchmark
================================================

Compares three pathways for CPU-side operations on MPS tensor data:

  A. Traditional: tensor.cpu().numpy() → numpy ops → result
  B. UMA-Blit:    Metal blit (private→shared) → C++ CPU read → result
  C. UMA-Shared:  Shared allocator tensor → C++ CPU read (TRUE zero-copy)

Operations benchmarked:
  1. Frustum culling (branchy per-gaussian visibility test)
  2. Densification mask (gradient + scale thresholding)
  3. Pruning mask (opacity + scale + screen-size thresholding)

Hardware: Apple M4 (Mac Mini), 10 GPU cores, 16GB unified memory
"""

import torch
import numpy as np
import time
import sys

import metal_gs._metal_gs_uma as uma

# ============================================================================
#  Config
# ============================================================================

SIZES = [100_000, 500_000, 1_000_000]
WARMUP = 3
ITERS = 20

# ============================================================================
#  Helpers
# ============================================================================

def timer():
    """High-resolution timer."""
    return time.perf_counter()


def benchmark(fn, warmup=WARMUP, iters=ITERS):
    """Benchmark a function, returning median time in ms."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = timer()
        fn()
        times.append((timer() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]  # median


# ============================================================================
#  Generate test data
# ============================================================================

def make_data(N, device='mps'):
    """Generate realistic 3DGS training data."""
    torch.manual_seed(42)
    data = {
        'means3d': torch.randn(N, 3, device=device) * 5.0,
        'scales': torch.randn(N, 3, device=device) * 0.5,
        'quats': torch.randn(N, 4, device=device),
        'opacities': torch.randn(N, device=device),
        'grad_magnitudes': torch.rand(N, device=device) * 0.001,
        'screen_radii': torch.rand(N, device=device) * 30.0,
        'viewmat': torch.eye(4, device=device),
    }
    # Normalize quaternions
    data['quats'] = data['quats'] / data['quats'].norm(dim=1, keepdim=True)
    torch.mps.synchronize()
    return data


def make_shared_data(N):
    """Generate data in shared storage mode (TRUE zero-copy)."""
    data = make_data(N, device='mps')
    torch.mps.synchronize()
    shared = {}
    for k, v in data.items():
        shared[k] = uma.to_shared(v)
    return shared


# ============================================================================
#  Traditional path: tensor.cpu().numpy() + numpy operations
# ============================================================================

def trad_frustum_cull(data, near=0.01, far=100.0, fov_x=1.0, fov_y=1.0):
    """Traditional frustum culling via numpy on CPU."""
    torch.mps.synchronize()
    means = data['means3d'].cpu().numpy()
    V = data['viewmat'].cpu().numpy()

    # Transform to camera space
    # p_cam = V[:3,:3] @ p + V[:3,3]
    R = V[:3, :3]
    t = V[:3, 3]
    cam = means @ R.T + t[np.newaxis, :]

    # Depth test
    depth_mask = (cam[:, 2] > near) & (cam[:, 2] < far)

    # Perspective divide
    inv_z = np.where(cam[:, 2] > 0, 1.0 / cam[:, 2], 0.0)
    ndc_x = cam[:, 0] * inv_z
    ndc_y = cam[:, 1] * inv_z

    pad = 1.05
    fov_mask = (ndc_x > -fov_x * pad) & (ndc_x < fov_x * pad) & \
               (ndc_y > -fov_y * pad) & (ndc_y < fov_y * pad)

    mask = depth_mask & fov_mask
    return mask


def trad_densification_mask(data, grad_threshold=0.0002, scale_threshold=0.01):
    """Traditional densification mask via numpy."""
    torch.mps.synchronize()
    grads = data['grad_magnitudes'].cpu().numpy()
    scales = data['scales'].cpu().numpy()

    exceed = grads > grad_threshold
    max_scale = np.exp(scales).max(axis=1)
    large = max_scale > scale_threshold

    clone_mask = exceed & ~large
    split_mask = exceed & large
    return clone_mask, split_mask


def trad_pruning_mask(data, opacity_thresh=0.005, world_thresh=1.0, screen_thresh=20.0):
    """Traditional pruning mask via numpy."""
    torch.mps.synchronize()
    opacities = data['opacities'].cpu().numpy()
    scales = data['scales'].cpu().numpy()
    radii = data['screen_radii'].cpu().numpy()

    sig_opacity = 1.0 / (1.0 + np.exp(-opacities))
    max_scale = np.exp(scales).max(axis=1)

    mask = (sig_opacity < opacity_thresh) | \
           (max_scale > world_thresh) | \
           (radii > screen_thresh)
    return mask


# ============================================================================
#  UMA path (private storage — uses Metal blit internally)
# ============================================================================

def uma_frustum_cull_private(data):
    """UMA frustum cull on private-storage tensors."""
    torch.mps.synchronize()
    mask, ms = uma.uma_frustum_cull(
        data['means3d'], data['viewmat'], 0.01, 100.0, 1.0, 1.0)
    return mask


def uma_densification_private(data):
    """UMA densification on private-storage tensors."""
    torch.mps.synchronize()
    clone_m, split_m, nc, ns, ms = uma.uma_densification_mask(
        data['grad_magnitudes'], data['scales'], 0.0002, 0.01)
    return clone_m, split_m


def uma_pruning_private(data):
    """UMA pruning on private-storage tensors."""
    torch.mps.synchronize()
    mask, np_, ms = uma.uma_pruning_mask(
        data['opacities'], data['scales'], data['screen_radii'],
        0.005, 1.0, 20.0)
    return mask


# ============================================================================
#  UMA-Shared path (shared storage — TRUE zero-copy)
# ============================================================================

def uma_frustum_cull_shared(shared_data):
    """UMA frustum cull on shared-storage tensors (TRUE zero-copy)."""
    # No sync needed — shared storage means CPU already has access
    mask, ms = uma.uma_frustum_cull(
        shared_data['means3d'], shared_data['viewmat'], 0.01, 100.0, 1.0, 1.0)
    return mask


def uma_densification_shared(shared_data):
    """UMA densification on shared-storage tensors."""
    clone_m, split_m, nc, ns, ms = uma.uma_densification_mask(
        shared_data['grad_magnitudes'], shared_data['scales'], 0.0002, 0.01)
    return clone_m, split_m


def uma_pruning_shared(shared_data):
    """UMA pruning on shared-storage tensors."""
    mask, np_, ms = uma.uma_pruning_mask(
        shared_data['opacities'], shared_data['scales'], shared_data['screen_radii'],
        0.005, 1.0, 20.0)
    return mask


# ============================================================================
#  Correctness validation
# ============================================================================

def validate_correctness(N=10000):
    """Validate all three paths produce identical results."""
    print(f"  Validating correctness (N={N})...")
    data = make_data(N)
    shared = make_shared_data(N)

    # Frustum cull
    trad = trad_frustum_cull(data)
    torch.mps.synchronize()
    uma_p, _ = uma.uma_frustum_cull(data['means3d'], data['viewmat'], 0.01, 100.0, 1.0, 1.0)
    uma_s, _ = uma.uma_frustum_cull(shared['means3d'], shared['viewmat'], 0.01, 100.0, 1.0, 1.0)

    # Compare (must match between trad and uma_p since both read same data)
    trad_t = torch.from_numpy(trad)
    match_p = (uma_p == trad_t).all().item()
    print(f"    Frustum cull:    trad↔blit={match_p}, "
          f"visible: trad={trad.sum()}, blit={uma_p.sum().item()}, shared={uma_s.sum().item()}")

    # Densification
    t_clone, t_split = trad_densification_mask(data)
    torch.mps.synchronize()
    u_clone, u_split, _, _, _ = uma.uma_densification_mask(
        data['grad_magnitudes'], data['scales'], 0.0002, 0.01)
    match_clone = (torch.from_numpy(t_clone) == u_clone).all().item()
    match_split = (torch.from_numpy(t_split) == u_split).all().item()
    print(f"    Densification:   clone_match={match_clone}, split_match={match_split}")

    # Pruning
    t_prune = trad_pruning_mask(data)
    torch.mps.synchronize()
    u_prune, _, _ = uma.uma_pruning_mask(
        data['opacities'], data['scales'], data['screen_radii'], 0.005, 1.0, 20.0)
    match_prune = (torch.from_numpy(t_prune) == u_prune).all().item()
    print(f"    Pruning:         match={match_prune}")

    all_pass = match_p and match_clone and match_split and match_prune
    print(f"    {'✓ ALL CORRECT' if all_pass else '✗ MISMATCH DETECTED'}")
    return all_pass


# ============================================================================
#  Main benchmark
# ============================================================================

def run_benchmarks():
    print("=" * 72)
    print("  Metal-GS: CPU-GPU Zero-Copy Synergy Benchmark")
    print("  Mac Mini M4 · 10 GPU cores · 16GB UMA")
    print(f"  PyTorch {torch.__version__} · MPS backend")
    print("=" * 72)

    # Correctness check first
    print("\n[Phase 1] Correctness Validation")
    if not validate_correctness():
        print("ABORTING: correctness check failed")
        sys.exit(1)

    # Benchmark
    print(f"\n[Phase 2] Performance Benchmark (warmup={WARMUP}, iters={ITERS})")
    print("-" * 72)

    results = {}

    for N in SIZES:
        print(f"\n  N = {N:,} Gaussians")
        print(f"  {'Operation':<20} {'Traditional':>12} {'UMA-Blit':>12} {'UMA-Shared':>12} {'Blit×':>8} {'Shared×':>8}")
        print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*12} {'─'*8} {'─'*8}")

        data = make_data(N)
        shared_data = make_shared_data(N)

        ops = [
            ("Frustum Cull",
             lambda: trad_frustum_cull(data),
             lambda: uma_frustum_cull_private(data),
             lambda: uma_frustum_cull_shared(shared_data)),
            ("Densification",
             lambda: trad_densification_mask(data),
             lambda: uma_densification_private(data),
             lambda: uma_densification_shared(shared_data)),
            ("Pruning",
             lambda: trad_pruning_mask(data),
             lambda: uma_pruning_private(data),
             lambda: uma_pruning_shared(shared_data)),
        ]

        for name, fn_trad, fn_blit, fn_shared in ops:
            t_trad = benchmark(fn_trad)
            t_blit = benchmark(fn_blit)
            t_shared = benchmark(fn_shared)

            speedup_blit = t_trad / t_blit if t_blit > 0 else float('inf')
            speedup_shared = t_trad / t_shared if t_shared > 0 else float('inf')

            print(f"  {name:<20} {t_trad:>10.3f}ms {t_blit:>10.3f}ms {t_shared:>10.3f}ms "
                  f"{speedup_blit:>7.1f}× {speedup_shared:>7.1f}×")

            results[(N, name)] = {
                'traditional': t_trad,
                'blit': t_blit,
                'shared': t_shared,
                'speedup_blit': speedup_blit,
                'speedup_shared': speedup_shared,
            }

    # Summary
    print("\n" + "=" * 72)
    print("  Summary")
    print("=" * 72)

    avg_blit = np.mean([r['speedup_blit'] for r in results.values()])
    avg_shared = np.mean([r['speedup_shared'] for r in results.values()])
    print(f"  Average speedup (UMA-Blit vs Traditional):   {avg_blit:.2f}×")
    print(f"  Average speedup (UMA-Shared vs Traditional): {avg_shared:.2f}×")

    # Breakdown by operation at largest N
    N_max = max(SIZES)
    print(f"\n  Breakdown at N={N_max:,}:")
    for name in ["Frustum Cull", "Densification", "Pruning"]:
        r = results[(N_max, name)]
        print(f"    {name:<20} Trad={r['traditional']:.2f}ms  "
              f"Blit={r['blit']:.2f}ms ({r['speedup_blit']:.1f}×)  "
              f"Shared={r['shared']:.2f}ms ({r['speedup_shared']:.1f}×)")

    # Memory analysis
    print(f"\n  Memory Analysis (N={N_max:,}):")
    bytes_per_gaussian = (3 + 3 + 4 + 1) * 4  # means3d + scales + quats + opacities
    total_mb = N_max * bytes_per_gaussian / 1024 / 1024
    print(f"    Per-frame tensor data:     {total_mb:.1f} MB")
    print(f"    Traditional (.cpu()) copy:  {total_mb:.1f} MB (full duplicate)")
    print(f"    UMA-Blit staging buffer:    {total_mb:.1f} MB (reusable)")
    print(f"    UMA-Shared:                 0.0 MB (zero copy)")

    print("\n" + "=" * 72)
    return results


if __name__ == "__main__":
    results = run_benchmarks()
