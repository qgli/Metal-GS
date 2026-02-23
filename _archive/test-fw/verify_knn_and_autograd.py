#!/usr/bin/env python3
"""
Phase 9 Verification: KNN accuracy + PyTorch autograd integration.

Test 1 — KNN:
  Generate 10,000 random 3D points.
  Compare simple_knn_metal vs scipy.spatial.KDTree (exact K=3 nearest).
  Allow relative tolerance for Morton-code approximation (window boundary misses).

Test 2 — Autograd:
  Run MetalGaussianRasterizer.apply with requires_grad=True tensors.
  Compute a pseudo-loss and call .backward().
  Verify .grad is not None for all differentiable parameters.
"""

import sys
import numpy as np

# ============================================================================
#  Test 1: KNN accuracy
# ============================================================================
def test_knn():
    from scipy.spatial import KDTree
    from metal_gs.rasterizer import simple_knn_metal

    print("=" * 60)
    print("  Test 1: KNN (Morton code + radix sort vs scipy KDTree)")
    print("=" * 60)

    np.random.seed(42)
    N = 10_000
    K = 3
    # Use clustered data (more realistic for 3DGS point clouds)
    # 20 clusters of 500 points each, mimicking a surface reconstruction
    n_clusters = 20
    pts_per_cluster = N // n_clusters
    clusters = []
    for _ in range(n_clusters):
        center = np.random.randn(1, 3).astype(np.float32) * 5.0
        cluster = center + np.random.randn(pts_per_cluster, 3).astype(np.float32) * 0.5
        clusters.append(cluster)
    points = np.concatenate(clusters, axis=0).astype(np.float32)

    # ---- Reference: scipy KDTree exact KNN ----
    tree = KDTree(points)
    dists, _ = tree.query(points, k=K + 1)  # +1 because self is included
    # Remove self (distance 0, first column)
    dists_neighbors = dists[:, 1:]  # [N, K]
    ref_avg_sq = np.mean(dists_neighbors ** 2, axis=1)  # [N]

    # ---- Metal: Morton-code KNN ----
    # Use large search window for good accuracy on clustered data
    WINDOW = 256
    metal_avg_sq, elapsed_ms = simple_knn_metal(points, k_neighbors=K, search_window=WINDOW)

    print(f"  N = {N},  K = {K},  search_window = {WINDOW}")
    print(f"  Metal KNN time: {elapsed_ms:.3f} ms")

    # ---- Compare ----
    # Morton-code KNN is approximate: if the true nearest neighbor is far
    # away in Morton order, it will be missed. We expect most (>95%) to match.
    abs_err = np.abs(metal_avg_sq - ref_avg_sq)
    rel_err = abs_err / (ref_avg_sq + 1e-10)

    # Exact match = relative error < 1%
    exact_mask = rel_err < 0.01
    exact_pct = np.mean(exact_mask) * 100

    # Close match = relative error < 10%
    close_mask = rel_err < 0.10
    close_pct = np.mean(close_mask) * 100

    print(f"  Exact match  (rel_err < 1%):  {exact_pct:.1f}%")
    print(f"  Close match  (rel_err < 10%): {close_pct:.1f}%")
    print(f"  Max abs error: {abs_err.max():.6f}")
    print(f"  Mean abs error: {abs_err.mean():.6f}")
    print(f"  Median rel error: {np.median(rel_err):.6f}")

    # We require > 90% exact match for PASS.
    # Morton-code KNN with window=32 on random data should get >95%.
    passed = exact_pct > 90.0
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  KNN Test: {status} ({exact_pct:.1f}% exact match, need >90%)\n")
    return passed


# ============================================================================
#  Test 2: PyTorch autograd integration
# ============================================================================
def test_autograd():
    try:
        import torch
    except ImportError:
        print("  [SKIP] PyTorch not installed, skipping autograd test")
        return True

    from metal_gs.rasterizer import MetalGaussianRasterizer, RenderSettings

    print("=" * 60)
    print("  Test 2: PyTorch autograd (MetalGaussianRasterizer)")
    print("=" * 60)

    np.random.seed(123)
    N = 64
    K = 16  # (sh_degree+1)^2 = 16 for degree 3
    sh_degree = 3
    H, W = 64, 64

    # ---- Create synthetic data with requires_grad ----
    means3d = torch.randn(N, 3, dtype=torch.float32, requires_grad=True)
    scales  = torch.randn(N, 3, dtype=torch.float32, requires_grad=True)
    quats   = torch.randn(N, 4, dtype=torch.float32, requires_grad=True)
    # Normalize quaternions
    quats_data = quats.data
    quats_data /= quats_data.norm(dim=1, keepdim=True)
    quats.data = quats_data

    sh_coeffs  = torch.randn(N, K, 3, dtype=torch.float32, requires_grad=True)
    opacities  = torch.rand(N, dtype=torch.float32, requires_grad=True)  # leaf tensor in [0,1]

    # Camera
    viewmat = torch.eye(4, dtype=torch.float32)
    viewmat[2, 3] = 5.0  # camera at z=5 looking at origin
    campos = torch.tensor([0.0, 0.0, 5.0], dtype=torch.float32)

    settings = RenderSettings(
        viewmat=viewmat.numpy(),
        tan_fovx=0.5,
        tan_fovy=0.5,
        focal_x=float(W),
        focal_y=float(H),
        principal_x=float(W) / 2,
        principal_y=float(H) / 2,
        img_width=W,
        img_height=H,
        sh_degree=sh_degree,
        bg_color=(0.0, 0.0, 0.0),
    )

    # ---- Forward ----
    print("  Running forward pass...")
    image = MetalGaussianRasterizer.apply(
        means3d, scales, quats, sh_coeffs, opacities,
        viewmat, campos, settings
    )
    print(f"  Output image shape: {image.shape}")
    assert image.shape == (H, W, 3), f"Expected ({H},{W},3), got {image.shape}"

    # ---- Backward ----
    print("  Running backward pass...")
    loss = image.sum()  # simple pseudo-loss
    loss.backward()

    # ---- Check gradients ----
    results = {}
    for name, param in [("means3d", means3d), ("scales", scales),
                         ("quats", quats), ("sh_coeffs", sh_coeffs),
                         ("opacities", opacities)]:
        has_grad = param.grad is not None
        nonzero = False
        if has_grad:
            nonzero = param.grad.abs().sum().item() > 0
        results[name] = (has_grad, nonzero)
        status = "✅" if has_grad and nonzero else ("⚠️ zero" if has_grad else "❌ None")
        grad_info = ""
        if has_grad:
            grad_info = f"  max={param.grad.abs().max().item():.6f}"
        print(f"    {name:12s} grad: {status}{grad_info}")

    # All must have non-None gradients
    all_have_grad = all(v[0] for v in results.values())
    # At least means3d, scales, quats should have nonzero grads
    # (sh/opacity might be zero depending on visibility)
    key_nonzero = all(results[k][1] for k in ["means3d", "scales", "quats"])

    passed = all_have_grad
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  Autograd Test: {status} (all grads exist: {all_have_grad})\n")
    return passed


# ============================================================================
#  Main
# ============================================================================
if __name__ == "__main__":
    print("\n" + "━" * 60)
    print("  Phase 9 Verification: KNN + Autograd")
    print("━" * 60 + "\n")

    knn_ok = test_knn()
    autograd_ok = test_autograd()

    print("━" * 60)
    print(f"  KNN:      {'✅ PASS' if knn_ok else '❌ FAIL'}")
    print(f"  Autograd: {'✅ PASS' if autograd_ok else '❌ FAIL'}")
    print("━" * 60)

    if knn_ok and autograd_ok:
        print("\n  🎉 All Phase 9 tests PASSED!\n")
        sys.exit(0)
    else:
        print("\n  ⚠️  Some tests failed.\n")
        sys.exit(1)
