"""
Verification script for Metal GPU Tile Binning.

Full pipeline test:
  1. Generate 100K virtual Gaussians
  2. Preprocess (Metal GPU) → means2d, cov2d, depths, radii, tile_min, tile_max
  3. Sort by depth (Metal GPU) → sorted_indices
  4. Tile binning (Metal GPU) → point_list, tile_bins
  5. Compare with NumPy reference

Target: "ALL TESTS PASSED" with zero discrepancy.
"""

import numpy as np
import time
import sys

try:
    from metal_gs._metal_gs_core import (
        preprocess_forward,
        radix_sort_by_depth,
        tile_binning,
    )
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Run 'pip install -e .' first to build the extension.")
    sys.exit(1)


# ---- Constants ----
TILE_SIZE = 16
IMG_WIDTH = 1280
IMG_HEIGHT = 720


def make_camera():
    """Pinhole camera at (0,0,5) looking at origin, 60° FOV."""
    fov_x = np.radians(60)
    fov_y = 2 * np.arctan(np.tan(fov_x / 2) * IMG_HEIGHT / IMG_WIDTH)
    fx = IMG_WIDTH / (2 * np.tan(fov_x / 2))
    fy = IMG_HEIGHT / (2 * np.tan(fov_y / 2))
    cx, cy = IMG_WIDTH / 2.0, IMG_HEIGHT / 2.0

    viewmat = np.eye(4, dtype=np.float32)
    viewmat[2, 3] = 5.0  # camera offset along Z

    return dict(
        viewmat=viewmat,
        tan_fovx=float(np.tan(fov_x / 2)),
        tan_fovy=float(np.tan(fov_y / 2)),
        focal_x=float(fx), focal_y=float(fy),
        cx=float(cx), cy=float(cy),
    )


def reference_tile_binning(sorted_indices, radii, tile_min, tile_max,
                           num_tiles_x, num_tiles_y):
    """
    NumPy reference implementation for tile binning.

    For each Gaussian in depth order, enumerate covered tiles.
    Stable sort by tile_id preserves depth ordering within each tile.
    """
    N = len(sorted_indices)
    num_tiles = num_tiles_x * num_tiles_y

    # Collect (tile_id, gaussian_original_index) pairs in depth order
    intersections = []
    for rank in range(N):
        idx = int(sorted_indices[rank])
        if radii[idx] == 0:
            continue
        tx_min = int(tile_min[idx, 0])
        ty_min = int(tile_min[idx, 1])
        tx_max = int(tile_max[idx, 0])
        ty_max = int(tile_max[idx, 1])
        for ty in range(ty_min, ty_max):
            for tx in range(tx_min, tx_max):
                tile_id = ty * num_tiles_x + tx
                intersections.append((tile_id, idx))

    num_isect = len(intersections)
    if num_isect == 0:
        return (np.array([], dtype=np.uint32),
                np.zeros((num_tiles, 2), dtype=np.uint32),
                0)

    # Stable sort by tile_id (preserves depth order within each tile)
    intersections.sort(key=lambda x: x[0])

    tile_ids   = np.array([x[0] for x in intersections], dtype=np.uint32)
    point_list = np.array([x[1] for x in intersections], dtype=np.uint32)

    # Build tile_bins[tile] = (start, end)
    tile_bins = np.zeros((num_tiles, 2), dtype=np.uint32)
    for i in range(num_isect):
        tid = tile_ids[i]
        if i == 0 or tile_ids[i - 1] != tid:
            tile_bins[tid, 0] = i           # start (inclusive)
        if i == num_isect - 1 or tile_ids[i + 1] != tid:
            tile_bins[tid, 1] = i + 1       # end (exclusive)

    return point_list, tile_bins, num_isect


def main():
    N = 100_000
    num_tiles_x = (IMG_WIDTH  + TILE_SIZE - 1) // TILE_SIZE   # 80
    num_tiles_y = (IMG_HEIGHT + TILE_SIZE - 1) // TILE_SIZE   # 45
    num_tiles   = num_tiles_x * num_tiles_y                   # 3600

    print("=" * 62)
    print("  Metal-GS Tile Binning — End-to-End Pipeline Verification")
    print("=" * 62)
    print(f"  Gaussians:  {N:,}")
    print(f"  Resolution: {IMG_WIDTH}×{IMG_HEIGHT}")
    print(f"  Tile grid:  {num_tiles_x}×{num_tiles_y} = {num_tiles:,} tiles")

    # ---- Generate random Gaussians ----
    np.random.seed(42)
    means3d = np.random.uniform(-2.0, 2.0, (N, 3)).astype(np.float32)
    scales  = np.exp(np.random.uniform(-4, -1.5, (N, 3))).astype(np.float32)
    quats   = np.random.randn(N, 4).astype(np.float32)
    quats  /= np.linalg.norm(quats, axis=1, keepdims=True)

    cam = make_camera()

    # ---- Step 1: Preprocess (Metal GPU) ----
    means2d, cov2d, depths, radii, tile_min, tile_max, prep_ms = \
        preprocess_forward(
            means3d, scales, quats, cam['viewmat'],
            cam['tan_fovx'], cam['tan_fovy'],
            cam['focal_x'], cam['focal_y'],
            cam['cx'], cam['cy'],
            IMG_WIDTH, IMG_HEIGHT
        )

    num_visible = int(np.sum(radii > 0))
    print(f"\n  ▸ Preprocess:  {prep_ms:7.3f} ms   ({num_visible:,} visible)")

    # ---- Step 2: Sort by depth (Metal GPU) ----
    sorted_indices, sort_ms = radix_sort_by_depth(depths)
    print(f"  ▸ Depth sort:  {sort_ms:7.3f} ms")

    # ---- Step 3: Tile Binning (Metal GPU) ----
    point_list, tile_bins, num_isect, bin_ms = tile_binning(
        sorted_indices.astype(np.uint32),
        radii.astype(np.uint32),
        tile_min.astype(np.uint32),
        tile_max.astype(np.uint32),
        np.uint32(num_tiles_x),
        np.uint32(num_tiles_y),
    )
    gpu_total_ms = prep_ms + sort_ms + bin_ms
    print(f"  ▸ Tile binning: {bin_ms:6.3f} ms   ({num_isect:,} intersections)")
    print(f"  ▸ GPU Pipeline: {gpu_total_ms:6.3f} ms total")

    # ---- Step 4: NumPy reference ----
    t0 = time.perf_counter()
    ref_point_list, ref_tile_bins, ref_num_isect = reference_tile_binning(
        sorted_indices, radii, tile_min, tile_max,
        num_tiles_x, num_tiles_y,
    )
    ref_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  ▸ NumPy ref:   {ref_ms:7.1f} ms")

    # ---- Step 5: Correctness checks ----
    print(f"\n{'─' * 62}")
    print(f"  Correctness Checks")
    print(f"{'─' * 62}")

    all_pass = True

    # 5a. Intersection count
    isect_ok = (num_isect == ref_num_isect)
    print(f"  num_intersections:  Metal={num_isect:,}  NumPy={ref_num_isect:,}  "
          f"{'✓' if isect_ok else '✗'}")
    all_pass &= isect_ok

    if isect_ok and num_isect > 0:
        # 5b. Point list (Gaussian IDs in tile-then-depth order)
        pl_match = np.array_equal(point_list, ref_point_list)
        print(f"  point_list match:   {pl_match}  {'✓' if pl_match else '✗'}")
        all_pass &= pl_match

        if not pl_match:
            mis = np.where(point_list != ref_point_list)[0]
            if len(mis) > 0:
                i = mis[0]
                print(f"    First mismatch @ [{i}]: Metal={point_list[i]}, "
                      f"NumPy={ref_point_list[i]}")

        # 5c. Tile bins (start, end per tile)
        tb_match = np.array_equal(tile_bins, ref_tile_bins)
        print(f"  tile_bins match:    {tb_match}  {'✓' if tb_match else '✗'}")
        all_pass &= tb_match

        if not tb_match:
            mis = np.where(np.any(tile_bins != ref_tile_bins, axis=1))[0]
            if len(mis) > 0:
                t = mis[0]
                print(f"    First mismatch @ tile {t}: Metal={tile_bins[t]}, "
                      f"NumPy={ref_tile_bins[t]}")

        # 5d. Statistics
        non_empty = int(np.sum(ref_tile_bins[:, 1] > ref_tile_bins[:, 0]))
        avg_per_tile = num_isect / max(non_empty, 1)
        print(f"\n  Non-empty tiles:    {non_empty:,} / {num_tiles:,}")
        print(f"  Avg Gaussians/tile: {avg_per_tile:.1f}")
        print(f"  Avg tiles/Gaussian: {num_isect / max(num_visible, 1):.1f}")

    # ---- Summary ----
    print(f"\n{'=' * 62}")
    if all_pass:
        print("  🎉 ALL TESTS PASSED — Tile binning is correct!")
        print(f"\n  GPU Pipeline Timing (100K Gaussians, 1280×720):")
        print(f"    Preprocess:   {prep_ms:7.3f} ms")
        print(f"    Depth Sort:   {sort_ms:7.3f} ms")
        print(f"    Tile Binning: {bin_ms:7.3f} ms")
        print(f"    ─────────────────────────")
        print(f"    Total:        {gpu_total_ms:7.3f} ms")
    else:
        print("  ❌ TESTS FAILED — check output above")
    print("=" * 62)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
