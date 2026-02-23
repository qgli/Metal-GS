"""
Verification script for Metal GPU Forward Rasterization.

Full end-to-end pipeline:
  1. Generate 100K colored Gaussians with random opacities
  2. render_forward() — preprocess → sort → bin → rasterize (all Metal GPU)
  3. Save rendered image as PNG
  4. NumPy reference rasterizer for correctness comparison
  5. Print per-stage timing + total FPS

Target: visually correct rendering, sub-100ms full pipeline.
"""

import numpy as np
import time
import sys
import os

try:
    from metal_gs._metal_gs_core import render_forward, rasterize_forward
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Run 'CC=/usr/bin/clang CXX=/usr/bin/clang++ pip install -e .' first.")
    sys.exit(1)


# ---- Constants ----
TILE_SIZE  = 16
IMG_WIDTH  = 1280
IMG_HEIGHT = 720
BG_COLOR   = (0.0, 0.0, 0.0)   # black background


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


def numpy_rasterize_reference(means2d, cov2d, colors, opacities,
                               tile_bins, point_list,
                               img_w, img_h, num_tiles_x, bg):
    """
    NumPy reference rasterizer for correctness comparison.
    Processes a small crop (center 64×64) for speed.
    """
    # Only check a small region for numerical comparison
    crop_x, crop_y = img_w // 2 - 32, img_h // 2 - 32
    crop_w, crop_h = 64, 64

    ref_img = np.zeros((crop_h, crop_w, 3), dtype=np.float32)

    for py in range(crop_y, crop_y + crop_h):
        for px in range(crop_x, crop_x + crop_w):
            pixel = np.array([px + 0.5, py + 0.5], dtype=np.float64)
            tx = px // TILE_SIZE
            ty = py // TILE_SIZE
            tile_id = ty * num_tiles_x + tx
            start = int(tile_bins[tile_id, 0])
            end   = int(tile_bins[tile_id, 1])

            T = 1.0
            C = np.zeros(3, dtype=np.float64)

            for k in range(start, end):
                if T < 1e-4:
                    break
                gid = int(point_list[k])

                mean = means2d[gid].astype(np.float64)
                a, b, c = [float(x) for x in cov2d[gid]]

                det = a * c - b * b
                if det < 1e-6:
                    continue
                det_inv = 1.0 / det
                inv_a =  c * det_inv
                inv_b = -b * det_inv
                inv_c =  a * det_inv

                d = pixel - mean
                maha = inv_a * d[0]**2 + 2.0 * inv_b * d[0] * d[1] + inv_c * d[1]**2

                if maha < 0 or maha > 18.0:
                    continue
                weight = np.exp(-0.5 * maha)

                alpha = min(0.999, float(opacities[gid]) * weight)
                if alpha < 1.0 / 255.0:
                    continue

                C += T * alpha * colors[gid].astype(np.float64)
                T *= (1.0 - alpha)

            C += T * np.array(bg, dtype=np.float64)
            ry = py - crop_y
            rx = px - crop_x
            ref_img[ry, rx] = C.astype(np.float32)

    return ref_img, crop_x, crop_y, crop_w, crop_h


def save_png(image, path):
    """Save float32 [H,W,3] image as PNG (0-255 uint8)."""
    try:
        from PIL import Image
        img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(img_u8).save(path)
        return True
    except ImportError:
        try:
            import matplotlib.pyplot as plt
            plt.imsave(path, np.clip(image, 0, 1))
            return True
        except ImportError:
            print("  ⚠ Neither PIL nor matplotlib available, skipping PNG save")
            return False


def main():
    N = 100_000
    num_tiles_x = (IMG_WIDTH  + TILE_SIZE - 1) // TILE_SIZE   # 80
    num_tiles_y = (IMG_HEIGHT + TILE_SIZE - 1) // TILE_SIZE   # 45

    print("=" * 62)
    print("  Metal-GS Forward Rasterization — Full Pipeline Test")
    print("=" * 62)
    print(f"  Gaussians:  {N:,}")
    print(f"  Resolution: {IMG_WIDTH}×{IMG_HEIGHT}")
    print(f"  Tile grid:  {num_tiles_x}×{num_tiles_y}")
    print(f"  Background: {BG_COLOR}")

    # ---- Generate random colored Gaussians ----
    np.random.seed(42)
    means3d = np.random.uniform(-2.0, 2.0, (N, 3)).astype(np.float32)
    scales  = np.exp(np.random.uniform(-4, -1.5, (N, 3))).astype(np.float32)
    quats   = np.random.randn(N, 4).astype(np.float32)
    quats  /= np.linalg.norm(quats, axis=1, keepdims=True)

    # Random vivid colors
    colors = np.random.uniform(0.1, 1.0, (N, 3)).astype(np.float32)
    # Random opacities (sigmoid-like range)
    opacities = np.random.uniform(0.3, 0.95, N).astype(np.float32)

    cam = make_camera()

    # ---- Run full pipeline (Metal GPU) ----
    print(f"\n{'─' * 62}")
    print(f"  Metal GPU Full Pipeline")
    print(f"{'─' * 62}")

    # Warmup run
    _ = render_forward(
        means3d, scales, quats, cam['viewmat'],
        colors, opacities,
        cam['tan_fovx'], cam['tan_fovy'],
        cam['focal_x'], cam['focal_y'],
        cam['cx'], cam['cy'],
        np.uint32(IMG_WIDTH), np.uint32(IMG_HEIGHT),
        BG_COLOR[0], BG_COLOR[1], BG_COLOR[2],
    )

    # Timed run
    result = render_forward(
        means3d, scales, quats, cam['viewmat'],
        colors, opacities,
        cam['tan_fovx'], cam['tan_fovy'],
        cam['focal_x'], cam['focal_y'],
        cam['cx'], cam['cy'],
        np.uint32(IMG_WIDTH), np.uint32(IMG_HEIGHT),
        BG_COLOR[0], BG_COLOR[1], BG_COLOR[2],
    )

    image     = result['image']
    prep_ms   = result['preprocess_ms']
    sort_ms   = result['sort_ms']
    bin_ms    = result['binning_ms']
    rast_ms   = result['rasterize_ms']
    total_ms  = result['total_ms']
    nvis      = result['num_visible']
    num_isect = result['num_intersections']
    fps       = 1000.0 / total_ms if total_ms > 0 else float('inf')

    print(f"  ▸ Preprocess:   {prep_ms:7.3f} ms   ({nvis:,} visible)")
    print(f"  ▸ Depth sort:   {sort_ms:7.3f} ms")
    print(f"  ▸ Tile binning: {bin_ms:7.3f} ms   ({num_isect:,} intersections)")
    print(f"  ▸ Rasterize:    {rast_ms:7.3f} ms")
    print(f"  ──────────────────────────────")
    print(f"  ▸ Total:        {total_ms:7.3f} ms   ({fps:.1f} FPS)")

    # ---- Save PNG ----
    png_path = os.path.join(os.path.dirname(__file__), "render_output.png")
    print(f"\n  Saving rendered image → {png_path}")
    if save_png(image, png_path):
        print(f"  ✓ PNG saved ({IMG_WIDTH}×{IMG_HEIGHT})")

    # ---- Basic sanity checks ----
    print(f"\n{'─' * 62}")
    print(f"  Sanity Checks")
    print(f"{'─' * 62}")

    all_pass = True

    # Check image shape
    shape_ok = image.shape == (IMG_HEIGHT, IMG_WIDTH, 3)
    print(f"  Image shape:     {image.shape}  {'✓' if shape_ok else '✗'}")
    all_pass &= shape_ok

    # Check no NaN/Inf
    finite_ok = np.all(np.isfinite(image))
    print(f"  All finite:      {finite_ok}  {'✓' if finite_ok else '✗'}")
    all_pass &= finite_ok

    # Check pixel range [0, 1] (approximately)
    range_ok = (image.min() >= -0.01 and image.max() <= 1.01)
    print(f"  Pixel range:     [{image.min():.4f}, {image.max():.4f}]  "
          f"{'✓' if range_ok else '✗'}")
    all_pass &= range_ok

    # Check not all black (should have some color)
    mean_val = image.mean()
    not_black = mean_val > 0.001
    print(f"  Mean pixel val:  {mean_val:.6f}  {'✓' if not_black else '✗'}")
    all_pass &= not_black

    # ---- Numerical comparison with NumPy reference (center 64×64 crop) ----
    print(f"\n{'─' * 62}")
    print(f"  Numerical Reference Check (center 64×64 crop)")
    print(f"{'─' * 62}")

    # Need intermediate results for reference — call individual stages
    from metal_gs._metal_gs_core import (
        preprocess_forward, radix_sort_by_depth, tile_binning
    )

    means2d, cov2d, depths, radii, tile_min, tile_max, _ = \
        preprocess_forward(
            means3d, scales, quats, cam['viewmat'],
            cam['tan_fovx'], cam['tan_fovy'],
            cam['focal_x'], cam['focal_y'],
            cam['cx'], cam['cy'],
            IMG_WIDTH, IMG_HEIGHT
        )
    sorted_indices, _ = radix_sort_by_depth(depths)
    point_list, tile_bins_np, num_isect_check, _ = tile_binning(
        sorted_indices.astype(np.uint32),
        radii.astype(np.uint32),
        tile_min.astype(np.uint32),
        tile_max.astype(np.uint32),
        np.uint32(num_tiles_x),
        np.uint32(num_tiles_y),
    )

    t0 = time.perf_counter()
    ref_crop, cx, cy, cw, ch = numpy_rasterize_reference(
        means2d, cov2d, colors, opacities,
        tile_bins_np, point_list,
        IMG_WIDTH, IMG_HEIGHT, num_tiles_x, BG_COLOR,
    )
    ref_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  NumPy reference: {ref_ms:.1f} ms (64×64 crop only)")

    # Compare Metal crop vs NumPy crop
    metal_crop = image[cy:cy+ch, cx:cx+cw]
    abs_err = np.abs(metal_crop - ref_crop)
    max_err = abs_err.max()
    mean_err = abs_err.mean()
    # Allow small FP32 tolerance (cooperative fetch + different eval order)
    err_ok = max_err < 0.01
    print(f"  Max abs error:   {max_err:.6f}  {'✓' if err_ok else '✗'}")
    print(f"  Mean abs error:  {mean_err:.6f}")
    all_pass &= err_ok

    # ---- Final verdict ----
    print(f"\n{'=' * 62}")
    if all_pass:
        print("  ✅ ALL TESTS PASSED — Forward Rasterization verified!")
    else:
        print("  ❌ SOME TESTS FAILED")
    print(f"{'=' * 62}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
