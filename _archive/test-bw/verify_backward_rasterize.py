"""
Metal-GS: Rasterize Backward Gradient Verification

Tests the backward rasterization kernel by:
  1. Generating a small set of Gaussians (N=100)
  2. Running the full forward pipeline (preprocess → sort → bin → rasterize)
  3. Running the backward kernel with a random dL/dC_pixel
  4. Computing reference gradients in NumPy (double precision)
  5. Comparing Metal vs NumPy gradients (max absolute error)

The NumPy reference implements the EXACT same alpha-blending forward pass
and then manually derives gradients using the reverse-traversal formulas.
"""

import numpy as np
import sys
import os

try:
    from metal_gs._metal_gs_core import (
        preprocess_forward, radix_sort_by_depth,
        tile_binning, rasterize_forward, rasterize_backward
    )
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Run 'CC=/usr/bin/clang CXX=/usr/bin/clang++ pip install -e .' first.")
    sys.exit(1)


# ---- Constants ----
TILE_SIZE  = 16
IMG_WIDTH  = 128
IMG_HEIGHT = 128
BG_COLOR   = (0.2, 0.3, 0.5)   # non-zero background for thorough testing
N_GAUSSIANS = 100


def make_camera():
    """Pinhole camera at (0,0,5) looking at origin, 60° FOV."""
    fov_x = np.radians(60)
    fov_y = 2 * np.arctan(np.tan(fov_x / 2) * IMG_HEIGHT / IMG_WIDTH)
    fx = IMG_WIDTH / (2 * np.tan(fov_x / 2))
    fy = IMG_HEIGHT / (2 * np.tan(fov_y / 2))
    cx, cy = IMG_WIDTH / 2.0, IMG_HEIGHT / 2.0

    viewmat = np.eye(4, dtype=np.float32)
    viewmat[2, 3] = 5.0

    return dict(
        viewmat=viewmat,
        tan_fovx=float(np.tan(fov_x / 2)),
        tan_fovy=float(np.tan(fov_y / 2)),
        focal_x=float(fx), focal_y=float(fy),
        cx=float(cx), cy=float(cy),
    )


def numpy_reference_backward(means2d, cov2d, colors, opacities,
                               tile_bins, point_list,
                               T_final, n_contrib,
                               dL_dC_pixel,
                               img_w, img_h, num_tiles_x, bg):
    """
    NumPy reference: compute exact gradients through alpha blending.
    All computation in float64 for maximum precision.
    
    Returns: dL_d_rgb, dL_d_opacity, dL_d_cov2d, dL_d_mean2d
    """
    N = len(opacities)
    dL_d_rgb  = np.zeros((N, 3), dtype=np.float64)
    dL_d_opac = np.zeros(N, dtype=np.float64)
    dL_d_cov  = np.zeros((N, 3), dtype=np.float64)
    dL_d_mean = np.zeros((N, 2), dtype=np.float64)
    
    means2d_64 = means2d.astype(np.float64)
    cov2d_64   = cov2d.astype(np.float64)
    colors_64  = colors.astype(np.float64)
    opac_64    = opacities.astype(np.float64)
    dL_dC_64   = dL_dC_pixel.astype(np.float64)
    bg_64      = np.array(bg, dtype=np.float64)
    
    for py in range(img_h):
        for px in range(img_w):
            pixel = np.array([px + 0.5, py + 0.5], dtype=np.float64)
            tx = px // TILE_SIZE
            ty = py // TILE_SIZE
            tile_id = ty * num_tiles_x + tx
            start = int(tile_bins[tile_id, 0])
            end   = int(tile_bins[tile_id, 1])
            pixel_idx = py * img_w + px
            
            dL_dC = dL_dC_64[pixel_idx]
            T_final_px = float(T_final[py, px])
            last_c = int(n_contrib[py, px])
            
            # ---- Collect contributing Gaussians (forward pass replay) ----
            # We need to know alpha_i for each contributing Gaussian
            contribs = []  # list of (global_sorted_idx, gauss_id, alpha_i)
            
            T_fwd = 1.0
            for k in range(start, end):
                if T_fwd < 1e-4:
                    break
                gid = int(point_list[k])
                
                mean = means2d_64[gid]
                a, b, c = cov2d_64[gid]
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
                alpha = min(0.999, opac_64[gid] * weight)
                if alpha < 1.0 / 255.0:
                    continue
                
                contribs.append((k, gid, alpha, maha, weight))
                T_fwd *= (1.0 - alpha)
            
            if len(contribs) == 0:
                continue
            
            # ---- Backward pass: reverse traversal ----
            T = T_final_px
            accum = bg_64 * T  # background * final_T
            
            for i in range(len(contribs) - 1, -1, -1):
                k_idx, gid, alpha, maha, weight = contribs[i]
                
                one_minus_alpha = 1.0 - alpha
                oma_safe = max(one_minus_alpha, 1e-5)
                T_i = T / oma_safe
                
                c_i = colors_64[gid]
                
                # dL/d_alpha
                dL_dalpha = np.dot(T_i * c_i - accum / oma_safe, dL_dC)
                
                # Update accum and T for next iteration
                accum += c_i * alpha * T_i
                T = T_i
                
                # dL/d_color
                dL_d_rgb[gid] += alpha * T_i * dL_dC
                
                # dL/d_opacity
                op = opac_64[gid]
                dL_d_opac[gid] += dL_dalpha * (alpha / max(op, 1e-8))
                
                # dL/d_sigma (Mahalanobis distance)
                dL_dsigma = dL_dalpha * (-0.5 * alpha)
                
                # Compute conic inverse for this Gaussian
                a_cov, b_cov, c_cov = cov2d_64[gid]
                det = a_cov * c_cov - b_cov * b_cov
                if det < 1e-6:
                    continue
                det_inv = 1.0 / det
                inv_a =  c_cov * det_inv
                inv_b = -b_cov * det_inv
                inv_c =  a_cov * det_inv
                
                mean = means2d_64[gid]
                d = pixel - mean
                
                # dL/d(conic) = dL/d(inv_cov)
                dL_d_inv_a = dL_dsigma * d[0]**2
                dL_d_inv_b = dL_dsigma * 2.0 * d[0] * d[1]
                dL_d_inv_c = dL_dsigma * d[1]**2
                
                # Convert dL/d(conic) to dL/d(cov2d) via:
                # dL/dΣ = -Σ^{-1} · dL/dΣ^{-1} · Σ^{-1}
                g_a = dL_d_inv_a
                g_b = dL_d_inv_b
                g_c = dL_d_inv_c
                
                dL_da = -(inv_a * inv_a * g_a + 2.0 * inv_a * inv_b * g_b + inv_b * inv_b * g_c)
                dL_db = -(inv_a * inv_b * g_a + (inv_a * inv_c + inv_b * inv_b) * g_b + inv_b * inv_c * g_c)
                dL_dc = -(inv_b * inv_b * g_a + 2.0 * inv_b * inv_c * g_b + inv_c * inv_c * g_c)
                
                dL_d_cov[gid, 0] += dL_da
                dL_d_cov[gid, 1] += dL_db
                dL_d_cov[gid, 2] += dL_dc
                
                # dL/d(mean2d)
                dL_dmx = dL_dsigma * (-2.0) * (inv_a * d[0] + inv_b * d[1])
                dL_dmy = dL_dsigma * (-2.0) * (inv_b * d[0] + inv_c * d[1])
                dL_d_mean[gid, 0] += dL_dmx
                dL_d_mean[gid, 1] += dL_dmy
    
    return dL_d_rgb, dL_d_opac, dL_d_cov, dL_d_mean


def main():
    np.random.seed(123)
    
    num_tiles_x = (IMG_WIDTH  + TILE_SIZE - 1) // TILE_SIZE
    num_tiles_y = (IMG_HEIGHT + TILE_SIZE - 1) // TILE_SIZE
    
    print("=" * 66)
    print("  Metal-GS: Rasterize Backward Gradient Verification")
    print("=" * 66)
    print(f"  Gaussians:  {N_GAUSSIANS}")
    print(f"  Resolution: {IMG_WIDTH}×{IMG_HEIGHT}")
    print(f"  Tile grid:  {num_tiles_x}×{num_tiles_y}")
    print(f"  Background: {BG_COLOR}")
    print(f"  Strategy:   A (naive global atomic add)")
    print()
    
    # ---- Generate random Gaussians ----
    cam = make_camera()
    
    # Place Gaussians in a compact region so they are visible
    means3d = np.random.uniform(-1.0, 1.0, (N_GAUSSIANS, 3)).astype(np.float32)
    means3d[:, 2] = np.random.uniform(-0.5, 0.5, N_GAUSSIANS)  # closer to center
    
    scales = np.exp(np.random.uniform(-3.0, -1.5, (N_GAUSSIANS, 3))).astype(np.float32)
    
    # Random unit quaternions
    quats = np.random.randn(N_GAUSSIANS, 4).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    
    colors = np.random.uniform(0.1, 0.9, (N_GAUSSIANS, 3)).astype(np.float32)
    opacities = np.random.uniform(0.3, 0.95, N_GAUSSIANS).astype(np.float32)
    
    # ---- Step 1: Preprocess ----
    print("  [1/5] Preprocess forward...")
    means2d, cov2d, depths, radii, tile_min, tile_max, prep_ms = \
        preprocess_forward(
            means3d, scales, quats, cam['viewmat'],
            cam['tan_fovx'], cam['tan_fovy'],
            cam['focal_x'], cam['focal_y'],
            cam['cx'], cam['cy'],
            IMG_WIDTH, IMG_HEIGHT
        )
    n_visible = int(np.sum(radii > 0))
    print(f"         {n_visible}/{N_GAUSSIANS} visible, {prep_ms:.2f} ms")
    
    # ---- Step 2: Sort ----
    print("  [2/5] Radix sort by depth...")
    sorted_idx, sort_ms = radix_sort_by_depth(depths)
    print(f"         {sort_ms:.2f} ms")
    
    # ---- Step 3: Tile binning ----
    print("  [3/5] Tile binning...")
    point_list, tile_bins, num_isect, bin_ms = tile_binning(
        sorted_idx, radii, tile_min, tile_max,
        num_tiles_x, num_tiles_y
    )
    print(f"         {num_isect:,} intersections, {bin_ms:.2f} ms")
    
    # ---- Step 4: Forward rasterize (with T_final, n_contrib) ----
    print("  [4/5] Forward rasterize...")
    out_img, T_final, n_contrib, rast_ms = rasterize_forward(
        means2d, cov2d, colors, opacities,
        tile_bins, point_list,
        IMG_WIDTH, IMG_HEIGHT,
        num_tiles_x, num_tiles_y,
        BG_COLOR[0], BG_COLOR[1], BG_COLOR[2]
    )
    print(f"         {rast_ms:.2f} ms")
    print(f"         T_final range: [{T_final.min():.6f}, {T_final.max():.6f}]")
    print(f"         n_contrib max: {n_contrib.max()}")
    
    # ---- Step 5: Backward rasterize ----
    print("  [5/5] Backward rasterize (Metal)...")
    
    # Generate random upstream gradient
    dL_dC_pixel = np.random.randn(IMG_HEIGHT, IMG_WIDTH, 3).astype(np.float32) * 0.1
    
    dL_rgb_metal, dL_opac_metal, dL_cov_metal, dL_mean_metal, bw_ms = \
        rasterize_backward(
            means2d, cov2d, colors, opacities,
            tile_bins, point_list,
            T_final, n_contrib,
            dL_dC_pixel.reshape(-1, 3),
            IMG_WIDTH, IMG_HEIGHT,
            num_tiles_x, num_tiles_y,
            BG_COLOR[0], BG_COLOR[1], BG_COLOR[2]
        )
    print(f"         {bw_ms:.2f} ms")
    
    # ---- NumPy reference backward ----
    print()
    print("  Computing NumPy reference gradients (float64)...")
    print("  (this may take a minute for 128×128...)")
    
    dL_dC_flat = dL_dC_pixel.reshape(-1, 3)
    
    ref_rgb, ref_opac, ref_cov, ref_mean = numpy_reference_backward(
        means2d, cov2d, colors, opacities,
        tile_bins, point_list,
        T_final, n_contrib,
        dL_dC_flat,
        IMG_WIDTH, IMG_HEIGHT, num_tiles_x, BG_COLOR
    )
    
    # ---- Compare gradients ----
    print()
    print("=" * 66)
    print("  Gradient Comparison: Metal vs NumPy (float64)")
    print("=" * 66)
    
    # Only compare visible Gaussians (others should have zero gradient)
    visible_mask = radii > 0
    
    def report(name, metal_vals, ref_vals, mask=None):
        if mask is not None:
            m = metal_vals[mask].astype(np.float64)
            r = ref_vals[mask]
        else:
            m = metal_vals.astype(np.float64).flatten()
            r = ref_vals.flatten()
        
        abs_err = np.abs(m - r)
        max_err = abs_err.max() if len(abs_err) > 0 else 0
        mean_err = abs_err.mean() if len(abs_err) > 0 else 0
        
        # Relative error (avoid div by zero)
        denom = np.maximum(np.abs(r), 1e-8)
        rel_err = (abs_err / denom)
        max_rel = rel_err.max() if len(rel_err) > 0 else 0
        
        status = "✅" if max_err < 1e-3 else ("⚠️" if max_err < 1e-2 else "❌")
        print(f"  {status} {name:20s}  MaxAbsErr={max_err:.2e}  "
              f"MeanAbsErr={mean_err:.2e}  MaxRelErr={max_rel:.2e}")
        return max_err
    
    max_errs = []
    max_errs.append(report("dL/d_rgb",      dL_rgb_metal, ref_rgb))
    max_errs.append(report("dL/d_opacity",   dL_opac_metal, ref_opac))
    max_errs.append(report("dL/d_cov2d",     dL_cov_metal, ref_cov))
    max_errs.append(report("dL/d_mean2d",    dL_mean_metal, ref_mean))
    
    overall_max = max(max_errs)
    print()
    print(f"  Overall max absolute error: {overall_max:.2e}")
    
    if overall_max < 1e-3:
        print("  🎉 ALL GRADIENTS PASS — backward rasterization is correct!")
    elif overall_max < 1e-2:
        print("  ⚠️  Marginal pass — consider investigating numerical issues")
    else:
        print("  ❌ GRADIENT MISMATCH — debug required!")
        
        # Print detailed diagnostics
        print()
        print("  Diagnostic details:")
        for name, m_arr, r_arr in [
            ("dL/d_rgb",    dL_rgb_metal,  ref_rgb),
            ("dL/d_opacity",dL_opac_metal, ref_opac),
            ("dL/d_cov2d",  dL_cov_metal,  ref_cov),
            ("dL/d_mean2d", dL_mean_metal, ref_mean),
        ]:
            m = m_arr.astype(np.float64).flatten()
            r = r_arr.flatten()
            abs_diff = np.abs(m - r)
            worst_idx = np.argmax(abs_diff)
            print(f"    {name}: worst at flat idx {worst_idx}, "
                  f"Metal={m[worst_idx]:.8f}, NumPy={r[worst_idx]:.8f}, "
                  f"diff={abs_diff[worst_idx]:.2e}")
    
    print()
    print(f"  Timing: forward={rast_ms:.2f}ms  backward={bw_ms:.2f}ms  "
          f"ratio={bw_ms/max(rast_ms,0.01):.1f}×")
    print("=" * 66)
    
    return overall_max < 1e-2


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
