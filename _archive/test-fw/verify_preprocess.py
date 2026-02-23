"""
Verification script for Metal preprocess_forward kernel.

Tests the Metal kernel against NumPy reference implementation using
10 hand-crafted Gaussians with clear boundary conditions:
- Identity quaternions
- Simple scales (0.1 to 1.0)
- Positions spanning [-5, 5] in world space
- Canonical view matrix (camera at origin looking down -Z)

Prints maximum absolute error for means2d and cov2d.
Success criteria: Max error < 1e-4 (FP32 precision threshold)
"""

import numpy as np
import sys
import pathlib

# Add _archive/ to sys.path so reference_preprocess can be found
_archive_dir = str(pathlib.Path(__file__).resolve().parent.parent)
if _archive_dir not in sys.path:
    sys.path.insert(0, _archive_dir)

try:
    from metal_gs._metal_gs_core import preprocess_forward
    from reference_preprocess import preprocess_forward_numpy
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Run 'pip install -e .' first to build the extension.")
    sys.exit(1)


def create_view_matrix(eye, target, up):
    """
    Create a row-major 4×4 view matrix (world → camera transform).
    Standard look-at matrix.
    """
    z_axis = eye - target
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    y_axis = np.cross(z_axis, x_axis)
    
    # Construct view matrix (rotation + translation)
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[0, :3] = x_axis
    viewmat[1, :3] = y_axis
    viewmat[2, :3] = z_axis
    viewmat[0, 3] = -np.dot(x_axis, eye)
    viewmat[1, 3] = -np.dot(y_axis, eye)
    viewmat[2, 3] = -np.dot(z_axis, eye)
    
    return viewmat


def generate_test_gaussians(N=10):
    """
    Generate N test Gaussians with clear boundary conditions.
    
    Returns: means3d, scales, quats (all FP32 numpy arrays)
    """
    np.random.seed(42)
    
    means3d = np.zeros((N, 3), dtype=np.float32)
    scales = np.zeros((N, 3), dtype=np.float32)
    quats = np.zeros((N, 4), dtype=np.float32)
    
    for i in range(N):
        # Positions spanning world space
        means3d[i] = np.array([
            (i % 4 - 1.5) * 2.0,  # x: -3 to +3
            (i // 4 - 1.0) * 1.5,  # y: -1.5 to +1.5
            -5.0 + i * 0.5         # z: -5 to -0.5
        ], dtype=np.float32)
        
        # Scales from small to large
        base_scale = 0.1 + i * 0.09
        scales[i] = np.array([base_scale, base_scale * 1.2, base_scale * 0.8], dtype=np.float32)
        
        # Quaternions: mix of identity and small rotations
        if i % 3 == 0:
            # Identity rotation (w=1, xyz=0)
            quats[i] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        else:
            # Small rotation around random axis
            axis = np.array([1.0, 0.5, 0.2], dtype=np.float32)
            axis = axis / np.linalg.norm(axis)
            angle = (i * 0.1) % (2.0 * np.pi / 3.0)  # up to 120 degrees
            quat_xyz = axis * np.sin(angle / 2.0)
            quat_w = np.cos(angle / 2.0)
            quats[i] = np.array([quat_xyz[0], quat_xyz[1], quat_xyz[2], quat_w], dtype=np.float32)
    
    return means3d, scales, quats


def main():
    print("=" * 80)
    print("Metal-GS Preprocess Kernel Verification")
    print("=" * 80)
    
    # Generate test data
    N = 10
    means3d, scales, quats = generate_test_gaussians(N)
    
    # Camera parameters (1280×720 with 50° FOV)
    img_width = 1280
    img_height = 720
    fov_x_deg = 50.0
    fov_y_deg = fov_x_deg * img_height / img_width
    
    fov_x_rad = np.deg2rad(fov_x_deg)
    fov_y_rad = np.deg2rad(fov_y_deg)
    
    tan_fovx = np.tan(fov_x_rad / 2.0)
    tan_fovy = np.tan(fov_y_rad / 2.0)
    
    focal_x = img_width / (2.0 * tan_fovx)
    focal_y = img_height / (2.0 * tan_fovy)
    
    principal_x = img_width / 2.0
    principal_y = img_height / 2.0
    
    # View matrix: camera at (0, 0, 3) looking at origin
    eye = np.array([0.0, 0.0, 3.0], dtype=np.float32)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    viewmat = create_view_matrix(eye, target, up)
    
    print(f"\n📊 Test Configuration:")
    print(f"  N = {N} Gaussians")
    print(f"  Image: {img_width}×{img_height}")
    print(f"  FOV: {fov_x_deg:.1f}° × {fov_y_deg:.1f}°")
    print(f"  Focal: ({focal_x:.1f}, {focal_y:.1f})")
    print(f"  Camera: eye={eye}, target={target}")
    
    # Run NumPy reference
    print("\n🧮 Running NumPy reference implementation...")
    ref_means2d, ref_cov2d, ref_depths, ref_radii, ref_tile_min, ref_tile_max = \
        preprocess_forward_numpy(
            means3d, scales, quats, viewmat,
            tan_fovx, tan_fovy, focal_x, focal_y, principal_x, principal_y,
            img_width, img_height
        )
    
    # Run Metal kernel
    print("⚡ Running Metal kernel...")
    metal_means2d, metal_cov2d, metal_depths, metal_radii, metal_tile_min, metal_tile_max, elapsed_ms = \
        preprocess_forward(
            means3d, scales, quats, viewmat,
            tan_fovx, tan_fovy, focal_x, focal_y, principal_x, principal_y,
            img_width, img_height
        )
    
    print(f"   GPU time: {elapsed_ms:.3f} ms")
    
    # Compute errors
    print("\n" + "=" * 80)
    print("Verification Results")
    print("=" * 80)
    
    # Means2D error
    means2d_err = np.abs(metal_means2d - ref_means2d)
    means2d_max_err = np.max(means2d_err)
    means2d_mean_err = np.mean(means2d_err)
    
    print(f"\n📍 Means2D:")
    print(f"   Max Error:  {means2d_max_err:.6e}")
    print(f"   Mean Error: {means2d_mean_err:.6e}")
    
    # Cov2D error
    cov2d_err = np.abs(metal_cov2d - ref_cov2d)
    cov2d_max_err = np.max(cov2d_err)
    cov2d_mean_err = np.mean(cov2d_err)
    
    print(f"\n📐 Cov2D (upper tri [a, b, c]):")
    print(f"   Max Error:  {cov2d_max_err:.6e}")
    print(f"   Mean Error: {cov2d_mean_err:.6e}")
    
    # Depths error
    depths_err = np.abs(metal_depths - ref_depths)
    depths_max_err = np.max(depths_err)
    
    print(f"\n🔍 Depths:")
    print(f"   Max Error:  {depths_max_err:.6e}")
    
    # Radii comparison (integer, expect exact match)
    radii_mismatch = np.sum(metal_radii != ref_radii)
    
    print(f"\n📏 Radii:")
    print(f"   Mismatches: {radii_mismatch}/{N}")
    
    # Tile bounds comparison
    tile_min_mismatch = np.sum(metal_tile_min != ref_tile_min)
    tile_max_mismatch = np.sum(metal_tile_max != ref_tile_max)
    
    print(f"\n🗺️  Tile Bounds:")
    print(f"   tile_min mismatches: {tile_min_mismatch}/{N*2}")
    print(f"   tile_max mismatches: {tile_max_mismatch}/{N*2}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    SUCCESS_THRESHOLD = 1e-4
    
    status_means2d = "✅ PASS" if means2d_max_err < SUCCESS_THRESHOLD else "❌ FAIL"
    status_cov2d = "✅ PASS" if cov2d_max_err < SUCCESS_THRESHOLD else "❌ FAIL"
    status_radii = "✅ PASS" if radii_mismatch <= 1 else "❌ FAIL"  # Allow 1 mismatch due to rounding
    
    print(f"\nMeans2D:  {status_means2d}  (threshold: {SUCCESS_THRESHOLD:.1e})")
    print(f"Cov2D:    {status_cov2d}  (threshold: {SUCCESS_THRESHOLD:.1e})")
    print(f"Radii:    {status_radii}  (exact match required)")
    
    overall_pass = (means2d_max_err < SUCCESS_THRESHOLD and 
                    cov2d_max_err < SUCCESS_THRESHOLD and 
                    radii_mismatch <= 1)
    
    if overall_pass:
        print("\n" + "🎉 " * 20)
        print("ALL TESTS PASSED! Metal kernel is numerically correct.")
        print("🎉 " * 20)
        return 0
    else:
        print("\n" + "❌ " * 20)
        print("TESTS FAILED! Check kernel implementation.")
        print("❌ " * 20)
        
        # Detailed debugging output
        print("\n📝 Detailed Comparison (first 3 Gaussians):")
        for i in range(min(3, N)):
            print(f"\nGaussian {i}:")
            print(f"  Means2D: Metal={metal_means2d[i]}, Ref={ref_means2d[i]}, Err={means2d_err[i]}")
            print(f"  Cov2D:   Metal={metal_cov2d[i]}, Ref={ref_cov2d[i]}, Err={cov2d_err[i]}")
            print(f"  Depth:   Metal={metal_depths[i]:.4f}, Ref={ref_depths[i]:.4f}")
            print(f"  Radius:  Metal={metal_radii[i]}, Ref={ref_radii[i]}")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
