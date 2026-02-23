"""Minimal loop to isolate Metal dispatch hang — no PyTorch optimizer."""
import sys, os, time
import numpy as np

os.environ.setdefault("METAL_GS_METALLIB_DIR",
    os.path.join(os.path.dirname(__file__), 'csrc', 'kernels'))

from metal_gs import _metal_gs_core as core

# Cat scene: 165K gaussians, 1032x688
N = 165000
np.random.seed(42)
means3d = np.random.randn(N, 3).astype(np.float32) * 0.5
scales  = np.full((N, 3), -3.0, dtype=np.float32)
quats   = np.zeros((N, 4), dtype=np.float32); quats[:, 3] = 1.0
opacities = np.full(N, 0.5, dtype=np.float32)

viewmat = np.eye(4, dtype=np.float32); viewmat[2, 3] = 3.0
campos = np.array([0.0, 0.0, -3.0], dtype=np.float32)

W, H = 1032, 688  # Cat scene resolution
tan_fovx, tan_fovy = 0.5, 0.33
focal_x, focal_y = W / (2.0 * tan_fovx), H / (2.0 * tan_fovy)

K = 1; sh_degree = 0
sh_coeffs = np.random.randn(N, K, 3).astype(np.float16)

# SH colors
dirs = means3d - campos[np.newaxis, :]
norms = np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-8)
dirs = (dirs / norms).astype(np.float32)
colors_fp16, _ = core.compute_sh_forward(dirs, sh_coeffs, N, K, sh_degree)
colors_fp16 = colors_fp16.view(np.float16)
colors_fp32 = colors_fp16.astype(np.float32)

print(f"Scene: {N} Gaussians, {W}x{H} image", flush=True)

for i in range(50):
    t0 = time.time()
    fwd = core.render_forward(
        means3d, scales, quats, viewmat,
        colors_fp32, opacities,
        tan_fovx, tan_fovy, focal_x, focal_y,
        W/2, H/2, W, H,
        0.0, 0.0, 0.0,
    )
    t_fwd = time.time() - t0

    grad_img = np.random.randn(H, W, 3).astype(np.float32) * 0.01

    t1 = time.time()
    bwd = core.render_backward(
        means3d, scales, quats, viewmat,
        colors_fp32, opacities,
        campos, sh_coeffs, colors_fp16,
        K, sh_degree,
        tan_fovx, tan_fovy, focal_x, focal_y,
        W/2, H/2, W, H,
        0.0, 0.0, 0.0,
        fwd["means2d"], fwd["cov2d"], fwd["radii"],
        fwd["tile_bins"], fwd["point_list"],
        fwd["T_final"], fwd["n_contrib"],
        grad_img,
    )
    t_bwd = time.time() - t1

    print(f"  iter {i:3d}  fwd={t_fwd*1000:.0f}ms  bwd={t_bwd*1000:.0f}ms  total={((t_fwd+t_bwd)*1000):.0f}ms", flush=True)

print("All 100 iterations completed — no hang")
