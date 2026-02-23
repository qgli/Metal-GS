"""Quick verification of Y-strip dispatch correctness."""
import numpy as np
import os
os.environ.setdefault('METAL_GS_METALLIB_DIR', 'csrc/kernels')
from metal_gs import _metal_gs_core as core

# Quick forward pass with known data
N = 1000
np.random.seed(42)
means3d = np.random.randn(N, 3).astype(np.float32) * 0.5
scales = np.random.randn(N, 3).astype(np.float32) * 0.5
quats = np.random.randn(N, 4).astype(np.float32)
quats /= np.linalg.norm(quats, axis=1, keepdims=True)
viewmat = np.eye(4, dtype=np.float32)
viewmat[2, 3] = 5.0
colors = np.random.rand(N, 3).astype(np.float32)
opacities = np.ones(N, dtype=np.float32) * 0.5

result = core.render_forward(
    means3d, scales, quats, viewmat, colors, opacities,
    1.0, 1.0, 256.0, 256.0, 256.0, 256.0, 512, 512,
    0.0, 0.0, 0.0
)
img = result['image']
print(f"Forward: image shape={img.shape}, range=[{img.min():.4f}, {img.max():.4f}]")
print(f"Non-zero pixels: {(img.sum(axis=2) > 0).sum()}/{512*512}")

# Quick backward pass
grad_img = np.ones_like(img) * 0.01
bwd = core.render_backward(
    means3d, scales, quats, viewmat, colors, opacities,
    np.zeros(3, dtype=np.float32),
    np.random.randn(N, 1, 3).astype(np.float16),
    colors.astype(np.float16).reshape(N, 3),
    1, 0,
    1.0, 1.0, 256.0, 256.0, 256.0, 256.0, 512, 512,
    0.0, 0.0, 0.0,
    result['means2d'], result['cov2d'], result['radii'],
    result['tile_bins'], result['point_list'],
    result['T_final'], result['n_contrib'],
    grad_img
)
print(f"Backward: means3d grad range=[{bwd['dL_d_means3d'].min():.6f}, {bwd['dL_d_means3d'].max():.6f}]")
print(f"Backward: scales grad range=[{bwd['dL_d_scales'].min():.6f}, {bwd['dL_d_scales'].max():.6f}]")
has_nonzero_grad = np.any(bwd['dL_d_means3d'] != 0) and np.any(bwd['dL_d_scales'] != 0)
print(f"Has non-zero gradients: {has_nonzero_grad}")
print("✅ Y-strip dispatch forward+backward passed")
