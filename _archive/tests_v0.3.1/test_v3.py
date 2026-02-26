"""Quick smoke test for Metal-GS v3 MPS custom ops."""
import torch
from metal_gs.rasterizer import _get_core

core = _get_core()

N = 100
device = 'mps'

means3d = torch.randn(N, 3, device=device)
scales = torch.randn(N, 3, device=device)
quats = torch.randn(N, 4, device=device)
quats = quats / quats.norm(dim=1, keepdim=True)
sh_coeffs = torch.randn(N, 16, 3, device=device, dtype=torch.float16)
opacities = torch.sigmoid(torch.randn(N, device=device))
viewmat = torch.eye(4, device=device)
viewmat[2, 3] = -5.0
campos = torch.tensor([0.0, 0.0, 0.0], device=device)

print("=== Forward Test ===")
results = core.render_forward(
    means3d, scales, quats, sh_coeffs, opacities,
    viewmat, campos,
    0.5, 0.5,
    512.0, 512.0,
    256.0, 256.0,
    512, 512,
    3,
    0.0, 0.0, 0.0,
    4096,
)

names = ['out_img', 'T_final', 'n_contrib', 'means2d', 'cov2d', 'depths', 'radii',
         'tile_bins', 'point_list', 'sorted_indices', 'colors_fp16', 'colors_fp32', 'directions']
print(f"Forward returned {len(results)} tensors:")
for i, (name, t) in enumerate(zip(names, results)):
    print(f"  [{i}] {name}: shape={t.shape}, dtype={t.dtype}, device={t.device}")

print("\n=== Backward Test ===")
out_img, T_final, n_contrib, means2d, cov2d, depths, radii, \
    tile_bins, point_list, sorted_indices, colors_fp16, colors_fp32, directions = results

grad_image = torch.randn_like(out_img)
bw_results = core.render_backward(
    means3d, scales, quats, sh_coeffs, opacities,
    viewmat, campos,
    means2d, cov2d, radii, colors_fp32, colors_fp16,
    tile_bins, point_list, T_final, n_contrib,
    grad_image,
    0.5, 0.5,
    512.0, 512.0,
    256.0, 256.0,
    512, 512,
    3,
    0.0, 0.0, 0.0,
    4096,
)

bw_names = ['dL_m3d_prep', 'dL_m3d_sh', 'dL_scales', 'dL_quats', 'dL_sh', 'dL_opac']
print(f"Backward returned {len(bw_results)} tensors:")
for name, t in zip(bw_names, bw_results):
    print(f"  {name}: shape={t.shape}, dtype={t.dtype}, device={t.device}")

print("\n=== Autograd Test ===")
means3d_ag = means3d.clone().requires_grad_(True)
scales_ag = scales.clone().requires_grad_(True)
quats_ag = quats.clone().requires_grad_(True)
sh_ag = sh_coeffs.clone().requires_grad_(True)
opac_ag = opacities.clone().requires_grad_(True)

from metal_gs.rasterizer import MetalGaussianRasterizer, RenderSettings
settings = RenderSettings(
    viewmat=viewmat,
    tan_fovx=0.5, tan_fovy=0.5,
    focal_x=512.0, focal_y=512.0,
    principal_x=256.0, principal_y=256.0,
    img_width=512, img_height=512,
    sh_degree=3,
)
image = MetalGaussianRasterizer.apply(
    means3d_ag, scales_ag, quats_ag, sh_ag, opac_ag,
    viewmat, campos, settings,
)
print(f"Rendered image: shape={image.shape}, device={image.device}")
loss = image.sum()
loss.backward()
print(f"means3d.grad: shape={means3d_ag.grad.shape}, norm={means3d_ag.grad.norm().item():.4f}")
print(f"scales.grad: shape={scales_ag.grad.shape}, norm={scales_ag.grad.norm().item():.4f}")
print(f"sh.grad: shape={sh_ag.grad.shape}, norm={sh_ag.grad.norm().item():.4f}")

print("\nAll tests PASSED!")
