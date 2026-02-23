"""Test MPS tensor flow through MetalGaussianRasterizer autograd."""
import os
import torch
import numpy as np

os.environ.setdefault('METAL_GS_METALLIB_DIR', 'csrc/kernels')
from metal_gs.rasterizer import MetalGaussianRasterizer, RenderSettings

# Check MPS availability
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

N = 500
torch.manual_seed(42)

# Create tensors on MPS
means3d  = torch.randn(N, 3, device=device, dtype=torch.float32) * 0.3
means3d.requires_grad_(True)
scales   = torch.randn(N, 3, device=device, dtype=torch.float32) * 0.3
scales.requires_grad_(True)
quats    = torch.randn(N, 4, device=device, dtype=torch.float32)
quats    = quats / quats.norm(dim=1, keepdim=True)
quats.requires_grad_(True)
sh_coeffs = torch.randn(N, 1, 3, device=device, dtype=torch.float16)
sh_coeffs.requires_grad_(True)
opacities = torch.rand(N, device=device, dtype=torch.float32)
opacities.requires_grad_(True)

viewmat = torch.eye(4, device=device, dtype=torch.float32)
viewmat[2, 3] = 5.0
campos = torch.tensor([0.0, 0.0, -5.0], device=device, dtype=torch.float32)

settings = RenderSettings(
    viewmat=viewmat.cpu().numpy(),
    tan_fovx=1.0, tan_fovy=1.0,
    focal_x=128.0, focal_y=128.0,
    principal_x=128.0, principal_y=128.0,
    img_width=256, img_height=256,
    sh_degree=0, bg_color=(0.0, 0.0, 0.0),
)

print(f"Input device: {means3d.device}")

# Forward
image = MetalGaussianRasterizer.apply(
    means3d, scales, quats, sh_coeffs, opacities,
    viewmat, campos, settings
)
print(f"Output device: {image.device}")
print(f"Output shape: {image.shape}, range=[{image.min():.4f}, {image.max():.4f}]")

# Backward
loss = image.sum()
loss.backward()

print(f"\nGradient devices:")
print(f"  means3d.grad: {means3d.grad.device}, non-zero={means3d.grad.abs().sum():.4f}")
print(f"  scales.grad:  {scales.grad.device}, non-zero={scales.grad.abs().sum():.4f}")
print(f"  quats.grad:   {quats.grad.device}, non-zero={quats.grad.abs().sum():.4f}")
print(f"  sh.grad:      {sh_coeffs.grad.device}, non-zero={sh_coeffs.grad.abs().sum():.4f}")
print(f"  opac.grad:    {opacities.grad.device}, non-zero={opacities.grad.abs().sum():.4f}")

# Test optimizer step on MPS
optimizer = torch.optim.Adam([means3d, scales, quats], lr=0.01)
optimizer.step()
print(f"\n✅ Adam optimizer.step() on {device} — success")
print(f"   means3d still on {means3d.device}")
