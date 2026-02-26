"""
Metal-GS v3: Zero-copy MPS autograd.Function wrapper.

All data stays on GPU (MPS device). No .cpu(), no numpy, no synchronize barriers.
The C++ extension dispatches Metal kernels directly on PyTorch's MPS command queue.

Usage:
    image = MetalGaussianRasterizer.apply(
        means3d, scales, quats, sh_coeffs, opacities,
        viewmat, campos, render_settings
    )
    loss = (image - target).pow(2).mean()
    loss.backward()
"""

import torch
from dataclasses import dataclass


def _get_core():
    """Lazy-load the compiled Metal extension."""
    try:
        from metal_gs import _metal_gs_core
        return _metal_gs_core
    except ImportError:
        import _metal_gs_core
        return _metal_gs_core


@dataclass
class RenderSettings:
    """Camera / rendering parameters for the Metal rasterizer."""
    viewmat: torch.Tensor       # [4, 4] float32 on MPS
    tan_fovx: float
    tan_fovy: float
    focal_x: float
    focal_y: float
    principal_x: float
    principal_y: float
    img_width: int
    img_height: int
    sh_degree: int = 3
    bg_color: tuple = (0.0, 0.0, 0.0)
    max_gaussians_per_tile: int = 4096


class MetalGaussianRasterizer(torch.autograd.Function):
    """
    Differentiable 3D Gaussian rasterizer — pure MPS, zero-copy.

    Forward: SH → preprocess → sort → tile_bin → rasterize (Metal compute encoders)
    Backward: rasterize_bw → preprocess_bw → sh_bw (single Metal encoder)

    All tensors stay on MPS device throughout.
    """

    @staticmethod
    def forward(ctx, means3d, scales, quats, sh_coeffs, opacities,
                viewmat, campos, settings: RenderSettings):
        core = _get_core()

        # Ensure inputs are on MPS and contiguous
        assert means3d.device.type == 'mps', "All tensors must be on MPS device"

        results = core.render_forward(
            means3d.contiguous(),
            scales.contiguous(),
            quats.contiguous(),
            sh_coeffs.contiguous(),
            opacities.contiguous(),
            viewmat.contiguous(),
            campos.contiguous(),
            settings.tan_fovx, settings.tan_fovy,
            settings.focal_x, settings.focal_y,
            settings.principal_x, settings.principal_y,
            settings.img_width, settings.img_height,
            settings.sh_degree,
            settings.bg_color[0], settings.bg_color[1], settings.bg_color[2],
            settings.max_gaussians_per_tile,
        )

        # Unpack: [out_img, T_final, n_contrib, means2d, cov2d, depths, radii,
        #          tile_bins, point_list, sorted_indices, colors_fp16, colors_fp32, directions]
        (out_img, T_final, n_contrib, means2d, cov2d, depths, radii,
         tile_bins, point_list, sorted_indices,
         colors_fp16, colors_fp32, directions) = results

        # Save for backward (all MPS tensors, no CPU copies)
        ctx.save_for_backward(
            means3d, scales, quats, sh_coeffs, opacities,
            viewmat, campos,
            means2d, cov2d, radii, colors_fp32, colors_fp16,
            tile_bins, point_list, T_final, n_contrib,
        )
        ctx.settings = settings

        return out_img

    @staticmethod
    def backward(ctx, grad_image):
        core = _get_core()
        settings = ctx.settings

        (means3d, scales, quats, sh_coeffs, opacities,
         viewmat, campos,
         means2d, cov2d, radii, colors_fp32, colors_fp16,
         tile_bins, point_list, T_final, n_contrib) = ctx.saved_tensors

        grads = core.render_backward(
            means3d, scales, quats, sh_coeffs, opacities,
            viewmat, campos,
            means2d, cov2d, radii, colors_fp32, colors_fp16,
            tile_bins, point_list, T_final, n_contrib,
            grad_image.contiguous(),
            settings.tan_fovx, settings.tan_fovy,
            settings.focal_x, settings.focal_y,
            settings.principal_x, settings.principal_y,
            settings.img_width, settings.img_height,
            settings.sh_degree,
            settings.bg_color[0], settings.bg_color[1], settings.bg_color[2],
            settings.max_gaussians_per_tile,
        )

        # grads = [dL_m3d_prep, dL_m3d_sh, dL_scales, dL_quats, dL_sh, dL_opac]
        dL_m3d_prep, dL_m3d_sh, dL_scales, dL_quats, dL_sh, dL_opac = grads

        # Combine means3d gradients from preprocess and SH paths
        dL_means3d = dL_m3d_prep + dL_m3d_sh

        # Return gradients for: means3d, scales, quats, sh_coeffs, opacities,
        #                       viewmat, campos, settings
        return (
            dL_means3d if ctx.needs_input_grad[0] else None,
            dL_scales  if ctx.needs_input_grad[1] else None,
            dL_quats   if ctx.needs_input_grad[2] else None,
            dL_sh      if ctx.needs_input_grad[3] else None,
            dL_opac    if ctx.needs_input_grad[4] else None,
            None, None, None,
        )


def simple_knn_metal(points, k_neighbors=3, search_window=32):
    """
    Compute average squared distance to K nearest neighbors.

    Parameters
    ----------
    points : torch.Tensor, float32, shape [N, 3], MPS device
    k_neighbors : int
    search_window : int

    Returns
    -------
    avg_sq_dist : torch.Tensor, float32, shape [N], MPS device
    """
    core = _get_core()
    if not isinstance(points, torch.Tensor):
        # Legacy compatibility: accept numpy and convert
        import numpy as np
        points = torch.from_numpy(np.ascontiguousarray(points, dtype=np.float32)).to('mps')
    assert points.device.type == 'mps', "points must be on MPS device"
    result = core.simple_knn_metal(points.contiguous(), k_neighbors, search_window)
    # Return (distances, time_ms) tuple for backward compatibility
    return result, 0.0
