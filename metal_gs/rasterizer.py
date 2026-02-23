"""
Metal-GS: PyTorch autograd.Function wrapper for the Metal Gaussian Splatting renderer.

Bridges PyTorch's autograd with the Metal compute pipeline:
  Forward:  PyTorch tensors → numpy → render_forward (Metal GPU) → PyTorch tensor
  Backward: grad_image → numpy → render_backward (Metal GPU) → PyTorch gradient tensors

Usage:
    image = MetalGaussianRasterizer.apply(
        means3d, scales, quats, sh_coeffs, opacities,
        viewmat, campos, render_settings
    )
    loss = (image - target).pow(2).mean()
    loss.backward()  # populates .grad for all input tensors
"""

import torch
import numpy as np
import gc
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
    viewmat: np.ndarray         # [4, 4] float32 row-major
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
    max_gaussians_per_tile: int = 1024  # hard cap per tile (0 = unlimited)


class MetalGaussianRasterizer(torch.autograd.Function):
    """
    Differentiable 3D Gaussian rasterizer backed by Metal compute shaders.

    Forward: full render pipeline (SH eval → preprocess → sort → bin → rasterize)
    Backward: rasterize_bw → preprocess_bw → sh_bw → combined gradients

    Inputs (all torch.Tensor, float32 unless noted):
        means3d:   [N, 3]   Gaussian centers
        scales:    [N, 3]   log-scale parameters
        quats:     [N, 4]   rotation quaternions (xyzw)
        sh_coeffs: [N, K, 3] SH coefficients (float16)
        opacities: [N]       opacity logits (sigmoid applied outside)
        viewmat:   [4, 4]   camera view matrix (row-major)
        campos:    [3]      camera position
        settings:  RenderSettings dataclass

    Output:
        image: [H, W, 3] rendered RGB image (float32)
    """

    @staticmethod
    def forward(ctx, means3d, scales, quats, sh_coeffs, opacities,
                viewmat, campos, settings: RenderSettings):
        core = _get_core()

        # Track input device (MPS/CPU) — we return the output on the same device
        input_device = means3d.device

        # Ensure MPS operations are complete before Metal dispatch
        if input_device.type == 'mps':
            torch.mps.synchronize()

        # ---- Convert to numpy (contiguous float32, except SH which is float16) ----
        means3d_np  = means3d.detach().cpu().contiguous().numpy().astype(np.float32)
        scales_np   = scales.detach().cpu().contiguous().numpy().astype(np.float32)
        quats_np    = quats.detach().cpu().contiguous().numpy().astype(np.float32)
        opacities_np = opacities.detach().cpu().contiguous().numpy().astype(np.float32)
        viewmat_np  = viewmat.detach().cpu().contiguous().numpy().astype(np.float32)
        campos_np   = campos.detach().cpu().contiguous().numpy().astype(np.float32)

        # SH coefficients: keep as float16 for SH forward kernel
        sh_coeffs_np = sh_coeffs.detach().cpu().contiguous().numpy()
        if sh_coeffs_np.dtype != np.float16:
            sh_coeffs_np = sh_coeffs_np.astype(np.float16)

        N = means3d_np.shape[0]
        K = sh_coeffs_np.shape[1]
        sh_degree = settings.sh_degree

        # ---- SH forward: compute RGB colors from SH coefficients ----
        # Compute view directions
        cam = campos_np  # [3]
        dirs = means3d_np - cam[np.newaxis, :]  # [N, 3]
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        dirs = dirs / norms  # normalized

        colors_fp16, _sh_ms = core.compute_sh_forward(
            dirs.astype(np.float32),
            sh_coeffs_np,
            N, K, sh_degree
        )
        colors_fp16 = colors_fp16.view(np.float16)  # [N, 3] fp16
        colors_fp32 = colors_fp16.astype(np.float32)  # [N, 3]

        # ---- Full forward render (preprocess → sort → bin → rasterize) ----
        fwd_result = core.render_forward(
            means3d_np, scales_np, quats_np, viewmat_np,
            colors_fp32, opacities_np,
            settings.tan_fovx, settings.tan_fovy,
            settings.focal_x, settings.focal_y,
            settings.principal_x, settings.principal_y,
            settings.img_width, settings.img_height,
            settings.bg_color[0], settings.bg_color[1], settings.bg_color[2],
            settings.max_gaussians_per_tile,
        )

        image_np = fwd_result["image"]  # [H, W, 3]

        # ---- Save for backward ----
        # We save numpy arrays as python objects via ctx (not save_for_backward,
        # which requires tensors). This is fine since Metal works with numpy anyway.
        ctx.metal_fwd = {
            "means3d": means3d_np,
            "scales": scales_np,
            "quats": quats_np,
            "viewmat": viewmat_np,
            "colors": colors_fp32,
            "opacities": opacities_np,
            "campos": campos_np,
            "sh_coeffs_fp16": sh_coeffs_np,
            "colors_fwd_fp16": colors_fp16,
            "K": K,
            "sh_degree": sh_degree,
            # Forward intermediates for backward
            "means2d": fwd_result["means2d"],
            "cov2d": fwd_result["cov2d"],
            "radii": fwd_result["radii"],
            "tile_bins": fwd_result["tile_bins"],
            "point_list": fwd_result["point_list"],
            "T_final": fwd_result["T_final"],
            "n_contrib": fwd_result["n_contrib"],
        }
        ctx.settings = settings

        # Mark which inputs need gradients
        ctx.needs_means3d_grad  = means3d.requires_grad
        ctx.needs_scales_grad   = scales.requires_grad
        ctx.needs_quats_grad    = quats.requires_grad
        ctx.needs_sh_grad       = sh_coeffs.requires_grad
        ctx.needs_opacity_grad  = opacities.requires_grad

        # Convert output to torch tensor on the original device
        image_tensor = torch.from_numpy(image_np.copy())
        if input_device.type != 'cpu':
            image_tensor = image_tensor.to(input_device)
            if input_device.type == 'mps':
                torch.mps.synchronize()

        return image_tensor

    @staticmethod
    def backward(ctx, grad_image):
        core = _get_core()
        fwd = ctx.metal_fwd
        settings = ctx.settings

        device = grad_image.device

        # Ensure MPS operations are complete before Metal dispatch
        if device.type == 'mps':
            torch.mps.synchronize()

        # ---- Convert upstream gradient to numpy ----
        grad_img_np = grad_image.detach().cpu().contiguous().numpy().astype(np.float32)

        # ---- Full backward pass ----
        bwd_result = core.render_backward(
            fwd["means3d"],
            fwd["scales"],
            fwd["quats"],
            fwd["viewmat"],
            fwd["colors"],
            fwd["opacities"],
            fwd["campos"],
            fwd["sh_coeffs_fp16"],
            fwd["colors_fwd_fp16"],
            fwd["K"], fwd["sh_degree"],
            settings.tan_fovx, settings.tan_fovy,
            settings.focal_x, settings.focal_y,
            settings.principal_x, settings.principal_y,
            settings.img_width, settings.img_height,
            settings.bg_color[0], settings.bg_color[1], settings.bg_color[2],
            settings.max_gaussians_per_tile,
            fwd["means2d"],
            fwd["cov2d"],
            fwd["radii"],
            fwd["tile_bins"],
            fwd["point_list"],
            fwd["T_final"],
            fwd["n_contrib"],
            grad_img_np,
        )

        device = grad_image.device

        # ---- Convert gradients back to PyTorch tensors ----
        grad_means3d = None
        if ctx.needs_means3d_grad:
            grad_means3d = torch.from_numpy(
                bwd_result["dL_d_means3d"].copy()
            ).to(device)

        grad_scales = None
        if ctx.needs_scales_grad:
            grad_scales = torch.from_numpy(
                bwd_result["dL_d_scales"].copy()
            ).to(device)

        grad_quats = None
        if ctx.needs_quats_grad:
            grad_quats = torch.from_numpy(
                bwd_result["dL_d_quats"].copy()
            ).to(device)

        grad_sh = None
        if ctx.needs_sh_grad:
            grad_sh = torch.from_numpy(
                bwd_result["dL_d_sh"].copy()
            ).to(device)

        grad_opacities = None
        if ctx.needs_opacity_grad:
            grad_opacities = torch.from_numpy(
                bwd_result["dL_d_opacities"].copy()
            ).to(device)

        # Ensure all MPS transfers complete before returning gradients
        if device.type == 'mps':
            torch.mps.synchronize()

        # Free saved forward data immediately to release numpy arrays
        del ctx.metal_fwd
        del bwd_result
        gc.collect()

        # Return gradients for each forward() argument:
        # means3d, scales, quats, sh_coeffs, opacities, viewmat, campos, settings
        return (grad_means3d, grad_scales, grad_quats, grad_sh, grad_opacities,
                None, None, None)


def simple_knn_metal(points_np, k_neighbors=3, search_window=32):
    """
    Compute average squared distance to K nearest neighbors for each 3D point.

    Uses Morton-code radix sort + window-based search on Metal GPU.

    Parameters
    ----------
    points_np : np.ndarray, float32, shape [N, 3]
    k_neighbors : int, default 3
    search_window : int, default 32 (half-window; searches ±window in sorted order)

    Returns
    -------
    avg_sq_dist : np.ndarray, float32, shape [N]
    elapsed_ms : float
    """
    core = _get_core()
    points_np = np.ascontiguousarray(points_np, dtype=np.float32)
    assert points_np.ndim == 2 and points_np.shape[1] == 3, \
        f"points must be [N, 3], got {points_np.shape}"
    return core.simple_knn_metal(points_np, k_neighbors, search_window)
