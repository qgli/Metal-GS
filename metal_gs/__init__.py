"""
Metal-GS: Apple Silicon-native Gaussian Splatting operators.
Deep-optimized for TBDR architecture, unified memory, and mixed precision.
"""

from .sh import compute_sh_colors_metal
from .rasterizer import MetalGaussianRasterizer, RenderSettings, simple_knn_metal

__all__ = [
    "compute_sh_colors_metal",
    "MetalGaussianRasterizer",
    "RenderSettings",
    "simple_knn_metal",
]
__version__ = "0.1.0"
