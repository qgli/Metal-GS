"""
Metal-GS: Apple Silicon-native Gaussian Splatting operators.
Deep-optimized for TBDR architecture, unified memory, and mixed precision.
"""

import os as _os

# ── Auto-discover metallib path ──────────────────────────────────────────
# The C++ extension (metal_wrapper.mm) looks for metal_gs.metallib via:
#   Strategy 1: __FILE__  (compile-time, relative to csrc/)
#   Strategy 2: METAL_GS_METALLIB_DIR env var
#
# Strategy 1 can fail if __FILE__ was a relative path at compile time and
# the working directory has changed.  We add a Python-side backstop:
# compute the absolute path from this package's location → ../csrc/kernels/
# and set the env var (if not already set) BEFORE the C extension loads.
_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
_metallib_dir = _os.path.join(_os.path.dirname(_pkg_dir), "csrc", "kernels")
if _os.path.isfile(_os.path.join(_metallib_dir, "metal_gs.metallib")):
    _os.environ.setdefault("METAL_GS_METALLIB_DIR", _metallib_dir)

from .sh import compute_sh_colors_metal
from .rasterizer import MetalGaussianRasterizer, RenderSettings, simple_knn_metal

__all__ = [
    "compute_sh_colors_metal",
    "MetalGaussianRasterizer",
    "RenderSettings",
    "simple_knn_metal",
]
__version__ = "0.1.0"
