"""
Python wrapper for the Metal SH colour kernel.
Loads the compiled C++ extension and provides a clean Python API.
"""

import os
import time
import importlib

def _load_extension():
    """Import the compiled _metal_gs_core extension module."""
    try:
        from metal_gs import _metal_gs_core
        return _metal_gs_core
    except ImportError:
        # Try to find it at package level
        import _metal_gs_core
        return _metal_gs_core

_core = None

def _get_core():
    global _core
    if _core is None:
        _core = _load_extension()
    return _core


def compute_sh_colors_metal(
    directions,      # numpy float32 array [N, 3]  (unit view directions)
    sh_coeffs,       # numpy float16 array [N, K, 3]  (SH coefficients, K bases)
    sh_degree: int = 3,
):
    """
    Evaluate Spherical Harmonics colour for N Gaussians using a Metal compute kernel.
    
    Parameters
    ----------
    directions : np.ndarray, float32, shape [N, 3]
        Normalised view directions (camera → Gaussian).
    sh_coeffs : np.ndarray, float16, shape [N, K, 3]
        SH coefficients. K = (sh_degree + 1)^2, with 3 colour channels.
    sh_degree : int
        SH degree (0-3). K must be >= (sh_degree+1)^2.
        
    Returns
    -------
    colors : np.ndarray, float16, shape [N, 3]
        Evaluated RGB colours (includes the +0.5 DC offset, clamped to [0,1]).
    elapsed_ms : float
        Kernel execution time in milliseconds.
    """
    import numpy as np
    
    core = _get_core()
    
    N = directions.shape[0]
    K = sh_coeffs.shape[1]
    
    # Validate
    assert directions.ndim == 2 and directions.shape[1] == 3, \
        f"directions must be [N, 3], got {directions.shape}"
    assert sh_coeffs.ndim == 3 and sh_coeffs.shape[2] == 3, \
        f"sh_coeffs must be [N, K, 3], got {sh_coeffs.shape}"
    assert sh_coeffs.shape[0] == N, "sh_coeffs batch size must match directions"
    assert K >= (sh_degree + 1) ** 2, \
        f"K={K} too small for degree {sh_degree} (need {(sh_degree+1)**2})"
    
    # Ensure correct dtypes (contiguous C-order)
    directions = np.ascontiguousarray(directions, dtype=np.float32)
    sh_coeffs = np.ascontiguousarray(sh_coeffs, dtype=np.float16)
    
    # Call Metal kernel — returns (uint16_array[N,3], elapsed_ms)
    # We pass sh_coeffs directly; its raw bytes are FP16 = uint16 bit patterns
    colors_u16, elapsed_ms = core.compute_sh_forward(
        directions,
        sh_coeffs,
        N,
        K,
        sh_degree,
    )
    
    # Reinterpret uint16 → float16
    colors_out = colors_u16.view(np.float16)
    
    return colors_out, elapsed_ms
