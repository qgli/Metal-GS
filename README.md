# Metal-GS

**3D Gaussian Splatting on Apple Silicon — pure Metal compute shaders, no CUDA required.**

Metal-GS is a fully differentiable 3D Gaussian Splatting renderer that runs entirely on Apple Silicon GPUs via Metal compute shaders. It implements the complete forward + backward pipeline (SH evaluation, projection, radix sort, tile binning, alpha-blending rasterization) with all kernels AOT-compiled into a single `.metallib`.

> **Status:** research prototype — trains 165K Gaussians at ~2.6 it/s on M1 16GB (516×344).

---

## The Problem: CUDA → Metal Is Not a Transliteration

Porting Gaussian Splatting from CUDA to Metal exposes a fundamental architectural difference: **Metal compute shaders dispatched via `threadgroup_barrier` are non-preemptible on Apple GPU.** On CUDA, a tile with 10,000 overlapping Gaussians simply runs longer. On Metal (M1, 7 GPU cores), the same tile triggers macOS's WindowServer watchdog ("Impacting System Interactivity"), which **kills the command buffer** after ~2 seconds.

This is not a bug — it is a design constraint of Apple's GPU architecture where the OS must maintain UI responsiveness.

## The Solution: Depth-Sorted Hard Capping

Since Gaussians are processed in strict front-to-back depth order, we apply a per-tile hard cap (`max_gaussians_per_tile`, default 1024). Dense tiles are truncated, discarding only the most distant (and most occluded) Gaussians. This:

- **Bounds worst-case GPU time** per tile to a fixed constant, eliminating watchdog timeouts
- **Preserves visual quality** — truncated Gaussians are behind hundreds of closer ones
- **Is fully differentiable** — backward pass applies the same cap, matching forward exactly

The cap is a runtime parameter (not a compile-time constant), tunable per scene via `RenderSettings.max_gaussians_per_tile`. Set to `0` for unlimited (at your own risk on M1).

---

## Architecture

```
PyTorch autograd (CPU/MPS)
    │
    ▼
metal_gs/rasterizer.py ── MetalGaussianRasterizer (autograd.Function)
    │
    ▼
_metal_gs_core (PyBind11)
    │
    ▼
csrc/metal_wrapper.mm ── ObjC++ dispatch layer
    │  ┌──────────────────────────────────────────────────────────────┐
    ├──│  Single MTLCommandQueue, one command buffer per dispatch     │
    │  │  MTLResourceStorageModeShared — zero-copy unified memory     │
    │  │  @autoreleasepool on all 9 entry points                      │
    │  └──────────────────────────────────────────────────────────────┘
    ▼
csrc/kernels/*.metal ── 15 PSOs, AOT-compiled metallib
    ├── sh_forward.metal        SH basis evaluation
    ├── preprocess.metal        3D→2D projection, cov2d, tile bounds
    ├── radix_sort.metal        32-bit radix sort (histogram, scan, scatter)
    ├── tile_binning.metal      Gaussian→tile assignment + tile_range
    ├── rasterize.metal         Forward alpha blending (cooperative fetch)
    ├── rasterize_backward.metal  Backward (reverse traversal, atomic grads)
    ├── preprocess_backward.metal Projection backward
    ├── sh_backward.metal       SH backward
    └── knn.metal               Morton-code KNN for scale initialization
```

**Key design decisions:**
- **Single encoder per dispatch** — no multi-pass command buffer splitting; `memoryBarrierWithScope` between stages
- **3-level prefix sum** for radix sort — block-level scan → block-sum scan → scatter; all in one metallib
- **FP32 everywhere** on M1 (`ENABLE_BF16=0`); set to `1` for M4+ with BF16 support
- **Naive atomic gradient accumulation** (Strategy A) — correctness-first; SIMD reduction is a future optimization

---

## Quick Start

### Prerequisites

- macOS 13+ with Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools (`xcode-select --install`)
- Python 3.10+, conda or venv

### Install

```bash
# Create environment
conda create -n metal-gs python=3.10 -y
conda activate metal-gs
pip install torch numpy pybind11 tqdm Pillow viser

# Build (AOT-compiles Metal shaders + C++/ObjC++ extension)
cd Metal-GS
CC=/usr/bin/clang CXX=/usr/bin/clang++ pip install -e .
```

### Train on COLMAP Data

Metal-GS includes **minGS**, a minimal training harness:

```bash
cd minGS
python example.py
```

This trains 500 iterations on the bundled COLMAP dataset (165K Gaussians, 179 cameras, 516×344 @ 2x downsample) with a live Viser viewer at [http://localhost:8080](http://localhost:8080), then saves `cat_mac_render.png`.

### Use as a Library

```python
import numpy as np
import torch
from metal_gs.rasterizer import MetalGaussianRasterizer, RenderSettings

settings = RenderSettings(
    viewmat=viewmat_np,            # [4,4] float32
    tan_fovx=tan_fovx,
    tan_fovy=tan_fovy,
    focal_x=focal_x,
    focal_y=focal_y,
    principal_x=W / 2.0,
    principal_y=H / 2.0,
    img_width=W,
    img_height=H,
    sh_degree=3,
    bg_color=(0.0, 0.0, 0.0),
    max_gaussians_per_tile=1024,   # tune for your GPU
)

# Fully differentiable — gradients flow through Metal compute shaders
image = MetalGaussianRasterizer.apply(
    means3d, scales, quats, sh_coeffs, opacities,
    viewmat_tensor, campos_tensor, settings
)
loss = (image - target).pow(2).mean()
loss.backward()
```

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `max_gaussians_per_tile` | 1024 | Hard cap per tile. Prevents watchdog timeout on M1. Increase for higher quality on M3/M4 Pro. Set `0` for unlimited. |
| `DOWNSAMPLE` | 2 | Image downsampling factor in `example.py`. Use 1 for full-res on M3+ with ≥32GB. |
| `ENABLE_BF16` | 0 | Set to 1 in `setup.py` for BF16 training on M4+ (Apple GPU Family 9+). |

---

## Performance (M1 16GB, 7 GPU cores)

| Dataset | Points | Resolution | Cap | Speed | Final Loss |
|---|---|---|---|---|---|
| Cat (COLMAP) | 165K | 516×344 (2x) | 1024 | ~2.6 it/s | 0.094 |
| Cat (COLMAP) | 165K | 1032×688 (1x) | 1024 | ~0.6 it/s | — |

---

## Hardware Tested

> **Metal-GS v1.0 has been developed and stress-tested exclusively on the weakest Apple Silicon chip: M1 with 7 GPU cores and 16GB unified memory (~4–6GB usable after system overhead).**

This is a deliberate design choice. If the code survives the M1's constraints — 7 GPU cores, no hardware ray tracing, no BF16, and macOS's aggressive 2-second GPU watchdog — it will run on anything Apple ships.

| Chip | GPU Cores | Memory | Status |
|---|---|---|---|
| **M1 (7-core)** | 7 | 16GB | ✅ **Fully tested** — 2000 iterations at full resolution |
| M1 Pro/Max/Ultra | 16–64 | 32–192GB | 🔜 Expected to work (same ISA, more headroom) |
| M2/M3/M4 family | 8–40 | 8–192GB | 🔜 Performance testing planned on M4 Max |

**For M3/M4 users:** You can safely increase `max_gaussians_per_tile` beyond 1024 (try 2048–4096) and set `DOWNSAMPLE=1` for full-resolution training. M4+ users can also enable `ENABLE_BF16=1` in `setup.py` for mixed-precision training.

---

## Known Limitations

- **Viser real-time visualization:** Verified working on M1 7-core GPU. Training + live Viser rendering runs concurrently without GPU watchdog warnings (~2.7 it/s with viewer active, ~5.5 it/s without). Frame drops may occur under extreme load.
- **No multi-GPU:** Metal-GS targets single-GPU Apple Silicon. Multi-GPU (Mac Pro with multiple M2 Ultra) is not supported.
- **FP32 only on M1:** BF16 requires Apple GPU Family 9+ (M4). M1/M2/M3 run all kernels in FP32.
- **Densification scale:** The bundled minGS trainer uses a simplified densification schedule (30 iterations). For production quality, increase `densify_until_iter`.

---

## Dataset

Metal-GS ships with a COLMAP-processed cat dataset (179 frames, 165K points) for quick testing.

**Download:** The dataset will be available on [Google Drive](https://drive.google.com/) (link coming soon).

### Expected COLMAP Directory Structure

```
data/cat/
├── images/                   # 179 source images (frame_00001.JPG … frame_00179.JPG)
│   ├── frame_00001.JPG
│   ├── frame_00002.JPG
│   └── ...
└── sparse/
    └── 0/
        ├── cameras.bin       # Camera intrinsics (COLMAP format)
        ├── images.bin        # Camera extrinsics (COLMAP format)
        ├── points3D.bin      # 3D point cloud (COLMAP format)
        └── points3D.ply      # Point cloud in PLY format (optional)
```

To use your own data, run COLMAP on your images and place the output in the same structure. Metal-GS reads the standard COLMAP binary format via `minGS/gs/io/colmap/`.

---

## Project Structure

```
Metal-GS/
├── csrc/
│   ├── kernels/          9 Metal shader files → AOT-compiled metallib
│   ├── metal_wrapper.mm  ObjC++ dispatch layer (9 public functions)
│   ├── metal_wrapper.h   C++ header
│   └── bindings.cpp      PyBind11 bindings
├── metal_gs/
│   ├── rasterizer.py     PyTorch autograd wrapper + RenderSettings
│   └── __init__.py
├── minGS/                Minimal training harness (COLMAP loader, 3DGS model)
│   ├── example.py        Entry point — train + render
│   ├── gs/               GaussianModel, trainers, visualization
│   └── data/cat/         Bundled COLMAP dataset
├── setup.py              Build script (Metal AOT + C++/ObjC++ compilation)
└── pyproject.toml        PEP 517 build-system declaration
```

---

## Acknowledgements

Metal-GS builds on the work of several excellent open-source projects:

- **[gsplat](https://github.com/nerfstudio-project/gsplat)** — reference CUDA kernels for the rasterization pipeline. Metal-GS reimplements the gsplat forward/backward kernels as Metal compute shaders with architecture-specific adaptations (cooperative fetch, SIMD prefix sums, hard capping for TBDR).
- **[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)** — Kerbl et al. 2023, the foundational algorithm.
- **[minGS](https://github.com/shiukaheng/minGS)** (Shiu Ka Heng) — minimal 3DGS training framework. Metal-GS uses a modified fork: replaced CUDA KNN with a Metal Morton-code KNN kernel, removed the CUDA/`diff-gaussian-rasterization` dependency, and wired up MPS tensor support.
- **[mlx-splat](https://github.com/daikiad/mlx-splat)** (daikiad) — MLX-native Gaussian Splatting. Studied for Apple Silicon tiling strategy and alpha-blending formulation; Metal-GS takes a different approach (PyTorch MPS + raw Metal compute shaders rather than MLX primitives).
- **[OpenSplat](https://github.com/pierotofy/OpenSplat)** (Piero Toffanin) — portable C++ 3DGS with Metal backend via gsplat-metal. Studied for Metal shader dispatch patterns; Metal-GS differs in using a single-encoder architecture with `memoryBarrierWithScope` rather than multi-encoder passes.

---

## License

MIT
