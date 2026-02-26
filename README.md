# Metal-GS

**3D Gaussian Splatting on Apple Silicon — pure Metal compute shaders, no CUDA required.**

Metal-GS is a fully differentiable 3D Gaussian Splatting renderer that runs entirely on Apple Silicon GPUs via Metal compute shaders. It implements the complete forward + backward pipeline (SH evaluation, projection, radix sort, tile binning, alpha-blending rasterization) with all kernels AOT-compiled into a single `.metallib`.

> **v0.2** — trains 165K Gaussians at **~11 it/s on M4** (BF16, 516×344) | ~2.6 it/s on M1 16GB.

---

## The Problem: CUDA → Metal Is Not a Transliteration

Porting Gaussian Splatting from CUDA to Metal exposes a fundamental architectural difference: **Metal compute shaders dispatched via `threadgroup_barrier` are non-preemptible on Apple GPU (M1/M2 family).** On CUDA, a tile with 10,000 overlapping Gaussians simply runs longer. On Metal, the same tile triggers macOS's WindowServer watchdog ("Impacting System Interactivity"), which **kills the command buffer** after ~2 seconds.

This is not a bug — it is a design constraint. On M1/M2 (including Pro/Max/Ultra), the GPU uses a static resource allocation model and a hardware-level non-preemptive scheduler. A single threadgroup that exceeds the OS's GPU time budget monopolises an entire GPU core, causing the WindowServer to forcibly terminate the command buffer to protect system interactivity.

## The Solution: Depth-Sorted Dynamic Hard Capping

Since Gaussians are processed in strict front-to-back depth order, we apply a per-tile hard cap (`max_gaussians_per_tile`, default 1024). Dense tiles are truncated, discarding only the most distant (and most occluded) Gaussians.

### Why Truncation Is Mathematically Safe

3DGS alpha blending accumulates in front-to-back order: $C = \sum_i T_i \alpha_i c_i$ where $T_i = \prod_{j<i}(1-\alpha_j)$. Due to the **alpha early-stopping** mechanism, the transmittance $T_i$ decays exponentially — after a few hundred Gaussians, $T_i < 10^{-4}$, meaning all subsequent contributions are sub-pixel-level noise. The hard cap exploits this property:

- **Bounds worst-case GPU time** per tile to a fixed constant, eliminating watchdog timeouts
- **Preserves visual quality** — truncated Gaussians contribute effectively zero to the final pixel color
- **Is fully differentiable** — backward pass applies the same cap, ensuring gradient consistency between forward and backward

The cap is a runtime parameter (not a compile-time constant), tunable per scene via `RenderSettings.max_gaussians_per_tile`.

### The Hardware Dividing Line: M1/M2 vs M3/M4

| Feature | M1/M2 (All variants) | M3/M4 (Family 9+) |
|---|---|---|
| GPU Resource Allocation | Static | **Dynamic Caching** |
| Threadgroup Preemption | Non-preemptive | Fine-grained preemption |
| Watchdog Sensitivity | High (~2s timeout) | **Immune** (tested unlimited) |
| Recommended Cap | 1024–4096 | **0 (unlimited)** |
| BF16 Hardware | ❌ | ✅ Native |

**M1/M2 family (including Pro/Max/Ultra):** These chips share the same Apple GPU Family 7/8 ISA architecture. The static register file allocation means a single long-running threadgroup cannot be preempted while the OS waits. You **must** set `max_gaussians_per_tile` to a finite value (recommended 1024–4096 depending on your core count).

**M3/M4 family (Apple GPU Family 9+):** The M3 introduced **Dynamic Caching** — a hardware-level feature that dynamically allocates register file and threadgroup memory on-the-fly, enabling fine-grained preemption of long-running threadgroups. This fundamentally eliminates the watchdog problem. M3/M4 users can safely set `max_gaussians_per_tile=0` (unlimited). On M4, this has been verified with 165K Gaussians across 500 training iterations with zero watchdog events.

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

- **macOS 14+ Sonoma** (macOS 13+ minimum, but 14+ recommended for Metal 3.1+ features)
- **Apple Silicon** (M1/M2/M3/M4 — any variant)
- **Xcode Command Line Tools:**
  ```bash
  xcode-select --install
  ```
- **Metal Toolchain** (critical — the AOT Metal shader compiler ships separately from Xcode CLT since Xcode 15.x):
  ```bash
  xcodebuild -downloadComponent MetalToolchain
  ```
  Without this step, compilation will fail with: `error: cannot execute tool 'metal' due to missing Metal Toolchain`.
- **Python 3.10+** via conda (recommended) or system Python
- **Verify** your toolchain:
  ```bash
  xcrun -sdk macosx metal --version   # Should print Apple metal version 3xxxx+
  ```

### Install (FP32 — all Apple Silicon)

```bash
# Create environment (conda recommended — do NOT use venv for Metal extensions)
conda create -n metal-gs python=3.10 -y
conda activate metal-gs
pip install torch numpy pybind11 tqdm Pillow viser plyfile

# Build (AOT-compiles Metal shaders + C++/ObjC++ extension)
cd Metal-GS
CC=/usr/bin/clang CXX=/usr/bin/clang++ pip install -e . --no-build-isolation
```

### Install with BF16 (M4+ only — Apple GPU Family 9+)

M4 and later chips include native `bfloat16` hardware instructions. Enabling BF16 provides ~5% training speedup with **zero gradient accuracy loss** (all gradients remain at $10^{-7}$ error vs float64 reference — identical to FP32).

```bash
# 1. Edit setup.py: change ENABLE_BF16 from "0" to "1"
#    This switches Metal Shading Language from metal3.0 → metal3.2
#    (bfloat type requires MSL 3.2+)

# 2. Clean build
cd Metal-GS
rm -rf build/ dist/ metal_gs.egg-info/
CC=/usr/bin/clang CXX=/usr/bin/clang++ pip install -e . --no-build-isolation --force-reinstall
```

Why BF16 is free on M4:
- The `ENABLE_BF16` flag only gates `AccumType` in the preprocess kernel's intermediate covariance accumulations
- The rasterization kernel (the precision-critical alpha-blending path) remains FP32 regardless
- M4's BF16 ALUs have identical throughput to FP16 but with FP32-range exponent (8 bits vs 5 bits)
- Measured gradient error: FP32 `MaxAbs=3.73e-07` vs BF16 `MaxAbs=4.37e-07` — both within $10^{-7}$ of float64 reference

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
| `ENABLE_BF16` | 0 | Set to 1 in `setup.py` for BF16 training on M4+ (Apple GPU Family 9+). Requires MSL 3.2 (auto-selected). |

---

## Performance

### M4 10-core GPU, 16GB (NEW)

| Dataset | Points | Resolution | Precision | Cap | Speed | Final Loss |
|---|---|---|---|---|---|---|
| Cat (COLMAP) | 165K | 516×344 (2x) | FP32 | 1024 | ~10.2 it/s | 0.137 |
| Cat (COLMAP) | 165K | 516×344 (2x) | **BF16** | 1024 | ~10.7 it/s | 0.130 |
| Cat (COLMAP) | 165K | 516×344 (2x) | BF16 | 4096 | ~11.0 it/s | 0.134 |
| Cat (COLMAP) | 165K | 516×344 (2x) | BF16 | 0 (∞) | ~11.3 it/s | 0.139 |

### M1 7-core GPU, 16GB

| Dataset | Points | Resolution | Precision | Cap | Speed | Final Loss |
|---|---|---|---|---|---|---|
| Cat (COLMAP) | 165K | 516×344 (2x) | FP32 | 1024 | ~2.6 it/s | 0.094 |
| Cat (COLMAP) | 165K | 1032×688 (1x) | FP32 | 1024 | ~0.6 it/s | — |

> **M4 delivers 4x the throughput of M1** with identical mathematical precision. BF16 is free performance on M4 — zero gradient accuracy loss. See [M4_PERFORMANCE_REPORT.md](M4_PERFORMANCE_REPORT.md) for the full analysis.

---

## Hardware Tested

| Chip | GPU Cores | Memory | GPU Family | Capping Required | BF16 | Status |
|---|---|---|---|---|---|---|
| **M1 (7-core)** | 7 | 16GB | Apple 7 | ✅ Yes (1024) | ❌ | ✅ **Fully tested** |
| M1 Pro/Max/Ultra | 16–64 | 32–192GB | Apple 7 | ✅ Yes (1024–4096) | ❌ | 🔜 Same ISA, needs cap |
| M2 family | 8–38 | 8–192GB | Apple 8 | ✅ Yes (1024–4096) | ❌ | 🔜 Same ISA, needs cap |
| M3 family | 10–40 | 8–128GB | **Apple 9** | ❌ Dynamic Caching | ❌ | 🔜 Cap=0 safe |
| **M4 (10-core)** | 10 | 16GB | **Apple 9** | ❌ Dynamic Caching | ✅ | ✅ **Fully tested** |
| M4 Pro/Max/Ultra | 14–40 | 24–192GB | **Apple 9** | ❌ Dynamic Caching | ✅ | 🔜 Expected faster |

> **Design philosophy:** Metal-GS v0.1 was developed exclusively on the weakest Apple Silicon (M1 7-core, 16GB) to ensure the code survives extreme constraints. v0.2 adds M4 BF16 native support and verifies the M3/M4 Dynamic Caching hypothesis.

**M4 users:** Set `ENABLE_BF16=1` in `setup.py`, `max_gaussians_per_tile=0` (unlimited). See [M4_PERFORMANCE_REPORT.md](M4_PERFORMANCE_REPORT.md) for full benchmark data.

**M1/M2 users:** Keep `ENABLE_BF16=0`, set `max_gaussians_per_tile=1024` (or up to 4096 on Pro/Max/Ultra with more GPU cores).

**M3 users:** Keep `ENABLE_BF16=0` (no hardware BF16), but set `max_gaussians_per_tile=0` — Dynamic Caching eliminates the watchdog problem.

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

## Architecture Notes & Failed Experiments

### ❌ simdgroup_matrix (8×8 MMA) for SH Evaluation

**Attempt:** Use Metal 3.0's `simdgroup_float8x8` matrix multiply-accumulate hardware (Apple GPU Family 9+, M3/M4) to accelerate the Spherical Harmonics forward kernel. Pack 8 Gaussians into MMA tiles via a "diagonal extraction" scheme — compute $D = A \times B$ and read only the diagonal $D_{gg}$ for correct per-Gaussian results.

**Result:** Mathematically correct (all precision tests passed, max error $4.88 \times 10^{-4}$ in FP16 output), but only **~1.07× average speedup** on M4 — effectively no benefit.

**Root cause:** SH evaluation is a **batched per-element dot product** ($\text{result}_c = \sum_k Y_k(\mathbf{d}) \cdot C_{k,c}$), where *both* operands are per-Gaussian. There is no shared matrix across the batch. To fit this into 8×8 MMA, we compute $D_{gg'} = \sum_k Y_k(\mathbf{d}_g) \cdot C_{g',k,c}$ — but only the diagonal ($g = g'$) is needed, wasting **87.5% of the MMA compute** on discarded cross-Gaussian products. The dedicated matrix hardware *is* approximately 8× faster per-operation, but this waste precisely cancels the advantage.

**Takeaway for future SIMD work:**
- `simdgroup_matrix` excels at **shared-operand** workloads (neural net inference, convolution) where one matrix (e.g., weights) is reused across a batch
- For per-element independent operations, the scalar 1-thread-per-Gaussian kernel with full ALU utilization remains optimal
- Better MMA candidates in this codebase: `rasterize.metal` (shared Gaussian data across tile pixels) and `preprocess.metal` (3×3 covariance matrix chains)

**Code reference:** Full implementation, benchmark script, and mathematical analysis archived on branch [`exp/simd-sh-mma`](https://github.com/qgli/Metal-GS/tree/exp/simd-sh-mma).

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
