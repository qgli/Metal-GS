# Metal-GS

**3D Gaussian Splatting on Apple Silicon — pure Metal compute shaders, no CUDA required.**

Metal-GS is a fully differentiable 3D Gaussian Splatting renderer that runs entirely on Apple Silicon GPUs via Metal compute shaders. It implements the complete forward + backward pipeline (SH evaluation, projection, radix sort, tile binning, alpha-blending rasterization) with all kernels AOT-compiled into a single `.metallib`.

> **v0.3.1** — trains 165K Gaussians at **~25 it/s on M4** (FP32, 516×344) via pure MPS Custom Op architecture. 2.5× faster than v0.2.

### Versions & Rollback

| Tag | Speed (M4) | Speed (M1) | Stability | Best For |
|-----|------------|------------|-----------|----------|
| **`v0.3.1`** (latest) | ~25 it/s | ~8.8 it/s | Aggressive | Max performance |
| **`v0.2.0`** | ~11 it/s | — | Rock-stable | Production, Viser UI |
| **`v0.1.0`** | ~6 it/s | ~5.3 it/s | Legacy | Fallback only |

Having issues? Roll back instantly: `git checkout v0.2.0 && python setup.py build_ext --inplace`
Full version details and rollback guide → [CHANGELOG.md](CHANGELOG.md)

---

## What's New in v0.3

v0.3 is a complete rewrite of the GPU dispatch layer. The entire CPU↔GPU data pipeline has been replaced with **MPS Custom Ops** — Metal kernels injected directly into PyTorch's internal MPS command stream, achieving true zero-copy execution.

| | v0.2 | v0.3 |
|---|---|---|
| Dispatch layer | `metal_wrapper.mm` (1553 LOC) | `mps_ops.mm` (855 LOC) |
| Data flow | numpy → `newBufferWithBytes:` → kernel → memcpy → numpy | `getMTLBufferStorage()` — direct buffer bind, zero copies |
| Sync barriers per iter | 9 (one per kernel) | **1** (read `num_intersections` only) |
| CPU↔GPU copies per iter | 18+ | **0** |
| Speed (M4, 165K, 516×344) | ~11 it/s | **~25 it/s** |

See [docs/reports/V0.3_ARCHITECTURE_JOURNEY.md](docs/reports/V0.3_ARCHITECTURE_JOURNEY.md) for the full architectural exploration story (4 phases, 3 dead ends, 1 breakthrough).

---

## The Problem: CUDA → Metal Is Not a Transliteration

Porting Gaussian Splatting from CUDA to Metal exposes a fundamental architectural difference: **Metal compute shaders dispatched via `threadgroup_barrier` are non-preemptible on Apple GPU (M1/M2 family).** On CUDA, a tile with 10,000 overlapping Gaussians simply runs longer. On Metal, the same tile triggers macOS's WindowServer watchdog ("Impacting System Interactivity"), which **kills the command buffer** after ~2 seconds.

This is not a bug — it is a design constraint. On M1/M2 (including Pro/Max/Ultra), the GPU uses a static resource allocation model and a hardware-level non-preemptive scheduler. A single threadgroup that exceeds the OS's GPU time budget monopolises an entire GPU core, causing the WindowServer to forcibly terminate the command buffer to protect system interactivity.

## The Solution: Depth-Sorted Dynamic Hard Capping

Since Gaussians are processed in strict front-to-back depth order, we apply a per-tile hard cap (`max_gaussians_per_tile`, default 4096). Dense tiles are truncated, discarding only the most distant (and most occluded) Gaussians.

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

## Architecture (v0.3)

```
PyTorch autograd (MPS device)
    │
    ▼
metal_gs/rasterizer.py ── MetalGaussianRasterizer (torch.autograd.Function)
    │                      All tensors remain on MPS. No .cpu(), no numpy.
    ▼
_metal_gs_core (pybind11, torch::Tensor API)
    │
    ▼
csrc/mps_ops.mm ── MPS Custom Op dispatch layer
    │  ┌──────────────────────────────────────────────────────────────────┐
    │  │  Uses PyTorch's internal MPSStream (shared command queue)        │
    │  │  getMTLBufferStorage() extracts id<MTLBuffer> from MPS tensors  │
    │  │  PSOs created on PyTorch's MPS device                           │
    │  │  Single encoder per phase — memoryBarrierWithScope between ops  │
    │  │  ONE unavoidable sync: read num_intersections for allocation    │
    │  └──────────────────────────────────────────────────────────────────┘
    ▼
csrc/kernels/*.metal ── 17 PSOs, AOT-compiled metallib
    ├── sh_forward.metal          SH basis evaluation (half precision I/O)
    ├── preprocess.metal          3D→2D projection, cov2d, tile bounds
    ├── radix_sort.metal          32-bit radix sort (histogram, scan, scatter)
    ├── tile_binning.metal        Gaussian→tile assignment + tile_range
    ├── rasterize.metal           Forward alpha blending (cooperative fetch)
    ├── rasterize_backward.metal  Backward (reverse traversal, atomic grads)
    ├── preprocess_backward.metal Projection backward
    ├── sh_backward.metal         SH backward (half precision I/O)
    └── knn.metal                 Morton-code KNN for scale initialization
```

**Key design decisions:**
- **MPS stream integration** — all kernels dispatch through `at::mps::getCurrentMPSStream()`, sharing PyTorch's `MTLCommandQueue`. No competing command queues.
- **Zero-copy buffer binding** — `getMTLBufferStorage()` extracts the `id<MTLBuffer>` that backs each MPS tensor, directly binding it to kernel arguments. No `newBufferWithBytes:`, no memcpy.
- **Single encoder per phase** — multiple kernels share one `MTLComputeCommandEncoder`, separated by `memoryBarrierWithScope`. One `synchronize(COMMIT_AND_WAIT)` per forward pass.
- **3-level prefix sum** for radix sort — block-level scan → block-sum scan → scatter; all in one metallib
- **SH precision gate** — SH kernels read/write `half*` (FP16). The dispatch layer explicitly converts `sh_coeffs` to float16 before forward and backward dispatch.
- **Naive atomic gradient accumulation** — correctness-first; SIMD reduction is a future optimization

---

## Installation & Environment

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
- **Verify** your toolchain:
  ```bash
  xcrun -sdk macosx metal --version   # Should print Apple metal version 3xxxx+
  ```

### Dual-Track Environments

Metal-GS provides two frozen Conda environments. **Choose based on your priorities:**

| | `environment_v0.3.yml` | `environment_v0.2.yml` |
|---|---|---|
| **Target** | v0.3.1 (latest) | v0.2.0 (stable) |
| **PyTorch** | `==2.10.0` (PINNED) | `>=2.0` (flexible) |
| **PyTorch API surface** | Internal C++ (`MPSStream.h`, `getMTLBufferStorage`) | Public Python only (`.cpu().numpy()`) |
| **Breakage risk** | ⚠️ High — torch version changes may break compilation | ✅ Low — public API, wide version tolerance |
| **Performance** | ~25 it/s (M4) | ~11 it/s (M4) |
| **Viser viewer** | ❌ Conflicts with MPS queue | ✅ Fully compatible |

#### Option A: v0.3.1 — Maximum Performance (⚠️ exact PyTorch version required)

> **WARNING:** v0.3 uses PyTorch's **internal, non-public C++ MPS API** (`ATen/mps/MPSStream.h`, `getMTLBufferStorage()`). These interfaces are NOT part of PyTorch's stability guarantees and may break across versions. The environment file pins the exact `torch==2.10.0` known to compile correctly. **Do not change the PyTorch version.**

```bash
# Create pinned environment
conda env create -f environment_v0.3.yml
conda activate metal-gs

# Build (M4 with BF16 — default)
CC=/usr/bin/clang CXX=/usr/bin/clang++ pip install -e . --no-build-isolation

# For M1/M2/M3: edit setup.py, change ENABLE_BF16 from "1" to "0", then build
```

#### Option B: v0.2.0 — Rock-Stable (any recent PyTorch)

If v0.3 fails to compile on your PyTorch version, or you need Viser viewer support, use the stable version:

```bash
# Switch to stable version
git checkout v0.2.0

# Create flexible environment
conda env create -f environment_v0.2.yml
conda activate metal-gs-v02

# Build
CC=/usr/bin/clang CXX=/usr/bin/clang++ pip install -e . --no-build-isolation
```

> **Why BF16 is free on M4:** The `ENABLE_BF16` flag only gates `AccumType` in the preprocess kernel's intermediate covariance accumulations. The rasterization kernel (the precision-critical alpha-blending path) remains FP32 regardless. M4's BF16 ALUs have identical throughput to FP16 but with FP32-range exponent (8 bits vs 5 bits). Measured gradient error: FP32 `MaxAbs=3.73e-07` vs BF16 `MaxAbs=4.37e-07` — both within $10^{-7}$ of float64 reference.

### Train on COLMAP Data

Metal-GS includes **minGS**, a minimal training harness:

```bash
cd minGS
python example.py
```

This trains 500 iterations on the bundled COLMAP dataset (165K Gaussians, 179 cameras, 516×344 @ 2x downsample) and saves `cat_mac_render.png`.

> **⚠️ Viser viewer:** The example enables `use_viewer=True` by default. On v0.3, the Viser real-time viewer may crash training due to GPU command queue contention. If you experience crashes, edit `example.py` and set `use_viewer=False`. See [Known Limitations](#known-limitations).

### Use as a Library

```python
import torch
from metal_gs.rasterizer import MetalGaussianRasterizer, RenderSettings

# All tensors must be on MPS device
device = torch.device("mps")

settings = RenderSettings(
    viewmat=viewmat_tensor,        # [4,4] torch.Tensor on MPS
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
    max_gaussians_per_tile=4096,   # tune for your GPU; 0 = unlimited (M3/M4)
)

# Fully differentiable — gradients flow through MPS Custom Op Metal kernels
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
| `max_gaussians_per_tile` | 4096 | Hard cap per tile. Prevents watchdog timeout on M1/M2. Set `0` for unlimited on M3/M4. |
| `DOWNSAMPLE` | 2 | Image downsampling factor in `example.py`. Use 1 for full-res on M3+ with ≥32GB. |
| `ENABLE_BF16` | 1 | Set to `0` in `setup.py` for M1/M2/M3 (no hardware BF16). Default `1` targets M4+. Requires MSL 3.2 (auto-selected). |

---

## Performance

### v0.3 — M4 10-core GPU, 16GB

| Dataset | Points | Resolution | Precision | Cap | Speed | Final Loss |
|---|---|---|---|---|---|---|
| Cat (COLMAP) | 165K | 516×344 (2x) | FP32 | 4096 | **~25.2 it/s** | 0.138 |

### v0.2 (previous) — M4 10-core GPU, 16GB

| Dataset | Points | Resolution | Precision | Cap | Speed | Final Loss |
|---|---|---|---|---|---|---|
| Cat (COLMAP) | 165K | 516×344 (2x) | FP32 | 1024 | ~10.2 it/s | 0.137 |
| Cat (COLMAP) | 165K | 516×344 (2x) | BF16 | 1024 | ~10.7 it/s | 0.130 |
| Cat (COLMAP) | 165K | 516×344 (2x) | BF16 | 0 (∞) | ~11.3 it/s | 0.139 |

### v0.3.1 — M1 7-core GPU, 16GB

| Dataset | Points | Resolution | Precision | Cap | Speed | vs v0.1 |
|---|---|---|---|---|---|---|
| Cat (COLMAP) | 165K | 516×344 (2x) | FP32 | 1024 | **~8.8 it/s** | +64% |
| Cat (COLMAP) | 165K | 1032×688 (1x) | FP32 | 1024 | **~3.7 it/s** | +39% |

Key timing (2x downsample): fwd 23ms (−43%), bwd 73ms (−42%), optimizer 16ms (unchanged).

### v0.1 (legacy) — M1 7-core GPU, 16GB

| Dataset | Points | Resolution | Precision | Cap | Speed | Final Loss |
|---|---|---|---|---|---|---|
| Cat (COLMAP) | 165K | 516×344 (2x) | FP32 | 1024 | ~5.3 it/s | 0.094 |

> **v0.3 on M4 is 2.5× faster than v0.2 and ~5× faster than v0.1.** On M1, v0.3.1 delivers **+39%–64% over v0.1** despite the 7-core GPU's limited bandwidth. The speedup comes entirely from eliminating CPU↔GPU synchronization — the GPU kernels are unchanged. Full M1 benchmark details and comparison report: [`exp/v0.3.1-m1-test`](https://github.com/qgli/Metal-GS/tree/exp/v0.3.1-m1-test).

---

## Hardware Tested

| Chip | GPU Cores | Memory | GPU Family | Capping Required | BF16 | Status |
|---|---|---|---|---|---|---|
| **M1 (7-core)** | 7 | 16GB | Apple 7 | ✅ Yes (1024) | ❌ | ✅ **Tested (v0.3.1)** |
| M1 Pro/Max/Ultra | 16–64 | 32–192GB | Apple 7 | ✅ Yes (1024–4096) | ❌ | 🔜 Same ISA, needs cap |
| M2 family | 8–38 | 8–192GB | Apple 8 | ✅ Yes (1024–4096) | ❌ | 🔜 Same ISA, needs cap |
| M3 family | 10–40 | 8–128GB | **Apple 9** | ❌ Dynamic Caching | ❌ | 🔜 Cap=0 safe |
| **M4 (10-core)** | 10 | 16GB | **Apple 9** | ❌ Dynamic Caching | ✅ | ✅ **Tested (v0.3)** |
| M4 Pro/Max/Ultra | 14–40 | 24–192GB | **Apple 9** | ❌ Dynamic Caching | ✅ | 🔜 Expected faster |

**M4 users:** Default config works out of the box (`ENABLE_BF16=1`). Set `max_gaussians_per_tile=0` (unlimited) for maximum quality.

**M1/M2 users:** Change `ENABLE_BF16=0` in `setup.py`, set `max_gaussians_per_tile=1024` (or up to 4096 on Pro/Max/Ultra with more GPU cores).

**M3 users:** Change `ENABLE_BF16=0` (no hardware BF16), but set `max_gaussians_per_tile=0` — Dynamic Caching eliminates the watchdog problem.

---

## Known Limitations

- **⚠️ Viser real-time viewer crashes training (v0.3):** The v0.3 MPS Custom Op pipeline monopolizes PyTorch's GPU command queue. When Viser's rendering loop submits concurrent Metal work, the two compete for the same queue, causing GPU contention crashes. **Workaround:** Set `use_viewer=False` in `example.py`. Headless training is fully stable. This is a known limitation of sharing PyTorch's MPS stream with external Metal consumers.
- **No multi-GPU:** Metal-GS targets single-GPU Apple Silicon. Multi-GPU (Mac Pro with multiple M2 Ultra) is not supported.
- **FP32 only on M1/M2/M3:** BF16 requires Apple GPU Family 9+ (M4). M1/M2/M3 run all kernels in FP32.
- **M1/M2 backward compatibility:** v0.3.1 has been verified on M1 (7-core, 16GB) with PyTorch 2.10.0. Requires `ENABLE_BF16=0` and `max_gaussians_per_tile=1024`. See branch [`exp/v0.3.1-m1-test`](https://github.com/qgli/Metal-GS/tree/exp/v0.3.1-m1-test) for M1-specific build adaptations and benchmark scripts.
- **Densification scale:** The bundled minGS trainer uses a simplified densification schedule (30 iterations). For production quality, increase `densify_until_iter`.

---

## Dataset

Metal-GS ships with a COLMAP-processed cat dataset (179 frames, 165K points, 1032×688 resolution) for quick testing.

### Download

The dataset is hosted on Google Drive:

📦 **[cat.zip on Google Drive](https://drive.google.com/drive/folders/1vmGAbOf69MsX1B6osLks0pNLfxsUDClQ?usp=drive_link)** (~83 MB)

Download `cat.zip` from the link above, then run:

```bash
# From the Metal-GS root directory:
unzip cat.zip -d .

# Verify the structure
ls minGS/data/cat/images/ | wc -l    # Should print 179
ls minGS/data/cat/sparse/0/          # Should contain cameras.bin, images.bin, points3D.bin, points3D.ply

# (Optional) Clean up macOS hidden files if present
find minGS/data/cat -name '.DS_Store' -delete
rm -rf minGS/data/cat/__MACOSX
```

> **Note:** The zip contains only the low-resolution training images (1032×688) and COLMAP sparse model — no raw camera originals. It extracts directly into `minGS/data/cat/` with the correct directory structure. After extraction, you can immediately run `cd minGS && python example.py`.

### Expected COLMAP Directory Structure

```
minGS/data/cat/
├── images/                   # 179 training images (1032×688, ~43 MB total)
│   ├── frame_00001.JPG
│   ├── frame_00002.JPG
│   └── ...
└── sparse/
    └── 0/
        ├── cameras.bin       # Camera intrinsics (COLMAP format)
        ├── images.bin        # Camera extrinsics (COLMAP format)
        ├── points3D.bin      # 3D point cloud (165K points, COLMAP format)
        └── points3D.ply      # Point cloud in PLY format (optional)
```

To use your own data, run COLMAP on your images and place the output in the same structure. Metal-GS reads the standard COLMAP binary format via `minGS/gs/io/colmap/`.

---

## Project Structure

```
Metal-GS/
├── csrc/
│   ├── kernels/              9 Metal shader files → AOT-compiled metallib (17 PSOs)
│   ├── mps_ops.mm            MPS Custom Op dispatch layer (v0.3)
│   ├── mps_bindings.cpp      pybind11 bindings (torch::Tensor API)
│   ├── metal_wrapper.mm      Legacy v0.2 dispatch (kept for reference, not compiled)
│   ├── metal_wrapper.h       Legacy v0.2 header
│   └── bindings.cpp          Legacy v0.2 bindings (numpy API)
├── metal_gs/
│   ├── rasterizer.py         PyTorch autograd wrapper + RenderSettings
│   └── __init__.py
├── minGS/                    Minimal training harness (COLMAP loader, 3DGS model)
│   ├── example.py            Entry point — train + render
│   ├── gs/                   GaussianModel, trainers, visualization
│   └── data/cat/             Bundled COLMAP dataset
├── docs/
│   └── reports/
│       └── V0.3_ARCHITECTURE_JOURNEY.md   Architecture exploration report
├── setup.py                  Build script (Metal AOT + C++/ObjC++ compilation)
└── pyproject.toml            PEP 517 build-system declaration
```

---

## Architecture Notes & Failed Experiments

### ❌ simdgroup_matrix (8×8 MMA) for SH Evaluation

**Attempt:** Use Metal 3.0's `simdgroup_float8x8` matrix multiply-accumulate hardware (Apple GPU Family 9+, M3/M4) to accelerate the Spherical Harmonics forward kernel. Pack 8 Gaussians into MMA tiles via a "diagonal extraction" scheme — compute $D = A \times B$ and read only the diagonal $D_{gg}$ for correct per-Gaussian results.

**Result:** Mathematically correct (all precision tests passed, max error $4.88 \times 10^{-4}$ in FP16 output), but only **~1.07× average speedup** on M4 — effectively no benefit.

**Root cause:** SH evaluation is a **batched per-element dot product** ($\text{result}_c = \sum_k Y_k(\mathbf{d}) \cdot C_{k,c}$), where *both* operands are per-Gaussian. There is no shared matrix across the batch. To fit this into 8×8 MMA, we compute $D_{gg'} = \sum_k Y_k(\mathbf{d}_g) \cdot C_{g',k,c}$ — but only the diagonal ($g = g'$) is needed, wasting **87.5% of the MMA compute** on discarded cross-Gaussian products. The dedicated matrix hardware *is* approximately 8× faster per-operation, but this waste precisely cancels the advantage.

**Code reference:** Full implementation, benchmark script, and mathematical analysis archived on branch [`exp/simd-sh-mma`](https://github.com/qgli/Metal-GS/tree/exp/simd-sh-mma).

### ❌ UMA Zero-Copy via `.cpu().numpy()` (v0.2+)

**Attempt:** Exploit Apple's Unified Memory Architecture to create `MTLBuffer`s backed by numpy array memory using `newBufferWithBytesNoCopy:`, eliminating the 19ms CPU↔GPU transfer measured in v0.2 profiling.

**Result:** ~10 it/s — actually *slower* than v0.2. The transfer time was real but irrelevant; 55% of the iteration was spent on pipeline drain latency from 9 `waitUntilCompleted` synchronization barriers. Amdahl's Law: optimizing 21% of a pipeline where 55% is serialized yields zero net benefit.

**The real insight:** UMA eliminates *hardware* copy cost, but software synchronization (`waitUntilCompleted`, `@autoreleasepool` cycling, Python↔C++ boundary crossings) dominates. This led directly to the v0.3 MPS Custom Op architecture.

See [docs/reports/V0.3_ARCHITECTURE_JOURNEY.md](docs/reports/V0.3_ARCHITECTURE_JOURNEY.md) for the complete analysis.

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
