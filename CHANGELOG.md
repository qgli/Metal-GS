# Changelog

All notable changes to Metal-GS are documented in this file.

> **Version Time-Machine:** If you encounter compatibility issues on your hardware, you can instantly rollback to any tagged version. See [How to Rollback](#how-to-rollback) below.

---

## How to Rollback

Each release is tagged in Git. Switch to any version with a single command:

```bash
# Roll back to the rock-stable v0.2.0 (recommended if v0.3.1 has issues)
git checkout v0.2.0

# Roll back to the legacy v0.1.0 (maximum compatibility)
git checkout v0.1.0

# Return to the latest release
git checkout main
```

After switching versions, rebuild:

```bash
python setup.py build_ext --inplace
```

### Which version should I use?

| Version | Codename | Speed | Best For |
|---------|----------|-------|----------|
| **v0.3.1** | *Aggressive* | ~25 it/s | M4 users who want maximum performance |
| **v0.2.0** | *Rock-Stable* | ~11 it/s | Production use, Viser UI, any Apple Silicon |
| **v0.1.0** | *Legacy* | ~6 it/s | M1/M2 with limited memory bandwidth |

---

## [v0.3.1] — 2025-xx-xx

**Pure GPU MPS Custom Op Architecture + Memory Leak Fix**

The entire CPU↔GPU dispatch layer has been replaced with MPS Custom Ops — Metal kernels injected directly into PyTorch's internal MPS command stream with zero-copy buffer binding. This is the most aggressive performance version.

### Performance
- **~25 it/s** on M4 (165K Gaussians, 516×344, FP32)
- 2.5× faster than v0.2.0
- Single sync barrier per iteration (vs. 9 in v0.2.0)
- Zero CPU↔GPU memory copies (vs. 18+ in v0.2.0)

### Architecture
- `mps_ops.mm` (855 LOC) replaces `metal_wrapper.mm` (1553 LOC)
- 17 pre-compiled Pipeline State Objects (PSOs) on PyTorch's MPS device
- `getMTLBufferStorage()` — direct MTLBuffer extraction from MPS tensors
- 3-phase pipeline: torch ops → Metal encoder (sort) → Metal encoder (rasterize)
- Single-encoder backward pass: `rasterize_bw → preprocess_bw → sh_bw`

### Bug Fixes (v0.3.0 → v0.3.1)
- **Fixed: Fatal memory leak** — 18 MTLBuffers leaked per iteration (~8 MB/iter) under non-ARC compilation. Phase 2 scratch now explicitly released after `COMMIT_AND_WAIT`; Phase 3 scratch deferred to next iteration's sync point.
- **Fixed: Backward pass segfault** — `sh_coeffs.to(kFloat16)` was called mid-encoder; PyTorch MPS ops can invalidate the current compute command encoder. Moved all torch ops before `stream->commandEncoder()`.
- Memory stress test: **0.00 MB delta** over 800 post-warmup iterations (1000 total).

### Known Limitations
- **Viser viewer crashes training** — the Viser WebSocket server competes with Metal-GS for the MPS command queue. Use `use_viewer=False`.
- Requires `torch >= 2.1` with MPS backend.

---

## [v0.2.0] — 2025-xx-xx

**M4 BF16 Native Support & Stable PyTorch CPU-GPU Pipeline**

The rock-stable version. Built on the traditional `.cpu().numpy()` data flow with explicit `newBufferWithBytes:` copies. Slower than v0.3.1, but battle-tested and fully compatible with Viser UI.

### Performance
- **~11 it/s** on M4 (165K Gaussians, 516×344, FP32)
- ~6 it/s on M1/M2

### Features
- **Native BF16 support** on M4+ (MSL `metal3.2`, `bfloat` type in SH evaluation)
- Dynamic Hard Capping analytics — runtime `max_gaussians_per_tile` tuning
- SH evaluation in FP16 (precision-gated: FP16 intermediate → FP32 final output)
- Full Viser viewer integration (works reliably in this version)
- Environment setup automation (conda YAML, requirements.txt)

### Architecture
- `metal_wrapper.mm` (1553 LOC) — explicit per-kernel dispatch with `newBufferWithBytes:` / `memcpy` transfers
- 9 sync barriers per iteration (one per kernel dispatch)
- 18+ CPU↔GPU memory copies per iteration
- Clean separation: Python (training loop) → NumPy (bridge) → ObjC++ (Metal dispatch)

### Compatibility
- M1, M2, M3, M4 (all Apple Silicon)
- Set `ENABLE_BF16=0` for M1/M2/M3 (no `bfloat` hardware)
- Viser UI: **fully supported**

---

## [v0.1.0] — 2025-xx-xx

**Initial M1 Apple Silicon Port with Dynamic Hard Capping**

The first working Gaussian Splatting renderer on Apple Silicon. Designed for M1/M2 hardware with limited memory bandwidth and non-preemptible Metal compute shaders.

### Features
- Complete forward + backward pipeline in Metal compute shaders
- **Dynamic Hard Capping** — `max_gaussians_per_tile` enforced to prevent WindowServer watchdog kills on M1/M2
- AOT-compiled `.metallib` (all shaders pre-compiled at build time)
- Radix sort, tile binning, alpha-blending rasterization — all in Metal
- Viser-based viewer for interactive visualization

### Architecture
- 10 Metal kernels compiled into single `metal_gs.metallib`
- Per-kernel CPU dispatch with explicit buffer management
- Conservative tile capping to stay within M1/M2 GPU timeout limits

### Compatibility
- Designed for M1/M2 (limited VRAM bandwidth, 8-16 GB unified memory)
- Works on M3/M4 but does not leverage newer hardware features

---

## Version History at a Glance

```
v0.1.0 ──── M1 survival: hard capping, explicit dispatch
  │
v0.2.0 ──── M4 BF16, stable pipeline, Viser works
  │
v0.3.1 ──── Pure MPS Custom Ops, 25+ it/s, zero-copy
  (latest)
```

---

*See [docs/reports/V0.3_ARCHITECTURE_JOURNEY.md](docs/reports/V0.3_ARCHITECTURE_JOURNEY.md) for the full technical story of how we got from v0.1 to v0.3.*
