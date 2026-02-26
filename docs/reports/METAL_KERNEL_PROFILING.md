# Metal Kernel Profiling Report

**Branch:** `exp/v1-kernel-profiling`  
**Date:** 2026-02-26  
**Hardware:** Mac Mini M4, 10 GPU cores, 10 CPU cores (4P+6E), 16GB UMA  
**Dataset:** minGS/data/cat, 179 cameras, 165,751 Gaussians, 516×344 (2× downsampled)  
**Config:** BF16 precision, `max_gaussians_per_tile=1024`, MPS device  

---

## 1. Executive Summary

We instrumented every stage of the Metal-GS rendering pipeline with high-precision timing. The results **fundamentally change our understanding of the performance bottleneck**:

| Category | Time (ms) | % of iteration |
|----------|-----------|----------------|
| **GPU Compute (Forward)** | 11.55 | 10.7% |
| **GPU Compute (Backward)** | 10.57 | 9.8% |
| **CPU↔GPU Transfer** | 35.32 | 32.8% |
| **Optimizer + Loss + Other** | 50.28 | 46.7% |
| **Total Wall-clock** | **107.72** | 100% |

**Key finding:** The GPU compute kernels (all 4 forward stages + 3 backward stages combined) account for only **~22 ms** — roughly **20% of the total iteration time**. The remaining 80% is CPU overhead: tensor conversion, PyTorch optimizer, and loss computation.

---

## 2. Per-Kernel Timing Breakdown

### 2.1 Forward Pass (40.14 ms total)

| Stage | Mean (ms) | Std | Min | Max | % of Forward |
|-------|-----------|-----|-----|-----|-------------|
| CPU→Numpy conversion | **17.64** | 1.86 | 16.41 | 33.83 | **44.0%** |
| SH Forward (GPU) | 4.72 | 0.66 | 4.29 | 10.58 | 11.8% |
| Preprocess (GPU) | 0.86 | 0.23 | 0.61 | 1.38 | 2.1% |
| Depth Sort (GPU) | 0.85 | 0.18 | 0.69 | 1.71 | 2.1% |
| Tile Binning (GPU+CPU) | 1.52 | 0.23 | 1.18 | 2.50 | 3.8% |
| Rasterize (GPU) | 3.60 | 1.04 | 0.68 | 6.48 | 9.0% |
| Numpy→Torch conversion | **10.95** | 1.64 | 7.67 | 20.49 | **27.3%** |

### 2.2 Backward Pass (17.31 ms total)

| Stage | Mean (ms) | Std | Min | Max | % of Backward |
|-------|-----------|-----|-----|-----|---------------|
| CPU→Numpy (grad) | 0.57 | 0.17 | 0.38 | 1.31 | 3.3% |
| Rasterize BW (GPU) | **7.91** | 2.26 | 1.05 | 14.15 | **45.7%** |
| Preprocess BW (GPU) | 0.73 | 0.17 | 0.66 | 2.22 | 4.2% |
| SH Backward (GPU) | 1.93 | 0.34 | 1.78 | 4.88 | 11.1% |
| Numpy→Torch (grad) | **6.17** | 0.68 | 5.10 | 8.31 | **35.6%** |

### 2.3 Top 5 Bottlenecks (Combined)

| Rank | Stage | Time (ms) | % of Iteration |
|------|-------|-----------|----------------|
| #1 | **CPU→Numpy** (forward) | 17.64 | 30.7% |
| #2 | **Numpy→Torch** (forward) | 10.95 | 19.1% |
| #3 | **Rasterize BW** (GPU) | 7.91 | 13.8% |
| #4 | **Numpy→Torch** (backward grad) | 6.17 | 10.7% |
| #5 | **SH Forward** (GPU) | 4.72 | 8.2% |

---

## 3. Scene Statistics

| Metric | Value |
|--------|-------|
| Total Gaussians | 165,751 |
| Visible (after frustum cull) | 132,361 (79.9%) |
| Tile Intersections | 575,159 |
| Tiles (16×16) | 33×22 = 726 |
| Avg intersections/tile | ~792 |

---

## 4. Architecture Diagnosis

### 4.1 The Real Bottleneck: CPU↔GPU Data Transfer (50%)

The #1 and #2 bottlenecks are **not GPU kernels** — they are the `.detach().cpu().contiguous().numpy()` forward conversion (17.64ms) and `torch.from_numpy(…).to(device)` backward conversion (10.95ms + 6.17ms). Combined, these account for **34.76 ms per iteration** — more than all GPU kernels combined.

**Root cause analysis:**

1. **Forward CPU→Numpy (17.64 ms):** Converts 6 tensors (means3d[165K,3], scales[165K,3], quats[165K,4], opacities[165K], viewmat[4,4], campos[3], sh_coeffs[165K,K,3]) from MPS tensors to numpy. Each `.cpu()` triggers an MPS→CPU transfer that must wait for the MPS command queue to flush. With 7 tensors, this creates 7 sequential synchronization points.

2. **Forward Numpy→Torch (10.95 ms):** The rendered image (516×344×3 float32 = ~2.1 MB) is copied back to MPS. The `.to(device)` + `torch.mps.synchronize()` adds a full GPU fence.

3. **Backward Numpy→Torch (6.17 ms):** 5 gradient tensors (means3d, scales, quats, sh_coeffs, opacities) copied back to MPS device.

**Optimization opportunities:**

- **Batch tensor download:** Instead of 7 individual `.cpu()` calls (each a sync point), pack all tensors into a single contiguous buffer on MPS, then do one `.cpu()` call and slice.
- **Keep CPU**: If we use `device="cpu"` instead of `"mps"`, these transfers become zero-cost. The optimizer runs slightly slower on CPU vs MPS, but we save ~35ms of transfer overhead.
- **Persistent numpy buffers:** Pre-allocate numpy arrays and reuse them across iterations instead of fresh allocations.

### 4.2 Bottleneck #3: Rasterize Backward (7.91 ms, 13.8%)

This is the heaviest GPU kernel. The rasterize backward kernel must:
- Walk through each tile's Gaussian list in reverse (back-to-front)
- For each pixel in the tile, recompute the alpha blending chain
- Accumulate gradients via per-Gaussian atomic adds (global memory)

**M4 TBDR architecture analysis:**

1. **Atomic contention:** The backward rasterizer uses `atomic_fetch_add_explicit` on global memory for gradient accumulation (`dL_d_rgb`, `dL_d_opacity`, `dL_d_cov2d`, `dL_d_mean2d`). Multiple tiles write gradients for the same Gaussian simultaneously. With 575K intersections across 726 tiles, this creates significant atomic contention on the M4's 10 GPU cores.

2. **Threadgroup memory pressure:** Each tile thread group likely loads the per-tile Gaussian list into threadgroup memory. With `max_gaussians_per_tile=1024` and each Gaussian needing ~40 bytes of data (means2d, cov2d, color, opacity), this is ~40KB — within M4's 32KB threadgroup limit only if carefully managed. Exceeding this forces register spilling.

3. **Divergent control flow:** The backward pass's early-exit conditions (skip if `T < 1e-4`, skip Gaussians after `n_contrib`) cause warp divergence within SIMD groups, reducing ALU utilization.

**Optimization hypotheses:**
- Replace global atomic adds with threadgroup-level reduction + single global write
- Use shared memory tile lists with coalesced access patterns
- Implement two-pass backward: first pass computes per-tile partial gradients, second pass reduces

### 4.3 Bottleneck #5: SH Forward (4.72 ms, 8.2%)

The SH evaluation kernel computes `SH_coeffs × SH_basis_functions → RGB` per Gaussian. With 165K Gaussians and degree-3 SH (16 bases × 3 channels = 48 evaluations per Gaussian), this is compute-intensive.

**Optimization hypotheses:**
- Start with SH degree 0 (constant color) for initial iterations — already implemented in training
- Use half-precision (FP16) SH evaluation since SH coefficients are already FP16
- Consider computing SH inline in the rasterize kernel to avoid a separate pass and the intermediate colors buffer

### 4.4 GPU Kernels That Are NOT Bottlenecks

| Kernel | Time | Assessment |
|--------|------|------------|
| Preprocess | 0.86 ms | Highly efficient — 165K Gaussians in <1ms |
| Depth Sort | 0.85 ms | Radix sort is well-optimized |
| Tile Binning | 1.52 ms | Includes CPU prefix sum, reasonable |
| Preprocess BW | 0.73 ms | 1:1 mapping, no atomics, efficient |

These kernels are already performing well and are not optimization targets.

---

## 5. The 50ms Gap: Where Does It Go?

The profiler captures 57.45 ms of forward+backward time, but wall-clock shows 107.72 ms. The missing **50.28 ms (46.7%)** is:

1. **PyTorch Optimizer** (`optimizer.step()` on MPS): Adam optimizer updates 6 parameter groups with momentum. On MPS, this involves GPU kernel launches + synchronization.
2. **Loss computation** (`mix_l1_ssim_loss`): L1 + SSIM loss on MPS tensors, including `F.conv2d` for SSIM windowing.
3. **`loss.backward()` overhead**: PyTorch autograd graph traversal, MPS kernel scheduling.
4. **`model.backprop_stats()`**: Gradient accumulation statistics for densification.
5. **Python overhead**: GC, function call overhead, tqdm.

**This is a critical finding:** Nearly half the iteration time is spent in PyTorch framework overhead, not in our Metal kernels. Optimizing Metal kernels alone cannot deliver more than ~2× improvement.

---

## 6. Revised Optimization Roadmap

Based on this profiling data, the optimization priorities are:

### Priority 1: Eliminate CPU↔GPU Transfer (save ~35 ms, 32%)
- **Option A:** Run model on CPU (`device="cpu"`) — zero transfer cost, optimizer is slightly slower
- **Option B:** Pre-pack all tensors into a single contiguous allocation, single `.cpu()` call
- **Option C:** Implement the full pipeline in Metal (avoid numpy round-trip entirely)

### Priority 2: Reduce PyTorch Overhead (save ~25-30 ms, ~25%)
- Profile `optimizer.step()` and `mix_l1_ssim_loss` separately
- Consider fused optimizer kernel (combined step + grad zero)
- Consider implementing L1+SSIM loss in Metal to avoid PyTorch overhead

### Priority 3: Optimize Rasterize Backward (save ~4-5 ms, 4-5%)  
- Replace global atomics with threadgroup reduction
- Profile actual GPU occupancy with Xcode Metal System Trace

### Priority 4: Optimize SH Forward (save ~2-3 ms, 2-3%)
- Fuse SH evaluation into preprocess or rasterize kernel
- Use FP16 throughout

---

## 7. Methodology

### 7.1 Timing Source

- **CPU-side timing:** `time.perf_counter()` — nanosecond resolution wall-clock
- **GPU kernel timing:** Already returned by the C++ Metal wrapper (`waitUntilCompleted` bracket timing) — measures actual GPU execution time including command buffer submission and completion

### 7.2 Test Configuration

```
iterations: 100 (10 warmup skipped)
dataset: minGS/data/cat (165,751 Gaussians, 179 cameras)
resolution: 516×344 (2× downsampled)
device: MPS
precision: BF16
max_gaussians_per_tile: 1024
```

### 7.3 Instrumentation Code

- [metal_gs/profiler.py](../../metal_gs/profiler.py) — `KernelProfiler` singleton with per-stage timing collection
- [metal_gs/rasterizer.py](../../metal_gs/rasterizer.py) — Instrumented forward/backward with `time.perf_counter()` + C++ kernel timing
- [profile_metal_kernels.py](../../profile_metal_kernels.py) — Standalone profiling script

---

## 8. Raw Data Reference

Full per-iteration timing data saved to: [`kernel_profiling_data.json`](kernel_profiling_data.json)

---

## 9. Conclusion

The profiling reveals that **Metal GPU kernels are NOT the primary bottleneck**. The 4 forward + 3 backward GPU kernels combined consume only ~22 ms (20% of iteration time). The dominant costs are:

1. **CPU↔GPU tensor conversion** (35 ms, 33%) — the numpy bridge tax
2. **PyTorch framework overhead** (50 ms, 47%) — optimizer, loss, autograd

This means the "Asymmetric Compute" direction (CPU-side culling to reduce GPU work) would have limited impact — even halving the GPU work saves only ~11 ms. The highest-leverage optimizations are:

1. **Eliminating the numpy round-trip** (Option C from §6: full Metal pipeline)
2. **Reducing PyTorch overhead** (fused optimizer, Metal-native loss)
3. **Running on CPU device** (trade ~5ms optimizer slowdown for ~35ms transfer savings)

The next experiment should be: `python profile_metal_kernels.py --device cpu` to measure the CPU-device baseline and quantify the actual transfer cost eliminated.
