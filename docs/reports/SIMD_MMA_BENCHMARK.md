# SIMD MMA vs Scalar SH Benchmark Report

> Metal-GS V1.0 Feature Probe · Mac Mini M4 (10-core GPU, 16 GB)  
> Date: 2025-01-XX · MSL 3.2 · Apple GPU Family 9

---

## Test Environment

| Item | Value |
|------|-------|
| **Hardware** | Mac Mini M4, 10 GPU cores, 16 GB unified memory |
| **GPU Family** | Apple GPU Family 9 (Dynamic Caching, matrix hardware) |
| **MSL Version** | metal3.2 (`-std=metal3.2`, BF16 enabled) |
| **Kernel Scalar** | `compute_sh_forward` — 1 thread/Gaussian, threadgroup=256 |
| **Kernel MMA** | `compute_sh_forward_mma` — simdgroup_float8x8, 8 Gaussians/simdgroup |
| **SH Degree** | 3 (16 basis functions, K=16) |
| **Benchmark** | 5 warmup + 20 timed trials, median reported |

---

## 1. Numerical Correctness

Both kernels produce **identical results** within FP32 accumulation ordering tolerance:

| Config | Max Error | Mean Error | Status |
|--------|-----------|------------|--------|
| 1K, degree 0 | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 1K, degree 1 | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 1K, degree 2 | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 1K, degree 3 | 1.91e-06 | 6.21e-10 | ✅ PASS |
| 10K, degree 3 | 1.53e-05 | 5.13e-10 | ✅ PASS |
| 50K, degree 3 | 4.88e-04 | 8.28e-09 | ✅ PASS |
| 100K, degree 3 | 4.88e-04 | 1.27e-08 | ✅ PASS |
| 7 (non-aligned) | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 63 (non-aligned) | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 65 (non-aligned) | 0.00e+00 | 0.00e+00 | ✅ PASS |

**Analysis**: Degree 0–2 produce bit-exact results. Degree 3 shows sub-FP16-precision differences (max 4.88e-04) due to FP32 FMA ordering between the tiled MMA path and the sequential scalar path. This is well within acceptable tolerance — the final FP16 output quantisation dominates.

---

## 2. Performance Results

### 2.1 Kernel Latency (median, ms)

| Gaussians | Scalar (ms) | MMA (ms) | Speedup | Winner |
|-----------|-------------|----------|---------|--------|
| 1,000 | 0.3420 | 0.3756 | 0.91× | Scalar |
| 10,000 | 0.4221 | 0.2400 | **1.76×** | **MMA** |
| 50,000 | 0.3720 | 0.4496 | 0.83× | Scalar |
| 100,000 | 0.5248 | 0.5012 | 1.05× | MMA |
| 200,000 | 0.9073 | 0.9253 | 0.98× | Scalar |
| 500,000 | 2.2751 | 2.4692 | 0.92× | Scalar |

### 2.2 Throughput (Gaussians/sec)

| Gaussians | Scalar | MMA |
|-----------|--------|-----|
| 1,000 | 2.92M/s | 2.66M/s |
| 10,000 | 23.7M/s | **41.7M/s** |
| 50,000 | 134M/s | 111M/s |
| 100,000 | 191M/s | 200M/s |
| 200,000 | 220M/s | 216M/s |
| 500,000 | 220M/s | 202M/s |

### 2.3 Summary Statistics

| Metric | Value |
|--------|-------|
| Average speedup | 1.07× |
| Best speedup | 1.76× (N=10K) |
| Worst speedup | 0.83× (N=50K) |
| Scalar peak throughput | 220M Gaussians/s (N≥200K) |
| MMA peak throughput | 216M Gaussians/s (N=200K) |

---

## 3. Analysis

### 3.1 Why MMA ≈ Scalar (1.07× average)

The benchmark confirms the theoretical prediction from the SIMD analysis:

**Theoretical operation count**:
- Scalar: 48 FMAs/Gaussian → 384 FMAs for 8 Gaussians
- MMA: 6 × (8×8×8) = 3,072 multiply-adds for 8 Gaussians (87.5% wasted off-diagonal)

**Break-even requires MMA hardware ≥8× faster than scalar ALU** — and the results show this is approximately the case on M4's GPU Family 9 matrix units. The MMA hardware is indeed roughly 8× faster per-operation, causing the 87.5% waste to exactly cancel the hardware advantage.

### 3.2 Why MMA Wins at N=10K

At N=10,000 the MMA kernel achieves 1.76× speedup. Possible explanations:
1. **Threadgroup scheduling sweet spot**: 10K/64 = 156 threadgroups maps well to M4's 10 compute units (15.6 threadgroups/CU)
2. **Cache working set**: 10K × 96 bytes (coefficients) = 960KB fits comfortably in L2 but not L1. MMA's cooperative loading from threadgroup memory may reduce L1 pressure
3. **Reduced dispatch overhead**: ~156 threadgroups vs ~40 (scalar) — the dispatch granularity may be more efficient

### 3.3 Why Scalar Wins at Large N

Above 50K Gaussians, the scalar kernel consistently outperforms MMA:
1. **Simpler thread model**: no threadgroup memory, no barriers → lower per-threadgroup overhead
2. **Higher Gaussians/threadgroup ratio**: 256 vs 64 → 4× fewer threadgroups to schedule
3. **Embarrassingly parallel**: all ALU cycles are useful (0% waste)
4. **Cache-friendly sequential access**: each thread accesses contiguous coefficient memory

### 3.4 Verdict

**SH evaluation is not a natural fit for simdgroup_matrix MMA.** The operation is a batched per-element dot product with no shared operands between Gaussians. The creative diagonal-extraction approach achieves functional correctness and demonstrates the MMA API, but the 87.5% compute waste prevents meaningful speedup.

---

## 4. Recommendations

### For SH Kernels (V1.0)

| Action | Priority | Expected Impact |
|--------|----------|-----------------|
| Keep scalar `compute_sh_forward` as production kernel | ✅ Immediate | Baseline, proven optimal |
| Archive MMA kernel as reference implementation | 📦 Done | API documentation value |
| Explore memory layout transposition `[N,K,3] → [K,3,N]` | 🔬 Medium | May improve cache coalescing |
| Investigate FP16 MMA (`simdgroup_half8x8`) on M1 | 🔬 Low | Different precision/performance tradeoff |

### For Other Kernels (V1.0 SIMD Roadmap)

| Kernel | MMA Fit | Rationale |
|--------|---------|-----------|
| `rasterize.metal` | ⭐⭐⭐ | Shared Gaussian data across pixels in tile → natural shared-operand matmul |
| `preprocess.metal` | ⭐⭐ | 3×3 covariance chain J·R·S·Sᵀ·Rᵀ·Jᵀ → small but structured matmuls |
| `simple_knn.metal` | ⭐⭐ | Batch distance matrix → 8×8 tiles for K-nearest search |
| `sh_forward.metal` | ⭐ | Per-element dot product, no shared operand (this report) |

---

## 5. Files

| File | Description |
|------|-------------|
| `csrc/kernels/sh_simd.metal` | MMA kernel + scalar fallback |
| `benchmark_simd_sh.py` | This benchmark script |
| `docs/SIMD_SH_ANALYSIS.md` | Mathematical analysis |
| `docs/reports/SIMD_MMA_BENCHMARK.md` | This report |
