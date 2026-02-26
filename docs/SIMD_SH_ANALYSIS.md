# SIMD SH Analysis — V1.0 Feature Probe

> Metal-GS simdgroup_matrix exploration for Spherical Harmonics acceleration  
> Target: Apple GPU Family 9+ (M3/M4) · MSL 3.0+ · `simdgroup_float8x8`

---

## 1. Executive Summary

| Item | Finding |
|------|---------|
| **Natural fit?** | ❌ SH evaluation is a *batched per-element dot product*, not a shared-operand matrix multiply |
| **Mapping used** | Diagonal Extraction — 8 Gaussians packed into 8×8 MMA, correct result on matrix diagonal |
| **Compute waste** | 87.5% (off-diagonal results discarded) |
| **Expected speedup** | Break-even requires MMA hardware ≥8× faster per-op than scalar ALU |
| **Recommendation** | Benchmark on M4; focus SIMD optimisation effort on covariance/rasterize kernels instead |

---

## 2. Problem Structure: Why SH ≠ Matrix Multiply

### 2.1 SH Forward Computation

For each Gaussian $g$ with view direction $\mathbf{d}_g$ and SH coefficients $C_{g,k,c}$:

$$\text{result}_{g,c} = \sum_{k=0}^{K-1} Y_k(\mathbf{d}_g) \cdot C_{g,k,c} \quad \text{for } c \in \{R, G, B\}$$

This is a **16-element dot product** done 3 times (once per channel), or equivalently a $[1 \times 16] \times [16 \times 3]$ matrix-vector product per Gaussian.

### 2.2 What simdgroup_matrix Expects

`simdgroup_multiply_accumulate(D, A, B, C)` computes $D = A \times B + C$ where $A, B, C, D$ are $8 \times 8$ matrices distributed across 32 SIMD lanes.

The canonical use case is **shared-operand** matrix multiplication:
- Neural net inference: $Y = X \cdot W$ where $W$ is constant across the batch
- Convolution: shared filter kernel applied to different spatial positions

### 2.3 The Fundamental Mismatch

In SH evaluation, **BOTH operands are per-Gaussian**:
- $Y_k(\mathbf{d}_g)$ depends on the per-Gaussian view direction
- $C_{g,k,c}$ is a per-Gaussian learned parameter

There is **no shared operand** across Gaussians. Standard matrix multiplication $D = A \times B$ computes all $D_{ij} = \sum_k A_{ik} B_{kj}$, mixing Gaussian $i$'s basis with Gaussian $j$'s coefficients. Only the diagonal $D_{gg}$ gives the correct result.

---

## 3. The Diagonal Extraction Approach

### 3.1 Mathematical Mapping

For 8 Gaussians ($g = 0 \ldots 7$), 8 basis functions ($k = 0 \ldots 7$), one channel $c$:

$$A_{gk} = Y_k(\mathbf{d}_g) \quad \text{(basis values — rows = Gaussians, cols = bases)}$$

$$B_{kg} = C_{g,k,c} \quad \text{(coefficients transposed — rows = bases, cols = Gaussians)}$$

$$D = A \times B \implies D_{gg'} = \sum_{k} Y_k(\mathbf{d}_g) \cdot C_{g',k,c}$$

**Diagonal extraction**: $D_{gg} = \sum_k Y_k(\mathbf{d}_g) \cdot C_{g,k,c}$ = correct result ✓

For degree 3 (16 bases): split into two 8×8 tiles and accumulate:
$$D_{\text{total}} = A_{\text{lo}} \times B_{\text{lo}} + A_{\text{hi}} \times B_{\text{hi}}$$

### 3.2 Operation Count

| Metric | Scalar Kernel | MMA Kernel | Ratio |
|--------|--------------|------------|-------|
| **Per-Gaussian FMAs** | 48 (16 bases × 3 channels) | — | — |
| **Per 8 Gaussians** | 384 FMAs | 6 MMA ops (2 tiles × 3 ch) | — |
| **Multiply-adds** | 384 | 3,072 (6 × 8×8×8) | 8× more |
| **Useful results** | 384 | 384 (diagonal only) | — |
| **Waste** | 0% | 87.5% | — |

### 3.3 Why It Might Still Win

Apple GPU Family 9 (M3/M4) has dedicated matrix multiply units. If these units can complete an 8×8×8 MMA in 1 cycle while scalar ALU processes 32 FMAs per cycle:

- **Scalar**: 384 ÷ 32 = **12 ALU cycles**
- **MMA**: 6 + loading/extraction overhead ≈ **6-10 cycles** (speculative)

The MMA hardware would otherwise be idle during SH evaluation, so the "waste" doesn't consume ALU resources — it consumes dedicated matrix hardware cycles that have no alternative use.

**This must be determined empirically.** The kernel in `sh_simd.metal` is designed for A/B benchmarking.

---

## 4. Implementation Details

### 4.1 Thread Model

```
┌─ Threadgroup (256 threads) ──────────────────────────────┐
│  Simdgroup 0 (32 threads) → Gaussians [0..7]            │
│  Simdgroup 1 (32 threads) → Gaussians [8..15]           │
│  ...                                                      │
│  Simdgroup 7 (32 threads) → Gaussians [56..63]          │
└──────────────────────────────────────────────────────────┘

Within each simdgroup:
  Threads 0-7:   Compute basis functions for Gaussians 0-7
  Threads 0-31:  Cooperatively load coefficient matrices (B)
  Threads 0-31:  Participate in simdgroup_matrix MMA operations
  Threads 0-7:   Extract diagonal, write output
```

### 4.2 Memory Layout

Per-simdgroup threadgroup scratch (struct `MmaScratch`):

| Field | Size | Purpose |
|-------|------|---------|
| `basis_lo[8][8]` | 256 B | A matrix: bases 0-7 |
| `basis_hi[8][8]` | 256 B | A matrix: bases 8-15 |
| `coeff[8][8]` | 256 B | B matrix: coefficients (per channel) |
| `result[8][8]` | 256 B | D matrix: MMA output |
| **Total per simdgroup** | **1 KB** | |
| **Total (8 simdgroups)** | **8 KB** | Well within 32 KB limit |

### 4.3 Execution Flow

```
Phase 1: Basis computation (once)
  ├── Zero-init basis arrays (32 threads cooperate)
  ├── Threads 0-7: compute Y_k(dir_g) for all k=0..15
  ├── Store to basis_lo[g][k] and basis_hi[g][k]
  ├── threadgroup_barrier
  └── simdgroup_load(matA_lo), simdgroup_load(matA_hi)

Phase 2: Per-channel MMA (×3 iterations: R, G, B)
  ├── Load B_lo: 32 threads fill coeff[k][g] for bases 0-7
  ├── threadgroup_barrier
  ├── simdgroup_load(matB) → MMA: D = A_lo × B_lo + 0
  ├── Load B_hi: 32 threads fill coeff[k][g] for bases 8-15
  ├── threadgroup_barrier
  ├── simdgroup_load(matB) → MMA: D += A_hi × B_hi
  ├── simdgroup_store(matD) → threadgroup_barrier
  └── Threads 0-7: result[g][g] + 0.5 → clamp → colors_out
```

### 4.4 Feature Guards

```metal
#if defined(__HAVE_SIMDGROUP_MATRIX__) && __METAL_VERSION__ >= 300
    // MMA path: simdgroup_float8x8 (Apple GPU Family 9+)
    kernel void compute_sh_forward_mma(...) { /* MMA impl */ }
#else
    // Scalar fallback: identical to compute_sh_forward
    kernel void compute_sh_forward_mma(...) { /* scalar impl */ }
#endif
```

Both branches define the same kernel name, ensuring the metallib always contains `compute_sh_forward_mma`. The wrapper can create this PSO unconditionally.

---

## 5. Alternative SIMD Approaches Explored

### 5.1 simd_shuffle Cooperative Reduction

**Idea**: 16 threads per Gaussian; thread $k$ evaluates $Y_k$, loads $C_k$; tree reduction via `simd_shuffle_xor`.

**Problem**: Each thread's basis function $Y_k$ requires different ALU instructions (different polynomial forms). With 16 threads computing 16 different functions, the SIMD group faces **16-way divergence** — the predicated execution serialises all paths.

**Verdict**: No benefit over scalar. The divergence cost exceeds any reduction speedup.

### 5.2 Cooperative Memory Loading

**Idea**: 32 threads cooperatively load coefficients into threadgroup memory, then each thread reads from low-latency local memory.

**Problem**: The coefficient access pattern `sh_coeffs[(tid * K + k) * 3 + c]` has stride $3K$ between bases but stride $3K^2$ between Gaussians. On M4 with 16+ MB L1, the access pattern likely stays in cache.

**Verdict**: Marginal benefit possible on memory-bound workloads. Not worth the threadgroup memory overhead and barrier cost.

### 5.3 Memory Layout Transposition

**Idea**: Transpose coefficients from `[N, K, 3]` to `[K, 3, N]` or `[K, N, 3]` for coalesced access.

**Analysis**: Adjacent threads would read adjacent memory addresses (stride = 1 instead of $3K$). This improves memory coalescing but requires a transposition pass on the host.

**Verdict**: Worth exploring as a separate optimisation, independent of SIMD instruction choice.

---

## 6. Where simdgroup_matrix WOULD Help

The following Metal-GS kernels have computation patterns that **naturally map to MMA**:

### 6.1 Covariance Computation (preprocess.metal)

The 3D→2D projection involves:
$$\Sigma_{2D} = J \cdot R \cdot S \cdot S^T \cdot R^T \cdot J^T$$

This is a chain of matrix multiplies with shared Jacobian structure. The $3 \times 3$ matrices can be packed into $8 \times 8$ tiles (processing 2-3 Gaussians per tile).

**Expected benefit**: Medium — the matrices are small ($3 \times 3$) but the operation is repeated for every Gaussian.

### 6.2 Tile-Based Alpha Blending (rasterize.metal)

The rasteriser processes multiple Gaussians per tile, with each pixel computing:
$$C_{\text{pixel}} = \sum_i \alpha_i \cdot T_i \cdot c_i$$

Multiple pixels in a tile share the same Gaussian set. This shared-operand structure maps naturally to:
- A: Gaussian color/opacity data (shared across pixels)
- B: Per-pixel blending weights

**Expected benefit**: High — this is the performance-critical inner loop.

### 6.3 Simple KNN (simple_knn.metal)

Distance matrix computation between Gaussian centroids could use MMA for batch distance calculations. $K \times K$ distance tiles map directly to $8 \times 8$ MMA.

**Expected benefit**: Medium — KNN is called infrequently (only during densification).

---

## 7. Benchmark Plan

### 7.1 Kernel Comparison

To wire up the benchmark, add PSO creation in `metal_wrapper.mm`:

```objc
// In init_metal():
c.sh_fwd_mma_pso = make_pso("compute_sh_forward_mma");
```

Dispatch with adjusted grid (64 Gaussians/threadgroup instead of 256):

```objc
uint tg_size = 256;
uint gaussians_per_tg = 64;  // 8 simdgroups × 8 Gaussians
uint grid = (N + gaussians_per_tg - 1) / gaussians_per_tg * tg_size;
```

### 7.2 Metrics to Collect

| Metric | Method |
|--------|--------|
| Kernel execution time | MTLCommandBuffer GPU timestamps |
| Throughput (Gaussians/sec) | N / kernel_time |
| Precision error | Max/mean absolute diff vs scalar kernel |
| Register usage | Metal shader profiler |
| Threadgroup memory | Reported by PSO `maxTotalThreadsPerThreadgroup` |

### 7.3 Expected Outcomes

| Scenario | Prediction |
|----------|------------|
| MMA faster | Dedicated matrix hardware completes 6 MMA ops faster than 384 scalar FMAs |
| MMA slower | Loading/barrier overhead and 87.5% waste overwhelm MMA throughput |
| MMA parity | MMA hardware fast enough to offset waste, but barriers add equal latency |

---

## 8. V1.0 Roadmap Recommendations

Based on this analysis:

1. **Benchmark** `compute_sh_forward_mma` vs `compute_sh_forward` on M4
   - If MMA wins: integrate into training pipeline, explore backward variant
   - If MMA loses: focus efforts on kernels with natural MMA mappings

2. **Priority targets for simdgroup_matrix**:
   - 🔴 `rasterize.metal` — highest impact, shared-operand structure
   - 🟡 `preprocess.metal` — covariance chain of 3×3 matmuls
   - 🟢 `sh_forward.metal` — only if MMA benchmark shows benefit

3. **Memory layout exploration** (independent of MMA):
   - Transpose SH coefficients `[N, K, 3]` → `[K, 3, N]` for coalesced access
   - Measure impact on both scalar and MMA kernels

4. **FP16 MMA variant** (`simdgroup_half8x8`):
   - Available on M1+ (GPU Family 7)
   - Matches coefficient precision (already FP16)
   - Worth building for comparison if float32 MMA shows promise

---

## Appendix: Kernel Files

| File | Purpose | Status |
|------|---------|--------|
| `csrc/kernels/sh_simd.metal` | MMA forward kernel + scalar fallback | ✅ Implemented |
| `csrc/kernels/sh_forward.metal` | Original scalar forward kernel | ✅ Production |
| `csrc/kernels/sh_backward.metal` | Scalar backward kernel | ✅ Production |
| `docs/SIMD_SH_ANALYSIS.md` | This analysis document | ✅ Complete |
