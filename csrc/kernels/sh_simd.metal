// ============================================================================
//  Metal-GS: SIMD Exploration Kernels for SH Evaluation (V1.0 Feature Probe)
//
//  Purpose:
//    Explore simdgroup_matrix 8×8 MMA (Matrix Multiply-Accumulate) for
//    accelerating Spherical Harmonics evaluation on Apple GPU Family 9+ (M3/M4).
//
//  Kernel variants compiled into this file:
//    compute_sh_forward_mma   — simdgroup_matrix 8×8 diagonal-extraction MMA
//                               (GPU Family 9+ / MSL 3.0+, float32 MMA)
//    [#else fallback]         — scalar clone of compute_sh_forward for M1/M2
//
//  Compilation:
//    Auto-compiled into metal_gs.metallib via setup.py glob("*.metal").
//    No changes to metal_wrapper.mm needed until benchmarking confirms benefit.
//
//  Feature guards:
//    __HAVE_SIMDGROUP_MATRIX__ && __METAL_VERSION__ >= 300
//      → MMA path  (Apple GPU Family 9+ / M3/M4)
//    else
//      → Scalar fallback (M1/M2, identical to compute_sh_forward)
//
//  Thread model (MMA variant):
//    256 threads = 8 simdgroups × 32 threads/simdgroup
//    Each simdgroup processes 8 Gaussians → 64 Gaussians per threadgroup
//    dispatch grid: ceil(N / 64) threadgroups
//
//  Architecture note:
//    SH evaluation is embarrassingly parallel — each Gaussian independently
//    computes result[c] = Σ_k Y_k(direction) × SH_coeff[k][c].
//    This is a per-element dot product, NOT a shared-operand matmul.
//
//    The MMA mapping packs 8 Gaussians into an 8×8 matrix:
//      A[g][k] = Y_k(dir_g)       — basis values  (rows = Gaussians)
//      B[k][g] = C[g, k, channel] — coefficients   (transposed: rows = bases)
//      D = A × B  →  D[g][g'] = Σ_k Y_k(dir_g) × C[g', k, c]
//      Diagonal D[g][g] = correct per-Gaussian result ✓
//      Off-diagonal D[g][g'] (g≠g') = cross-terms, discarded (87.5% waste)
//
//    For 16 bases (degree 3): two 8×8 tiles accumulated.
//    For 3 channels (RGB): sequential iteration, reloading B each time.
//    Total: 6 MMA ops per 8 Gaussians (2 tiles × 3 channels).
//
//    See docs/SIMD_SH_ANALYSIS.md for full mathematical analysis.
// ============================================================================

#include <metal_stdlib>
using namespace metal;

#if __METAL_VERSION__ >= 300
#include <metal_simdgroup_matrix>
#endif

// ---------------------------------------------------------------------------
//  SH normalisation constants (duplicated for self-containment)
//  Real spherical harmonics, Condon-Shortley phase — must match sh_forward.metal
// ---------------------------------------------------------------------------
constant float SH_C0 = 0.28209479177387814f;
constant float SH_C1 = 0.4886025119029199f;

constant float SH_C2_0 =  1.0925484305920792f;
constant float SH_C2_1 = -1.0925484305920792f;
constant float SH_C2_2 =  0.31539156525252005f;
constant float SH_C2_3 = -1.0925484305920792f;
constant float SH_C2_4 =  0.5462742152960396f;

constant float SH_C3_0 = -0.5900435899266435f;
constant float SH_C3_1 =  2.890611442640554f;
constant float SH_C3_2 = -0.4570457994644658f;
constant float SH_C3_3 =  0.3731763325901154f;
constant float SH_C3_4 = -0.4570457994644658f;
constant float SH_C3_5 =  1.445305721320277f;
constant float SH_C3_6 = -0.5900435899266435f;

// ---------------------------------------------------------------------------
//  Parameters — must match sh_forward.metal byte-for-byte
// ---------------------------------------------------------------------------
struct SHParamsSIMD {
    uint num_points;
    uint num_bases;
    uint sh_degree;
};

// ---------------------------------------------------------------------------
//  Thread model constants
// ---------------------------------------------------------------------------
constant uint SIMD_WIDTH             = 32;
constant uint MMA_TILE               = 8;
constant uint GAUSSIANS_PER_SIMD_MMA = MMA_TILE;       // 8
constant uint MAX_SIMDGROUPS_PER_TG  = 8;               // 256 / 32
constant uint GAUSSIANS_PER_TG       = MAX_SIMDGROUPS_PER_TG * GAUSSIANS_PER_SIMD_MMA;  // 64


// ============================================================================
//  Variant 1: simdgroup_matrix 8×8 MMA (diagonal extraction)
//
//  Complexity analysis (degree 3, 8 Gaussians):
//    Scalar:  48 FMA/Gaussian × 8 = 384 FMAs
//    MMA:     6 × (8×8×8) = 3072 multiply-adds on dedicated matrix hardware
//    Waste:   87.5% (off-diagonal results discarded)
//    Break-even: requires MMA hardware ≥8× faster per-op than scalar ALU
//
//  Memory per simdgroup: 4 × 8×8 × sizeof(float) = 1024 bytes
//  Total threadgroup memory: 8 simdgroups × 1024 = 8192 bytes (< 32KB limit)
// ============================================================================

#if defined(__HAVE_SIMDGROUP_MATRIX__) && __METAL_VERSION__ >= 300

// Per-simdgroup scratch in threadgroup memory
struct MmaScratch {
    float basis_lo[MMA_TILE][MMA_TILE];   // A matrix: first 8 basis values  [gauss × basis]
    float basis_hi[MMA_TILE][MMA_TILE];   // A matrix: last 8 basis values   [gauss × basis]
    float coeff[MMA_TILE][MMA_TILE];      // B matrix: coefficients           [basis × gauss]
    float result[MMA_TILE][MMA_TILE];     // D matrix: MMA output             [gauss × gauss']
};

kernel void compute_sh_forward_mma(
    device const float3*      directions  [[buffer(0)]],
    device const half*        sh_coeffs   [[buffer(1)]],
    device half*              colors_out  [[buffer(2)]],
    constant SHParamsSIMD&    params      [[buffer(3)]],
    uint tid              [[thread_position_in_grid]],
    uint simd_lane        [[thread_index_in_simdgroup]],
    uint simd_gid         [[simdgroup_index_in_threadgroup]],
    uint tg_pos           [[threadgroup_position_in_grid]]
)
{
    const uint N      = params.num_points;
    const uint K      = params.num_bases;
    const uint degree = params.sh_degree;

    // Which 8 Gaussians does this simdgroup handle?
    uint gauss_base = tg_pos * GAUSSIANS_PER_TG
                    + simd_gid * GAUSSIANS_PER_SIMD_MMA;

    // Early exit: entire simdgroup out of range
    if (gauss_base >= N) return;

    // ---- Threadgroup scratch (one per simdgroup) ----
    threadgroup MmaScratch scratch[MAX_SIMDGROUPS_PER_TG];

    // ================================================================
    //  Phase 1: Compute basis functions (threads 0-7, one per Gaussian)
    //  Threads 8-31 zero-init in parallel, then idle during basis eval.
    // ================================================================

    // Zero-init both basis matrices (32 threads → 128 elements)
    for (uint idx = simd_lane; idx < MMA_TILE * MMA_TILE * 2; idx += SIMD_WIDTH) {
        uint matrix_id = idx / (MMA_TILE * MMA_TILE);  // 0 = lo, 1 = hi
        uint flat = idx % (MMA_TILE * MMA_TILE);
        uint r = flat / MMA_TILE;
        uint c = flat % MMA_TILE;
        if (matrix_id == 0)
            scratch[simd_gid].basis_lo[r][c] = 0.0f;
        else
            scratch[simd_gid].basis_hi[r][c] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Threads 0-7 each compute all basis functions for one Gaussian
    if (simd_lane < MMA_TILE) {
        uint gid = gauss_base + simd_lane;
        if (gid < N) {
            float3 dir = directions[gid];
            float inv_len = rsqrt(max(dot(dir, dir), 1e-8f));
            float x = dir.x * inv_len;
            float y = dir.y * inv_len;
            float z = dir.z * inv_len;

            float xx = x*x, yy = y*y, zz = z*z;
            float xy = x*y, yz = y*z, xz = x*z;

            uint g = simd_lane;  // row index in the A matrix

            // ---- First tile: bases 0-7 ----
            scratch[simd_gid].basis_lo[g][0] = SH_C0;

            if (degree >= 1) {
                scratch[simd_gid].basis_lo[g][1] = -SH_C1 * y;
                scratch[simd_gid].basis_lo[g][2] =  SH_C1 * z;
                scratch[simd_gid].basis_lo[g][3] = -SH_C1 * x;
            }

            if (degree >= 2) {
                scratch[simd_gid].basis_lo[g][4] = SH_C2_0 * xy;
                scratch[simd_gid].basis_lo[g][5] = SH_C2_1 * yz;
                scratch[simd_gid].basis_lo[g][6] = SH_C2_2 * (2.f*zz - xx - yy);
                scratch[simd_gid].basis_lo[g][7] = SH_C2_3 * xz;
            }

            // ---- Second tile: bases 8-15 ----
            if (degree >= 2) {
                scratch[simd_gid].basis_hi[g][0] = SH_C2_4 * (xx - yy);
            }

            if (degree >= 3) {
                scratch[simd_gid].basis_hi[g][1] = SH_C3_0 * y * (3.f*xx - yy);
                scratch[simd_gid].basis_hi[g][2] = SH_C3_1 * xy * z;
                scratch[simd_gid].basis_hi[g][3] = SH_C3_2 * y * (4.f*zz - xx - yy);
                scratch[simd_gid].basis_hi[g][4] = SH_C3_3 * z * (2.f*zz - 3.f*xx - 3.f*yy);
                scratch[simd_gid].basis_hi[g][5] = SH_C3_4 * x * (4.f*zz - xx - yy);
                scratch[simd_gid].basis_hi[g][6] = SH_C3_5 * z * (xx - yy);
                scratch[simd_gid].basis_hi[g][7] = SH_C3_6 * x * (xx - 3.f*yy);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Load A matrices into simdgroup registers (shared across 3 channels) ----
    simdgroup_float8x8 matA_lo, matA_hi;
    simdgroup_load(matA_lo, &scratch[simd_gid].basis_lo[0][0], MMA_TILE);
    simdgroup_load(matA_hi, &scratch[simd_gid].basis_hi[0][0], MMA_TILE);

    // ================================================================
    //  Phase 2: Per-channel MMA — iterate R, G, B
    //
    //  For each channel c:
    //    1. Cooperative load of B[k][g] = C[gauss_base+g, k, c]
    //    2. MMA: D = A_lo × B_lo + A_hi × B_hi
    //    3. Extract diagonal D[g][g] → result for Gaussian g, channel c
    // ================================================================

    for (uint c = 0; c < 3; c++) {

        // ---- Load coefficient tile (bases 0-7): B_lo[k][g] ----
        // 32 threads cooperatively load 64 elements
        for (uint idx = simd_lane; idx < MMA_TILE * MMA_TILE; idx += SIMD_WIDTH) {
            uint k = idx / MMA_TILE;   // basis index (row of B)
            uint g = idx % MMA_TILE;   // Gaussian index (col of B)
            uint gid = gauss_base + g;
            scratch[simd_gid].coeff[k][g] =
                (gid < N && k < K)
                    ? float(sh_coeffs[(gid * K + k) * 3 + c])
                    : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- MMA: D = A_lo × B_lo ----
        simdgroup_float8x8 matB, matD;
        simdgroup_float8x8 zero_mat(0.0f);
        simdgroup_load(matB, &scratch[simd_gid].coeff[0][0], MMA_TILE);
        simdgroup_multiply_accumulate(matD, matA_lo, matB, zero_mat);

        // ---- Load coefficient tile (bases 8-15): B_hi[k][g] ----
        for (uint idx = simd_lane; idx < MMA_TILE * MMA_TILE; idx += SIMD_WIDTH) {
            uint k      = idx / MMA_TILE;
            uint g      = idx % MMA_TILE;
            uint gid    = gauss_base + g;
            uint real_k = k + MMA_TILE;  // bases 8..15
            scratch[simd_gid].coeff[k][g] =
                (gid < N && real_k < K)
                    ? float(sh_coeffs[(gid * K + real_k) * 3 + c])
                    : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- MMA: D += A_hi × B_hi ----
        simdgroup_load(matB, &scratch[simd_gid].coeff[0][0], MMA_TILE);
        simdgroup_multiply_accumulate(matD, matA_hi, matB, matD);

        // ---- Store D and extract diagonal ----
        simdgroup_store(matD, &scratch[simd_gid].result[0][0], MMA_TILE);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Threads 0-7 read diagonal element → result for their Gaussian
        if (simd_lane < MMA_TILE) {
            uint gid = gauss_base + simd_lane;
            if (gid < N) {
                float val = scratch[simd_gid].result[simd_lane][simd_lane];
                val += 0.5f;              // DC offset (3DGS convention)
                val = clamp(val, 0.0f, 1.0f);
                colors_out[gid * 3 + c] = half(val);
            }
        }

        // Barrier before next channel reuses coeff[] scratch
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#else  // Scalar fallback for M1/M2 (no simdgroup_matrix or MSL < 3.0)

// ============================================================================
//  Scalar fallback: identical to compute_sh_forward but with _mma name
//
//  Ensures the metallib always contains `compute_sh_forward_mma` regardless
//  of target GPU family. The wrapper can unconditionally create this PSO.
//
//  Thread model: 1 thread per Gaussian, threadgroup = 256.
//  dispatch grid: ceil(N / 256) threadgroups × 256 threads.
// ============================================================================

kernel void compute_sh_forward_mma(
    device const float3*      directions  [[buffer(0)]],
    device const half*        sh_coeffs   [[buffer(1)]],
    device half*              colors_out  [[buffer(2)]],
    constant SHParamsSIMD&    params      [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
)
{
    if (tid >= params.num_points) return;

    float3 dir = directions[tid];
    float inv_len = rsqrt(max(dot(dir, dir), 1e-8f));
    float x = dir.x * inv_len;
    float y = dir.y * inv_len;
    float z = dir.z * inv_len;

    uint base_offset = tid * params.num_bases * 3;

    #define READ_SH(basis) float3( \
        float(sh_coeffs[base_offset + (basis) * 3 + 0]), \
        float(sh_coeffs[base_offset + (basis) * 3 + 1]), \
        float(sh_coeffs[base_offset + (basis) * 3 + 2])  \
    )

    // ---- Degree 0 ----
    float3 result = SH_C0 * READ_SH(0);

    // ---- Degree 1 ----
    if (params.sh_degree >= 1) {
        result += SH_C1 * (-y * READ_SH(1) + z * READ_SH(2) - x * READ_SH(3));
    }

    // ---- Degree 2 ----
    if (params.sh_degree >= 2) {
        float xx = x*x, yy = y*y, zz = z*z;
        float xy = x*y, yz = y*z, xz = x*z;

        result += SH_C2_0 * xy          * READ_SH(4)
                + SH_C2_1 * yz          * READ_SH(5)
                + SH_C2_2 * (2.f*zz - xx - yy) * READ_SH(6)
                + SH_C2_3 * xz          * READ_SH(7)
                + SH_C2_4 * (xx - yy)   * READ_SH(8);
    }

    // ---- Degree 3 ----
    if (params.sh_degree >= 3) {
        float xx = x*x, yy = y*y, zz = z*z;
        float xy = x*y, yz = y*z, xz = x*z;

        result += SH_C3_0 * y * (3.f*xx - yy)       * READ_SH(9)
                + SH_C3_1 * xy * z                    * READ_SH(10)
                + SH_C3_2 * y * (4.f*zz - xx - yy)   * READ_SH(11)
                + SH_C3_3 * z * (2.f*zz - 3.f*xx - 3.f*yy) * READ_SH(12)
                + SH_C3_4 * x * (4.f*zz - xx - yy)   * READ_SH(13)
                + SH_C3_5 * z * (xx - yy)             * READ_SH(14)
                + SH_C3_6 * x * (xx - 3.f*yy)         * READ_SH(15);
    }

    #undef READ_SH

    result += 0.5f;
    result = clamp(result, 0.f, 1.f);

    colors_out[tid * 3 + 0] = half(result.x);
    colors_out[tid * 3 + 1] = half(result.y);
    colors_out[tid * 3 + 2] = half(result.z);
}

#endif // __HAVE_SIMDGROUP_MATRIX__ && __METAL_VERSION__ >= 300
