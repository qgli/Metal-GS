// ============================================================================
//  Metal-GS: Spherical Harmonics Compute Shader
//  Optimised for Apple Silicon TBDR — mixed precision (FP32 geometry, FP16 colour)
//
//  Convention follows the original 3DGS (inria):
//    - Degree 0-3  (16 basis functions max)
//    - Coefficient layout: [N, K, 3]  (basis-major, then RGB)
//    - Output includes +0.5 DC offset, clamped to [0, 1]
//
//  Design choices for M1 (7-core GPU, 16 GB unified memory):
//    ▸ View directions kept in FP32 for numerical stability of normalisation
//    ▸ SH coefficients stored/read as FP16 → 2× bandwidth saving
//    ▸ Accumulation in FP32 to avoid catastrophic cancellation across bases
//    ▸ Final colour converted to FP16 for downstream alpha-blending
//    ▸ Threadgroup size = 256 (sweet spot for M1 occupancy)
//    ▸ No shared memory needed — each thread is independent (embarrassingly parallel)
//    ▸ Future: BF16 path gated on Metal GPU family check (M2+)
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
//  SH normalisation constants  (real spherical harmonics, Condon-Shortley phase)
// ---------------------------------------------------------------------------
constant float SH_C0 = 0.28209479177387814f;       // 1 / (2 * sqrt(pi))
constant float SH_C1 = 0.4886025119029199f;        // sqrt(3) / (2 * sqrt(pi))

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
//  Parameters passed via constant buffer
// ---------------------------------------------------------------------------
struct SHParams {
    uint num_points;    // N
    uint num_bases;     // K  (total SH coefficients per point)
    uint sh_degree;     // 0-3
};

// ---------------------------------------------------------------------------
//  Kernel: compute_sh_forward
//
//  Inputs:
//    directions  — float3[N]  (unit view directions, FP32)
//    sh_coeffs   — half[N * K * 3]  (SH coefficients, FP16, layout [N, K, 3])
//  Output:
//    colors_out  — half3[N]   (RGB, FP16)
//
//  One thread per Gaussian.  Threadgroup = 256 threads.
// ---------------------------------------------------------------------------
kernel void compute_sh_forward(
    device const packed_float3* directions  [[buffer(0)]],
    device const half*    sh_coeffs   [[buffer(1)]],
    device half*          colors_out  [[buffer(2)]],
    constant SHParams&    params      [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
)
{
    if (tid >= params.num_points) return;

    // ---- Read view direction (FP32 for stable normalisation) ----
    float3 dir = directions[tid];
    // Re-normalise in case the caller didn't (cheap insurance)
    float inv_len = rsqrt(max(dot(dir, dir), 1e-8f));
    float x = dir.x * inv_len;
    float y = dir.y * inv_len;
    float z = dir.z * inv_len;

    // ---- Pointer to this point's SH coefficients ----
    // Layout: sh_coeffs[ (tid * K + basis) * 3 + channel ]
    uint base_offset = tid * params.num_bases * 3;

    // ---- Helper: read one SH basis's RGB as FP32 for accumulation ----
    #define READ_SH(basis) float3( \
        float(sh_coeffs[base_offset + (basis) * 3 + 0]), \
        float(sh_coeffs[base_offset + (basis) * 3 + 1]), \
        float(sh_coeffs[base_offset + (basis) * 3 + 2])  \
    )

    // ---- Degree 0 (DC) — always present ----
    float3 result = SH_C0 * READ_SH(0);

    // ---- Degree 1 ----
    if (params.sh_degree >= 1) {
        // Basis 1: Y_1^{-1} = -y,   Basis 2: Y_1^0 = z,   Basis 3: Y_1^1 = -x
        result += SH_C1 * (-y * READ_SH(1) + z * READ_SH(2) - x * READ_SH(3));
    }

    // ---- Degree 2 ----
    if (params.sh_degree >= 2) {
        float xx = x * x, yy = y * y, zz = z * z;
        float xy = x * y, yz = y * z, xz = x * z;

        result += SH_C2_0 * xy          * READ_SH(4)
                + SH_C2_1 * yz          * READ_SH(5)
                + SH_C2_2 * (2.f*zz - xx - yy) * READ_SH(6)
                + SH_C2_3 * xz          * READ_SH(7)
                + SH_C2_4 * (xx - yy)   * READ_SH(8);
    }

    // ---- Degree 3 ----
    if (params.sh_degree >= 3) {
        float xx = x * x, yy = y * y, zz = z * z;
        float xy = x * y, yz = y * z, xz = x * z;

        result += SH_C3_0 * y * (3.f*xx - yy)       * READ_SH(9)
                + SH_C3_1 * xy * z                    * READ_SH(10)
                + SH_C3_2 * y * (4.f*zz - xx - yy)   * READ_SH(11)
                + SH_C3_3 * z * (2.f*zz - 3.f*xx - 3.f*yy) * READ_SH(12)
                + SH_C3_4 * x * (4.f*zz - xx - yy)   * READ_SH(13)
                + SH_C3_5 * z * (xx - yy)             * READ_SH(14)
                + SH_C3_6 * x * (xx - 3.f*yy)         * READ_SH(15);
    }

    #undef READ_SH

    // ---- Apply +0.5 offset (standard 3DGS convention) and clamp ----
    result += 0.5f;
    result = clamp(result, 0.f, 1.f);

    // ---- Write output as FP16 ----
    colors_out[tid * 3 + 0] = half(result.x);
    colors_out[tid * 3 + 1] = half(result.y);
    colors_out[tid * 3 + 2] = half(result.z);
}
