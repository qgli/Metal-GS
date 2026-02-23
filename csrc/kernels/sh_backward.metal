// ============================================================================
//  Metal-GS: Spherical Harmonics Backward Kernel (M4)
//
//  Computes gradients from color back to SH coefficients and view directions:
//    dL/d_colors → dL/d_sh_coeffs + dL/d_directions → dL/d_means3d
//
//  1:1 mapping — one thread per Gaussian, NO atomic operations needed.
//
//  Gradient chain:
//    color_c = clamp(SH_eval_c + 0.5, 0, 1)
//    dL/d_SH_eval = dL/d_color * (1 if 0 < color < 1, else 0)   [clamp grad]
//    dL/d_sh_k = Y_k(d) * dL/d_SH_eval
//    dL/d_d = Σ_k (∂Y_k/∂d) * sh_k · dL/d_SH_eval
//    dL/d_mean3d += (I - d dᵀ) / ||v|| * dL/d_d
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ---- SH constants (must match forward) ----
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

// ---- Parameters ----
struct SHParams {
    uint num_points;
    uint num_bases;
    uint sh_degree;
};

// =========================================================================
//  Kernel: sh_backward
//
//  Inputs:
//    means3d         — float[N*3]   (Gaussian centers, world space)
//    campos          — float[3]     (camera position, world space)
//    sh_coeffs       — half[N*K*3]  (SH coefficients, FP16)
//    colors_fwd      — half[N*3]    (forward output colors, to detect clamp)
//    dL_d_colors     — float[N*3]   (upstream gradient from rasterize_bw)
//
//  Outputs:
//    dL_d_sh         — float[N*K*3] (gradients w.r.t. SH coefficients)
//    dL_d_means3d    — float[N*3]   (gradients from direction → position)
// =========================================================================
kernel void sh_backward(
    device const float*   means3d       [[buffer(0)]],   // [N*3]
    constant float*       campos        [[buffer(1)]],   // [3]
    device const half*    sh_coeffs     [[buffer(2)]],   // [N*K*3]
    device const half*    colors_fwd    [[buffer(3)]],   // [N*3] forward output
    device const float*   dL_d_colors   [[buffer(4)]],   // [N*3]
    constant SHParams&    params        [[buffer(5)]],
    // ---- Outputs ----
    device float*         dL_d_sh       [[buffer(6)]],   // [N*K*3]
    device float*         dL_d_means3d  [[buffer(7)]],   // [N*3]

    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.num_points) return;

    uint K = params.num_bases;
    uint degree = params.sh_degree;

    // ---- Read upstream color gradient ----
    float3 dL_dc = float3(dL_d_colors[tid*3], dL_d_colors[tid*3+1], dL_d_colors[tid*3+2]);

    // ---- Clamp gradient: zero out where forward clamped ----
    // Forward: color = clamp(SH_eval + 0.5, 0, 1)
    // If color_fwd == 0 or color_fwd == 1, gradient is zero
    float3 color_fwd = float3(
        float(colors_fwd[tid*3]),
        float(colors_fwd[tid*3+1]),
        float(colors_fwd[tid*3+2])
    );

    // Use a small tolerance for FP16 boundary detection
    if (color_fwd.x <= 0.0f || color_fwd.x >= 1.0f) dL_dc.x = 0.0f;
    if (color_fwd.y <= 0.0f || color_fwd.y >= 1.0f) dL_dc.y = 0.0f;
    if (color_fwd.z <= 0.0f || color_fwd.z >= 1.0f) dL_dc.z = 0.0f;

    // ---- Compute view direction (same as forward) ----
    float3 pw = float3(means3d[tid*3], means3d[tid*3+1], means3d[tid*3+2]);
    float3 cp = float3(campos[0], campos[1], campos[2]);
    float3 v = pw - cp;
    float v_len = length(v);
    float inv_len = 1.0f / max(v_len, 1e-8f);
    float x = v.x * inv_len;
    float y = v.y * inv_len;
    float z = v.z * inv_len;

    // ---- SH coefficient base offset ----
    uint base_offset = tid * K * 3;

    #define READ_SH(basis) float3( \
        float(sh_coeffs[base_offset + (basis) * 3 + 0]), \
        float(sh_coeffs[base_offset + (basis) * 3 + 1]), \
        float(sh_coeffs[base_offset + (basis) * 3 + 2])  \
    )

    // ===================================================================
    //  dL/d_sh: for each basis function k, dL/d_sh[k] = Y_k(d) * dL_dc
    //  (this is per-channel, so 3 floats per basis)
    //
    //  dL/d_d: accumulated directional gradient from all basis functions
    //  dL/d_d[c] = Σ_k (∂Y_k/∂d) * sh_k[c] * dL_dc[c]  (per channel)
    // ===================================================================

    float3 dL_dx = float3(0.0f);
    float3 dL_dy = float3(0.0f);
    float3 dL_dz = float3(0.0f);

    // ---- Degree 0: Y_0 = SH_C0 (constant) ----
    {
        float Y0 = SH_C0;
        float3 dL_dsh0 = Y0 * dL_dc;
        dL_d_sh[base_offset + 0] = dL_dsh0.x;
        dL_d_sh[base_offset + 1] = dL_dsh0.y;
        dL_d_sh[base_offset + 2] = dL_dsh0.z;
        // No directional gradient from degree 0 (constant)
    }

    // ---- Degree 1: Y1 = -SH_C1*y, Y2 = SH_C1*z, Y3 = -SH_C1*x ----
    if (degree >= 1) {
        float Y1 = -SH_C1 * y;
        float Y2 =  SH_C1 * z;
        float Y3 = -SH_C1 * x;

        float3 dL_dsh1 = Y1 * dL_dc;
        float3 dL_dsh2 = Y2 * dL_dc;
        float3 dL_dsh3 = Y3 * dL_dc;

        dL_d_sh[base_offset + 1*3 + 0] = dL_dsh1.x;
        dL_d_sh[base_offset + 1*3 + 1] = dL_dsh1.y;
        dL_d_sh[base_offset + 1*3 + 2] = dL_dsh1.z;
        dL_d_sh[base_offset + 2*3 + 0] = dL_dsh2.x;
        dL_d_sh[base_offset + 2*3 + 1] = dL_dsh2.y;
        dL_d_sh[base_offset + 2*3 + 2] = dL_dsh2.z;
        dL_d_sh[base_offset + 3*3 + 0] = dL_dsh3.x;
        dL_d_sh[base_offset + 3*3 + 1] = dL_dsh3.y;
        dL_d_sh[base_offset + 3*3 + 2] = dL_dsh3.z;

        // ∂Y1/∂y = -SH_C1, ∂Y2/∂z = SH_C1, ∂Y3/∂x = -SH_C1
        float3 sh1 = READ_SH(1);
        float3 sh2 = READ_SH(2);
        float3 sh3 = READ_SH(3);

        dL_dx += -SH_C1 * sh3 * dL_dc;
        dL_dy += -SH_C1 * sh1 * dL_dc;
        dL_dz +=  SH_C1 * sh2 * dL_dc;
    }

    // ---- Degree 2 ----
    if (degree >= 2) {
        float xx = x*x, yy = y*y, zz = z*z;
        float xy = x*y, yz = y*z, xz = x*z;

        // Y4 = SH_C2_0 * xy
        // Y5 = SH_C2_1 * yz
        // Y6 = SH_C2_2 * (2zz - xx - yy)
        // Y7 = SH_C2_3 * xz
        // Y8 = SH_C2_4 * (xx - yy)
        float Y4 = SH_C2_0 * xy;
        float Y5 = SH_C2_1 * yz;
        float Y6 = SH_C2_2 * (2.f*zz - xx - yy);
        float Y7 = SH_C2_3 * xz;
        float Y8 = SH_C2_4 * (xx - yy);

        float3 dL_dsh4 = Y4 * dL_dc;
        float3 dL_dsh5 = Y5 * dL_dc;
        float3 dL_dsh6 = Y6 * dL_dc;
        float3 dL_dsh7 = Y7 * dL_dc;
        float3 dL_dsh8 = Y8 * dL_dc;

        dL_d_sh[base_offset + 4*3+0] = dL_dsh4.x;
        dL_d_sh[base_offset + 4*3+1] = dL_dsh4.y;
        dL_d_sh[base_offset + 4*3+2] = dL_dsh4.z;
        dL_d_sh[base_offset + 5*3+0] = dL_dsh5.x;
        dL_d_sh[base_offset + 5*3+1] = dL_dsh5.y;
        dL_d_sh[base_offset + 5*3+2] = dL_dsh5.z;
        dL_d_sh[base_offset + 6*3+0] = dL_dsh6.x;
        dL_d_sh[base_offset + 6*3+1] = dL_dsh6.y;
        dL_d_sh[base_offset + 6*3+2] = dL_dsh6.z;
        dL_d_sh[base_offset + 7*3+0] = dL_dsh7.x;
        dL_d_sh[base_offset + 7*3+1] = dL_dsh7.y;
        dL_d_sh[base_offset + 7*3+2] = dL_dsh7.z;
        dL_d_sh[base_offset + 8*3+0] = dL_dsh8.x;
        dL_d_sh[base_offset + 8*3+1] = dL_dsh8.y;
        dL_d_sh[base_offset + 8*3+2] = dL_dsh8.z;

        float3 sh4 = READ_SH(4);
        float3 sh5 = READ_SH(5);
        float3 sh6 = READ_SH(6);
        float3 sh7 = READ_SH(7);
        float3 sh8 = READ_SH(8);

        // ∂Y4/∂x = SH_C2_0*y,  ∂Y4/∂y = SH_C2_0*x
        // ∂Y5/∂y = SH_C2_1*z,  ∂Y5/∂z = SH_C2_1*y
        // ∂Y6/∂x = -2*SH_C2_2*x,  ∂Y6/∂y = -2*SH_C2_2*y,  ∂Y6/∂z = 4*SH_C2_2*z
        // ∂Y7/∂x = SH_C2_3*z,  ∂Y7/∂z = SH_C2_3*x
        // ∂Y8/∂x = 2*SH_C2_4*x,  ∂Y8/∂y = -2*SH_C2_4*y
        dL_dx += (SH_C2_0 * y * sh4 - 2.f * SH_C2_2 * x * sh6
                + SH_C2_3 * z * sh7 + 2.f * SH_C2_4 * x * sh8) * dL_dc;
        dL_dy += (SH_C2_0 * x * sh4 + SH_C2_1 * z * sh5
                - 2.f * SH_C2_2 * y * sh6 - 2.f * SH_C2_4 * y * sh8) * dL_dc;
        dL_dz += (SH_C2_1 * y * sh5 + 4.f * SH_C2_2 * z * sh6
                + SH_C2_3 * x * sh7) * dL_dc;
    }

    // ---- Degree 3 ----
    if (degree >= 3) {
        float xx = x*x, yy = y*y, zz = z*z;
        float xy = x*y, yz = y*z, xz = x*z;

        // Y9  = SH_C3_0 * y*(3xx - yy)
        // Y10 = SH_C3_1 * xy*z
        // Y11 = SH_C3_2 * y*(4zz - xx - yy)
        // Y12 = SH_C3_3 * z*(2zz - 3xx - 3yy)
        // Y13 = SH_C3_4 * x*(4zz - xx - yy)
        // Y14 = SH_C3_5 * z*(xx - yy)
        // Y15 = SH_C3_6 * x*(xx - 3yy)

        float Y9  = SH_C3_0 * y * (3.f*xx - yy);
        float Y10 = SH_C3_1 * xy * z;
        float Y11 = SH_C3_2 * y * (4.f*zz - xx - yy);
        float Y12 = SH_C3_3 * z * (2.f*zz - 3.f*xx - 3.f*yy);
        float Y13 = SH_C3_4 * x * (4.f*zz - xx - yy);
        float Y14 = SH_C3_5 * z * (xx - yy);
        float Y15 = SH_C3_6 * x * (xx - 3.f*yy);

        float3 dL_dsh9  = Y9  * dL_dc;
        float3 dL_dsh10 = Y10 * dL_dc;
        float3 dL_dsh11 = Y11 * dL_dc;
        float3 dL_dsh12 = Y12 * dL_dc;
        float3 dL_dsh13 = Y13 * dL_dc;
        float3 dL_dsh14 = Y14 * dL_dc;
        float3 dL_dsh15 = Y15 * dL_dc;

        dL_d_sh[base_offset +  9*3+0] = dL_dsh9.x;
        dL_d_sh[base_offset +  9*3+1] = dL_dsh9.y;
        dL_d_sh[base_offset +  9*3+2] = dL_dsh9.z;
        dL_d_sh[base_offset + 10*3+0] = dL_dsh10.x;
        dL_d_sh[base_offset + 10*3+1] = dL_dsh10.y;
        dL_d_sh[base_offset + 10*3+2] = dL_dsh10.z;
        dL_d_sh[base_offset + 11*3+0] = dL_dsh11.x;
        dL_d_sh[base_offset + 11*3+1] = dL_dsh11.y;
        dL_d_sh[base_offset + 11*3+2] = dL_dsh11.z;
        dL_d_sh[base_offset + 12*3+0] = dL_dsh12.x;
        dL_d_sh[base_offset + 12*3+1] = dL_dsh12.y;
        dL_d_sh[base_offset + 12*3+2] = dL_dsh12.z;
        dL_d_sh[base_offset + 13*3+0] = dL_dsh13.x;
        dL_d_sh[base_offset + 13*3+1] = dL_dsh13.y;
        dL_d_sh[base_offset + 13*3+2] = dL_dsh13.z;
        dL_d_sh[base_offset + 14*3+0] = dL_dsh14.x;
        dL_d_sh[base_offset + 14*3+1] = dL_dsh14.y;
        dL_d_sh[base_offset + 14*3+2] = dL_dsh14.z;
        dL_d_sh[base_offset + 15*3+0] = dL_dsh15.x;
        dL_d_sh[base_offset + 15*3+1] = dL_dsh15.y;
        dL_d_sh[base_offset + 15*3+2] = dL_dsh15.z;

        float3 sh9  = READ_SH(9);
        float3 sh10 = READ_SH(10);
        float3 sh11 = READ_SH(11);
        float3 sh12 = READ_SH(12);
        float3 sh13 = READ_SH(13);
        float3 sh14 = READ_SH(14);
        float3 sh15 = READ_SH(15);

        // ∂Y9/∂x  = SH_C3_0 * 6xy
        // ∂Y9/∂y  = SH_C3_0 * (3xx - 3yy)
        // ∂Y10/∂x = SH_C3_1 * yz
        // ∂Y10/∂y = SH_C3_1 * xz
        // ∂Y10/∂z = SH_C3_1 * xy
        // ∂Y11/∂x = SH_C3_2 * (-2xy)
        // ∂Y11/∂y = SH_C3_2 * (4zz - 3yy - xx)    ← (4zz-xx-yy) + y*(-2y)
        // ∂Y11/∂z = SH_C3_2 * 8yz
        // ∂Y12/∂x = SH_C3_3 * (-6xz)
        // ∂Y12/∂y = SH_C3_3 * (-6yz)
        // ∂Y12/∂z = SH_C3_3 * (6zz - 3xx - 3yy)   ← (2zz-3xx-3yy) + z*4z
        // ∂Y13/∂x = SH_C3_4 * (4zz - 3xx - yy)    ← (4zz-xx-yy) + x*(-2x)
        // ∂Y13/∂y = SH_C3_4 * (-2xy)
        // ∂Y13/∂z = SH_C3_4 * 8xz
        // ∂Y14/∂x = SH_C3_5 * 2xz
        // ∂Y14/∂y = SH_C3_5 * (-2yz)
        // ∂Y14/∂z = SH_C3_5 * (xx - yy)
        // ∂Y15/∂x = SH_C3_6 * (3xx - 3yy)
        // ∂Y15/∂y = SH_C3_6 * (-6xy)

        dL_dx += (SH_C3_0 * 6.f*xy                     * sh9
                + SH_C3_1 * yz                           * sh10
                + SH_C3_2 * (-2.f*xy)                    * sh11
                + SH_C3_3 * (-6.f*xz)                    * sh12
                + SH_C3_4 * (4.f*zz - 3.f*xx - yy)      * sh13
                + SH_C3_5 * 2.f*xz                       * sh14
                + SH_C3_6 * (3.f*xx - 3.f*yy)            * sh15) * dL_dc;

        dL_dy += (SH_C3_0 * (3.f*xx - 3.f*yy)           * sh9
                + SH_C3_1 * xz                            * sh10
                + SH_C3_2 * (4.f*zz - xx - 3.f*yy)       * sh11
                + SH_C3_3 * (-6.f*yz)                     * sh12
                + SH_C3_4 * (-2.f*xy)                     * sh13
                + SH_C3_5 * (-2.f*yz)                     * sh14
                + SH_C3_6 * (-6.f*xy)                     * sh15) * dL_dc;

        dL_dz += (SH_C3_1 * xy                            * sh10
                + SH_C3_2 * 8.f*yz                         * sh11
                + SH_C3_3 * (6.f*zz - 3.f*xx - 3.f*yy)    * sh12
                + SH_C3_4 * 8.f*xz                         * sh13
                + SH_C3_5 * (xx - yy)                      * sh14) * dL_dc;
    }

    #undef READ_SH

    // Zero out unused SH coefficient gradients
    for (uint k = (degree >= 3 ? 16 : (degree >= 2 ? 9 : (degree >= 1 ? 4 : 1))); k < K; k++) {
        dL_d_sh[base_offset + k*3 + 0] = 0.0f;
        dL_d_sh[base_offset + k*3 + 1] = 0.0f;
        dL_d_sh[base_offset + k*3 + 2] = 0.0f;
    }

    // ===================================================================
    //  dL/d_direction → dL/d_means3d
    //
    //  d = v / ||v||,  v = mean3d - campos
    //  ∂d/∂v = (I - d dᵀ) / ||v||
    //
    //  dL/d_v = sum_c (dL_dx[c], dL_dy[c], dL_dz[c]) projected through ∂d/∂v
    // ===================================================================

    // Sum over color channels
    float dL_ddx = dL_dx.x + dL_dx.y + dL_dx.z;
    float dL_ddy = dL_dy.x + dL_dy.y + dL_dy.z;
    float dL_ddz = dL_dz.x + dL_dz.y + dL_dz.z;

    // dL/d_v = (I - d*d^T) * dL/d_d / ||v||
    float3 dL_dd = float3(dL_ddx, dL_ddy, dL_ddz);
    float3 d = float3(x, y, z);
    float3 dL_dv = inv_len * (dL_dd - d * dot(d, dL_dd));

    // dL/d_means3d += dL/d_v (since v = mean3d - campos)
    dL_d_means3d[tid*3]     = dL_dv.x;
    dL_d_means3d[tid*3 + 1] = dL_dv.y;
    dL_d_means3d[tid*3 + 2] = dL_dv.z;
}
