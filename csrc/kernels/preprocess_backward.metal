// ============================================================================
//  Metal-GS: Fused Preprocess + Cov3D Backward Kernel
//
//  Computes gradients from 2D parameters back to 3D Gaussian parameters:
//    dL/d_cov2d, dL/d_mean2d  →  dL/d_means3d, dL/d_scales, dL/d_quats
//
//  Fuses the preprocess_backward and cov3d_backward stages (M3).
//  1:1 mapping — one thread per Gaussian, NO atomic operations needed.
//
//  Gradient chain:
//    dL/d_cov2d → dL/d_cov3d → dL/d_M → dL/d_scales + dL/d_R → dL/d_quats
//    dL/d_cov2d → dL/d_T_eff → dL/d_J → dL/d_t → dL/d_means3d (part 1)
//    dL/d_mean2d → dL/d_t → dL/d_means3d (part 2)
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ---- Constants ----
// (CLIP_THRESH and AA_BLUR not needed in backward — forward-only)

// ---- Params struct (same as forward preprocess) ----
struct PreprocessParams {
    float    tan_fovx;
    float    tan_fovy;
    float    focal_x;
    float    focal_y;
    float    principal_x;
    float    principal_y;
    uint     img_width;
    uint     img_height;
    uint     num_points;
};

// =========================================================================
//  Helper: quaternion → rotation matrix (same as forward, column-major)
// =========================================================================
inline float3x3 quat_to_rotmat(float4 q) {
    float inv_n = rsqrt(max(dot(q, q), 1e-8f));
    q *= inv_n;
    float x = q.x, y = q.y, z = q.z, w = q.w;
    return float3x3(
        1.f-2.f*(y*y+z*z),  2.f*(x*y+w*z),      2.f*(x*z-w*y),
        2.f*(x*y-w*z),      1.f-2.f*(x*x+z*z),   2.f*(y*z+w*x),
        2.f*(x*z+w*y),      2.f*(y*z-w*x),       1.f-2.f*(x*x+y*y)
    );
}

// =========================================================================
//  Helper: affine transform (same as forward)
// =========================================================================
inline float3 transform_4x3(constant float* m, float3 p) {
    return float3(
        m[0]*p.x + m[1]*p.y + m[ 2]*p.z + m[ 3],
        m[4]*p.x + m[5]*p.y + m[ 6]*p.z + m[ 7],
        m[8]*p.x + m[9]*p.y + m[10]*p.z + m[11]
    );
}

// =========================================================================
//  Kernel: preprocess_backward
//
//  1 thread per Gaussian. Invisible Gaussians (radii=0) are skipped.
// =========================================================================
kernel void preprocess_backward(
    // ---- Forward inputs (needed to recompute intermediates) ----
    device const float*   means3d      [[buffer(0)]],   // [N*3]
    device const float*   scales       [[buffer(1)]],   // [N*3]
    device const float*   quats        [[buffer(2)]],   // [N*4]
    constant float*       viewmat      [[buffer(3)]],   // [16] row-major
    device const uint*    radii        [[buffer(4)]],   // [N]
    constant PreprocessParams& params  [[buffer(5)]],
    // ---- Upstream gradients (from rasterize_backward) ----
    device const float*   dL_d_cov2d   [[buffer(6)]],   // [N*3] upper-tri
    device const float*   dL_d_mean2d  [[buffer(7)]],   // [N*2]
    // ---- Output gradients ----
    device float*         dL_d_means3d [[buffer(8)]],   // [N*3]
    device float*         dL_d_scales  [[buffer(9)]],   // [N*3]
    device float*         dL_d_quats   [[buffer(10)]],  // [N*4]

    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.num_points) return;

    // Skip invisible Gaussians
    if (radii[tid] == 0) {
        dL_d_means3d[tid*3]   = 0.f;
        dL_d_means3d[tid*3+1] = 0.f;
        dL_d_means3d[tid*3+2] = 0.f;
        dL_d_scales[tid*3]    = 0.f;
        dL_d_scales[tid*3+1]  = 0.f;
        dL_d_scales[tid*3+2]  = 0.f;
        dL_d_quats[tid*4]     = 0.f;
        dL_d_quats[tid*4+1]   = 0.f;
        dL_d_quats[tid*4+2]   = 0.f;
        dL_d_quats[tid*4+3]   = 0.f;
        return;
    }

    // ---- Read inputs ----
    float3 pw = float3(means3d[tid*3], means3d[tid*3+1], means3d[tid*3+2]);
    float3 sc = float3(scales[tid*3],  scales[tid*3+1],  scales[tid*3+2]);
    float4 qt = float4(quats[tid*4],   quats[tid*4+1],   quats[tid*4+2], quats[tid*4+3]);

    float dL_da = dL_d_cov2d[tid*3];
    float dL_db = dL_d_cov2d[tid*3+1];
    float dL_dc = dL_d_cov2d[tid*3+2];

    float dL_du = dL_d_mean2d[tid*2];
    float dL_dv = dL_d_mean2d[tid*2+1];

    // ===================================================================
    //  RECOMPUTE forward intermediates
    // ===================================================================

    // Step 1: world → camera
    float3 p_cam = transform_4x3(viewmat, pw);
    float depth = p_cam.z;

    // Step 2: rotation matrix from quaternion
    float3x3 R = quat_to_rotmat(qt);

    // Step 3: M = R * diag(s)
    float3x3 M = float3x3(R[0] * sc.x, R[1] * sc.y, R[2] * sc.z);

    // Step 4: cov3d = M * M^T (stored as 3x3 symmetric)
    float3x3 V = M * transpose(M);

    // Step 5: FOV clamping
    float lx = 1.3f * params.tan_fovx;
    float ly = 1.3f * params.tan_fovy;
    float rz  = 1.f / depth;
    float rz2 = rz * rz;

    float tx_raw = p_cam.x * rz;
    float ty_raw = p_cam.y * rz;
    float tx = clamp(tx_raw, -lx, lx) * depth;
    float ty = clamp(ty_raw, -ly, ly) * depth;
    bool  tx_clamped = (tx_raw < -lx || tx_raw > lx);
    bool  ty_clamped = (ty_raw < -ly || ty_raw > ly);

    float fx = params.focal_x;
    float fy = params.focal_y;

    // Step 6: Jacobian J (3x3, col-major, row 2 = 0)
    float3x3 J = float3x3(
        fx*rz,      0.f,        0.f,
        0.f,        fy*rz,      0.f,
        -fx*tx*rz2, -fy*ty*rz2, 0.f
    );

    // Step 7: W = rotation part of viewmat (col-major)
    float3x3 W = float3x3(
        float3(viewmat[0], viewmat[4], viewmat[8]),
        float3(viewmat[1], viewmat[5], viewmat[9]),
        float3(viewmat[2], viewmat[6], viewmat[10])
    );

    // Step 8: T_eff = J * W
    float3x3 T_eff = J * W;

    // ===================================================================
    //  BACKWARD: dL/d_cov2d → dL/d_cov3d
    // ===================================================================

    // cov2d = T_eff * V * T_eff^T + blur*I
    // dL/d_cov2d is [a, b, c] = [Σ00, Σ01, Σ11]
    // Full 2x2 gradient matrix:
    //   dL/dΣ2d = [[dL_da, dL_db], [dL_db, dL_dc]]
    // But cov2d is computed from T_eff * V * T_eff^T where T_eff is 3x3 (row 2 = 0)
    //
    // dL/dV = T_eff^T * dL/dΣ2d * T_eff
    //
    // Since T_eff row 2 is zero, we only need the 2x2 upper-left block:
    //   T = T_eff[:2, :] (2x3)
    //   dL/dV = T^T * G * T   where G = [[dL_da, dL_db], [dL_db, dL_dc]]

    // T_eff rows (extracting from column-major matrix)
    float3 T0 = float3(T_eff[0][0], T_eff[1][0], T_eff[2][0]);  // row 0
    float3 T1 = float3(T_eff[0][1], T_eff[1][1], T_eff[2][1]);  // row 1

    // G * T → two row vectors
    // GT0 = dL_da * T0 + dL_db * T1
    // GT1 = dL_db * T0 + dL_dc * T1
    float3 GT0 = dL_da * T0 + dL_db * T1;
    float3 GT1 = dL_db * T0 + dL_dc * T1;

    // dL/dV = T^T * G * T (3x3 symmetric)
    // dV[i][j] = T0[i]*GT0[j] + T1[i]*GT1[j]
    float3x3 dL_dV;
    dL_dV[0] = T0.x * GT0 + T1.x * GT1;  // col 0
    dL_dV[1] = T0.y * GT0 + T1.y * GT1;  // col 1
    dL_dV[2] = T0.z * GT0 + T1.z * GT1;  // col 2

    // ===================================================================
    //  BACKWARD: dL/d_cov3d → dL/d_M → dL/d_scales, dL/d_R
    // ===================================================================

    // V = M * M^T → dL/dM = (dL/dV + dL/dV^T) * M = 2 * symm(dL/dV) * M
    // Since dL/dV may not be perfectly symmetric due to floating point,
    // symmetrize it first
    float3x3 dL_dV_sym = 0.5f * (dL_dV + transpose(dL_dV));
    float3x3 dL_dM = 2.0f * dL_dV_sym * M;

    // M = R * diag(s), so M[:,j] = R[:,j] * s[j]
    // dL/ds[j] = dot(R[:,j], dL/dM[:,j])
    // (In column-major: dL/ds[j] = dot(R[j], dL_dM[j]))
    float3 dL_dscales = float3(
        dot(R[0], dL_dM[0]),
        dot(R[1], dL_dM[1]),
        dot(R[2], dL_dM[2])
    );

    // dL/dR[:,j] = dL/dM[:,j] * s[j]
    // (In column-major: dL/dR[j] = dL_dM[j] * s[j])
    float3x3 dL_dR = float3x3(
        dL_dM[0] * sc.x,
        dL_dM[1] * sc.y,
        dL_dM[2] * sc.z
    );

    // ===================================================================
    //  BACKWARD: dL/d_R → dL/d_quats  (quaternion VJP)
    // ===================================================================

    // Normalize quaternion first (same as forward)
    float inv_n = rsqrt(max(dot(qt, qt), 1e-8f));
    float4 qn = qt * inv_n;
    float qx = qn.x, qy = qn.y, qz = qn.z, qw = qn.w;

    // dL/dR is a 3x3 matrix. We need dL/dq = trace(dL/dR^T * dR/dq)
    // Using the analytical derivatives of R w.r.t. normalized quaternion components.
    //
    // R = [[1-2(yy+zz), 2(xy-wz), 2(xz+wy)],
    //      [2(xy+wz), 1-2(xx+zz), 2(yz-wx)],
    //      [2(xz-wy), 2(yz+wx), 1-2(xx+yy)]]
    //
    // dL/dR[i][j] in column-major = dL_dR[j][i]

    // Extract dL/dR in row-major notation for clarity
    float r00 = dL_dR[0][0], r01 = dL_dR[1][0], r02 = dL_dR[2][0];
    float r10 = dL_dR[0][1], r11 = dL_dR[1][1], r12 = dL_dR[2][1];
    float r20 = dL_dR[0][2], r21 = dL_dR[1][2], r22 = dL_dR[2][2];

    // dR/dx contributions (partial of each R element w.r.t. x):
    // dR00/dx = 0
    // dR01/dx = 2y,   dR02/dx = 2z
    // dR10/dx = 2y,   dR11/dx = -4x,  dR12/dx = -2w
    // dR20/dx = 2z,   dR21/dx = 2w,   dR22/dx = -4x
    float dL_dqx = 2.0f * (
          r01 * qy + r02 * qz
        + r10 * qy - r11 * 2.0f * qx - r12 * qw
        + r20 * qz + r21 * qw - r22 * 2.0f * qx
    );

    // dR/dy contributions:
    // dR00/dy = -4y,  dR01/dy = 2x,   dR02/dy = -2w
    // dR10/dy = 2x,   dR11/dy = 0
    // dR12/dy = 2z
    // dR20/dy = 2w,   dR21/dy = 2z,   dR22/dy = -4y
    float dL_dqy = 2.0f * (
        - r00 * 2.0f * qy + r01 * qx - r02 * qw
        + r10 * qx
        + r12 * qz
        + r20 * qw + r21 * qz - r22 * 2.0f * qy
    );

    // dR/dz contributions:
    // dR00/dz = -4z,  dR01/dz = -2w,  dR02/dz = 2x
    // dR10/dz = 2w,   dR11/dz = -4z,  dR12/dz = 2y
    // dR20/dz = 2x,   dR21/dz = 2y,   dR22/dz = 0
    float dL_dqz = 2.0f * (
        - r00 * 2.0f * qz - r01 * qw + r02 * qx
        + r10 * qw - r11 * 2.0f * qz + r12 * qy
        + r20 * qx + r21 * qy
    );

    // dR/dw contributions:
    // dR00/dw = 0,    dR01/dw = -2z,  dR02/dw = 2y
    // dR10/dw = 2z,   dR11/dw = 0,    dR12/dw = -2x
    // dR20/dw = -2y,  dR21/dw = 2x,   dR22/dw = 0
    float dL_dqw = 2.0f * (
        - r01 * qz + r02 * qy
        + r10 * qz - r12 * qx
        - r20 * qy + r21 * qx
    );

    // Backprop through quaternion normalization:
    // qn = qt / ||qt||
    // d(qn)/d(qt) = (I - qn * qn^T) / ||qt||
    float4 dL_dqn = float4(dL_dqx, dL_dqy, dL_dqz, dL_dqw);
    float4 dL_dqt = inv_n * (dL_dqn - qn * dot(qn, dL_dqn));

    // ===================================================================
    //  BACKWARD: dL/d_cov2d → dL/d_T_eff → dL/d_J → dL/d_t (camera pos)
    // ===================================================================

    // cov2d = T_eff * V * T_eff^T
    // dL/d_T_eff = G * T_eff * V^T + G^T * T_eff * V
    //            = 2 * G * T_eff * V   (since V is symmetric and G is symmetric)
    // But we only care about rows 0,1 of T_eff (row 2 is zero)
    //
    // Actually: dL/dT = (dL/dΣ2d + dL/dΣ2d^T) * T * V = 2*G*T*V
    // where G = [[dL_da, dL_db],[dL_db, dL_dc]] (already symmetric)

    // dL/dT[i][j] = sum over k,l of: 2 * G[i][k] * sum_m T[k][m] * V[m][j]
    // Easier: dL/dT_eff = 2 * G_full * T_eff * V (matrix multiply)

    // G_full in 3x3 (with zeros in row/col 2):
    // [[dL_da, dL_db, 0], [dL_db, dL_dc, 0], [0, 0, 0]]
    // But since T_eff row 2 = 0, we can work with 2x3 directly

    // dL/dT_row0 = 2 * (dL_da * (T0 * V) + dL_db * (T1 * V))
    //             = 2 * (dL_da * T0·V + dL_db * T1·V)  → where T·V is a row vec times matrix
    // Wait, T0 is a row vector [T00, T01, T02]. T0 * V (row × matrix) gives a row vector.
    // Actually for matrix notation: T is 2x3. (G * T * V)[i,j] = G[i,:] * T * V[:,j]

    // Let's compute T*V first (2x3 times 3x3 → 2x3)
    float3 TV0 = float3(dot(T0, V[0]), dot(T0, V[1]), dot(T0, V[2]));  // T0 * V (row 0)
    float3 TV1 = float3(dot(T1, V[0]), dot(T1, V[1]), dot(T1, V[2]));  // T1 * V (row 1)

    // G * (TV) → 2x3
    float3 dL_dT0 = 2.0f * (dL_da * TV0 + dL_db * TV1);
    float3 dL_dT1 = 2.0f * (dL_db * TV0 + dL_dc * TV1);

    // T_eff = J * W → dL/dJ = dL/dT_eff * W^T
    // J is 3x3 col-major with row 2 = 0
    // J rows: J_row0 = [fx/z, 0, -fx*tx/z^2], J_row1 = [0, fy/z, -fy*ty/z^2]
    //
    // dL/dJ_row0 = dL_dT0 * W^T (row vec times matrix)
    // dL/dJ_row1 = dL_dT1 * W^T

    float3x3 Wt = transpose(W);
    float3 dL_dJ0 = float3(dot(dL_dT0, Wt[0]), dot(dL_dT0, Wt[1]), dot(dL_dT0, Wt[2]));
    float3 dL_dJ1 = float3(dot(dL_dT1, Wt[0]), dot(dL_dT1, Wt[1]), dot(dL_dT1, Wt[2]));

    // J depends on (tx, ty, tz=depth):
    //   J_row0 = [fx/z, 0, -fx*tx/z^2]
    //   J_row1 = [0, fy/z, -fy*ty/z^2]
    //
    // Partial derivatives of J w.r.t. tx, ty, tz:
    //   dJ_row0/dtx = [0, 0, -fx/z^2]         → dL/dtx += dL_dJ0[2] * (-fx/z^2)
    //   dJ_row1/dty = [0, 0, -fy/z^2]         → dL/dty += dL_dJ1[2] * (-fy/z^2)
    //   dJ_row0/dtz = [-fx/z^2, 0, 2*fx*tx/z^3] → dL/dtz += dL_dJ0[0]*(-fx/z^2) + dL_dJ0[2]*(2*fx*tx/z^3)
    //   dJ_row1/dtz = [0, -fy/z^2, 2*fy*ty/z^3] → dL/dtz += dL_dJ1[1]*(-fy/z^2) + dL_dJ1[2]*(2*fy*ty/z^3)

    float rz3 = rz2 * rz;

    float dL_dtx = dL_dJ0.z * (-fx * rz2);
    float dL_dty = dL_dJ1.z * (-fy * rz2);
    float dL_dtz = dL_dJ0.x * (-fx * rz2) + dL_dJ0.z * (2.0f * fx * tx * rz3)
                 + dL_dJ1.y * (-fy * rz2) + dL_dJ1.z * (2.0f * fy * ty * rz3);

    // FOV clamping: if tx or ty was clamped, zero their gradients
    if (tx_clamped) dL_dtx = 0.0f;
    if (ty_clamped) dL_dty = 0.0f;

    // ===================================================================
    //  BACKWARD: dL/d_mean2d → dL/d_t (camera-space position)
    // ===================================================================

    // mean2d_x = fx * p_cam.x / depth + cx
    // mean2d_y = fy * p_cam.y / depth + cy
    // (using original p_cam, NOT clamped tx/ty)

    float dL_d_pcamx = dL_du * fx * rz;
    float dL_d_pcamy = dL_dv * fy * rz;
    float dL_d_pcamz = dL_du * (-fx * p_cam.x * rz2)
                     + dL_dv * (-fy * p_cam.y * rz2);

    // ===================================================================
    //  Combine: total dL/d_p_cam and transform to dL/d_means3d
    // ===================================================================

    // Note: dL_dtx/dty/dtz come from the Jacobian backward (through cov2d)
    // and dL_d_pcamx/y/z come from the projection backward (through mean2d)
    // The Jacobian uses (tx, ty, depth) which are clamped versions
    // but the gradient routes differently:
    //   - For the Jacobian path: tx/ty may be clamped, but they map back to p_cam components
    //     If NOT clamped: tx = p_cam.x, ty = p_cam.y, so dL/d_pcam += dL/dt
    //     If clamped: the clamped tx/ty are constants, gradient doesn't flow to p_cam x/y
    //     But dL/dtz always flows (tz = depth = p_cam.z always)

    float3 dL_d_pcam;
    dL_d_pcam.x = dL_d_pcamx + dL_dtx;
    dL_d_pcam.y = dL_d_pcamy + dL_dty;
    dL_d_pcam.z = dL_d_pcamz + dL_dtz;

    // p_cam = W_rot * pw + t_view → dL/d_pw = W_rot^T * dL/d_pcam
    // W is already the rotation part (col-major)
    float3 dL_dmean3d = float3(
        W[0][0] * dL_d_pcam.x + W[0][1] * dL_d_pcam.y + W[0][2] * dL_d_pcam.z,
        W[1][0] * dL_d_pcam.x + W[1][1] * dL_d_pcam.y + W[1][2] * dL_d_pcam.z,
        W[2][0] * dL_d_pcam.x + W[2][1] * dL_d_pcam.y + W[2][2] * dL_d_pcam.z
    );

    // ---- Write outputs ----
    dL_d_means3d[tid*3]   = dL_dmean3d.x;
    dL_d_means3d[tid*3+1] = dL_dmean3d.y;
    dL_d_means3d[tid*3+2] = dL_dmean3d.z;

    dL_d_scales[tid*3]    = dL_dscales.x;
    dL_d_scales[tid*3+1]  = dL_dscales.y;
    dL_d_scales[tid*3+2]  = dL_dscales.z;

    dL_d_quats[tid*4]     = dL_dqt.x;
    dL_d_quats[tid*4+1]   = dL_dqt.y;
    dL_d_quats[tid*4+2]   = dL_dqt.z;
    dL_d_quats[tid*4+3]   = dL_dqt.w;
}
