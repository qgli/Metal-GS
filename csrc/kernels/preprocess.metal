// ============================================================================
//  Metal-GS: 3D Gaussian Splatting Preprocessing Kernel (v2 — all bugs fixed)
//
//  Transforms 3D Gaussians to 2D screen space:
//    ▸ 2D projected means
//    ▸ 2D covariance via EWA splatting (J · W · Σ3D · Wᵀ · Jᵀ + blur·I)
//    ▸ Eigenvalue-based 3σ radius & tile bounding boxes
//
//  Precision: FP32 geometry | BF16 gated by ENABLE_BF16 (M4+)
//
//  Buffer layout: ALL arrays are tightly-packed raw float / uint.
//    float3 types are NOT used in buffer pointers (16-byte stride ≠ 12-byte data).
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ---- Constants ----
constant uint  TILE_SIZE   = 16;
constant float CLIP_THRESH = 0.01f;
constant float AA_BLUR     = 0.3f;

// ---- BF16 precision gate ----
#if ENABLE_BF16
    typedef bfloat AccumType;
#else
    typedef float  AccumType;
#endif

// ---- Params (must match C++ PreprocessParams byte-for-byte) ----
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
//  Helper: quaternion (x,y,z,w) → 3×3 rotation matrix (column-major)
// =========================================================================
inline float3x3 quat_to_rotmat(float4 q) {
    float inv_n = rsqrt(max(dot(q, q), 1e-8f));
    q *= inv_n;
    float x = q.x, y = q.y, z = q.z, w = q.w;

    return float3x3(
        // column 0               column 1                column 2
        1.f-2.f*(y*y+z*z),       2.f*(x*y+w*z),          2.f*(x*z-w*y),
        2.f*(x*y-w*z),           1.f-2.f*(x*x+z*z),      2.f*(y*z+w*x),
        2.f*(x*z+w*y),           2.f*(y*z-w*x),          1.f-2.f*(x*x+y*y)
    );
}

// =========================================================================
//  Helper: Σ3D = R · diag(s) · diag(s)ᵀ · Rᵀ  =  M · Mᵀ
//  where M = R · diag(s)   (scale each COLUMN of R by s_j)
//  Output: upper-triangular [Σ00 Σ01 Σ02 Σ11 Σ12 Σ22]
// =========================================================================
inline void compute_cov3d(float3x3 R, float3 s, thread float* c) {
    float3x3 M = float3x3(
        R[0] * s.x,        // col 0 × sx
        R[1] * s.y,        // col 1 × sy
        R[2] * s.z          // col 2 × sz
    );

    // Σ_{ij} = Σ_k M_{ik} · M_{jk}    (col-major: M_{ik} = M[k][i])
    AccumType s00 = AccumType(M[0].x*M[0].x + M[1].x*M[1].x + M[2].x*M[2].x);
    AccumType s01 = AccumType(M[0].x*M[0].y + M[1].x*M[1].y + M[2].x*M[2].y);
    AccumType s02 = AccumType(M[0].x*M[0].z + M[1].x*M[1].z + M[2].x*M[2].z);
    AccumType s11 = AccumType(M[0].y*M[0].y + M[1].y*M[1].y + M[2].y*M[2].y);
    AccumType s12 = AccumType(M[0].y*M[0].z + M[1].y*M[1].z + M[2].y*M[2].z);
    AccumType s22 = AccumType(M[0].z*M[0].z + M[1].z*M[1].z + M[2].z*M[2].z);

    c[0]=float(s00); c[1]=float(s01); c[2]=float(s02);
    c[3]=float(s11); c[4]=float(s12); c[5]=float(s22);
}

// =========================================================================
//  Helper: affine transform  p_cam = W · [p, 1]  (row-major 4×4 view matrix)
// =========================================================================
inline float3 transform_4x3(constant float* m, float3 p) {
    return float3(
        m[0]*p.x + m[1]*p.y + m[ 2]*p.z + m[ 3],
        m[4]*p.x + m[5]*p.y + m[ 6]*p.z + m[ 7],
        m[8]*p.x + m[9]*p.y + m[10]*p.z + m[11]
    );
}

// =========================================================================
//  Kernel:  preprocess_forward   (1 thread per Gaussian)
//
//  All buffer pointers are raw float* / uint* (packed, 4-byte stride)
//  to avoid Metal float3 padding (16-byte vs 12-byte) alignment bugs.
// =========================================================================
kernel void preprocess_forward(
    device const float*   means3d      [[buffer(0)]],   // [N*3]
    device const float*   scales       [[buffer(1)]],   // [N*3]
    device const float*   quats        [[buffer(2)]],   // [N*4]
    constant float*       viewmat      [[buffer(3)]],   // [16]
    constant PreprocessParams& params  [[buffer(4)]],

    device float*         means2d_out  [[buffer(5)]],   // [N*2]
    device float*         cov2d_out    [[buffer(6)]],   // [N*3] upper-tri [a b c]
    device float*         depths_out   [[buffer(7)]],   // [N]
    device uint*          radii_out    [[buffer(8)]],   // [N]
    device uint*          tile_min_out [[buffer(9)]],   // [N*2]
    device uint*          tile_max_out [[buffer(10)]],  // [N*2]

    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.num_points) return;

    // ---- Read packed inputs ----
    float3 pw = float3(means3d[tid*3], means3d[tid*3+1], means3d[tid*3+2]);
    float3 sc = float3(scales[tid*3],  scales[tid*3+1],  scales[tid*3+2]);
    float4 qt = float4(quats[tid*4],   quats[tid*4+1],   quats[tid*4+2], quats[tid*4+3]);

    // ---- Step 1: world → camera ----
    float3 p_cam = transform_4x3(viewmat, pw);
    float  depth = p_cam.z;

    if (depth <= CLIP_THRESH) { radii_out[tid] = 0; return; }
    depths_out[tid] = depth;

    // ---- Step 2: 3D covariance ----
    float3x3 R = quat_to_rotmat(qt);
    float cov3d[6];
    compute_cov3d(R, sc, cov3d);

    // ---- Step 3: FOV guard-band clamping (for Jacobian only) ----
    float lx = 1.3f * params.tan_fovx;
    float ly = 1.3f * params.tan_fovy;
    float tx = clamp(p_cam.x / depth, -lx, lx) * depth;
    float ty = clamp(p_cam.y / depth, -ly, ly) * depth;

    // ---- Step 4: Jacobian of perspective projection ----
    float rz  = 1.f / depth;
    float rz2 = rz * rz;
    float fx  = params.focal_x;
    float fy  = params.focal_y;

    // J is 2×3 embedded in a 3×3 (third row = 0)
    float3x3 J = float3x3(
        fx*rz,          0.f,            0.f,          // col 0
        0.f,            fy*rz,          0.f,          // col 1
        -fx*tx*rz2,     -fy*ty*rz2,     0.f           // col 2
    );

    // ---- Step 5: extract W (rotation part of view matrix) ----
    // viewmat is row-major → transpose into Metal column-major float3x3
    float3x3 W = float3x3(
        float3(viewmat[0], viewmat[4], viewmat[8]),   // col 0
        float3(viewmat[1], viewmat[5], viewmat[9]),   // col 1
        float3(viewmat[2], viewmat[6], viewmat[10])   // col 2
    );

    // ---- Step 6: EWA splatting  Σ2D = T · Σ3D · Tᵀ + blur·I ----
    float3x3 T = J * W;                  // 3×3 (row 2 all zero)
    float3x3 V = float3x3(               // symmetric Σ3D
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5]
    );
    float3x3 cov = T * V * transpose(T);

    float a = cov[0][0] + AA_BLUR;       // Σ2D[0,0]
    float b = cov[0][1];                  // Σ2D[0,1] = Σ2D[1,0]
    float c = cov[1][1] + AA_BLUR;       // Σ2D[1,1]

    cov2d_out[tid*3]   = a;
    cov2d_out[tid*3+1] = b;
    cov2d_out[tid*3+2] = c;

    // ---- Step 7: eigenvalue → 3σ radius ----
    float det    = max(0.f, a * c - b * b);
    float b_half = 0.5f * (a + c);
    float disc   = max(0.1f, b_half * b_half - det);   // gsplat uses 0.1
    float lam    = b_half + sqrt(disc);
    float rad_f  = ceil(3.f * sqrt(lam));
    uint  rad    = uint(rad_f);

    if (rad == 0) { radii_out[tid] = 0; return; }
    radii_out[tid] = rad;

    // ---- Step 8: screen projection (original p_cam, NOT clamped) ----
    float u = fx * p_cam.x * rz + params.principal_x;
    float v = fy * p_cam.y * rz + params.principal_y;
    means2d_out[tid*2]   = u;
    means2d_out[tid*2+1] = v;

    // ---- Step 9: tile bounding box ----
    uint tgx = (params.img_width  + TILE_SIZE - 1) / TILE_SIZE;
    uint tgy = (params.img_height + TILE_SIZE - 1) / TILE_SIZE;

    float tf = float(TILE_SIZE);
    int2 tmin = int2(floor((float2(u, v) - rad_f) / tf));
    int2 tmax = int2( ceil((float2(u, v) + rad_f) / tf));

    tmin = clamp(tmin, int2(0), int2(tgx, tgy));
    tmax = clamp(tmax, int2(0), int2(tgx, tgy));

    if (tmin.x >= tmax.x || tmin.y >= tmax.y) { radii_out[tid] = 0; return; }

    tile_min_out[tid*2]   = uint(tmin.x);
    tile_min_out[tid*2+1] = uint(tmin.y);
    tile_max_out[tid*2]   = uint(tmax.x);
    tile_max_out[tid*2+1] = uint(tmax.y);
}
