// ============================================================================
//  Metal-GS: Backward Rasterization Kernel
//
//  Computes gradients through the alpha-blending compositing equation.
//  Strategy A: Naive global atomic add (correctness-first, no SIMD reduction).
//
//  Architecture:
//    ▸ 1 threadgroup (16×16 = 256 threads) per screen tile
//    ▸ Reverse-order traversal: from n_contrib back to range_start
//    ▸ Recovers T_i from T_{i+1} / (1 - α_i) with div-zero protection
//    ▸ Global atomic_fetch_add_explicit for gradient accumulation
//
//  Gradient outputs (accumulated via atomics):
//    dL/d_rgb      [N, 3]  — color gradient per Gaussian
//    dL/d_opacity  [N]     — opacity gradient per Gaussian
//    dL/d_cov2d    [N, 3]  — conic (inverse cov2d) gradient per Gaussian
//    dL/d_mean2d   [N, 2]  — 2D mean gradient per Gaussian
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ---- Constants ----
constant uint TILE_SIZE  = 16;
constant uint BLOCK_SIZE = 256;

// ---- Params struct (same as forward, must match C++ byte-for-byte) ----
struct RasterizeParams {
    uint   img_width;
    uint   img_height;
    uint   num_tiles_x;
    uint   num_tiles_y;
    float  bg_r;
    float  bg_g;
    float  bg_b;
    uint   max_gaussians_per_tile;  // hard cap per tile (0 = unlimited)
};

// ---- Shared memory for cooperative fetch (backward) ----
struct SharedGaussianBW {
    float2 mean;
    float3 cov_upper;      // raw cov2d upper-tri [a, b, c] (NOT inverted)
    float3 color;
    float  opacity;
    uint   gauss_id;       // original Gaussian index for atomic writes
};

// =========================================================================
//  Kernel: rasterize_backward
//
//  Grid:   (num_tiles_x, num_tiles_y, 1) threadgroups
//  TG:     (16, 16, 1) → 256 threads = 1 pixel per thread
//
//  Traverses Gaussians in REVERSE order (back-to-front) to recover
//  per-Gaussian gradients from the chain rule of alpha compositing.
// =========================================================================
kernel void rasterize_backward(
    // ---- Forward data ----
    device const float*  means2d     [[buffer(0)]],    // [N*2]
    device const float*  cov2d       [[buffer(1)]],    // [N*3] upper-tri (raw, not inverted)
    device const float*  colors      [[buffer(2)]],    // [N*3]
    device const float*  opacities   [[buffer(3)]],    // [N]
    device const uint*   tile_bins   [[buffer(4)]],    // [num_tiles*2]
    device const uint*   point_list  [[buffer(5)]],    // [num_isect]
    // ---- Forward-saved state ----
    device const float*  T_final     [[buffer(6)]],    // [H*W]
    device const uint*   n_contrib   [[buffer(7)]],    // [H*W]
    // ---- Upstream gradient ----
    device const float*  dL_dC_pixel [[buffer(8)]],    // [H*W*3]
    // ---- Params ----
    constant RasterizeParams& params [[buffer(9)]],
    // ---- Output gradients (atomic accumulation) ----
    device atomic_float* dL_d_rgb    [[buffer(10)]],   // [N*3]
    device atomic_float* dL_d_opacity[[buffer(11)]],   // [N]
    device atomic_float* dL_d_cov2d  [[buffer(12)]],   // [N*3]
    device atomic_float* dL_d_mean2d [[buffer(13)]],   // [N*2]

    uint2 tg_id     [[threadgroup_position_in_grid]],
    uint2 tid_in_tg [[thread_position_in_threadgroup]],
    uint  flat_tid   [[thread_index_in_threadgroup]])
{
    // ---- Compute pixel coordinates ----
    uint px = tg_id.x * TILE_SIZE + tid_in_tg.x;
    uint py = tg_id.y * TILE_SIZE + tid_in_tg.y;
    bool inside = (px < params.img_width && py < params.img_height);

    float2 pixel = float2(float(px) + 0.5f, float(py) + 0.5f);

    // ---- Get tile's Gaussian range ----
    uint tile_id = tg_id.y * params.num_tiles_x + tg_id.x;
    uint range_start = tile_bins[tile_id * 2];
    uint range_end   = tile_bins[tile_id * 2 + 1];
    uint num_gaussians = range_end - range_start;

    // ---- Per-pixel state ----
    uint pixel_idx = py * params.img_width + px;

    // Load upstream gradient and forward-saved state
    float3 dL_dC = float3(0.0f);
    float T = 0.0f;
    uint last_contrib = 0;

    if (inside) {
        dL_dC = float3(dL_dC_pixel[pixel_idx * 3],
                        dL_dC_pixel[pixel_idx * 3 + 1],
                        dL_dC_pixel[pixel_idx * 3 + 2]);
        T = T_final[pixel_idx];
        last_contrib = n_contrib[pixel_idx];
    }

    // Running sum: accumulates c_j * α_j * T_j for j > i
    // Initialized with the background contribution tracked via T_final
    float3 accum = float3(params.bg_r, params.bg_g, params.bg_b) * T;

    // ---- Shared memory for cooperative fetch ----
    threadgroup SharedGaussianBW shared_gs[BLOCK_SIZE];

    // ---- Process Gaussians in REVERSE order, in batches ----
    // Hard-cap matching forward: only iterate over gaussians the forward saw.
    uint cap = params.max_gaussians_per_tile;
    uint capped_gaussians = (cap > 0) ? min(num_gaussians, cap) : num_gaussians;
    uint num_batches = (capped_gaussians + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int batch = (int)num_batches - 1; batch >= 0; batch--) {

        // ---- Cooperative fetch: load Gaussians for this batch ----
        // In reverse traversal, batch 0 has the front-most Gaussians.
        // We load in forward order within the batch, but process in reverse.
        uint fetch_idx = (uint)batch * BLOCK_SIZE + flat_tid;
        if (fetch_idx < capped_gaussians) {
            uint gauss_id = point_list[range_start + fetch_idx];

            shared_gs[flat_tid].mean    = float2(means2d[gauss_id * 2],
                                                  means2d[gauss_id * 2 + 1]);
            shared_gs[flat_tid].cov_upper = float3(cov2d[gauss_id * 3],
                                                    cov2d[gauss_id * 3 + 1],
                                                    cov2d[gauss_id * 3 + 2]);
            shared_gs[flat_tid].color   = float3(colors[gauss_id * 3],
                                                  colors[gauss_id * 3 + 1],
                                                  colors[gauss_id * 3 + 2]);
            shared_gs[flat_tid].opacity = opacities[gauss_id];
            shared_gs[flat_tid].gauss_id = gauss_id;
        } else {
            shared_gs[flat_tid].opacity = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Process Gaussians in this batch in REVERSE order ----
        if (inside) {
            uint count = min(BLOCK_SIZE, capped_gaussians - (uint)batch * BLOCK_SIZE);

            for (int s = (int)count - 1; s >= 0; s--) {
                // Global index in the sorted list
                uint global_idx = range_start + (uint)batch * BLOCK_SIZE + (uint)s;

                // Skip Gaussians that were not reached in forward pass
                if (global_idx > last_contrib) continue;

                float op = shared_gs[s].opacity;
                if (op < 1e-6f) continue;

                float2 d = pixel - shared_gs[s].mean;
                float3 cov_raw = shared_gs[s].cov_upper;
                float a_cov = cov_raw.x;  // Σ00
                float b_cov = cov_raw.y;  // Σ01
                float c_cov = cov_raw.z;  // Σ11

                // Invert 2×2 covariance
                float det = a_cov * c_cov - b_cov * b_cov;
                if (det < 1e-6f) continue;
                float det_inv = 1.0f / det;

                float inv_a =  c_cov * det_inv;
                float inv_b = -b_cov * det_inv;
                float inv_c =  a_cov * det_inv;

                // Mahalanobis distance
                float maha = inv_a * d.x * d.x
                           + 2.0f * inv_b * d.x * d.y
                           + inv_c * d.y * d.y;

                if (maha < 0.0f || maha > 18.0f) continue;
                float weight = exp(-0.5f * maha);

                float alpha = min(0.999f, op * weight);
                if (alpha < 1.0f / 255.0f) continue;

                // ---- Recover T_i ----
                // T after this Gaussian (T_{i+1}) is the current T
                // T_i = T_{i+1} / (1 - α_i)
                float one_minus_alpha = 1.0f - alpha;
                float T_i = T / max(one_minus_alpha, 1e-5f);

                // ---- Gradient of loss w.r.t. alpha_i ----
                // dL/dα = T_i * (c_i · dL/dC) - 1/(1-α) * (accum · dL/dC)
                float3 c_i = shared_gs[s].color;
                float dL_dalpha = dot(T_i * c_i - accum / max(one_minus_alpha, 1e-5f), dL_dC);

                // ---- Update running accum BEFORE moving to next ----
                // accum tracks Σ_{j>i} c_j * α_j * T_j
                // When going from i to i-1, we add the contribution of i:
                accum += c_i * alpha * T_i;

                // ---- Update T for the next iteration (i-1) ----
                // Move T from T_{i+1} to T_i (going backward means T increases)
                T = T_i;

                // ---- Gradient of α w.r.t. opacity ----
                // α = o * exp(-σ/2), so dα/do = exp(-σ/2) = α/o
                float dL_d_opac = dL_dalpha * (alpha / max(op, 1e-8f));

                // ---- Gradient of α w.r.t. σ (Mahalanobis distance) ----
                // dα/dσ = -α/2  (when α = o * exp(-σ/2))
                float dL_dsigma = dL_dalpha * (-0.5f * alpha);

                // ---- Gradient of σ w.r.t. conic (inverse cov2d) ----
                // σ = d^T · Σ^{-1} · d = inv_a*dx² + 2*inv_b*dx*dy + inv_c*dy²
                // dσ/d(inv_a) = dx²,  dσ/d(inv_b) = 2*dx*dy,  dσ/d(inv_c) = dy²
                float dL_d_inv_a = dL_dsigma * d.x * d.x;
                float dL_d_inv_b = dL_dsigma * 2.0f * d.x * d.y;
                float dL_d_inv_c = dL_dsigma * d.y * d.y;

                // ---- Convert dL/d(conic) to dL/d(cov2d) via chain rule ----
                // conic = inv(cov2d), so dL/d(cov2d) = -conic^T * dL/d(conic) * conic^T
                // For symmetric 2x2:
                //   Σ^{-1} = [[inv_a, inv_b], [inv_b, inv_c]]
                //   dL/dΣ = -Σ^{-1} · dL/dΣ^{-1} · Σ^{-1}
                //
                // Expand:  G = [[dL_d_inv_a, dL_d_inv_b], [dL_d_inv_b, dL_d_inv_c]]
                //   -(Σ^{-1} · G · Σ^{-1})_00 = -(inv_a² g_a + 2 inv_a inv_b g_b + inv_b² g_c)
                //   -(Σ^{-1} · G · Σ^{-1})_01 = -(inv_a inv_b g_a + (inv_a inv_c + inv_b²) g_b + inv_b inv_c g_c)
                //   -(Σ^{-1} · G · Σ^{-1})_11 = -(inv_b² g_a + 2 inv_b inv_c g_b + inv_c² g_c)
                float g_a = dL_d_inv_a;
                float g_b = dL_d_inv_b;
                float g_c = dL_d_inv_c;

                float dL_da = -(inv_a * inv_a * g_a + 2.0f * inv_a * inv_b * g_b + inv_b * inv_b * g_c);
                float dL_db = -(inv_a * inv_b * g_a + (inv_a * inv_c + inv_b * inv_b) * g_b + inv_b * inv_c * g_c);
                float dL_dc = -(inv_b * inv_b * g_a + 2.0f * inv_b * inv_c * g_b + inv_c * inv_c * g_c);

                // ---- Gradient w.r.t. 2D mean ----
                // σ = d^T · Σ^{-1} · d, where d = pixel - mean
                // dσ/d(mean) = -2 * Σ^{-1} · d
                float dL_dmx = dL_dsigma * (-2.0f) * (inv_a * d.x + inv_b * d.y);
                float dL_dmy = dL_dsigma * (-2.0f) * (inv_b * d.x + inv_c * d.y);

                // ---- Gradient w.r.t. color ----
                // C_pixel = Σ α_i T_i c_i + T_N bg
                // dL/dc_i = α_i * T_i * dL/dC_pixel
                float3 dL_dc_i = alpha * T_i * dL_dC;

                // ---- Atomic accumulation (Strategy A: naive) ----
                uint gid = shared_gs[s].gauss_id;

                atomic_fetch_add_explicit(&dL_d_rgb[gid * 3],     dL_dc_i.x, memory_order_relaxed);
                atomic_fetch_add_explicit(&dL_d_rgb[gid * 3 + 1], dL_dc_i.y, memory_order_relaxed);
                atomic_fetch_add_explicit(&dL_d_rgb[gid * 3 + 2], dL_dc_i.z, memory_order_relaxed);

                atomic_fetch_add_explicit(&dL_d_opacity[gid], dL_d_opac, memory_order_relaxed);

                atomic_fetch_add_explicit(&dL_d_cov2d[gid * 3],     dL_da, memory_order_relaxed);
                atomic_fetch_add_explicit(&dL_d_cov2d[gid * 3 + 1], dL_db, memory_order_relaxed);
                atomic_fetch_add_explicit(&dL_d_cov2d[gid * 3 + 2], dL_dc, memory_order_relaxed);

                atomic_fetch_add_explicit(&dL_d_mean2d[gid * 2],     dL_dmx, memory_order_relaxed);
                atomic_fetch_add_explicit(&dL_d_mean2d[gid * 2 + 1], dL_dmy, memory_order_relaxed);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
