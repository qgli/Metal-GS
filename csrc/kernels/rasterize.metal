// ============================================================================
//  Metal-GS: Forward Rasterization Kernel
//
//  Per-tile alpha blending of 2D Gaussians.
//
//  Architecture:
//    ▸ 1 threadgroup (16×16 = 256 threads) per screen tile
//    ▸ Cooperative fetch: all 256 threads batch-load Gaussian params
//      into threadgroup shared memory via threadgroup_barrier
//    ▸ Front-to-back alpha blending with early termination (T < 1e-4)
//    ▸ Numerical stability: epsilon guard on cov2d determinant inversion
//
//  Buffer layout: raw float* / uint* (packed, no float3 stride issues)
//  Precision: FP32 for all geometry (ENABLE_BF16=0 on M1)
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ---- Constants ----
constant uint TILE_SIZE  = 16;
constant uint BLOCK_SIZE = 256;   // TILE_SIZE * TILE_SIZE

// ---- Params (must match C++ RasterizeParams byte-for-byte) ----
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

// ---- Shared memory struct for one cached Gaussian ----
struct SharedGaussian {
    float2 mean;          // 2D center (px, py)
    float3 cov_inv;       // inverse cov2d upper-tri [inv_a, inv_b, inv_c]
    float3 color;         // RGB
    float  opacity;       // sigmoid(raw_opacity)
};

// =========================================================================
//  Kernel: rasterize_forward
//
//  Grid:   (num_tiles_x, num_tiles_y, 1) threadgroups
//  TG:     (16, 16, 1) → 256 threads = 1 pixel per thread
//
//  Inputs:
//    means2d    [N*2]          — 2D screen positions
//    cov2d      [N*3]          — upper-tri [a, b, c] of Σ2D
//    colors     [N*3]          — RGB per Gaussian
//    opacities  [N]            — opacity per Gaussian (already sigmoid'd)
//    tile_bins  [num_tiles*2]  — (start, end) indices into point_list
//    point_list [num_isect]    — Gaussian indices in tile-then-depth order
//
//  Output:
//    out_img    [H*W*3]        — rendered image (row-major, RGB float)
// =========================================================================
kernel void rasterize_forward(
    device const float*  means2d     [[buffer(0)]],    // [N*2]
    device const float*  cov2d       [[buffer(1)]],    // [N*3] upper-tri
    device const float*  colors      [[buffer(2)]],    // [N*3]
    device const float*  opacities   [[buffer(3)]],    // [N]
    device const uint*   tile_bins   [[buffer(4)]],    // [num_tiles*2]
    device const uint*   point_list  [[buffer(5)]],    // [num_isect]
    constant RasterizeParams& params [[buffer(6)]],
    device float*        out_img     [[buffer(7)]],    // [H*W*3]
    device float*        T_final     [[buffer(8)]],    // [H*W] — final transmittance
    device uint*         n_contrib   [[buffer(9)]],    // [H*W] — last contributing idx

    uint2 tg_id     [[threadgroup_position_in_grid]],   // tile (tx, ty)
    uint2 tid_in_tg [[thread_position_in_threadgroup]], // local (lx, ly)
    uint  flat_tid   [[thread_index_in_threadgroup]])
{
    // ---- Compute pixel coordinates ----
    uint px = tg_id.x * TILE_SIZE + tid_in_tg.x;
    uint py = tg_id.y * TILE_SIZE + tid_in_tg.y;
    float2 pixel = float2(float(px) + 0.5f, float(py) + 0.5f);

    bool inside = (px < params.img_width && py < params.img_height);

    // ---- Get tile's Gaussian range ----
    uint tile_id = tg_id.y * params.num_tiles_x + tg_id.x;
    uint range_start = tile_bins[tile_id * 2];
    uint range_end   = tile_bins[tile_id * 2 + 1];
    uint num_gaussians = range_end - range_start;

    // ---- Per-pixel accumulation state ----
    float T = 1.0f;                    // transmittance
    float3 C = float3(0.0f);          // accumulated color
    uint last_contributor = 0;         // global index of last contributing Gaussian

    // ---- Threadgroup shared memory for cooperative fetch ----
    threadgroup SharedGaussian shared_gs[BLOCK_SIZE];

    // ---- Process Gaussians in batches of BLOCK_SIZE ----
    // Hard-cap: bound worst-case GPU time per tile to prevent Watchdog timeouts.
    // Since gaussians are depth-sorted, truncation only discards distant ones.
    uint cap = params.max_gaussians_per_tile;
    uint capped_gaussians = (cap > 0) ? min(num_gaussians, cap) : num_gaussians;
    uint num_batches = (capped_gaussians + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (uint batch = 0; batch < num_batches; batch++) {

        // ---- Cooperative fetch: each thread loads one Gaussian ----
        uint fetch_idx = batch * BLOCK_SIZE + flat_tid;
        if (fetch_idx < capped_gaussians) {
            uint gauss_id = point_list[range_start + fetch_idx];

            float mx = means2d[gauss_id * 2];
            float my = means2d[gauss_id * 2 + 1];

            // Cov2d upper-tri: a = Σ00, b = Σ01, c = Σ11
            float a = cov2d[gauss_id * 3];
            float b = cov2d[gauss_id * 3 + 1];
            float c_val = cov2d[gauss_id * 3 + 2];

            // ---- Invert 2×2 covariance with numerical stability ----
            // det(Σ) = a*c - b²
            float det = a * c_val - b * b;
            // Guard: if det too small, this Gaussian is degenerate → skip
            // Use reciprocal with epsilon clamping
            float det_inv = (det > 1e-6f) ? (1.0f / det) : 0.0f;

            shared_gs[flat_tid].mean    = float2(mx, my);
            shared_gs[flat_tid].cov_inv = float3(c_val * det_inv,     // inv_a =  c/det
                                                  -b    * det_inv,     // inv_b = -b/det
                                                  a     * det_inv);    // inv_c =  a/det
            shared_gs[flat_tid].color   = float3(colors[gauss_id * 3],
                                                  colors[gauss_id * 3 + 1],
                                                  colors[gauss_id * 3 + 2]);
            shared_gs[flat_tid].opacity = (det > 1e-6f) ? opacities[gauss_id] : 0.0f;
        } else {
            // Padding: mark as invisible
            shared_gs[flat_tid].opacity = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Each thread evaluates all Gaussians in this batch ----
        if (inside) {
            uint count = min(BLOCK_SIZE, capped_gaussians - batch * BLOCK_SIZE);
            for (uint s = 0; s < count; s++) {
                // Early termination: if transmittance is negligible, stop
                if (T < 1e-4f) break;

                float op = shared_gs[s].opacity;
                if (op < 1e-6f) continue;  // skip degenerate/invisible

                float2 d = pixel - shared_gs[s].mean;
                float3 cinv = shared_gs[s].cov_inv;

                // Mahalanobis distance: d^T · Σ^{-1} · d
                // Σ^{-1} = [[inv_a, inv_b], [inv_b, inv_c]]
                float maha = cinv.x * d.x * d.x
                           + 2.0f * cinv.y * d.x * d.y
                           + cinv.z * d.y * d.y;

                // Gaussian weight: exp(-0.5 * maha)
                // Skip if too far (maha > ~18 → weight < 1e-4)
                if (maha < 0.0f || maha > 18.0f) continue;
                float weight = exp(-0.5f * maha);

                // Alpha = opacity * gaussian_weight, clamped to [0, 0.999]
                float alpha = min(0.999f, op * weight);
                if (alpha < 1.0f / 255.0f) continue;  // negligible contribution

                // ---- Front-to-back alpha blending ----
                // C += T * alpha * color
                // T *= (1 - alpha)
                C += T * alpha * shared_gs[s].color;
                T *= (1.0f - alpha);

                // Track global index for backward pass
                last_contributor = range_start + batch * BLOCK_SIZE + s;
            }
        }

        // Barrier before next batch overwrites shared memory
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Write final pixel color (with background) ----
    if (inside) {
        // Background contribution: T * bg_color
        float3 bg = float3(params.bg_r, params.bg_g, params.bg_b);
        C += T * bg;

        uint pixel_idx = py * params.img_width + px;
        out_img[pixel_idx * 3]     = C.x;
        out_img[pixel_idx * 3 + 1] = C.y;
        out_img[pixel_idx * 3 + 2] = C.z;

        // Save for backward pass
        T_final[pixel_idx]   = T;
        n_contrib[pixel_idx] = last_contributor;
    }
}
