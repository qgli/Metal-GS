// Metal-GS: C++ header for the Metal wrapper functions
#pragma once

#include <cstdint>

// Compute SH forward pass on Metal GPU.
// Returns elapsed time in ms, or -1.0 on error.
double metal_compute_sh_forward(
    const float*    directions,     // [N, 3]  FP32
    const uint16_t* sh_coeffs,      // [N, K, 3]  FP16 (as raw uint16 bits)
    uint16_t*       colors_out,     // [N, 3]  FP16 (as raw uint16 bits)
    uint32_t        N,
    uint32_t        K,
    uint32_t        sh_degree
);

// Compute SH forward pass using simdgroup_matrix MMA (V1.0 exploration).
// Identical interface to metal_compute_sh_forward, different dispatch grid.
// Returns elapsed time in ms, or -1.0 on error.
double metal_compute_sh_forward_mma(
    const float*    directions,     // [N, 3]  FP32
    const uint16_t* sh_coeffs,      // [N, K, 3]  FP16 (as raw uint16 bits)
    uint16_t*       colors_out,     // [N, 3]  FP16 (as raw uint16 bits)
    uint32_t        N,
    uint32_t        K,
    uint32_t        sh_degree
);

// Preprocess 3D Gaussians to 2D screen space.
// Returns elapsed time in ms, or -1.0 on error.
double metal_preprocess_forward(
    const float*    means3d,        // [N, 3]  FP32
    const float*    scales,         // [N, 3]  FP32
    const float*    quats,          // [N, 4]  FP32 (xyzw order)
    const float*    viewmat,        // [4, 4]  FP32 row-major
    float           tan_fovx,
    float           tan_fovy,
    float           focal_x,
    float           focal_y,
    float           principal_x,
    float           principal_y,
    uint32_t        img_width,
    uint32_t        img_height,
    uint32_t        N,
    float*          means2d_out,    // [N, 2]  FP32
    float*          cov2d_out,      // [N, 3]  FP32 (upper tri: [a, b, c])
    float*          depths_out,     // [N]     FP32
    uint32_t*       radii_out,      // [N]     uint32
    uint32_t*       tile_min_out,   // [N, 2]  uint32
    uint32_t*       tile_max_out    // [N, 2]  uint32
);

// Radix sort by depth: returns sorted indices.
// Returns elapsed time in ms, or -1.0 on error.
double metal_radix_sort_by_depth(
    const float*    depths,              // [N] FP32
    uint32_t*       sorted_indices_out,  // [N] uint32
    uint32_t        N
);

// Tile binning: assign depth-sorted Gaussians to screen tiles.
// offsets[i] = exclusive prefix sum of per-Gaussian tile counts (CPU-computed).
// Returns elapsed time in ms, or -1.0 on error.
double metal_tile_binning(
    const uint32_t* sorted_indices,   // [N] depth-sorted order
    const uint32_t* radii,            // [N] visibility (0 = invisible)
    const uint32_t* tile_min,         // [N*2] (tx_min, ty_min)
    const uint32_t* tile_max,         // [N*2] (tx_max, ty_max)
    const uint32_t* offsets,          // [N] exclusive prefix sum
    uint32_t        N,
    uint32_t        num_tiles_x,
    uint32_t        num_tiles_y,
    uint32_t        num_intersections,
    uint32_t*       point_list_out,   // [num_intersections]
    uint32_t*       tile_bins_out     // [num_tiles * 2] (start, end)
);

// Forward rasterization: alpha-blend 2D Gaussians per tile.
// Returns elapsed time in ms, or -1.0 on error.
double metal_rasterize_forward(
    const float*    means2d,          // [N*2]
    const float*    cov2d,            // [N*3] upper-tri
    const float*    colors,           // [N*3] RGB
    const float*    opacities,        // [N]
    const uint32_t* tile_bins,        // [num_tiles*2] (start, end)
    const uint32_t* point_list,       // [num_isect]
    uint32_t        num_points,
    uint32_t        num_intersections,
    uint32_t        img_width,
    uint32_t        img_height,
    uint32_t        num_tiles_x,
    uint32_t        num_tiles_y,
    float           bg_r,
    float           bg_g,
    float           bg_b,
    uint32_t        max_gaussians_per_tile,  // 0 = unlimited
    float*          out_img,          // [H*W*3]
    float*          T_final_out,      // [H*W] final transmittance
    uint32_t*       n_contrib_out     // [H*W] last contributing index
);

// Backward rasterization: compute gradients for 2D Gaussian params.
// Uses Strategy A (naive global atomic add) for correctness verification.
// Returns elapsed time in ms, or -1.0 on error.
double metal_rasterize_backward(
    const float*    means2d,          // [N*2]
    const float*    cov2d,            // [N*3] upper-tri
    const float*    colors,           // [N*3] RGB
    const float*    opacities,        // [N]
    const uint32_t* tile_bins,        // [num_tiles*2] (start, end)
    const uint32_t* point_list,       // [num_isect]
    const float*    T_final,          // [H*W] from forward
    const uint32_t* n_contrib,        // [H*W] from forward
    const float*    dL_dC_pixel,      // [H*W*3] upstream gradient
    uint32_t        num_points,
    uint32_t        num_intersections,
    uint32_t        img_width,
    uint32_t        img_height,
    uint32_t        num_tiles_x,
    uint32_t        num_tiles_y,
    float           bg_r,
    float           bg_g,
    float           bg_b,
    uint32_t        max_gaussians_per_tile,  // 0 = unlimited
    float*          dL_d_rgb_out,     // [N*3]
    float*          dL_d_opacity_out, // [N]
    float*          dL_d_cov2d_out,   // [N*3]
    float*          dL_d_mean2d_out   // [N*2]
);

// Preprocess backward: compute gradients from 2D params to 3D Gaussian params.
// Fused preprocess_backward + cov3d_backward (M3). 1:1 mapping, no atomics.
// Returns elapsed time in ms, or -1.0 on error.
double metal_preprocess_backward(
    const float*    means3d,          // [N*3]
    const float*    scales,           // [N*3]
    const float*    quats,            // [N*4]
    const float*    viewmat,          // [16] row-major
    const uint32_t* radii,            // [N]
    float           tan_fovx,
    float           tan_fovy,
    float           focal_x,
    float           focal_y,
    float           principal_x,
    float           principal_y,
    uint32_t        img_width,
    uint32_t        img_height,
    uint32_t        N,
    const float*    dL_d_cov2d,       // [N*3]
    const float*    dL_d_mean2d,      // [N*2]
    float*          dL_d_means3d_out, // [N*3]
    float*          dL_d_scales_out,  // [N*3]
    float*          dL_d_quats_out    // [N*4]
);

// SH backward: compute gradients from colors to SH coefficients and means3d.
// Returns elapsed time in ms, or -1.0 on error.
double metal_sh_backward(
    const float*    means3d,          // [N*3]
    const float*    campos,           // [3]
    const uint16_t* sh_coeffs,        // [N*K*3] FP16
    const uint16_t* colors_fwd,       // [N*3] FP16 (forward output)
    const float*    dL_d_colors,      // [N*3]
    uint32_t        N,
    uint32_t        K,
    uint32_t        sh_degree,
    float*          dL_d_sh_out,      // [N*K*3]
    float*          dL_d_means3d_out  // [N*3]
);

// Simple KNN using Morton codes + radix sort.
// Returns average squared distance to K (=3) nearest neighbors for each point.
// Returns elapsed time in ms, or -1.0 on error.
double metal_simple_knn(
    const float*    points,           // [N*3] FP32 (xyz)
    float*          avg_sq_dist_out,  // [N]   FP32
    uint32_t        N,
    uint32_t        k_neighbors,      // typically 3
    uint32_t        search_window     // half-window, typically 32
);

