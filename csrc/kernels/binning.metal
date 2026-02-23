// ============================================================================
//  Metal-GS: Tile Binning — Assign depth-sorted Gaussians to screen tiles
//
//  Pipeline:
//    1. generate_intersections  — expand each Gaussian into per-tile entries
//    2. (external) stable radix sort by tile_id
//    3. identify_tile_ranges   — find [start, end) per tile in sorted list
//
//  Key invariant: Gaussians arrive in depth-sorted order. The stable radix
//  sort on tile_id preserves depth ordering within each tile. This gives us
//  the exact per-tile rendering order needed for front-to-back alpha blending.
//
//  Memory: MTLResourceStorageModeShared (zero CPU↔GPU copy on Apple Silicon).
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ---- Params (match C++ BinningParams byte-for-byte) ----
struct BinningParams {
    uint num_points;           // N (total Gaussians in sorted order)
    uint num_tiles_x;
    uint num_tiles_y;
    uint num_intersections;    // total (tile, Gaussian) pairs
};

// =========================================================================
//  Kernel 1: Generate intersection entries
//
//  For each visible Gaussian (in depth-sorted order), enumerate the tiles
//  it covers and write (tile_id, gaussian_original_index) pairs.
//
//  offsets[tid] = exclusive prefix sum of per-Gaussian tile counts,
//  computed on CPU (O(N), negligible). Each thread writes a contiguous
//  range ⟹ zero write conflicts, no atomics needed.
// =========================================================================
kernel void generate_intersections(
    device const uint* sorted_indices  [[buffer(0)]],   // [N]
    device const uint* radii           [[buffer(1)]],   // [N]
    device const uint* tile_min        [[buffer(2)]],   // [N * 2] (tx_min, ty_min)
    device const uint* tile_max        [[buffer(3)]],   // [N * 2] (tx_max, ty_max)
    device const uint* offsets         [[buffer(4)]],   // [N] exclusive prefix sum
    device uint*       isect_tile_ids  [[buffer(5)]],   // [num_isect] output: tile_id
    device uint*       isect_gauss_ids [[buffer(6)]],   // [num_isect] output: gaussian_id
    constant BinningParams& params     [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.num_points) return;

    uint idx = sorted_indices[tid];
    if (radii[idx] == 0) return;        // invisible → contributes 0 tiles

    uint base   = offsets[tid];
    uint tx_min = tile_min[idx * 2];
    uint ty_min = tile_min[idx * 2 + 1];
    uint tx_max = tile_max[idx * 2];
    uint ty_max = tile_max[idx * 2 + 1];

    uint pos = 0;
    for (uint ty = ty_min; ty < ty_max; ty++) {
        for (uint tx = tx_min; tx < tx_max; tx++) {
            uint tile_id = ty * params.num_tiles_x + tx;
            isect_tile_ids [base + pos] = tile_id;
            isect_gauss_ids[base + pos] = idx;     // original Gaussian index
            pos++;
        }
    }
}

// =========================================================================
//  Kernel 2: Identify tile ranges in the sorted intersection list
//
//  After stable sort by tile_id, intersections are grouped by tile.
//  This kernel finds the [start, end) boundary of each tile's block.
//
//  tile_bins must be zero-initialized before dispatch (CPU memset).
//  Empty tiles keep (0, 0) → start == end → no Gaussians to render.
// =========================================================================
kernel void identify_tile_ranges(
    device const uint* sorted_tile_ids [[buffer(0)]],   // [num_isect]
    device uint*       tile_bins       [[buffer(1)]],   // [num_tiles * 2]
    constant uint&     num_isect       [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_isect) return;

    uint cur = sorted_tile_ids[tid];

    // Start of a new tile group
    if (tid == 0 || sorted_tile_ids[tid - 1] != cur) {
        tile_bins[cur * 2] = tid;           // start (inclusive)
    }
    // End of a tile group
    if (tid == num_isect - 1 || sorted_tile_ids[tid + 1] != cur) {
        tile_bins[cur * 2 + 1] = tid + 1;  // end (exclusive)
    }
}
