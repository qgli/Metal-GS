// ============================================================================
//  tile_count.metal — Count per-Gaussian tile intersections on GPU
//
//  Replaces the CPU-side prefix sum loop that required a GPU→CPU sync.
//  Two kernels:
//    1. count_tile_intersections: computes tile_count[i] for each sorted Gaussian
//    2. (prefix sum is handled by the existing scan_exclusive/scan_add_block_sums)
// ============================================================================

#include <metal_stdlib>
using namespace metal;

struct TileCountParams {
    uint num_points;
    uint num_tiles_x;
    uint num_tiles_y;
};

// ---------------------------------------------------------------------------
//  count_tile_intersections
//
//  For each depth-sorted Gaussian i:
//    idx = sorted_indices[i]
//    if radii[idx] > 0:
//      tile_count[i] = (tile_max[idx*2] - tile_min[idx*2]) *
//                       (tile_max[idx*2+1] - tile_min[idx*2+1])
//    else:
//      tile_count[i] = 0
//
//  This replaces the CPU loop in bindings.cpp that computed offsets.
// ---------------------------------------------------------------------------
kernel void count_tile_intersections(
    device const uint* sorted_indices [[buffer(0)]],
    device const uint* radii          [[buffer(1)]],
    device const uint* tile_min       [[buffer(2)]],  // [N*2]
    device const uint* tile_max       [[buffer(3)]],  // [N*2]
    device uint*       tile_counts    [[buffer(4)]],  // [N] output
    constant TileCountParams& params  [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
)
{
    if (tid >= params.num_points) return;

    uint idx = sorted_indices[tid];
    if (radii[idx] == 0) {
        tile_counts[tid] = 0;
        return;
    }

    uint w = tile_max[idx * 2 + 0] - tile_min[idx * 2 + 0];
    uint h = tile_max[idx * 2 + 1] - tile_min[idx * 2 + 1];
    tile_counts[tid] = w * h;
}

// ---------------------------------------------------------------------------
//  reduce_total — sum all tile_counts to get total num_intersections.
//  Uses single-threadgroup reduction.  Call with 1 threadgroup of 256 threads.
//  Input: tile_counts[N], Output: total_out[0] = sum(tile_counts).
//
//  For large N, we do a two-pass approach:
//    Pass 1: each thread sums stride elements → threadgroup reduce → block_sums
//    Pass 2: sum block_sums (CPU or tiny GPU kernel)
//
//  This kernel handles up to N=256*65536 = 16M (far more than needed).
// ---------------------------------------------------------------------------
struct ReduceParams {
    uint num_elements;
};

kernel void reduce_sum_uint(
    device const uint*   input   [[buffer(0)]],
    device uint*         output  [[buffer(1)]],  // [num_threadgroups]
    constant ReduceParams& params [[buffer(2)]],
    uint tid  [[thread_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint gid  [[threadgroup_position_in_grid]],
    uint tgs  [[threads_per_threadgroup]]
)
{
    threadgroup uint shared_data[256];
    
    // Each thread accumulates a strided range
    uint block_start = gid * tgs;
    uint block_end = min(block_start + tgs, params.num_elements);
    
    uint sum = 0;
    for (uint i = block_start + lid; i < block_end; i += tgs) {
        if (i < params.num_elements) {
            sum += input[i];
        }
    }
    shared_data[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce within threadgroup
    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared_data[lid] += shared_data[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (lid == 0) {
        output[gid] = shared_data[0];
    }
}
