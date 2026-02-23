// ============================================================================
//  Metal-GS: Morton-code based simple K-nearest-neighbor search
//
//  Architecture:
//    1. compute_morton_codes — AABB normalize → 10-bit grid → 30-bit Morton
//    2. (reuse existing radix sort to sort by Morton code)
//    3. knn_search — window-based brute-force in sorted Morton order
//
//  For 3DGS training initialization: computes average squared distance
//  to the 3 nearest neighbors for each point → used to initialize scales.
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
//  Morton code helpers: 10-bit → 30-bit interleaving via magic numbers
// ---------------------------------------------------------------------------

// Expand a 10-bit integer into 30 bits by inserting 2 zero bits between each bit.
// Uses the standard "magic number" bit-spreading technique.
inline uint expand_bits_10(uint v) {
    v = (v | (v << 16)) & 0x030000FFu;  // ---- ----  ---- ---- 0000 00xx  xxxx xxxx
    v = (v | (v <<  8)) & 0x0300F00Fu;  // ---- ----  xxxx ----  ---- xxxx  ---- xxxx
    v = (v | (v <<  4)) & 0x030C30C3u;  // ---- xx--  xx-- xx--  xx-- xx--  xx-- xx--
    v = (v | (v <<  2)) & 0x09249249u;  // x--x --x-  -x-- x--x  --x- -x--  x--x --x-
    return v;
}

// Compute 30-bit 3D Morton code from 10-bit x, y, z coordinates.
// Interleaves bits: z2 y2 x2 z1 y1 x1 z0 y0 x0 ...
inline uint morton_3d(uint x, uint y, uint z) {
    return (expand_bits_10(z) << 2) | (expand_bits_10(y) << 1) | expand_bits_10(x);
}

// ---------------------------------------------------------------------------
//  Params struct
// ---------------------------------------------------------------------------
struct KNNParams {
    uint num_points;
    uint search_window;   // half-window size (search ±window in sorted order)
    uint k_neighbors;     // number of nearest neighbors (3)
};

// ---------------------------------------------------------------------------
//  Kernel 1: compute_morton_codes
//
//  Pass 1 — find AABB (done on CPU for simplicity)
//  This kernel: normalize to [0, 1023] grid → compute 30-bit Morton code
//
//  Inputs:
//    points   — float[N*3]
//    aabb_min — float[3]  (bounding box minimum)
//    aabb_inv — float[3]  (1.0 / (aabb_max - aabb_min))
//  Outputs:
//    morton_codes — uint[N]
//    indices      — uint[N]  (identity permutation for sort)
// ---------------------------------------------------------------------------
kernel void compute_morton_codes(
    device const float*  points       [[buffer(0)]],   // [N*3]
    constant float*      aabb_min     [[buffer(1)]],   // [3]
    constant float*      aabb_inv     [[buffer(2)]],   // [3]
    device uint*         morton_codes [[buffer(3)]],   // [N]
    device uint*         indices      [[buffer(4)]],   // [N]
    constant KNNParams&  params       [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.num_points) return;

    float px = points[tid * 3 + 0];
    float py = points[tid * 3 + 1];
    float pz = points[tid * 3 + 2];

    // Normalize to [0, 1]
    float nx = (px - aabb_min[0]) * aabb_inv[0];
    float ny = (py - aabb_min[1]) * aabb_inv[1];
    float nz = (pz - aabb_min[2]) * aabb_inv[2];

    // Clamp to [0, 1] for safety
    nx = clamp(nx, 0.0f, 1.0f);
    ny = clamp(ny, 0.0f, 1.0f);
    nz = clamp(nz, 0.0f, 1.0f);

    // Quantize to 10-bit integers [0, 1023]
    uint ix = min((uint)(nx * 1023.0f), 1023u);
    uint iy = min((uint)(ny * 1023.0f), 1023u);
    uint iz = min((uint)(nz * 1023.0f), 1023u);

    morton_codes[tid] = morton_3d(ix, iy, iz);
    indices[tid] = tid;
}

// ---------------------------------------------------------------------------
//  Kernel 2: knn_search
//
//  After sorting by Morton code, spatially nearby points are adjacent.
//  For each point, search ±window in sorted order, compute L2 distances,
//  find the K (=3) nearest neighbors, output average squared distance.
//
//  Inputs:
//    sorted_indices — uint[N]   (original point indices in Morton order)
//    points         — float[N*3] (original point positions)
//  Output:
//    avg_sq_dist    — float[N]  (average squared distance to K nearest)
// ---------------------------------------------------------------------------
kernel void knn_search(
    device const uint*   sorted_indices [[buffer(0)]],  // [N] Morton-sorted
    device const float*  points         [[buffer(1)]],  // [N*3]
    device float*        avg_sq_dist    [[buffer(2)]],  // [N]
    constant KNNParams&  params         [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.num_points) return;

    uint N = params.num_points;
    uint W = params.search_window;
    uint K = params.k_neighbors;

    // This point's original index and position
    uint my_idx = sorted_indices[tid];
    float3 my_pos = float3(
        points[my_idx * 3 + 0],
        points[my_idx * 3 + 1],
        points[my_idx * 3 + 2]
    );

    // Track K nearest squared distances (simple insertion sort)
    // Initialize with large values
    // NOTE: K is always 3 for 3DGS scale initialization.
    // Metal does not support VLAs, so we fix the array to 3 and clamp K.
    K = min(K, 3u);
    float best[3] = { 1e30f, 1e30f, 1e30f };

    // Search window: [tid - W, tid + W] in sorted order
    uint lo = (tid >= W) ? (tid - W) : 0u;
    uint hi = min(tid + W, N - 1);

    for (uint j = lo; j <= hi; j++) {
        if (j == tid) continue;  // skip self

        uint other_idx = sorted_indices[j];
        float3 other_pos = float3(
            points[other_idx * 3 + 0],
            points[other_idx * 3 + 1],
            points[other_idx * 3 + 2]
        );

        float3 diff = my_pos - other_pos;
        float sq_dist = dot(diff, diff);

        // Insert into sorted best[0..K-1] (ascending)
        if (sq_dist < best[K - 1]) {
            best[K - 1] = sq_dist;
            // Bubble down
            for (int m = (int)K - 2; m >= 0; m--) {
                if (best[m + 1] < best[m]) {
                    float tmp = best[m];
                    best[m] = best[m + 1];
                    best[m + 1] = tmp;
                } else {
                    break;
                }
            }
        }
    }

    // Average of K nearest squared distances
    float sum = 0.0f;
    uint count = 0;
    for (uint m = 0; m < K; m++) {
        if (best[m] < 1e29f) {
            sum += best[m];
            count++;
        }
    }

    avg_sq_dist[my_idx] = (count > 0) ? (sum / float(count)) : 0.0f;
}
