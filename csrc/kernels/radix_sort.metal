// ============================================================================
//  Metal-GS: GPU Radix Sort for 3DGS depth ordering
//
//  4-bit radix (16 buckets) × 8 passes = 32-bit sort on sortable depth keys.
//  Stable scatter via simd_prefix_exclusive_sum (no ballot hacks).
//
//  Kernels:
//    1. radix_generate_keys   — float→sortable-uint conversion
//    2. radix_histogram       — per-block digit frequency counting
//    3. scan_exclusive        — Blelloch exclusive prefix sum
//    4. scan_add_block_sums   — propagate block totals
//    5. radix_scatter_stable  — SIMD prefix-sum-based stable scatter
//
//  Memory: MTLResourceStorageModeShared (zero CPU↔GPU copy on Apple Silicon).
//
// ── Architecture Decision: Why Not One-Sweep / Decoupled Look-back? ──
//
//  The SOTA GPU radix sort (Onesweep, Merrill-Grimshaw 2022) fuses
//  histogram + prefix-scan + scatter into ONE kernel per pass via
//  "decoupled look-back": each threadgroup atomically publishes partial
//  aggregates and scans backwards through predecessor blocks.
//  This reduces dispatches from ~5→1 per pass.
//
//  Why we DON'T use it on Metal / Apple Silicon:
//
//    ❶ No forward-progress guarantee between threadgroups.
//       M1 (7 cores, ~28 concurrent TGs) cannot guarantee all TGs
//       run concurrently. A look-back chain longer than ~28 blocks
//       DEADLOCKS: a later TG spins waiting for an earlier one that
//       hasn't been scheduled yet. For 1M keys → 4096 blocks → unsafe.
//
//    ❷ Metal atomic ordering limitations.
//       The packed status+value atomic pattern (INVALID/AGGREGATE/PREFIX)
//       relies on total-store-order semantics. Metal lacks
//       memory_order_seq_cst on device atomics, complicating correctness.
//
//  Instead we optimise the dispatch overhead via:
//    ✓ Single MTLComputeCommandEncoder + memoryBarrierWithScope:
//      (eliminates ~40 encoder creation/teardown cycles)
//    ✓ setBytes for small constant params (no buffer allocation per pass)
//    ✓ CPU fallback for N ≤ 16K (avoids GPU launch overhead entirely)
//    ✓ dispatch_radix_sort_kv helper with configurable pass count
//      (tile binning uses 4 passes for 16-bit tile IDs → 2× faster)
// ============================================================================

#include <metal_stdlib>
using namespace metal;

// ---- Sort tuning constants ----
constant uint RADIX_BITS = 4u;
constant uint RADIX_SIZE = 16u;
constant uint RADIX_MASK = 0xFu;
constant uint SORT_BLOCK = 256u;
constant uint SCAN_BLOCK = 256u;

// ---- Parameter structs (match C++ byte-for-byte) ----
struct SortParams {
    uint num_keys;
    uint num_blocks;
    uint bit_offset;
};

struct ScanParams {
    uint num_elements;
};

// =========================================================================
//  IEEE-754 float → monotonically-increasing uint
//  Positive: flip sign bit.  Negative: flip ALL bits.
//  Preserves total order: a < b  ⟹  key(a) < key(b)
// =========================================================================
inline uint float_to_sort_key(float f) {
    uint bits = as_type<uint>(f);
    uint mask = -int(bits >> 31) | 0x80000000u;
    return bits ^ mask;
}

// =========================================================================
//  Kernel 1: generate sort keys from float depths
// =========================================================================
kernel void radix_generate_keys(
    device const float* depths   [[buffer(0)]],
    device uint*        keys_out [[buffer(1)]],
    device uint*        vals_out [[buffer(2)]],
    constant uint&      num_keys [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_keys) return;
    keys_out[tid] = float_to_sort_key(depths[tid]);
    vals_out[tid] = tid;
}

// =========================================================================
//  Kernel 2: per-block histogram (column-major output)
//  Output: histograms[digit * num_blocks + block_id]
// =========================================================================
kernel void radix_histogram(
    device const uint*   keys       [[buffer(0)]],
    device uint*         histograms [[buffer(1)]],
    constant SortParams& params     [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    threadgroup atomic_uint local_hist[16];  // RADIX_SIZE

    if (lid < RADIX_SIZE)
        atomic_store_explicit(&local_hist[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < params.num_keys) {
        uint digit = (keys[tid] >> params.bit_offset) & RADIX_MASK;
        atomic_fetch_add_explicit(&local_hist[digit], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < RADIX_SIZE)
        histograms[lid * params.num_blocks + gid] =
            atomic_load_explicit(&local_hist[lid], memory_order_relaxed);
}

// =========================================================================
//  Kernel 3: Blelloch work-efficient exclusive prefix sum
//  Each threadgroup scans SCAN_BLOCK elements, outputs block total.
// =========================================================================
kernel void scan_exclusive(
    device uint*         data       [[buffer(0)]],
    device uint*         block_sums [[buffer(1)]],
    constant ScanParams& params     [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    threadgroup uint temp[256];  // SCAN_BLOCK

    uint idx = gid * SCAN_BLOCK + lid;
    temp[lid] = (idx < params.num_elements) ? data[idx] : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Up-sweep (reduce) ----
    for (uint stride = 1; stride < SCAN_BLOCK; stride <<= 1) {
        uint ai = (lid + 1) * (stride << 1) - 1;
        if (ai < SCAN_BLOCK)
            temp[ai] += temp[ai - stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        block_sums[gid] = temp[SCAN_BLOCK - 1];
        temp[SCAN_BLOCK - 1] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Down-sweep ----
    for (uint stride = SCAN_BLOCK >> 1; stride >= 1; stride >>= 1) {
        uint ai = (lid + 1) * (stride << 1) - 1;
        if (ai < SCAN_BLOCK) {
            uint t = temp[ai - stride];
            temp[ai - stride] = temp[ai];
            temp[ai] += t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (idx < params.num_elements)
        data[idx] = temp[lid];
}

// =========================================================================
//  Kernel 4: propagate block sums into scanned data
// =========================================================================
kernel void scan_add_block_sums(
    device uint*         data       [[buffer(0)]],
    device const uint*   block_sums [[buffer(1)]],
    constant ScanParams& params     [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint gid [[threadgroup_position_in_grid]])
{
    if (gid > 0 && tid < params.num_elements)
        data[tid] += block_sums[gid];
}

// =========================================================================
//  Kernel 5: stable scatter via SIMD prefix sum
//
//  For each of the RADIX_SIZE (16) possible digits:
//    1. flag = (my_digit == d) ? 1 : 0
//    2. rank_in_simd = simd_prefix_exclusive_sum(flag)
//    3. total_in_simd = simd_sum(flag)
//    4. Cross-SIMD prefix sum → full block rank
//    5. output_pos = global_offset[digit][block] + rank
//
//  This is stable: simd_prefix_exclusive_sum preserves thread ordering.
// =========================================================================
kernel void radix_scatter_stable(
    device const uint*   keys_in  [[buffer(0)]],
    device const uint*   vals_in  [[buffer(1)]],
    device uint*         keys_out [[buffer(2)]],
    device uint*         vals_out [[buffer(3)]],
    device const uint*   offsets  [[buffer(4)]],
    constant SortParams& params   [[buffer(5)]],
    uint tid       [[thread_position_in_grid]],
    uint lid       [[thread_position_in_threadgroup]],
    uint gid       [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    // 256 threads / 32 SIMD width = 8 SIMD groups
    threadgroup uint simd_totals[8][16];     // [simd_group][digit]
    threadgroup uint simd_prefix[8][16];     // cross-SIMD prefix sums

    // ---- Load element ----
    bool valid = tid < params.num_keys;
    uint key   = valid ? keys_in[tid] : 0u;
    uint val   = valid ? vals_in[tid] : 0u;
    uint my_digit = valid ? ((key >> params.bit_offset) & RADIX_MASK) : RADIX_SIZE;

    // ---- Per-digit ranking using SIMD prefix sum ----
    uint my_rank_in_simd = 0;

    for (uint d = 0; d < RADIX_SIZE; d++) {
        uint flag  = (my_digit == d) ? 1u : 0u;
        uint rank  = simd_prefix_exclusive_sum(flag);
        uint total = simd_sum(flag);

        if (my_digit == d)
            my_rank_in_simd = rank;

        if (simd_lane == 0)
            simd_totals[simd_id][d] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Cross-SIMD exclusive prefix sum (first 16 threads) ----
    if (lid < RADIX_SIZE) {
        uint d = lid;
        uint running = 0;
        for (uint sg = 0; sg < 8; sg++) {
            simd_prefix[sg][d] = running;
            running += simd_totals[sg][d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Scatter to output ----
    if (valid) {
        uint global_off = offsets[my_digit * params.num_blocks + gid];
        uint local_rank = simd_prefix[simd_id][my_digit] + my_rank_in_simd;
        uint out_pos    = global_off + local_rank;

        keys_out[out_pos] = key;
        vals_out[out_pos] = val;
    }
}
