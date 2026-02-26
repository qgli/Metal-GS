// ============================================================================
//  Metal-GS: C++ / Objective-C++ wrapper for Metal compute dispatch
//
//  Key design decisions for Apple Silicon unified memory:
//    ▸ MTLResourceStorageModeShared — zero-copy CPU↔GPU buffer sharing
//    ▸ Single MTLCommandQueue reused across calls (avoid queue creation overhead)
//    ▸ AOT-compiled .metallib loaded once at init time
//    ▸ Threadgroup size 256 (optimal for M1 7-core GPU occupancy)
// ============================================================================

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_wrapper.h"
#include <cstring>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cmath>
#include <string>
#include <mach/mach_time.h>

// ---------------------------------------------------------------------------
//  SHParams must match the struct in sh_forward.metal exactly
// ---------------------------------------------------------------------------
struct SHParams {
    uint32_t num_points;
    uint32_t num_bases;
    uint32_t sh_degree;
};

// ---------------------------------------------------------------------------
//  PreprocessParams must match the struct in preprocess.metal exactly
// ---------------------------------------------------------------------------
struct PreprocessParams {
    float    tan_fovx;
    float    tan_fovy;
    float    focal_x;
    float    focal_y;
    float    principal_x;
    float    principal_y;
    uint32_t img_width;
    uint32_t img_height;
    uint32_t num_points;
};

// ---------------------------------------------------------------------------
//  Radix sort parameter structs (must match radix_sort.metal byte-for-byte)
// ---------------------------------------------------------------------------
struct SortParams {
    uint32_t num_keys;
    uint32_t num_blocks;
    uint32_t bit_offset;
};

struct ScanParams {
    uint32_t num_elements;
};

struct BinningParams {
    uint32_t num_points;
    uint32_t num_tiles_x;
    uint32_t num_tiles_y;
    uint32_t num_intersections;
};

struct RasterizeParams {
    uint32_t img_width;
    uint32_t img_height;
    uint32_t num_tiles_x;
    uint32_t num_tiles_y;
    float    bg_r;
    float    bg_g;
    float    bg_b;
    uint32_t max_gaussians_per_tile;  // hard cap per tile (0 = unlimited)
};

// ---------------------------------------------------------------------------
//  Singleton Metal context
// ---------------------------------------------------------------------------
namespace {

struct MetalContext {
    id<MTLDevice>               device           = nil;
    id<MTLCommandQueue>         queue            = nil;
    id<MTLComputePipelineState> sh_fwd_pso       = nil;
    id<MTLComputePipelineState> preprocess_pso   = nil;
    // Radix sort PSOs
    id<MTLComputePipelineState> gen_keys_pso     = nil;
    id<MTLComputePipelineState> histogram_pso    = nil;
    id<MTLComputePipelineState> scan_pso         = nil;
    id<MTLComputePipelineState> scan_add_pso     = nil;
    id<MTLComputePipelineState> scatter_pso      = nil;
    // Tile binning PSOs
    id<MTLComputePipelineState> gen_isect_pso    = nil;
    id<MTLComputePipelineState> tile_range_pso   = nil;
    // Rasterize PSO
    id<MTLComputePipelineState> rasterize_pso    = nil;
    // Rasterize backward PSO
    id<MTLComputePipelineState> rasterize_bw_pso = nil;
    // Preprocess backward PSO (M3)
    id<MTLComputePipelineState> preprocess_bw_pso = nil;
    // SH backward PSO (M4)
    id<MTLComputePipelineState> sh_bw_pso         = nil;
    // Simple KNN PSOs
    id<MTLComputePipelineState> morton_codes_pso  = nil;
    id<MTLComputePipelineState> knn_search_pso    = nil;
    // SIMD MMA exploration PSO (v1.0)
    id<MTLComputePipelineState> sh_fwd_mma_pso   = nil;
    bool                        initialised      = false;
    std::string                 error_msg;
};

MetalContext& ctx() {
    static MetalContext c;
    return c;
}

// Find the pre-compiled metal_gs.metallib next to this translation unit
std::string find_metallib_path() {
    @autoreleasepool {
        // Strategy 1: relative to __FILE__ (baked at compile time)
        std::string src_dir = __FILE__;
        auto slash = src_dir.rfind('/');
        if (slash != std::string::npos) {
            std::string candidate = src_dir.substr(0, slash) + "/kernels/metal_gs.metallib";
            if ([[NSFileManager defaultManager]
                    fileExistsAtPath:[NSString stringWithUTF8String:candidate.c_str()]]) {
                return candidate;
            }
        }
        // Strategy 2: METAL_GS_METALLIB_DIR env var
        const char* env = getenv("METAL_GS_METALLIB_DIR");
        if (env) {
            std::string candidate = std::string(env) + "/metal_gs.metallib";
            if ([[NSFileManager defaultManager]
                    fileExistsAtPath:[NSString stringWithUTF8String:candidate.c_str()]]) {
                return candidate;
            }
        }
        return "";
    }
}

bool ensure_init() {
    MetalContext& c = ctx();
    if (c.initialised) return true;

    @autoreleasepool {
        c.device = MTLCreateSystemDefaultDevice();
        if (!c.device) {
            c.error_msg = "No Metal device found";
            return false;
        }

        c.queue = [c.device newCommandQueue];
        if (!c.queue) {
            c.error_msg = "Failed to create command queue";
            return false;
        }

        // ---- Load pre-compiled metallib (AOT) ----
        std::string lib_path = find_metallib_path();
        if (lib_path.empty()) {
            c.error_msg = "Cannot find metal_gs.metallib (AOT compiled library)";
            return false;
        }

        NSError* err = nil;
        NSString* ns_path = [NSString stringWithUTF8String:lib_path.c_str()];
        NSURL* url = [NSURL fileURLWithPath:ns_path];
        id<MTLLibrary> library = [c.device newLibraryWithURL:url error:&err];
        if (!library) {
            c.error_msg = std::string("Failed to load metallib: ") +
                          [[err localizedDescription] UTF8String];
            return false;
        }

        fprintf(stderr, "[Metal-GS] Loaded AOT metallib: %s\n", lib_path.c_str());

        id<MTLFunction> sh_fn = [library newFunctionWithName:@"compute_sh_forward"];
        if (!sh_fn) {
            c.error_msg = "Function 'compute_sh_forward' not found in metallib";
            return false;
        }

        c.sh_fwd_pso = [c.device newComputePipelineStateWithFunction:sh_fn error:&err];
        if (!c.sh_fwd_pso) {
            c.error_msg = std::string("Failed to create PSO: ") +
                          [[err localizedDescription] UTF8String];
            return false;
        }

        // ---- Load preprocess kernel ----
        id<MTLFunction> prep_fn = [library newFunctionWithName:@"preprocess_forward"];
        if (!prep_fn) {
            c.error_msg = "Function 'preprocess_forward' not found in metallib";
            return false;
        }

        c.preprocess_pso = [c.device newComputePipelineStateWithFunction:prep_fn error:&err];
        if (!c.preprocess_pso) {
            c.error_msg = std::string("Failed to create preprocess PSO: ") +
                          [[err localizedDescription] UTF8String];
            return false;
        }

        // ---- Load radix sort kernels (unrolled for ARC compatibility) ----
        auto make_pso = [&](const char* name) -> id<MTLComputePipelineState> {
            NSString* ns = [NSString stringWithUTF8String:name];
            id<MTLFunction> fn = [library newFunctionWithName:ns];
            if (!fn) { c.error_msg = std::string("Function '") + name + "' not found"; return nil; }
            NSError* e2 = nil;
            id<MTLComputePipelineState> p = [c.device newComputePipelineStateWithFunction:fn error:&e2];
            if (!p) { c.error_msg = std::string("PSO failed: '") + name + "'"; return nil; }
            return p;
        };

        c.gen_keys_pso  = make_pso("radix_generate_keys");   if (!c.gen_keys_pso)  return false;
        c.histogram_pso = make_pso("radix_histogram");        if (!c.histogram_pso) return false;
        c.scan_pso      = make_pso("scan_exclusive");         if (!c.scan_pso)      return false;
        c.scan_add_pso  = make_pso("scan_add_block_sums");    if (!c.scan_add_pso)  return false;
        c.scatter_pso   = make_pso("radix_scatter_stable");   if (!c.scatter_pso)   return false;
        c.gen_isect_pso = make_pso("generate_intersections"); if (!c.gen_isect_pso) return false;
        c.tile_range_pso= make_pso("identify_tile_ranges");  if (!c.tile_range_pso)return false;
        c.rasterize_pso = make_pso("rasterize_forward");     if (!c.rasterize_pso) return false;
        c.rasterize_bw_pso = make_pso("rasterize_backward");  if (!c.rasterize_bw_pso) return false;
        c.preprocess_bw_pso = make_pso("preprocess_backward"); if (!c.preprocess_bw_pso) return false;
        c.sh_bw_pso         = make_pso("sh_backward");         if (!c.sh_bw_pso)         return false;
        c.morton_codes_pso  = make_pso("compute_morton_codes"); if (!c.morton_codes_pso)  return false;
        c.knn_search_pso    = make_pso("knn_search");           if (!c.knn_search_pso)    return false;
        c.sh_fwd_mma_pso    = make_pso("compute_sh_forward_mma"); if (!c.sh_fwd_mma_pso) return false;

        fprintf(stderr, "[Metal-GS] All 16 PSOs created successfully\n");

        c.initialised = true;
        return true;
    }
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
//  Public C++ API
// ---------------------------------------------------------------------------
double metal_compute_sh_forward(
    const float*    directions,
    const uint16_t* sh_coeffs,
    uint16_t*       colors_out,
    uint32_t        N,
    uint32_t        K,
    uint32_t        sh_degree
)
{
    @autoreleasepool {
        if (!ensure_init()) {
            fprintf(stderr, "[Metal-GS] Init failed: %s\n", ctx().error_msg.c_str());
            return -1.0;
        }

        MetalContext& c = ctx();

        size_t dir_buf_bytes = (size_t)N * sizeof(float) * 4;   // float3 → float4 stride
        size_t sh_bytes      = (size_t)N * K * 3 * sizeof(uint16_t);
        size_t out_bytes     = (size_t)N * 3 * sizeof(uint16_t);

        // ---- Shared buffers (unified memory, zero-copy) ----
        id<MTLBuffer> dir_buf   = [c.device newBufferWithLength:dir_buf_bytes
                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> sh_buf    = [c.device newBufferWithBytes:(const void*)sh_coeffs
                                            length:sh_bytes
                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buf   = [c.device newBufferWithLength:out_bytes
                                            options:MTLResourceStorageModeShared];

        SHParams params = { N, K, sh_degree };
        id<MTLBuffer> param_buf = [c.device newBufferWithBytes:&params
                                             length:sizeof(SHParams)
                                             options:MTLResourceStorageModeShared];

        // Pack directions: float[N*3] → float4[N] (pad w=0)
        float* dir_dst = (float*)[dir_buf contents];
        for (uint32_t i = 0; i < N; i++) {
            dir_dst[i * 4 + 0] = directions[i * 3 + 0];
            dir_dst[i * 4 + 1] = directions[i * 3 + 1];
            dir_dst[i * 4 + 2] = directions[i * 3 + 2];
            dir_dst[i * 4 + 3] = 0.0f;
        }

        // ---- Encode & dispatch ----
        id<MTLCommandBuffer>         cmd = [c.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:c.sh_fwd_pso];
        [enc setBuffer:dir_buf   offset:0 atIndex:0];
        [enc setBuffer:sh_buf    offset:0 atIndex:1];
        [enc setBuffer:out_buf   offset:0 atIndex:2];
        [enc setBuffer:param_buf offset:0 atIndex:3];

        NSUInteger tg_size = std::min((NSUInteger)256,
                                       [c.sh_fwd_pso maxTotalThreadsPerThreadgroup]);
        MTLSize threads_per_group = MTLSizeMake(tg_size, 1, 1);
        MTLSize grid_size         = MTLSizeMake(N, 1, 1);

        [enc dispatchThreads:grid_size threadsPerThreadgroup:threads_per_group];
        [enc endEncoding];

        // ---- Time the GPU execution ----
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        uint64_t t0 = mach_absolute_time();

        [cmd commit];
        [cmd waitUntilCompleted];

        uint64_t t1 = mach_absolute_time();
        double elapsed_ns = (double)(t1 - t0) * tb.numer / tb.denom;
        double elapsed_ms = elapsed_ns / 1e6;

        if ([cmd status] == MTLCommandBufferStatusError) {
            fprintf(stderr, "[Metal-GS] Command buffer error: %s\n",
                    [[cmd.error localizedDescription] UTF8String]);
            return -1.0;
        }

        // Copy result
        memcpy(colors_out, [out_buf contents], out_bytes);

        return elapsed_ms;
    }
}

// ---------------------------------------------------------------------------
//  SH forward MMA variant — simdgroup_matrix 8×8 (V1.0 exploration)
//  Identical interface to metal_compute_sh_forward but dispatches the MMA kernel.
//  Thread model: 8 Gaussians per simdgroup, 64 per threadgroup (256 threads).
// ---------------------------------------------------------------------------
double metal_compute_sh_forward_mma(
    const float*    directions,
    const uint16_t* sh_coeffs,
    uint16_t*       colors_out,
    uint32_t        N,
    uint32_t        K,
    uint32_t        sh_degree
)
{
    @autoreleasepool {
        if (!ensure_init()) {
            fprintf(stderr, "[Metal-GS] Init failed: %s\n", ctx().error_msg.c_str());
            return -1.0;
        }

        MetalContext& c = ctx();

        size_t dir_buf_bytes = (size_t)N * sizeof(float) * 4;
        size_t sh_bytes      = (size_t)N * K * 3 * sizeof(uint16_t);
        size_t out_bytes     = (size_t)N * 3 * sizeof(uint16_t);

        id<MTLBuffer> dir_buf   = [c.device newBufferWithLength:dir_buf_bytes
                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> sh_buf    = [c.device newBufferWithBytes:(const void*)sh_coeffs
                                            length:sh_bytes
                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buf   = [c.device newBufferWithLength:out_bytes
                                            options:MTLResourceStorageModeShared];

        SHParams params = { N, K, sh_degree };
        id<MTLBuffer> param_buf = [c.device newBufferWithBytes:&params
                                             length:sizeof(SHParams)
                                             options:MTLResourceStorageModeShared];

        float* dir_dst = (float*)[dir_buf contents];
        for (uint32_t i = 0; i < N; i++) {
            dir_dst[i * 4 + 0] = directions[i * 3 + 0];
            dir_dst[i * 4 + 1] = directions[i * 3 + 1];
            dir_dst[i * 4 + 2] = directions[i * 3 + 2];
            dir_dst[i * 4 + 3] = 0.0f;
        }

        id<MTLCommandBuffer>         cmd = [c.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:c.sh_fwd_mma_pso];
        [enc setBuffer:dir_buf   offset:0 atIndex:0];
        [enc setBuffer:sh_buf    offset:0 atIndex:1];
        [enc setBuffer:out_buf   offset:0 atIndex:2];
        [enc setBuffer:param_buf offset:0 atIndex:3];

        // MMA thread model: 64 Gaussians per threadgroup (8 simdgroups × 8 each)
        NSUInteger tg_size = 256;
        NSUInteger gaussians_per_tg = 64;
        uint32_t num_tgs = (N + (uint32_t)gaussians_per_tg - 1) / (uint32_t)gaussians_per_tg;
        MTLSize tg_count = MTLSizeMake(num_tgs, 1, 1);
        MTLSize tg_dim   = MTLSizeMake(tg_size, 1, 1);

        [enc dispatchThreadgroups:tg_count threadsPerThreadgroup:tg_dim];
        [enc endEncoding];

        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        uint64_t t0 = mach_absolute_time();

        [cmd commit];
        [cmd waitUntilCompleted];

        uint64_t t1 = mach_absolute_time();
        double elapsed_ns = (double)(t1 - t0) * tb.numer / tb.denom;
        double elapsed_ms = elapsed_ns / 1e6;

        if ([cmd status] == MTLCommandBufferStatusError) {
            fprintf(stderr, "[Metal-GS] MMA Command buffer error: %s\n",
                    [[cmd.error localizedDescription] UTF8String]);
            return -1.0;
        }

        memcpy(colors_out, [out_buf contents], out_bytes);
        return elapsed_ms;
    }
}

// ---------------------------------------------------------------------------
//  Preprocess forward: 3D Gaussians → 2D screen space projection
// ---------------------------------------------------------------------------
double metal_preprocess_forward(
    const float*    means3d,
    const float*    scales,
    const float*    quats,
    const float*    viewmat,
    float           tan_fovx,
    float           tan_fovy,
    float           focal_x,
    float           focal_y,
    float           principal_x,
    float           principal_y,
    uint32_t        img_width,
    uint32_t        img_height,
    uint32_t        N,
    float*          means2d_out,
    float*          cov2d_out,
    float*          depths_out,
    uint32_t*       radii_out,
    uint32_t*       tile_min_out,
    uint32_t*       tile_max_out
)
{
    @autoreleasepool {
        if (!ensure_init()) {
            fprintf(stderr, "[Metal-GS] Init failed: %s\n", ctx().error_msg.c_str());
            return -1.0;
        }

        MetalContext& c = ctx();

        // ---- Allocate Metal buffers (unified memory, zero-copy) ----
        size_t means3d_bytes   = (size_t)N * 3 * sizeof(float);
        size_t scales_bytes    = (size_t)N * 3 * sizeof(float);
        size_t quats_bytes     = (size_t)N * 4 * sizeof(float);
        size_t viewmat_bytes   = 16 * sizeof(float);
        size_t means2d_bytes   = (size_t)N * 2 * sizeof(float);
        size_t cov2d_bytes     = (size_t)N * 3 * sizeof(float);  // upper tri [a, b, c]
        size_t depths_bytes    = (size_t)N * sizeof(float);
        size_t radii_bytes     = (size_t)N * sizeof(uint32_t);
        size_t tile_min_bytes  = (size_t)N * 2 * sizeof(uint32_t);
        size_t tile_max_bytes  = (size_t)N * 2 * sizeof(uint32_t);

        id<MTLBuffer> means3d_buf   = [c.device newBufferWithBytes:(const void*)means3d
                                                 length:means3d_bytes
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> scales_buf    = [c.device newBufferWithBytes:(const void*)scales
                                                 length:scales_bytes
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> quats_buf     = [c.device newBufferWithBytes:(const void*)quats
                                                 length:quats_bytes
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> viewmat_buf   = [c.device newBufferWithBytes:(const void*)viewmat
                                                 length:viewmat_bytes
                                                 options:MTLResourceStorageModeShared];

        id<MTLBuffer> means2d_buf   = [c.device newBufferWithLength:means2d_bytes
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> cov2d_buf     = [c.device newBufferWithLength:cov2d_bytes
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> depths_buf    = [c.device newBufferWithLength:depths_bytes
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> radii_buf     = [c.device newBufferWithLength:radii_bytes
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> tile_min_buf  = [c.device newBufferWithLength:tile_min_bytes
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> tile_max_buf  = [c.device newBufferWithLength:tile_max_bytes
                                                 options:MTLResourceStorageModeShared];

        PreprocessParams params;
        params.tan_fovx     = tan_fovx;
        params.tan_fovy     = tan_fovy;
        params.focal_x      = focal_x;
        params.focal_y      = focal_y;
        params.principal_x  = principal_x;
        params.principal_y  = principal_y;
        params.img_width    = img_width;
        params.img_height   = img_height;
        params.num_points   = N;

        id<MTLBuffer> param_buf = [c.device newBufferWithBytes:&params
                                             length:sizeof(PreprocessParams)
                                             options:MTLResourceStorageModeShared];

        // ---- Encode & dispatch ----
        id<MTLCommandBuffer>         cmd = [c.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:c.preprocess_pso];
        [enc setBuffer:means3d_buf   offset:0 atIndex:0];
        [enc setBuffer:scales_buf    offset:0 atIndex:1];
        [enc setBuffer:quats_buf     offset:0 atIndex:2];
        [enc setBuffer:viewmat_buf   offset:0 atIndex:3];
        [enc setBuffer:param_buf     offset:0 atIndex:4];
        [enc setBuffer:means2d_buf   offset:0 atIndex:5];
        [enc setBuffer:cov2d_buf     offset:0 atIndex:6];
        [enc setBuffer:depths_buf    offset:0 atIndex:7];
        [enc setBuffer:radii_buf     offset:0 atIndex:8];
        [enc setBuffer:tile_min_buf  offset:0 atIndex:9];
        [enc setBuffer:tile_max_buf  offset:0 atIndex:10];

        NSUInteger tg_size = std::min((NSUInteger)256,
                                       [c.preprocess_pso maxTotalThreadsPerThreadgroup]);
        MTLSize threads_per_group = MTLSizeMake(tg_size, 1, 1);
        MTLSize grid_size         = MTLSizeMake(N, 1, 1);

        [enc dispatchThreads:grid_size threadsPerThreadgroup:threads_per_group];
        [enc endEncoding];

        // ---- Time the GPU execution ----
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        uint64_t t0 = mach_absolute_time();

        [cmd commit];
        [cmd waitUntilCompleted];

        uint64_t t1 = mach_absolute_time();
        double elapsed_ns = (double)(t1 - t0) * tb.numer / tb.denom;
        double elapsed_ms = elapsed_ns / 1e6;

        if ([cmd status] == MTLCommandBufferStatusError) {
            fprintf(stderr, "[Metal-GS] Command buffer error: %s\n",
                    [[cmd.error localizedDescription] UTF8String]);
            return -1.0;
        }

        // Copy results back
        memcpy(means2d_out,  [means2d_buf contents],  means2d_bytes);
        memcpy(cov2d_out,    [cov2d_buf contents],    cov2d_bytes);
        memcpy(depths_out,   [depths_buf contents],   depths_bytes);
        memcpy(radii_out,    [radii_buf contents],    radii_bytes);
        memcpy(tile_min_out, [tile_min_buf contents], tile_min_bytes);
        memcpy(tile_max_out, [tile_max_buf contents], tile_max_bytes);

        return elapsed_ms;
    }
}

// ---------------------------------------------------------------------------
//  Radix sort by depth — full GPU sort with 8 × 4-bit radix passes
// ---------------------------------------------------------------------------

static const uint32_t RADIX_SIZE  = 16;
static const uint32_t SORT_BLOCK  = 256;
static const uint32_t SCAN_BLOCK  = 256;

// Helper: multi-level exclusive prefix sum on a SINGLE compute encoder.
// Supports up to 256^3 = 16M elements (3-level scan hierarchy).
// Uses memoryBarrierWithScope between dispatches for data visibility.
static void dispatch_prefix_sum_se(
    MetalContext& c,
    id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> data_buf,
    uint32_t n,
    id<MTLBuffer> bsum1_buf,
    id<MTLBuffer> bsum2_buf,
    id<MTLBuffer> bsum3_buf)
{
    uint32_t nblk1 = (n + SCAN_BLOCK - 1) / SCAN_BLOCK;
    uint32_t nblk2 = (nblk1 + SCAN_BLOCK - 1) / SCAN_BLOCK;
    uint32_t nblk3 = (nblk2 + SCAN_BLOCK - 1) / SCAN_BLOCK;

    // Level 1: scan data → block sums
    {
        ScanParams sp = { n };
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:c.scan_pso];
        [enc setBuffer:data_buf  offset:0 atIndex:0];
        [enc setBuffer:bsum1_buf offset:0 atIndex:1];
        [enc setBytes:&sp length:sizeof(ScanParams) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(nblk1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(SCAN_BLOCK, 1, 1)];
    }

    // Level 2: scan block sums → level-2 block sums
    {
        ScanParams sp = { nblk1 };
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:c.scan_pso];
        [enc setBuffer:bsum1_buf offset:0 atIndex:0];
        [enc setBuffer:bsum2_buf offset:0 atIndex:1];
        [enc setBytes:&sp length:sizeof(ScanParams) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(nblk2, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(SCAN_BLOCK, 1, 1)];
    }

    // Level 3 (if needed): scan level-2 block sums
    if (nblk2 > 1) {
        {
            ScanParams sp = { nblk2 };
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:c.scan_pso];
            [enc setBuffer:bsum2_buf offset:0 atIndex:0];
            [enc setBuffer:bsum3_buf offset:0 atIndex:1];
            [enc setBytes:&sp length:sizeof(ScanParams) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(nblk3, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(SCAN_BLOCK, 1, 1)];
        }

        // Propagate level-3 → level-2 (add scanned bsum2 into bsum1)
        {
            ScanParams sp = { nblk1 };
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:c.scan_add_pso];
            [enc setBuffer:bsum1_buf offset:0 atIndex:0];
            [enc setBuffer:bsum2_buf offset:0 atIndex:1];
            [enc setBytes:&sp length:sizeof(ScanParams) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(nblk2, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(SCAN_BLOCK, 1, 1)];
        }
    }

    // Propagate level-2 → level-1 (add block sums back into data)
    if (nblk1 > 1) {
        ScanParams sp = { n };
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:c.scan_add_pso];
        [enc setBuffer:data_buf  offset:0 atIndex:0];
        [enc setBuffer:bsum1_buf offset:0 atIndex:1];
        [enc setBytes:&sp length:sizeof(ScanParams) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(nblk1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(SCAN_BLOCK, 1, 1)];
    }
}

// ---------------------------------------------------------------------------
//  Helper: encode ONE radix pass (histogram + prefix_sum + scatter)
//  into the given encoder.  Caller manages command buffer submission.
// ---------------------------------------------------------------------------
static void encode_one_radix_pass(
    MetalContext& c,
    id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> k_src, id<MTLBuffer> k_dst,
    id<MTLBuffer> v_src, id<MTLBuffer> v_dst,
    id<MTLBuffer> hist_buf,
    id<MTLBuffer> bsum1_buf, id<MTLBuffer> bsum2_buf,
    id<MTLBuffer> bsum3_buf,
    uint32_t N, uint32_t pass_idx)
{
    uint32_t num_blocks = (N + SORT_BLOCK - 1) / SORT_BLOCK;
    uint32_t hist_size  = RADIX_SIZE * num_blocks;
    SortParams sp = { N, num_blocks, pass_idx * 4 };

    // Histogram
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [enc setComputePipelineState:c.histogram_pso];
    [enc setBuffer:k_src    offset:0 atIndex:0];
    [enc setBuffer:hist_buf offset:0 atIndex:1];
    [enc setBytes:&sp       length:sizeof(SortParams) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(N, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(SORT_BLOCK, 1, 1)];

    // Prefix sum on histogram
    dispatch_prefix_sum_se(c, enc, hist_buf, hist_size,
                           bsum1_buf, bsum2_buf, bsum3_buf);

    // Scatter
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [enc setComputePipelineState:c.scatter_pso];
    [enc setBuffer:k_src    offset:0 atIndex:0];
    [enc setBuffer:v_src    offset:0 atIndex:1];
    [enc setBuffer:k_dst    offset:0 atIndex:2];
    [enc setBuffer:v_dst    offset:0 atIndex:3];
    [enc setBuffer:hist_buf offset:0 atIndex:4];
    [enc setBytes:&sp       length:sizeof(SortParams) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(N, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(SORT_BLOCK, 1, 1)];
}

// ---------------------------------------------------------------------------
//  Helper: radix sort uint32 key-value pairs on a SINGLE compute encoder.
//  Uses memoryBarrierWithScope between passes for data visibility.
//  After num_passes (must be even), results are in keys_a / vals_a.
// ---------------------------------------------------------------------------
static void dispatch_radix_sort_kv(
    MetalContext& c,
    id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> keys_a, id<MTLBuffer> keys_b,
    id<MTLBuffer> vals_a, id<MTLBuffer> vals_b,
    id<MTLBuffer> hist_buf,
    id<MTLBuffer> bsum1_buf, id<MTLBuffer> bsum2_buf,
    id<MTLBuffer> bsum3_buf,
    uint32_t N, uint32_t num_passes)
{
    id<MTLBuffer> k_src = keys_a, k_dst = keys_b;
    id<MTLBuffer> v_src = vals_a, v_dst = vals_b;

    for (uint32_t pass = 0; pass < num_passes; pass++) {
        encode_one_radix_pass(c, enc,
                              k_src, k_dst, v_src, v_dst,
                              hist_buf, bsum1_buf, bsum2_buf, bsum3_buf,
                              N, pass);
        std::swap(k_src, k_dst);
        std::swap(v_src, v_dst);
    }
    // After even passes: result in keys_a, vals_a ✓
}

double metal_radix_sort_by_depth(
    const float*    depths,
    uint32_t*       sorted_indices_out,
    uint32_t        N)
{
    @autoreleasepool {
        if (!ensure_init()) {
            fprintf(stderr, "[Metal-GS] Init failed: %s\n", ctx().error_msg.c_str());
            return -1.0;
        }
        MetalContext& c = ctx();

        if (N == 0) return 0.0;

        // ---- CPU fallback for small arrays (GPU launch overhead > compute) ----
        if (N <= 16384) {
            mach_timebase_info_data_t tb;
            mach_timebase_info(&tb);
            uint64_t t0 = mach_absolute_time();
            for (uint32_t i = 0; i < N; i++) sorted_indices_out[i] = i;
            std::sort(sorted_indices_out, sorted_indices_out + N,
                      [&](uint32_t a, uint32_t b){ return depths[a] < depths[b]; });
            uint64_t t1 = mach_absolute_time();
            return (double)(t1 - t0) * tb.numer / tb.denom / 1e6;
        }

        uint32_t num_blocks  = (N + SORT_BLOCK - 1) / SORT_BLOCK;
        uint32_t hist_size   = RADIX_SIZE * num_blocks;
        uint32_t scan_blocks = (hist_size + SCAN_BLOCK - 1) / SCAN_BLOCK;

        size_t key_bytes  = (size_t)N * sizeof(uint32_t);
        size_t hist_bytes = (size_t)hist_size * sizeof(uint32_t);
        size_t sb1_bytes  = (size_t)scan_blocks * sizeof(uint32_t);
        size_t sb2_bytes  = sizeof(uint32_t) * ((scan_blocks + SCAN_BLOCK - 1) / SCAN_BLOCK);
        size_t sb3_bytes  = sizeof(uint32_t) * std::max(1u,
            ((scan_blocks + SCAN_BLOCK - 1) / SCAN_BLOCK + SCAN_BLOCK - 1) / SCAN_BLOCK);

        id<MTLBuffer> depth_buf = [c.device newBufferWithBytes:(const void*)depths
                                             length:N * sizeof(float)
                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> keys_a = [c.device newBufferWithLength:key_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> keys_b = [c.device newBufferWithLength:key_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> vals_a = [c.device newBufferWithLength:key_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> vals_b = [c.device newBufferWithLength:key_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> hist_buf  = [c.device newBufferWithLength:hist_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bsum1_buf = [c.device newBufferWithLength:sb1_bytes  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bsum2_buf = [c.device newBufferWithLength:sb2_bytes  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bsum3_buf = [c.device newBufferWithLength:sb3_bytes  options:MTLResourceStorageModeShared];

        // ---- Single command buffer: gen_keys + 8 radix passes ----
        id<MTLCommandBuffer>         cmd = [c.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:c.gen_keys_pso];
        [enc setBuffer:depth_buf offset:0 atIndex:0];
        [enc setBuffer:keys_a    offset:0 atIndex:1];
        [enc setBuffer:vals_a    offset:0 atIndex:2];
        [enc setBytes:&N         length:sizeof(uint32_t) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(N, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(SORT_BLOCK, 1, 1)];

        dispatch_radix_sort_kv(c, enc,
                               keys_a, keys_b, vals_a, vals_b,
                               hist_buf, bsum1_buf, bsum2_buf, bsum3_buf,
                               N, 8);

        [enc endEncoding];

        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        uint64_t t0 = mach_absolute_time();

        [cmd commit];
        [cmd waitUntilCompleted];

        uint64_t t1 = mach_absolute_time();
        double elapsed_ms = (double)(t1 - t0) * tb.numer / tb.denom / 1e6;

        if ([cmd status] == MTLCommandBufferStatusError) {
            fprintf(stderr, "[Metal-GS] Radix sort error: %s\n",
                    [[cmd.error localizedDescription] UTF8String]);
            return -1.0;
        }

        memcpy(sorted_indices_out, [vals_a contents], N * sizeof(uint32_t));
        return elapsed_ms;
    }
}

// ---------------------------------------------------------------------------
//  Tile binning: assign depth-sorted Gaussians to screen tiles
//
//  Pipeline (single command buffer, single encoder):
//    1. generate_intersections → (tile_id, gauss_id) pairs
//    2. stable radix sort by tile_id  (≤4 passes for 16-bit tile IDs)
//    3. identify_tile_ranges → tile_bins[tile] = (start, end)
//
//  offsets[] is CPU-computed exclusive prefix sum of per-Gaussian tile
//  counts. This avoids GPU-side count+scan, keeping a single submit.
// ---------------------------------------------------------------------------
double metal_tile_binning(
    const uint32_t* sorted_indices,
    const uint32_t* radii,
    const uint32_t* tile_min,
    const uint32_t* tile_max,
    const uint32_t* offsets,
    uint32_t        N,
    uint32_t        num_tiles_x,
    uint32_t        num_tiles_y,
    uint32_t        num_intersections,
    uint32_t*       point_list_out,
    uint32_t*       tile_bins_out)
{
    @autoreleasepool {
        if (!ensure_init()) {
            fprintf(stderr, "[Metal-GS] Init failed: %s\n", ctx().error_msg.c_str());
            return -1.0;
        }
        MetalContext& c = ctx();

        uint32_t num_tiles = num_tiles_x * num_tiles_y;
        size_t tbins_bytes = (size_t)num_tiles * 2 * sizeof(uint32_t);
        memset(tile_bins_out, 0, tbins_bytes);

        if (num_intersections == 0 || N == 0) return 0.0;

        // ---- Determine radix passes based on tile_id range ----
        uint32_t max_tile_id = num_tiles - 1;
        uint32_t needed_bits = 0;
        { uint32_t tmp = max_tile_id; while (tmp > 0) { needed_bits++; tmp >>= 1; } }
        uint32_t num_passes = (needed_bits + 3) / 4;
        if (num_passes == 0) num_passes = 2;
        if (num_passes % 2 != 0) num_passes++;

        // ---- Sort buffer sizing ----
        uint32_t sort_blocks = (num_intersections + SORT_BLOCK - 1) / SORT_BLOCK;
        uint32_t hist_size   = RADIX_SIZE * sort_blocks;
        uint32_t sblk1       = (hist_size + SCAN_BLOCK - 1) / SCAN_BLOCK;
        size_t isect_bytes   = (size_t)num_intersections * sizeof(uint32_t);
        size_t hist_bytes    = (size_t)hist_size * sizeof(uint32_t);
        size_t sb1_bytes     = (size_t)sblk1 * sizeof(uint32_t);
        size_t sb2_bytes     = (size_t)((sblk1 + SCAN_BLOCK - 1) / SCAN_BLOCK) * sizeof(uint32_t);
        uint32_t sblk2       = (sblk1 + SCAN_BLOCK - 1) / SCAN_BLOCK;
        size_t sb3_bytes     = (size_t)std::max(1u, (sblk2 + SCAN_BLOCK - 1) / SCAN_BLOCK) * sizeof(uint32_t);

        // ---- Input buffers ----
        size_t idx_bytes  = (size_t)N * sizeof(uint32_t);
        size_t tile_bytes = (size_t)N * 2 * sizeof(uint32_t);

        id<MTLBuffer> si_buf   = [c.device newBufferWithBytes:sorted_indices length:idx_bytes  options:MTLResourceStorageModeShared];
        id<MTLBuffer> rad_buf  = [c.device newBufferWithBytes:radii          length:idx_bytes  options:MTLResourceStorageModeShared];
        id<MTLBuffer> tmin_buf = [c.device newBufferWithBytes:tile_min       length:tile_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> tmax_buf = [c.device newBufferWithBytes:tile_max       length:tile_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> off_buf  = [c.device newBufferWithBytes:offsets        length:idx_bytes  options:MTLResourceStorageModeShared];

        // Intersection ping-pong buffers
        id<MTLBuffer> keys_a = [c.device newBufferWithLength:isect_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> keys_b = [c.device newBufferWithLength:isect_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> vals_a = [c.device newBufferWithLength:isect_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> vals_b = [c.device newBufferWithLength:isect_bytes options:MTLResourceStorageModeShared];

        // Sort helper buffers
        id<MTLBuffer> hist_buf  = [c.device newBufferWithLength:hist_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bsum1_buf = [c.device newBufferWithLength:sb1_bytes  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bsum2_buf = [c.device newBufferWithLength:sb2_bytes  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bsum3_buf = [c.device newBufferWithLength:sb3_bytes  options:MTLResourceStorageModeShared];

        // Tile bins (zero-initialized, GPU writes non-empty tiles)
        id<MTLBuffer> tbins_buf = [c.device newBufferWithLength:tbins_bytes options:MTLResourceStorageModeShared];
        memset([tbins_buf contents], 0, tbins_bytes);

        BinningParams bp = { N, num_tiles_x, num_tiles_y, num_intersections };

        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        uint64_t t0 = mach_absolute_time();

        // ---- Single command buffer: gen_isect + sort + tile_range ----
        id<MTLCommandBuffer>         cmd = [c.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        // Step 1: Generate (tile_id, gauss_id) intersection pairs
        [enc setComputePipelineState:c.gen_isect_pso];
        [enc setBuffer:si_buf   offset:0 atIndex:0];
        [enc setBuffer:rad_buf  offset:0 atIndex:1];
        [enc setBuffer:tmin_buf offset:0 atIndex:2];
        [enc setBuffer:tmax_buf offset:0 atIndex:3];
        [enc setBuffer:off_buf  offset:0 atIndex:4];
        [enc setBuffer:keys_a   offset:0 atIndex:5];
        [enc setBuffer:vals_a   offset:0 atIndex:6];
        [enc setBytes:&bp       length:sizeof(BinningParams) atIndex:7];
        [enc dispatchThreads:MTLSizeMake(N, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(SORT_BLOCK, 1, 1)];

        // Step 2: Stable radix sort by tile_id
        dispatch_radix_sort_kv(c, enc,
                               keys_a, keys_b, vals_a, vals_b,
                               hist_buf, bsum1_buf, bsum2_buf, bsum3_buf,
                               num_intersections, num_passes);

        // Step 3: Identify tile ranges in sorted list
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:c.tile_range_pso];
        [enc setBuffer:keys_a    offset:0 atIndex:0];
        [enc setBuffer:tbins_buf offset:0 atIndex:1];
        [enc setBytes:&num_intersections length:sizeof(uint32_t) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(num_intersections, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(SORT_BLOCK, 1, 1)];

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        uint64_t t1 = mach_absolute_time();
        double elapsed_ms = (double)(t1 - t0) * tb.numer / tb.denom / 1e6;

        if ([cmd status] == MTLCommandBufferStatusError) {
            fprintf(stderr, "[Metal-GS] Tile binning error: %s\n",
                    [[cmd.error localizedDescription] UTF8String]);
            return -1.0;
        }

        // Copy results (zero-copy read from unified memory)
        memcpy(point_list_out, [vals_a contents], num_intersections * sizeof(uint32_t));
        memcpy(tile_bins_out,  [tbins_buf contents], tbins_bytes);

        return elapsed_ms;
    }
}

// ---------------------------------------------------------------------------
//  Forward rasterization: per-tile alpha blending of 2D Gaussians
//
//  Dispatches one threadgroup (16×16 = 256 threads) per screen tile.
//  Each thread renders one pixel using cooperative Gaussian fetching
//  into threadgroup shared memory, then front-to-back alpha blending.
// ---------------------------------------------------------------------------
double metal_rasterize_forward(
    const float*    means2d,
    const float*    cov2d,
    const float*    colors,
    const float*    opacities,
    const uint32_t* tile_bins,
    const uint32_t* point_list,
    uint32_t        num_points,
    uint32_t        num_intersections,
    uint32_t        img_width,
    uint32_t        img_height,
    uint32_t        num_tiles_x,
    uint32_t        num_tiles_y,
    float           bg_r,
    float           bg_g,
    float           bg_b,
    uint32_t        max_gaussians_per_tile,
    float*          out_img,
    float*          T_final_out,
    uint32_t*       n_contrib_out)
{
    @autoreleasepool {
        if (!ensure_init()) {
            fprintf(stderr, "[Metal-GS] Init failed: %s\n", ctx().error_msg.c_str());
            return -1.0;
        }
        MetalContext& c = ctx();

        uint32_t num_tiles   = num_tiles_x * num_tiles_y;
        size_t means2d_bytes = (size_t)num_points * 2 * sizeof(float);
        size_t cov2d_bytes   = (size_t)num_points * 3 * sizeof(float);
        size_t colors_bytes  = (size_t)num_points * 3 * sizeof(float);
        size_t opac_bytes    = (size_t)num_points * sizeof(float);
        size_t tbins_bytes   = (size_t)num_tiles * 2 * sizeof(uint32_t);
        size_t plist_bytes   = (size_t)num_intersections * sizeof(uint32_t);
        size_t img_bytes     = (size_t)img_width * img_height * 3 * sizeof(float);
        size_t tfinal_bytes  = (size_t)img_width * img_height * sizeof(float);
        size_t ncontrib_bytes= (size_t)img_width * img_height * sizeof(uint32_t);

        // ---- Metal buffers (unified memory, zero-copy) ----
        id<MTLBuffer> means2d_buf = [c.device newBufferWithBytes:means2d
                                               length:means2d_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> cov2d_buf   = [c.device newBufferWithBytes:cov2d
                                               length:cov2d_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> colors_buf  = [c.device newBufferWithBytes:colors
                                               length:colors_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> opac_buf    = [c.device newBufferWithBytes:opacities
                                               length:opac_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> tbins_buf   = [c.device newBufferWithBytes:tile_bins
                                               length:tbins_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> plist_buf   = [c.device newBufferWithBytes:point_list
                                               length:plist_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> img_buf     = [c.device newBufferWithLength:img_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> tfinal_buf  = [c.device newBufferWithLength:tfinal_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> ncontrib_buf= [c.device newBufferWithLength:ncontrib_bytes
                                               options:MTLResourceStorageModeShared];
        memset([ncontrib_buf contents], 0, ncontrib_bytes);

        RasterizeParams params;
        params.img_width   = img_width;
        params.img_height  = img_height;
        params.num_tiles_x = num_tiles_x;
        params.num_tiles_y = num_tiles_y;
        params.bg_r        = bg_r;
        params.bg_g        = bg_g;
        params.bg_b        = bg_b;
        params.max_gaussians_per_tile = max_gaussians_per_tile;

        // ---- Single command buffer: rasterize all tiles ----
        id<MTLCommandBuffer>         cmd = [c.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:c.rasterize_pso];
        [enc setBuffer:means2d_buf  offset:0 atIndex:0];
        [enc setBuffer:cov2d_buf    offset:0 atIndex:1];
        [enc setBuffer:colors_buf   offset:0 atIndex:2];
        [enc setBuffer:opac_buf     offset:0 atIndex:3];
        [enc setBuffer:tbins_buf    offset:0 atIndex:4];
        [enc setBuffer:plist_buf    offset:0 atIndex:5];
        [enc setBytes:&params       length:sizeof(RasterizeParams) atIndex:6];
        [enc setBuffer:img_buf      offset:0 atIndex:7];
        [enc setBuffer:tfinal_buf   offset:0 atIndex:8];
        [enc setBuffer:ncontrib_buf offset:0 atIndex:9];

        MTLSize threadgroup_size  = MTLSizeMake(16, 16, 1);
        MTLSize threadgroup_count = MTLSizeMake(num_tiles_x, num_tiles_y, 1);

        [enc dispatchThreadgroups:threadgroup_count
            threadsPerThreadgroup:threadgroup_size];
        [enc endEncoding];

        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        uint64_t t0 = mach_absolute_time();

        [cmd commit];
        [cmd waitUntilCompleted];

        uint64_t t1 = mach_absolute_time();
        double elapsed_ms = (double)(t1 - t0) * tb.numer / tb.denom / 1e6;

        if ([cmd status] == MTLCommandBufferStatusError) {
            fprintf(stderr, "[Metal-GS] Rasterize forward error: %s\n",
                    [[cmd.error localizedDescription] UTF8String]);
            return -1.0;
        }

        memcpy(out_img, [img_buf contents], img_bytes);
        memcpy(T_final_out, [tfinal_buf contents], tfinal_bytes);
        memcpy(n_contrib_out, [ncontrib_buf contents], ncontrib_bytes);
        return elapsed_ms;
    }
}

// ---------------------------------------------------------------------------
//  Backward rasterization: compute gradients for 2D Gaussian params
//
//  Strategy A: Naive global atomic add for correctness verification.
//  Dispatches one threadgroup (16×16) per screen tile, reverse traversal.
// ---------------------------------------------------------------------------
double metal_rasterize_backward(
    const float*    means2d,
    const float*    cov2d,
    const float*    colors,
    const float*    opacities,
    const uint32_t* tile_bins,
    const uint32_t* point_list,
    const float*    T_final,
    const uint32_t* n_contrib,
    const float*    dL_dC_pixel,
    uint32_t        num_points,
    uint32_t        num_intersections,
    uint32_t        img_width,
    uint32_t        img_height,
    uint32_t        num_tiles_x,
    uint32_t        num_tiles_y,
    float           bg_r,
    float           bg_g,
    float           bg_b,
    uint32_t        max_gaussians_per_tile,
    float*          dL_d_rgb_out,
    float*          dL_d_opacity_out,
    float*          dL_d_cov2d_out,
    float*          dL_d_mean2d_out)
{
    @autoreleasepool {
        if (!ensure_init()) {
            fprintf(stderr, "[Metal-GS] Init failed: %s\n", ctx().error_msg.c_str());
            return -1.0;
        }
        MetalContext& c = ctx();

        uint32_t num_tiles     = num_tiles_x * num_tiles_y;
        uint32_t num_pixels    = img_width * img_height;
        size_t means2d_bytes   = (size_t)num_points * 2 * sizeof(float);
        size_t cov2d_bytes     = (size_t)num_points * 3 * sizeof(float);
        size_t colors_bytes    = (size_t)num_points * 3 * sizeof(float);
        size_t opac_bytes      = (size_t)num_points * sizeof(float);
        size_t tbins_bytes     = (size_t)num_tiles * 2 * sizeof(uint32_t);
        size_t plist_bytes     = (size_t)num_intersections * sizeof(uint32_t);
        size_t tfinal_bytes    = (size_t)num_pixels * sizeof(float);
        size_t ncontrib_bytes  = (size_t)num_pixels * sizeof(uint32_t);
        size_t dLdC_bytes      = (size_t)num_pixels * 3 * sizeof(float);

        // Output gradient sizes
        size_t grad_rgb_bytes  = (size_t)num_points * 3 * sizeof(float);
        size_t grad_opac_bytes = (size_t)num_points * sizeof(float);
        size_t grad_cov_bytes  = (size_t)num_points * 3 * sizeof(float);
        size_t grad_mean_bytes = (size_t)num_points * 2 * sizeof(float);

        // ---- Input buffers ----
        id<MTLBuffer> means2d_buf = [c.device newBufferWithBytes:means2d
                                               length:means2d_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> cov2d_buf   = [c.device newBufferWithBytes:cov2d
                                               length:cov2d_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> colors_buf  = [c.device newBufferWithBytes:colors
                                               length:colors_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> opac_buf    = [c.device newBufferWithBytes:opacities
                                               length:opac_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> tbins_buf   = [c.device newBufferWithBytes:tile_bins
                                               length:tbins_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> plist_buf   = [c.device newBufferWithBytes:point_list
                                               length:plist_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> tfinal_buf  = [c.device newBufferWithBytes:T_final
                                               length:tfinal_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> ncontrib_buf= [c.device newBufferWithBytes:n_contrib
                                               length:ncontrib_bytes
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> dLdC_buf    = [c.device newBufferWithBytes:dL_dC_pixel
                                               length:dLdC_bytes
                                               options:MTLResourceStorageModeShared];

        // ---- Output gradient buffers (zero-initialized for atomic adds) ----
        id<MTLBuffer> grad_rgb_buf  = [c.device newBufferWithLength:grad_rgb_bytes
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> grad_opac_buf = [c.device newBufferWithLength:grad_opac_bytes
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> grad_cov_buf  = [c.device newBufferWithLength:grad_cov_bytes
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> grad_mean_buf = [c.device newBufferWithLength:grad_mean_bytes
                                                 options:MTLResourceStorageModeShared];

        // Zero gradient buffers (for atomic adds)
        memset([grad_rgb_buf contents],  0, grad_rgb_bytes);
        memset([grad_opac_buf contents], 0, grad_opac_bytes);
        memset([grad_cov_buf contents],  0, grad_cov_bytes);
        memset([grad_mean_buf contents], 0, grad_mean_bytes);

        RasterizeParams params;
        params.img_width   = img_width;
        params.img_height  = img_height;
        params.num_tiles_x = num_tiles_x;
        params.num_tiles_y = num_tiles_y;
        params.bg_r        = bg_r;
        params.bg_g        = bg_g;
        params.bg_b        = bg_b;
        params.max_gaussians_per_tile = max_gaussians_per_tile;

        // ---- Single command buffer: rasterize backward all tiles ----
        id<MTLCommandBuffer>         cmd = [c.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:c.rasterize_bw_pso];
        [enc setBuffer:means2d_buf   offset:0 atIndex:0];
        [enc setBuffer:cov2d_buf     offset:0 atIndex:1];
        [enc setBuffer:colors_buf    offset:0 atIndex:2];
        [enc setBuffer:opac_buf      offset:0 atIndex:3];
        [enc setBuffer:tbins_buf     offset:0 atIndex:4];
        [enc setBuffer:plist_buf     offset:0 atIndex:5];
        [enc setBuffer:tfinal_buf    offset:0 atIndex:6];
        [enc setBuffer:ncontrib_buf  offset:0 atIndex:7];
        [enc setBuffer:dLdC_buf      offset:0 atIndex:8];
        [enc setBytes:&params        length:sizeof(RasterizeParams) atIndex:9];
        [enc setBuffer:grad_rgb_buf  offset:0 atIndex:10];
        [enc setBuffer:grad_opac_buf offset:0 atIndex:11];
        [enc setBuffer:grad_cov_buf  offset:0 atIndex:12];
        [enc setBuffer:grad_mean_buf offset:0 atIndex:13];

        MTLSize threadgroup_size  = MTLSizeMake(16, 16, 1);
        MTLSize threadgroup_count = MTLSizeMake(num_tiles_x, num_tiles_y, 1);

        [enc dispatchThreadgroups:threadgroup_count
            threadsPerThreadgroup:threadgroup_size];
        [enc endEncoding];

        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        uint64_t t0 = mach_absolute_time();

        [cmd commit];
        [cmd waitUntilCompleted];

        uint64_t t1 = mach_absolute_time();
        double elapsed_ms = (double)(t1 - t0) * tb.numer / tb.denom / 1e6;

        if ([cmd status] == MTLCommandBufferStatusError) {
            fprintf(stderr, "[Metal-GS] Rasterize backward error: %s\n",
                    [[cmd.error localizedDescription] UTF8String]);
            return -1.0;
        }

        memcpy(dL_d_rgb_out,     [grad_rgb_buf contents],  grad_rgb_bytes);
        memcpy(dL_d_opacity_out, [grad_opac_buf contents], grad_opac_bytes);
        memcpy(dL_d_cov2d_out,   [grad_cov_buf contents],  grad_cov_bytes);
        memcpy(dL_d_mean2d_out,  [grad_mean_buf contents], grad_mean_bytes);

        return elapsed_ms;
    }
}

// ---------------------------------------------------------------------------
//  Preprocess backward dispatch (M3 — fused preprocess + cov3d backward)
// ---------------------------------------------------------------------------
double metal_preprocess_backward(
    const float*    means3d,
    const float*    scales,
    const float*    quats,
    const float*    viewmat,
    const uint32_t* radii,
    float           tan_fovx,
    float           tan_fovy,
    float           focal_x,
    float           focal_y,
    float           principal_x,
    float           principal_y,
    uint32_t        img_width,
    uint32_t        img_height,
    uint32_t        N,
    const float*    dL_d_cov2d,
    const float*    dL_d_mean2d,
    float*          dL_d_means3d_out,
    float*          dL_d_scales_out,
    float*          dL_d_quats_out)
{
    @autoreleasepool {
        if (!ensure_init()) {
            fprintf(stderr, "[Metal-GS] Init failed: %s\n", ctx().error_msg.c_str());
            return -1.0;
        }
        MetalContext& c = ctx();

        size_t m3d_bytes   = (size_t)N * 3 * sizeof(float);
        size_t sc_bytes    = (size_t)N * 3 * sizeof(float);
        size_t qt_bytes    = (size_t)N * 4 * sizeof(float);
        size_t vm_bytes    = 16 * sizeof(float);
        size_t rad_bytes   = (size_t)N * sizeof(uint32_t);
        size_t cov_bytes   = (size_t)N * 3 * sizeof(float);
        size_t mean_bytes  = (size_t)N * 2 * sizeof(float);
        size_t out3d_bytes = (size_t)N * 3 * sizeof(float);
        size_t outsc_bytes = (size_t)N * 3 * sizeof(float);
        size_t outqt_bytes = (size_t)N * 4 * sizeof(float);

        id<MTLBuffer> m3d_buf = [c.device newBufferWithBytes:means3d
                                           length:m3d_bytes
                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> sc_buf  = [c.device newBufferWithBytes:scales
                                           length:sc_bytes
                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> qt_buf  = [c.device newBufferWithBytes:quats
                                           length:qt_bytes
                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> vm_buf  = [c.device newBufferWithBytes:viewmat
                                           length:vm_bytes
                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> rad_buf = [c.device newBufferWithBytes:radii
                                           length:rad_bytes
                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> cov_buf = [c.device newBufferWithBytes:dL_d_cov2d
                                           length:cov_bytes
                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> mean_buf= [c.device newBufferWithBytes:dL_d_mean2d
                                           length:mean_bytes
                                           options:MTLResourceStorageModeShared];

        id<MTLBuffer> out3d_buf = [c.device newBufferWithLength:out3d_bytes
                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> outsc_buf = [c.device newBufferWithLength:outsc_bytes
                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> outqt_buf = [c.device newBufferWithLength:outqt_bytes
                                             options:MTLResourceStorageModeShared];

        PreprocessParams params;
        params.tan_fovx     = tan_fovx;
        params.tan_fovy     = tan_fovy;
        params.focal_x      = focal_x;
        params.focal_y      = focal_y;
        params.principal_x  = principal_x;
        params.principal_y  = principal_y;
        params.img_width    = img_width;
        params.img_height   = img_height;
        params.num_points   = N;

        id<MTLCommandBuffer>         cmd = [c.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:c.preprocess_bw_pso];
        [enc setBuffer:m3d_buf   offset:0 atIndex:0];
        [enc setBuffer:sc_buf    offset:0 atIndex:1];
        [enc setBuffer:qt_buf    offset:0 atIndex:2];
        [enc setBuffer:vm_buf    offset:0 atIndex:3];
        [enc setBuffer:rad_buf   offset:0 atIndex:4];
        [enc setBytes:&params    length:sizeof(PreprocessParams) atIndex:5];
        [enc setBuffer:cov_buf   offset:0 atIndex:6];
        [enc setBuffer:mean_buf  offset:0 atIndex:7];
        [enc setBuffer:out3d_buf offset:0 atIndex:8];
        [enc setBuffer:outsc_buf offset:0 atIndex:9];
        [enc setBuffer:outqt_buf offset:0 atIndex:10];

        NSUInteger tg_size = std::min((NSUInteger)256,
                                       c.preprocess_bw_pso.maxTotalThreadsPerThreadgroup);
        NSUInteger grid = N;
        [enc dispatchThreads:MTLSizeMake(grid, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];

        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        uint64_t t0 = mach_absolute_time();
        [cmd commit];
        [cmd waitUntilCompleted];
        uint64_t t1 = mach_absolute_time();
        double elapsed_ms = (double)(t1 - t0) * tb.numer / tb.denom / 1e6;

        if ([cmd status] == MTLCommandBufferStatusError) {
            fprintf(stderr, "[Metal-GS] Preprocess backward error: %s\n",
                    [[cmd.error localizedDescription] UTF8String]);
            return -1.0;
        }

        memcpy(dL_d_means3d_out, [out3d_buf contents], out3d_bytes);
        memcpy(dL_d_scales_out,  [outsc_buf contents],  outsc_bytes);
        memcpy(dL_d_quats_out,   [outqt_buf contents],  outqt_bytes);

        return elapsed_ms;
    }
}

// ---------------------------------------------------------------------------
//  SH backward dispatch (M4)
// ---------------------------------------------------------------------------
double metal_sh_backward(
    const float*    means3d,
    const float*    campos,
    const uint16_t* sh_coeffs,
    const uint16_t* colors_fwd,
    const float*    dL_d_colors,
    uint32_t        N,
    uint32_t        K,
    uint32_t        sh_degree,
    float*          dL_d_sh_out,
    float*          dL_d_means3d_out)
{
    @autoreleasepool {
        if (!ensure_init()) {
            fprintf(stderr, "[Metal-GS] Init failed: %s\n", ctx().error_msg.c_str());
            return -1.0;
        }
        MetalContext& c = ctx();

        size_t m3d_bytes  = (size_t)N * 3 * sizeof(float);
        size_t cam_bytes  = 3 * sizeof(float);
        size_t sh_bytes   = (size_t)N * K * 3 * sizeof(uint16_t);
        size_t col_bytes  = (size_t)N * 3 * sizeof(uint16_t);
        size_t dLc_bytes  = (size_t)N * 3 * sizeof(float);
        size_t dLsh_bytes = (size_t)N * K * 3 * sizeof(float);

        id<MTLBuffer> m3d_buf  = [c.device newBufferWithBytes:means3d
                                            length:m3d_bytes
                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> cam_buf  = [c.device newBufferWithBytes:campos
                                            length:cam_bytes
                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> sh_buf   = [c.device newBufferWithBytes:sh_coeffs
                                            length:sh_bytes
                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> col_buf  = [c.device newBufferWithBytes:colors_fwd
                                            length:col_bytes
                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> dLc_buf  = [c.device newBufferWithBytes:dL_d_colors
                                            length:dLc_bytes
                                            options:MTLResourceStorageModeShared];

        id<MTLBuffer> dLsh_buf = [c.device newBufferWithLength:dLsh_bytes
                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> dLm_buf  = [c.device newBufferWithLength:m3d_bytes
                                            options:MTLResourceStorageModeShared];

        SHParams params;
        params.num_points = N;
        params.num_bases  = K;
        params.sh_degree  = sh_degree;

        id<MTLCommandBuffer>         cmd = [c.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:c.sh_bw_pso];
        [enc setBuffer:m3d_buf  offset:0 atIndex:0];
        [enc setBuffer:cam_buf  offset:0 atIndex:1];
        [enc setBuffer:sh_buf   offset:0 atIndex:2];
        [enc setBuffer:col_buf  offset:0 atIndex:3];
        [enc setBuffer:dLc_buf  offset:0 atIndex:4];
        [enc setBytes:&params   length:sizeof(SHParams) atIndex:5];
        [enc setBuffer:dLsh_buf offset:0 atIndex:6];
        [enc setBuffer:dLm_buf  offset:0 atIndex:7];

        NSUInteger tg_size = std::min((NSUInteger)256,
                                       c.sh_bw_pso.maxTotalThreadsPerThreadgroup);
        NSUInteger grid = N;
        [enc dispatchThreads:MTLSizeMake(grid, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];

        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        uint64_t t0 = mach_absolute_time();
        [cmd commit];
        [cmd waitUntilCompleted];
        uint64_t t1 = mach_absolute_time();
        double elapsed_ms = (double)(t1 - t0) * tb.numer / tb.denom / 1e6;

        if ([cmd status] == MTLCommandBufferStatusError) {
            fprintf(stderr, "[Metal-GS] SH backward error: %s\n",
                    [[cmd.error localizedDescription] UTF8String]);
            return -1.0;
        }

        memcpy(dL_d_sh_out,      [dLsh_buf contents], dLsh_bytes);
        memcpy(dL_d_means3d_out, [dLm_buf contents],  m3d_bytes);

        return elapsed_ms;
    }
}

// ---------------------------------------------------------------------------
//  KNN params must match the struct in simple_knn.metal
// ---------------------------------------------------------------------------
struct KNNParams {
    uint32_t num_points;
    uint32_t search_window;
    uint32_t k_neighbors;
};

// ---------------------------------------------------------------------------
//  Simple KNN using Morton codes + radix sort
//
//  Pipeline (single command buffer):
//    1. CPU: compute bounding box
//    2. GPU: compute_morton_codes → morton_codes[N], indices[N]
//    3. GPU: radix sort by morton code (reuse existing infrastructure)
//    4. GPU: knn_search → avg_sq_dist[N]
// ---------------------------------------------------------------------------
double metal_simple_knn(
    const float*    points,
    float*          avg_sq_dist_out,
    uint32_t        N,
    uint32_t        k_neighbors,
    uint32_t        search_window)
{
    @autoreleasepool {
        if (!ensure_init()) {
            fprintf(stderr, "[Metal-GS] Init failed: %s\n", ctx().error_msg.c_str());
            return -1.0;
        }
        MetalContext& c = ctx();

        if (N == 0) return 0.0;

        // ---- CPU fallback for small arrays ----
        if (N <= 256) {
            mach_timebase_info_data_t tb;
            mach_timebase_info(&tb);
            uint64_t t0 = mach_absolute_time();
            for (uint32_t i = 0; i < N; i++) {
                float px = points[i * 3 + 0];
                float py = points[i * 3 + 1];
                float pz = points[i * 3 + 2];
                // Brute force K nearest
                float best[3] = { 1e30f, 1e30f, 1e30f };
                for (uint32_t j = 0; j < N; j++) {
                    if (j == i) continue;
                    float dx = px - points[j * 3 + 0];
                    float dy = py - points[j * 3 + 1];
                    float dz = pz - points[j * 3 + 2];
                    float sq = dx*dx + dy*dy + dz*dz;
                    if (sq < best[k_neighbors - 1]) {
                        best[k_neighbors - 1] = sq;
                        for (int m = (int)k_neighbors - 2; m >= 0; m--) {
                            if (best[m+1] < best[m]) std::swap(best[m], best[m+1]);
                            else break;
                        }
                    }
                }
                float sum = 0; uint32_t cnt = 0;
                for (uint32_t m = 0; m < k_neighbors; m++) {
                    if (best[m] < 1e29f) { sum += best[m]; cnt++; }
                }
                avg_sq_dist_out[i] = cnt > 0 ? sum / (float)cnt : 0.0f;
            }
            uint64_t t1 = mach_absolute_time();
            return (double)(t1 - t0) * tb.numer / tb.denom / 1e6;
        }

        // ---- Step 1: CPU compute bounding box ----
        float aabb_min[3] = { 1e30f,  1e30f,  1e30f};
        float aabb_max[3] = {-1e30f, -1e30f, -1e30f};
        for (uint32_t i = 0; i < N; i++) {
            for (int d = 0; d < 3; d++) {
                float v = points[i * 3 + d];
                if (v < aabb_min[d]) aabb_min[d] = v;
                if (v > aabb_max[d]) aabb_max[d] = v;
            }
        }
        float aabb_inv[3];
        for (int d = 0; d < 3; d++) {
            float range = aabb_max[d] - aabb_min[d];
            aabb_inv[d] = (range > 1e-8f) ? (1.0f / range) : 0.0f;
        }

        // ---- Allocate buffers ----
        size_t pts_bytes = (size_t)N * 3 * sizeof(float);
        size_t key_bytes = (size_t)N * sizeof(uint32_t);
        size_t out_bytes = (size_t)N * sizeof(float);

        uint32_t num_blocks  = (N + SORT_BLOCK - 1) / SORT_BLOCK;
        uint32_t hist_size   = RADIX_SIZE * num_blocks;
        uint32_t scan_blocks = (hist_size + SCAN_BLOCK - 1) / SCAN_BLOCK;

        size_t hist_bytes = (size_t)hist_size * sizeof(uint32_t);
        size_t sb1_bytes  = (size_t)scan_blocks * sizeof(uint32_t);
        size_t sb2_bytes  = sizeof(uint32_t) * ((scan_blocks + SCAN_BLOCK - 1) / SCAN_BLOCK);
        size_t sb3_bytes  = sizeof(uint32_t) * std::max(1u,
            ((scan_blocks + SCAN_BLOCK - 1) / SCAN_BLOCK + SCAN_BLOCK - 1) / SCAN_BLOCK);

        id<MTLBuffer> pts_buf   = [c.device newBufferWithBytes:points
                                             length:pts_bytes
                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> keys_a    = [c.device newBufferWithLength:key_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> keys_b    = [c.device newBufferWithLength:key_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> vals_a    = [c.device newBufferWithLength:key_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> vals_b    = [c.device newBufferWithLength:key_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> hist_buf  = [c.device newBufferWithLength:hist_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bsum1_buf = [c.device newBufferWithLength:sb1_bytes  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bsum2_buf = [c.device newBufferWithLength:sb2_bytes  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bsum3_buf = [c.device newBufferWithLength:sb3_bytes  options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buf   = [c.device newBufferWithLength:out_bytes  options:MTLResourceStorageModeShared];

        id<MTLBuffer> aabb_min_buf = [c.device newBufferWithBytes:aabb_min
                                                length:3 * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> aabb_inv_buf = [c.device newBufferWithBytes:aabb_inv
                                                length:3 * sizeof(float)
                                                options:MTLResourceStorageModeShared];

        KNNParams knn_params;
        knn_params.num_points    = N;
        knn_params.search_window = search_window;
        knn_params.k_neighbors   = k_neighbors;

        // ---- Single command buffer: morton codes + sort + KNN search ----
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        uint64_t t0 = mach_absolute_time();

        id<MTLCommandBuffer>         cmd = [c.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        // Step 2: compute morton codes + identity indices
        [enc setComputePipelineState:c.morton_codes_pso];
        [enc setBuffer:pts_buf       offset:0 atIndex:0];
        [enc setBuffer:aabb_min_buf  offset:0 atIndex:1];
        [enc setBuffer:aabb_inv_buf  offset:0 atIndex:2];
        [enc setBuffer:keys_a        offset:0 atIndex:3];  // morton codes
        [enc setBuffer:vals_a        offset:0 atIndex:4];  // indices
        [enc setBytes:&knn_params    length:sizeof(KNNParams) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(N, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        // Step 3: radix sort by morton code (8 passes)
        dispatch_radix_sort_kv(c, enc,
                               keys_a, keys_b, vals_a, vals_b,
                               hist_buf, bsum1_buf, bsum2_buf, bsum3_buf,
                               N, 8);

        // Step 4: KNN search in sorted order
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:c.knn_search_pso];
        [enc setBuffer:vals_a    offset:0 atIndex:0];  // sorted indices
        [enc setBuffer:pts_buf   offset:0 atIndex:1];  // original points
        [enc setBuffer:out_buf   offset:0 atIndex:2];  // avg_sq_dist output
        [enc setBytes:&knn_params length:sizeof(KNNParams) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(N, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        uint64_t t1 = mach_absolute_time();
        double elapsed_ms = (double)(t1 - t0) * tb.numer / tb.denom / 1e6;

        if ([cmd status] == MTLCommandBufferStatusError) {
            fprintf(stderr, "[Metal-GS] KNN error: %s\n",
                    [[cmd.error localizedDescription] UTF8String]);
            return -1.0;
        }

        memcpy(avg_sq_dist_out, [out_buf contents], out_bytes);
        return elapsed_ms;
    }
}
