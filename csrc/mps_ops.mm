// ============================================================================
//  Metal-GS v3: MPS Custom Op Dispatch Layer
//
//  Architecture:
//    Phase 1 (torch ops):  direction computation, fp16→fp32 color conversion
//    Phase 2 (Metal enc):  preprocess → sort → tile_count → prefix_sum
//                         [sync: read num_isect scalar from GPU]
//    Phase 3 (Metal enc):  prefix_sum offsets → gen_isect → sort → tile_range → rasterize
//    Backward (Metal enc): rasterize_bw → preprocess_bw → sh_bw (single encoder)
//
//  Zero-copy: extract id<MTLBuffer> from MPS tensors via getMTLBufferStorage()
//  PSOs created on PyTorch's MPS device (same command queue).
//
//  Memory management (non-ARC / -fno-objc-arc):
//    - Phase 2 scratch: explicit [release] after COMMIT_AND_WAIT (GPU done)
//    - Phase 3 scratch: deferred release via s_deferred_release (freed at
//      next iteration's sync point when GPU is guaranteed idle)
//    - Backward: all tensors PyTorch-managed, no raw buffer allocations
//    - CRITICAL: never call torch ops (e.g. .to()) while holding an
//      encoder handle — MPS stream may invalidate the encoder.
// ============================================================================

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#include <algorithm>
#include <cstdio>
#include <string>

// ---------------------------------------------------------------------------
//  Parameter structs (must match .metal kernel structs byte-for-byte)
// ---------------------------------------------------------------------------
struct SHParams {
    uint32_t num_points;
    uint32_t num_bases;
    uint32_t sh_degree;
};

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
    uint32_t max_gaussians_per_tile;
};

struct TileCountParams {
    uint32_t num_points;
    uint32_t num_tiles_x;
    uint32_t num_tiles_y;
};

struct ReduceParams {
    uint32_t num_elements;
};

struct KNNParams {
    uint32_t num_points;
    uint32_t search_window;
    uint32_t k_neighbors;
};

// ---------------------------------------------------------------------------
//  Singleton PSO registry — uses PyTorch's MPS device
// ---------------------------------------------------------------------------
namespace {

static const uint32_t RADIX_SIZE  = 16;
static const uint32_t SORT_BLOCK  = 256;
static const uint32_t SCAN_BLOCK  = 256;

struct MPSMetalContext {
    id<MTLDevice>               device            = nil;
    id<MTLComputePipelineState> sh_fwd_pso        = nil;
    id<MTLComputePipelineState> preprocess_pso    = nil;
    id<MTLComputePipelineState> gen_keys_pso      = nil;
    id<MTLComputePipelineState> histogram_pso     = nil;
    id<MTLComputePipelineState> scan_pso          = nil;
    id<MTLComputePipelineState> scan_add_pso      = nil;
    id<MTLComputePipelineState> scatter_pso       = nil;
    id<MTLComputePipelineState> gen_isect_pso     = nil;
    id<MTLComputePipelineState> tile_range_pso    = nil;
    id<MTLComputePipelineState> rasterize_pso     = nil;
    id<MTLComputePipelineState> rasterize_bw_pso  = nil;
    id<MTLComputePipelineState> preprocess_bw_pso = nil;
    id<MTLComputePipelineState> sh_bw_pso         = nil;
    id<MTLComputePipelineState> morton_codes_pso  = nil;
    id<MTLComputePipelineState> knn_search_pso    = nil;
    id<MTLComputePipelineState> tile_count_pso    = nil;
    id<MTLComputePipelineState> reduce_sum_pso    = nil;
    bool                        initialised       = false;
    std::string                 error_msg;
};

MPSMetalContext& mps_ctx() {
    static MPSMetalContext c;
    return c;
}

std::string find_metallib_path() {
    @autoreleasepool {
        std::string src_dir = __FILE__;
        auto slash = src_dir.rfind('/');
        if (slash != std::string::npos) {
            std::string candidate = src_dir.substr(0, slash) + "/kernels/metal_gs.metallib";
            if ([[NSFileManager defaultManager]
                    fileExistsAtPath:[NSString stringWithUTF8String:candidate.c_str()]]) {
                return candidate;
            }
        }
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

bool ensure_mps_init() {
    MPSMetalContext& c = mps_ctx();
    if (c.initialised) return true;

    @autoreleasepool {
        c.device = at::mps::getCurrentMPSStream()->device();
        if (!c.device) {
            c.error_msg = "Cannot get MPS device from PyTorch";
            return false;
        }

        std::string lib_path = find_metallib_path();
        if (lib_path.empty()) {
            c.error_msg = "Cannot find metal_gs.metallib";
            return false;
        }

        NSError* err = nil;
        id<MTLLibrary> library = [c.device
            newLibraryWithURL:[NSURL fileURLWithPath:
                [NSString stringWithUTF8String:lib_path.c_str()]]
            error:&err];
        if (!library) {
            c.error_msg = std::string("Failed to load metallib: ") +
                          [[err localizedDescription] UTF8String];
            return false;
        }

        fprintf(stderr, "[Metal-GS v3] Loaded metallib: %s (MPS device)\n", lib_path.c_str());

        auto make_pso = [&](const char* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [library newFunctionWithName:
                [NSString stringWithUTF8String:name]];
            if (!fn) { c.error_msg = std::string("Function '") + name + "' not found"; return nil; }
            NSError* e2 = nil;
            auto p = [c.device newComputePipelineStateWithFunction:fn error:&e2];
            if (!p) { c.error_msg = std::string("PSO failed: '") + name + "'"; return nil; }
            return p;
        };

        c.sh_fwd_pso        = make_pso("compute_sh_forward");      if (!c.sh_fwd_pso)        return false;
        c.preprocess_pso    = make_pso("preprocess_forward");      if (!c.preprocess_pso)    return false;
        c.gen_keys_pso      = make_pso("radix_generate_keys");     if (!c.gen_keys_pso)      return false;
        c.histogram_pso     = make_pso("radix_histogram");         if (!c.histogram_pso)     return false;
        c.scan_pso          = make_pso("scan_exclusive");          if (!c.scan_pso)          return false;
        c.scan_add_pso      = make_pso("scan_add_block_sums");     if (!c.scan_add_pso)      return false;
        c.scatter_pso       = make_pso("radix_scatter_stable");    if (!c.scatter_pso)       return false;
        c.gen_isect_pso     = make_pso("generate_intersections");  if (!c.gen_isect_pso)     return false;
        c.tile_range_pso    = make_pso("identify_tile_ranges");    if (!c.tile_range_pso)    return false;
        c.rasterize_pso     = make_pso("rasterize_forward");       if (!c.rasterize_pso)     return false;
        c.rasterize_bw_pso  = make_pso("rasterize_backward");      if (!c.rasterize_bw_pso)  return false;
        c.preprocess_bw_pso = make_pso("preprocess_backward");     if (!c.preprocess_bw_pso) return false;
        c.sh_bw_pso         = make_pso("sh_backward");             if (!c.sh_bw_pso)         return false;
        c.morton_codes_pso  = make_pso("compute_morton_codes");     if (!c.morton_codes_pso)  return false;
        c.knn_search_pso    = make_pso("knn_search");              if (!c.knn_search_pso)    return false;
        c.tile_count_pso    = make_pso("count_tile_intersections"); if (!c.tile_count_pso)    return false;
        c.reduce_sum_pso    = make_pso("reduce_sum_uint");          if (!c.reduce_sum_pso)    return false;

        fprintf(stderr, "[Metal-GS v3] All 17 PSOs created (MPS device)\n");
        c.initialised = true;
        return true;
    }
}

// ---------------------------------------------------------------------------
//  Helpers
// ---------------------------------------------------------------------------
static inline void set_buf(id<MTLComputeCommandEncoder> enc,
                           const torch::Tensor& t, uint32_t idx) {
    [enc setBuffer:at::native::mps::getMTLBufferStorage(t)
           offset:t.storage_offset() * t.element_size()
          atIndex:idx];
}

static inline id<MTLBuffer> get_buf(const torch::Tensor& t) {
    return at::native::mps::getMTLBufferStorage(t);
}

static id<MTLBuffer> alloc_private(size_t bytes) {
    return [mps_ctx().device newBufferWithLength:bytes
                     options:MTLResourceStorageModePrivate];
}

static id<MTLBuffer> alloc_shared(size_t bytes) {
    return [mps_ctx().device newBufferWithLength:bytes
                     options:MTLResourceStorageModeShared];
}

// ---------------------------------------------------------------------------
//  Radix sort: prefix sum on single encoder
// ---------------------------------------------------------------------------
static void prefix_sum(MPSMetalContext& c, id<MTLComputeCommandEncoder> enc,
                       id<MTLBuffer> data, uint32_t n,
                       id<MTLBuffer> b1, id<MTLBuffer> b2, id<MTLBuffer> b3)
{
    uint32_t nblk1 = (n + SCAN_BLOCK - 1) / SCAN_BLOCK;
    uint32_t nblk2 = (nblk1 + SCAN_BLOCK - 1) / SCAN_BLOCK;
    uint32_t nblk3 = (nblk2 + SCAN_BLOCK - 1) / SCAN_BLOCK;
    auto tg = MTLSizeMake(SCAN_BLOCK, 1, 1);

    auto scan_level = [&](id<MTLBuffer> d, id<MTLBuffer> bs, uint32_t cnt) {
        ScanParams sp = { cnt };
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:c.scan_pso];
        [enc setBuffer:d  offset:0 atIndex:0];
        [enc setBuffer:bs offset:0 atIndex:1];
        [enc setBytes:&sp length:sizeof(sp) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((cnt + SCAN_BLOCK - 1) / SCAN_BLOCK, 1, 1)
            threadsPerThreadgroup:tg];
    };

    auto add_level = [&](id<MTLBuffer> d, id<MTLBuffer> bs, uint32_t cnt) {
        ScanParams sp = { cnt };
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:c.scan_add_pso];
        [enc setBuffer:d  offset:0 atIndex:0];
        [enc setBuffer:bs offset:0 atIndex:1];
        [enc setBytes:&sp length:sizeof(sp) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((cnt + SCAN_BLOCK - 1) / SCAN_BLOCK, 1, 1)
            threadsPerThreadgroup:tg];
    };

    scan_level(data, b1, n);
    scan_level(b1, b2, nblk1);
    if (nblk2 > 1) {
        scan_level(b2, b3, nblk2);
        add_level(b1, b2, nblk1);
    }
    if (nblk1 > 1) {
        add_level(data, b1, n);
    }
}

// ---------------------------------------------------------------------------
//  Radix sort: one pass (histogram → prefix sum → scatter)
// ---------------------------------------------------------------------------
static void radix_pass(MPSMetalContext& c, id<MTLComputeCommandEncoder> enc,
                       id<MTLBuffer> k_src, id<MTLBuffer> k_dst,
                       id<MTLBuffer> v_src, id<MTLBuffer> v_dst,
                       id<MTLBuffer> hist, id<MTLBuffer> b1,
                       id<MTLBuffer> b2, id<MTLBuffer> b3,
                       uint32_t N, uint32_t pass_idx)
{
    uint32_t num_blocks = (N + SORT_BLOCK - 1) / SORT_BLOCK;
    SortParams sp = { N, num_blocks, pass_idx * 4 };

    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [enc setComputePipelineState:c.histogram_pso];
    [enc setBuffer:k_src offset:0 atIndex:0];
    [enc setBuffer:hist  offset:0 atIndex:1];
    [enc setBytes:&sp    length:sizeof(sp) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(N, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(SORT_BLOCK, 1, 1)];

    prefix_sum(c, enc, hist, RADIX_SIZE * num_blocks, b1, b2, b3);

    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [enc setComputePipelineState:c.scatter_pso];
    [enc setBuffer:k_src offset:0 atIndex:0];
    [enc setBuffer:v_src offset:0 atIndex:1];
    [enc setBuffer:k_dst offset:0 atIndex:2];
    [enc setBuffer:v_dst offset:0 atIndex:3];
    [enc setBuffer:hist  offset:0 atIndex:4];
    [enc setBytes:&sp    length:sizeof(sp) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(N, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(SORT_BLOCK, 1, 1)];
}

// ---------------------------------------------------------------------------
//  Radix sort: 8-pass full sort (results end in keys_a, vals_a)
// ---------------------------------------------------------------------------
static void radix_sort(MPSMetalContext& c, id<MTLComputeCommandEncoder> enc,
                       id<MTLBuffer> ka, id<MTLBuffer> kb,
                       id<MTLBuffer> va, id<MTLBuffer> vb,
                       id<MTLBuffer> hist, id<MTLBuffer> b1,
                       id<MTLBuffer> b2, id<MTLBuffer> b3,
                       uint32_t N, uint32_t num_passes)
{
    id<MTLBuffer> ks = ka, kd = kb, vs = va, vd = vb;
    for (uint32_t p = 0; p < num_passes; p++) {
        radix_pass(c, enc, ks, kd, vs, vd, hist, b1, b2, b3, N, p);
        std::swap(ks, kd);
        std::swap(vs, vd);
    }
}

// ---------------------------------------------------------------------------
//  Sort scratch buffer allocation helper
// ---------------------------------------------------------------------------
struct SortScratch {
    id<MTLBuffer> keys_a, keys_b, vals_a, vals_b;
    id<MTLBuffer> hist, b1, b2, b3;

    void release_all() {
        [keys_a release]; [keys_b release];
        [vals_a release]; [vals_b release];
        [hist release]; [b1 release]; [b2 release]; [b3 release];
    }
};

static SortScratch alloc_sort_scratch(uint32_t N) {
    uint32_t nb  = (N + SORT_BLOCK - 1) / SORT_BLOCK;
    uint32_t hs  = RADIX_SIZE * nb;
    uint32_t sb1 = (hs + SCAN_BLOCK - 1) / SCAN_BLOCK;
    uint32_t sb2 = (sb1 + SCAN_BLOCK - 1) / SCAN_BLOCK;
    uint32_t sb3 = std::max(1u, (sb2 + SCAN_BLOCK - 1) / SCAN_BLOCK);
    size_t k = (size_t)N * sizeof(uint32_t);
    return {
        alloc_private(k), alloc_private(k),
        alloc_private(k), alloc_private(k),
        alloc_private(hs * sizeof(uint32_t)),
        alloc_private(sb1 * sizeof(uint32_t)),
        alloc_private(sb2 * sizeof(uint32_t)),
        alloc_private(sb3 * sizeof(uint32_t))
    };
}

// ---------------------------------------------------------------------------
//  Deferred release: Phase 3 buffers freed at next sync point
// ---------------------------------------------------------------------------
static std::vector<id<MTLBuffer>> s_deferred_release;

static void flush_deferred_release() {
    for (auto buf : s_deferred_release) {
        if (buf) [buf release];
    }
    s_deferred_release.clear();
}

}  // anonymous namespace


// ============================================================================
//  PUBLIC API
// ============================================================================

// ---------------------------------------------------------------------------
//  mps_render_forward
//    Full pipeline: SH → preprocess → sort → tile_bin → rasterize
//    Returns vector of tensors for backward pass
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> mps_render_forward(
    const torch::Tensor& means3d,      // [N, 3] f32 mps
    const torch::Tensor& scales,        // [N, 3] f32
    const torch::Tensor& quats,         // [N, 4] f32
    const torch::Tensor& sh_coeffs,     // [N, K, 3] f16
    const torch::Tensor& opacities,     // [N] f32
    const torch::Tensor& viewmat,       // [4, 4] f32
    const torch::Tensor& campos,        // [3] f32
    float tan_fovx, float tan_fovy,
    float focal_x, float focal_y,
    float principal_x, float principal_y,
    int64_t img_width, int64_t img_height,
    int64_t sh_degree,
    float bg_r, float bg_g, float bg_b,
    int64_t max_gaussians_per_tile)
{
    @autoreleasepool {
        TORCH_CHECK(ensure_mps_init(), mps_ctx().error_msg);
        MPSMetalContext& c = mps_ctx();
        auto* stream = at::mps::getCurrentMPSStream();

        uint32_t N = (uint32_t)means3d.size(0);
        uint32_t K = (uint32_t)sh_coeffs.size(1);
        uint32_t ntx = ((uint32_t)img_width  + 15) / 16;
        uint32_t nty = ((uint32_t)img_height + 15) / 16;
        uint32_t num_tiles = ntx * nty;

        auto f32 = means3d.options();
        auto u32 = f32.dtype(torch::kInt32);
        auto f16 = f32.dtype(torch::kFloat16);

        // ── Phase 1: torch ops (async on MPS command queue) ─────────────
        // Directions + SH forward + fp16→fp32 color cast
        torch::Tensor dirs = means3d - campos.unsqueeze(0);
        torch::Tensor directions = dirs / dirs.norm(2, 1, true).clamp_min(1e-8f);

        // SH kernel expects half* — convert float32 sh_coeffs to float16
        torch::Tensor sh_fp16 = sh_coeffs.to(torch::kFloat16);

        torch::Tensor colors_fp16 = torch::empty({(int64_t)N, 3}, f16);
        {
            // Encode SH forward kernel via MPS stream
            id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
            SHParams sp = { N, K, (uint32_t)sh_degree };
            [enc setComputePipelineState:c.sh_fwd_pso];
            set_buf(enc, directions, 0);
            set_buf(enc, sh_fp16, 1);
            set_buf(enc, colors_fp16, 2);
            [enc setBytes:&sp length:sizeof(sp) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(N, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            // Don't endEncoding — stream manages encoder lifecycle
        }

        // fp16 → fp32 color conversion (torch MPS op, async)
        torch::Tensor colors_fp32 = colors_fp16.to(torch::kFloat32);

        // ── Phase 2: preprocess → sort → tile_count ─────────────────────
        torch::Tensor means2d  = torch::empty({(int64_t)N, 2}, f32);
        torch::Tensor cov2d    = torch::empty({(int64_t)N, 3}, f32);
        torch::Tensor depths   = torch::empty({(int64_t)N}, f32);
        torch::Tensor radii    = torch::empty({(int64_t)N}, u32);
        torch::Tensor tile_min = torch::empty({(int64_t)N, 2}, u32);
        torch::Tensor tile_max = torch::empty({(int64_t)N, 2}, u32);

        // Pre-allocate sorted_indices as tensor so sort writes directly into it
        torch::Tensor sorted_indices = torch::empty({(int64_t)N}, u32);
        torch::Tensor tile_counts    = torch::empty({(int64_t)N}, u32);

        // Sort scratch: use sorted_indices tensor as vals_a, alloc rest as private
        uint32_t nb  = (N + SORT_BLOCK - 1) / SORT_BLOCK;
        uint32_t hs  = RADIX_SIZE * nb;
        uint32_t sb1 = (hs + SCAN_BLOCK - 1) / SCAN_BLOCK;
        uint32_t sb2 = (sb1 + SCAN_BLOCK - 1) / SCAN_BLOCK;
        uint32_t sb3 = std::max(1u, (sb2 + SCAN_BLOCK - 1) / SCAN_BLOCK);
        size_t k_sz  = (size_t)N * sizeof(uint32_t);

        id<MTLBuffer> sort_keys_a = alloc_private(k_sz);
        id<MTLBuffer> sort_keys_b = alloc_private(k_sz);
        id<MTLBuffer> sort_vals_b = alloc_private(k_sz);
        id<MTLBuffer> sort_hist   = alloc_private(hs * sizeof(uint32_t));
        id<MTLBuffer> sort_b1     = alloc_private(sb1 * sizeof(uint32_t));
        id<MTLBuffer> sort_b2     = alloc_private(sb2 * sizeof(uint32_t));
        id<MTLBuffer> sort_b3     = alloc_private(sb3 * sizeof(uint32_t));

        // Block sums for reduce (Shared for CPU readback)
        uint32_t reduce_tgs = (N + 255) / 256;
        id<MTLBuffer> block_sums = alloc_shared(reduce_tgs * sizeof(uint32_t));

        {
            id<MTLComputeCommandEncoder> enc = stream->commandEncoder();

            // Preprocess
            PreprocessParams pp = {
                tan_fovx, tan_fovy, focal_x, focal_y,
                principal_x, principal_y,
                (uint32_t)img_width, (uint32_t)img_height, N
            };
            [enc setComputePipelineState:c.preprocess_pso];
            set_buf(enc, means3d, 0);
            set_buf(enc, scales, 1);
            set_buf(enc, quats, 2);
            set_buf(enc, viewmat, 3);
            [enc setBytes:&pp length:sizeof(pp) atIndex:4];
            set_buf(enc, means2d, 5);
            set_buf(enc, cov2d, 6);
            set_buf(enc, depths, 7);
            set_buf(enc, radii, 8);
            set_buf(enc, tile_min, 9);
            set_buf(enc, tile_max, 10);
            [enc dispatchThreads:MTLSizeMake(N, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Depth sort: generate keys + 8-pass radix sort
            // vals_a = sorted_indices tensor buffer (result ends here after even # passes)
            id<MTLBuffer> sort_vals_a = get_buf(sorted_indices);
            uint32_t N32 = N;
            [enc setComputePipelineState:c.gen_keys_pso];
            set_buf(enc, depths, 0);
            [enc setBuffer:sort_keys_a offset:0 atIndex:1];
            [enc setBuffer:sort_vals_a offset:0 atIndex:2];
            [enc setBytes:&N32 length:sizeof(N32) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(N, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(SORT_BLOCK, 1, 1)];

            radix_sort(c, enc, sort_keys_a, sort_keys_b, sort_vals_a, sort_vals_b,
                       sort_hist, sort_b1, sort_b2, sort_b3, N, 8);

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Tile count: per-Gaussian tile intersection count
            TileCountParams tcp = { N, ntx, nty };
            [enc setComputePipelineState:c.tile_count_pso];
            set_buf(enc, sorted_indices, 0);  // sort result is in the tensor
            set_buf(enc, radii, 1);
            set_buf(enc, tile_min, 2);
            set_buf(enc, tile_max, 3);
            set_buf(enc, tile_counts, 4);
            [enc setBytes:&tcp length:sizeof(tcp) atIndex:5];
            [enc dispatchThreads:MTLSizeMake(N, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            // Reduce to get total num_intersections (need CPU readback)
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            ReduceParams rp = { N };
            [enc setComputePipelineState:c.reduce_sum_pso];
            set_buf(enc, tile_counts, 0);
            [enc setBuffer:block_sums offset:0 atIndex:1];
            [enc setBytes:&rp length:sizeof(rp) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(reduce_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            // Let stream end encoder and commit+wait
        }

        // === SYNC: read num_isect scalar (unavoidable) ===
        stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

        // GPU done — release Phase 2 scratch + any deferred Phase 3 from last iteration
        flush_deferred_release();
        [sort_keys_a release]; [sort_keys_b release]; [sort_vals_b release];
        [sort_hist release]; [sort_b1 release]; [sort_b2 release]; [sort_b3 release];

        uint32_t num_isect = 0;
        {
            uint32_t* bs = (uint32_t*)[block_sums contents];
            for (uint32_t i = 0; i < reduce_tgs; i++) num_isect += bs[i];
        }
        [block_sums release];  // CPU readback done, safe to free

        // ── Phase 3: prefix_sum → gen_isect → sort → tile_range → rasterize ──
        torch::Tensor tile_bins  = torch::zeros({(int64_t)num_tiles, 2}, u32);
        torch::Tensor point_list = torch::empty({std::max((int64_t)num_isect, (int64_t)1)}, u32);
        torch::Tensor out_img    = torch::empty({(int64_t)img_height, (int64_t)img_width, 3}, f32);
        torch::Tensor T_final    = torch::empty({(int64_t)img_height, (int64_t)img_width}, f32);
        torch::Tensor n_contrib  = torch::zeros({(int64_t)img_height, (int64_t)img_width}, u32);

        if (num_isect > 0) {
            id<MTLComputeCommandEncoder> enc = stream->commandEncoder();

            // Prefix sum tile_counts → offsets
            {
                uint32_t ps1 = (N + SCAN_BLOCK - 1) / SCAN_BLOCK;
                uint32_t ps2 = (ps1 + SCAN_BLOCK - 1) / SCAN_BLOCK;
                id<MTLBuffer> pb1 = alloc_private(ps1 * sizeof(uint32_t));
                id<MTLBuffer> pb2 = alloc_private(ps2 * sizeof(uint32_t));
                id<MTLBuffer> pb3 = alloc_private(
                    std::max(1u, (ps2 + SCAN_BLOCK - 1) / SCAN_BLOCK) * sizeof(uint32_t));
                prefix_sum(c, enc, get_buf(tile_counts), N, pb1, pb2, pb3);
                // Defer release — encoder still references these buffers
                s_deferred_release.push_back(pb1);
                s_deferred_release.push_back(pb2);
                s_deferred_release.push_back(pb3);
            }

            // Determine radix passes for tile sort
            uint32_t max_tid = num_tiles - 1;
            uint32_t bits = 0;
            { uint32_t t = max_tid; while (t) { bits++; t >>= 1; } }
            uint32_t tile_passes = (bits + 3) / 4;
            if (tile_passes == 0) tile_passes = 2;
            if (tile_passes % 2) tile_passes++;

            // Intersection sort scratch — use point_list tensor as vals_a
            size_t ib = (size_t)num_isect * sizeof(uint32_t);
            uint32_t snb  = (num_isect + SORT_BLOCK - 1) / SORT_BLOCK;
            uint32_t shs  = RADIX_SIZE * snb;
            uint32_t ssb1 = (shs + SCAN_BLOCK - 1) / SCAN_BLOCK;
            uint32_t ssb2 = (ssb1 + SCAN_BLOCK - 1) / SCAN_BLOCK;
            id<MTLBuffer> ika = alloc_private(ib);
            id<MTLBuffer> ikb = alloc_private(ib);
            id<MTLBuffer> ivb = alloc_private(ib);
            id<MTLBuffer> ih  = alloc_private(shs * sizeof(uint32_t));
            id<MTLBuffer> ib1 = alloc_private(ssb1 * sizeof(uint32_t));
            id<MTLBuffer> ib2 = alloc_private(ssb2 * sizeof(uint32_t));
            id<MTLBuffer> ib3 = alloc_private(
                std::max(1u, (ssb2 + SCAN_BLOCK - 1) / SCAN_BLOCK) * sizeof(uint32_t));

            // Generate intersections (using prefix-summed offsets in tile_counts)
            id<MTLBuffer> iva = get_buf(point_list);  // sort result → point_list tensor
            BinningParams bp = { N, ntx, nty, num_isect };
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:c.gen_isect_pso];
            set_buf(enc, sorted_indices, 0);
            set_buf(enc, radii, 1);
            set_buf(enc, tile_min, 2);
            set_buf(enc, tile_max, 3);
            set_buf(enc, tile_counts, 4);  // now contains offsets
            [enc setBuffer:ika offset:0 atIndex:5];
            [enc setBuffer:iva offset:0 atIndex:6];
            [enc setBytes:&bp length:sizeof(bp) atIndex:7];
            [enc dispatchThreads:MTLSizeMake(N, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(SORT_BLOCK, 1, 1)];

            // Sort intersections by tile_id
            radix_sort(c, enc, ika, ikb, iva, ivb, ih, ib1, ib2, ib3,
                       num_isect, tile_passes);

            // Identify tile ranges
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:c.tile_range_pso];
            [enc setBuffer:ika offset:0 atIndex:0];
            set_buf(enc, tile_bins, 1);
            [enc setBytes:&num_isect length:sizeof(num_isect) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(num_isect, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(SORT_BLOCK, 1, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Rasterize forward
            RasterizeParams ras;
            ras.img_width   = (uint32_t)img_width;
            ras.img_height  = (uint32_t)img_height;
            ras.num_tiles_x = ntx;
            ras.num_tiles_y = nty;
            ras.bg_r = bg_r; ras.bg_g = bg_g; ras.bg_b = bg_b;
            ras.max_gaussians_per_tile = (uint32_t)max_gaussians_per_tile;

            [enc setComputePipelineState:c.rasterize_pso];
            set_buf(enc, means2d, 0);
            set_buf(enc, cov2d, 1);
            set_buf(enc, colors_fp32, 2);
            set_buf(enc, opacities, 3);
            set_buf(enc, tile_bins, 4);
            set_buf(enc, point_list, 5);  // sort result is in the tensor
            [enc setBytes:&ras length:sizeof(ras) atIndex:6];
            set_buf(enc, out_img, 7);
            set_buf(enc, T_final, 8);
            set_buf(enc, n_contrib, 9);

            [enc dispatchThreadgroups:MTLSizeMake(ntx, nty, 1)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

            // Defer release of Phase 3 scratch — GPU hasn't finished yet
            // Will be freed at next iteration's sync point
            s_deferred_release.push_back(ika);
            s_deferred_release.push_back(ikb);
            s_deferred_release.push_back(ivb);
            s_deferred_release.push_back(ih);
            s_deferred_release.push_back(ib1);
            s_deferred_release.push_back(ib2);
            s_deferred_release.push_back(ib3);

            // Don't end encoder — stream manages lifecycle
        } else {
            // No intersections → fill bg
            out_img.select(2, 0).fill_(bg_r);
            out_img.select(2, 1).fill_(bg_g);
            out_img.select(2, 2).fill_(bg_b);
            T_final.fill_(1.0f);
        }

        return {out_img, T_final, n_contrib,
                means2d, cov2d, depths, radii,
                tile_bins, point_list, sorted_indices,
                colors_fp16, colors_fp32, directions};
    }
}


// ---------------------------------------------------------------------------
//  mps_render_backward
//    rasterize_bw → preprocess_bw → sh_bw (single encoder, no syncs)
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> mps_render_backward(
    const torch::Tensor& means3d,
    const torch::Tensor& scales,
    const torch::Tensor& quats,
    const torch::Tensor& sh_coeffs,
    const torch::Tensor& opacities,
    const torch::Tensor& viewmat,
    const torch::Tensor& campos,
    const torch::Tensor& means2d,
    const torch::Tensor& cov2d,
    const torch::Tensor& radii,
    const torch::Tensor& colors_fp32,
    const torch::Tensor& colors_fp16,
    const torch::Tensor& tile_bins,
    const torch::Tensor& point_list,
    const torch::Tensor& T_final,
    const torch::Tensor& n_contrib,
    const torch::Tensor& dL_dC_pixel,
    float tan_fovx, float tan_fovy,
    float focal_x, float focal_y,
    float principal_x, float principal_y,
    int64_t img_width, int64_t img_height,
    int64_t sh_degree,
    float bg_r, float bg_g, float bg_b,
    int64_t max_gaussians_per_tile)
{
    @autoreleasepool {
        TORCH_CHECK(ensure_mps_init(), mps_ctx().error_msg);
        MPSMetalContext& c = mps_ctx();
        auto* stream = at::mps::getCurrentMPSStream();

        uint32_t N   = (uint32_t)means3d.size(0);
        uint32_t K   = (uint32_t)sh_coeffs.size(1);
        uint32_t ntx = ((uint32_t)img_width  + 15) / 16;
        uint32_t nty = ((uint32_t)img_height + 15) / 16;
        uint32_t num_isect = (uint32_t)point_list.size(0);

        auto f32 = means3d.options();

        // Gradient outputs (zero-init for atomics in rasterize_bw)
        torch::Tensor dL_rgb   = torch::zeros({(int64_t)N, 3}, f32);
        torch::Tensor dL_opac  = torch::zeros({(int64_t)N},    f32);
        torch::Tensor dL_cov   = torch::zeros({(int64_t)N, 3}, f32);
        torch::Tensor dL_mean  = torch::zeros({(int64_t)N, 2}, f32);
        torch::Tensor dL_m3d_p = torch::empty({(int64_t)N, 3}, f32);
        torch::Tensor dL_sc    = torch::empty({(int64_t)N, 3}, f32);
        torch::Tensor dL_qt    = torch::empty({(int64_t)N, 4}, f32);
        torch::Tensor dL_sh    = torch::empty({(int64_t)N, (int64_t)K, 3}, f32);
        torch::Tensor dL_m3d_s = torch::empty({(int64_t)N, 3}, f32);

        // Pre-compute SH fp16 cast BEFORE opening encoder — PyTorch MPS ops
        // can invalidate the current compute command encoder, so all torch ops
        // must complete before we grab the encoder handle.
        torch::Tensor sh_fp16 = sh_coeffs.to(torch::kFloat16);

        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();

        // Stage 1: rasterize backward
        if (num_isect > 0) {
            RasterizeParams rp;
            rp.img_width = (uint32_t)img_width;
            rp.img_height = (uint32_t)img_height;
            rp.num_tiles_x = ntx;
            rp.num_tiles_y = nty;
            rp.bg_r = bg_r; rp.bg_g = bg_g; rp.bg_b = bg_b;
            rp.max_gaussians_per_tile = (uint32_t)max_gaussians_per_tile;

            [enc setComputePipelineState:c.rasterize_bw_pso];
            set_buf(enc, means2d, 0);
            set_buf(enc, cov2d, 1);
            set_buf(enc, colors_fp32, 2);
            set_buf(enc, opacities, 3);
            set_buf(enc, tile_bins, 4);
            set_buf(enc, point_list, 5);
            set_buf(enc, T_final, 6);
            set_buf(enc, n_contrib, 7);
            set_buf(enc, dL_dC_pixel, 8);
            [enc setBytes:&rp length:sizeof(rp) atIndex:9];
            set_buf(enc, dL_rgb, 10);
            set_buf(enc, dL_opac, 11);
            set_buf(enc, dL_cov, 12);
            set_buf(enc, dL_mean, 13);

            [enc dispatchThreadgroups:MTLSizeMake(ntx, nty, 1)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        }

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // Stage 2: preprocess backward
        {
            PreprocessParams pp = {
                tan_fovx, tan_fovy, focal_x, focal_y,
                principal_x, principal_y,
                (uint32_t)img_width, (uint32_t)img_height, N
            };
            [enc setComputePipelineState:c.preprocess_bw_pso];
            set_buf(enc, means3d, 0);
            set_buf(enc, scales, 1);
            set_buf(enc, quats, 2);
            set_buf(enc, viewmat, 3);
            set_buf(enc, radii, 4);
            [enc setBytes:&pp length:sizeof(pp) atIndex:5];
            set_buf(enc, dL_cov, 6);
            set_buf(enc, dL_mean, 7);
            set_buf(enc, dL_m3d_p, 8);
            set_buf(enc, dL_sc, 9);
            set_buf(enc, dL_qt, 10);

            [enc dispatchThreads:MTLSizeMake(N, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // Stage 3: SH backward (sh_fp16 pre-computed above)
        {
            SHParams sp = { N, K, (uint32_t)sh_degree };
            [enc setComputePipelineState:c.sh_bw_pso];
            set_buf(enc, means3d, 0);
            set_buf(enc, campos, 1);
            set_buf(enc, sh_fp16, 2);
            set_buf(enc, colors_fp16, 3);
            set_buf(enc, dL_rgb, 4);
            [enc setBytes:&sp length:sizeof(sp) atIndex:5];
            set_buf(enc, dL_sh, 6);
            set_buf(enc, dL_m3d_s, 7);

            [enc dispatchThreads:MTLSizeMake(N, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        // Don't end encoder — MPS stream manages lifecycle

        // Return individual grad components (Python adds dL_m3d_p + dL_m3d_s)
        return {dL_m3d_p, dL_m3d_s, dL_sc, dL_qt, dL_sh, dL_opac};
    }
}


// ---------------------------------------------------------------------------
//  mps_simple_knn — Morton code KNN on MPS tensors
// ---------------------------------------------------------------------------
torch::Tensor mps_simple_knn(
    const torch::Tensor& points,
    int64_t k_neighbors,
    int64_t search_window)
{
    @autoreleasepool {
        TORCH_CHECK(ensure_mps_init(), mps_ctx().error_msg);
        MPSMetalContext& c = mps_ctx();
        auto* stream = at::mps::getCurrentMPSStream();

        uint32_t N = (uint32_t)points.size(0);
        auto f32 = points.options();
        auto u32 = f32.dtype(torch::kInt32);

        // Bounding box via torch (GPU, async)
        auto pts_min = std::get<0>(points.min(0));
        auto pts_max = std::get<0>(points.max(0));
        auto pts_inv = 1.0f / (pts_max - pts_min).clamp_min(1e-8f);

        torch::Tensor avg_sq = torch::empty({(int64_t)N}, f32);
        auto ss = alloc_sort_scratch(N);

        id<MTLComputeCommandEncoder> enc = stream->commandEncoder();

        KNNParams kp = { N, (uint32_t)search_window, (uint32_t)k_neighbors };

        // Morton codes
        [enc setComputePipelineState:c.morton_codes_pso];
        set_buf(enc, points, 0);
        set_buf(enc, pts_min, 1);
        set_buf(enc, pts_inv, 2);
        [enc setBuffer:ss.keys_a offset:0 atIndex:3];
        [enc setBuffer:ss.vals_a offset:0 atIndex:4];
        [enc setBytes:&kp length:sizeof(kp) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(N, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        // Radix sort
        radix_sort(c, enc, ss.keys_a, ss.keys_b, ss.vals_a, ss.vals_b,
                   ss.hist, ss.b1, ss.b2, ss.b3, N, 8);

        // KNN search
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:c.knn_search_pso];
        [enc setBuffer:ss.vals_a offset:0 atIndex:0];
        set_buf(enc, points, 1);
        set_buf(enc, avg_sq, 2);
        [enc setBytes:&kp length:sizeof(kp) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(N, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        // KNN is called once at init — sync and release scratch immediately
        stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
        ss.release_all();

        return avg_sq;
    }
}
