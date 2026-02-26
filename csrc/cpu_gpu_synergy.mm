// ============================================================================
//  Metal-GS: CPU-GPU Zero-Copy Synergy via Apple Silicon UMA
//
//  FINDING: PyTorch MPS uses MTLResourceStorageModePrivate by default.
//  This means [buffer contents] returns null — CPU cannot directly read.
//
//  TWO pathways implemented:
//    A. Private → Shared blit: Metal GPU blit copies private buffer to a
//       shared-mode staging buffer, then CPU reads via [contents]. This
//       eliminates Python-level `.cpu()` overhead and autograd bookkeeping.
//    B. Shared allocator: create tensors directly in shared mode via
//       at::mps::GetMPSAllocator(true). TRUE zero-copy — CPU and GPU see
//       the same memory through [buffer contents].
//
//  Operations:
//    1. uma_read_test          — correctness validation
//    2. uma_frustum_cull       — view-frustum visibility (branchy/CPU-optimal)
//    3. uma_densification_mask — gradient + scale thresholding
//    4. uma_pruning_mask       — opacity + scale + screen-size pruning
//    5. uma_buffer_identity_check — inspect MTLBuffer properties
//    6. create_shared_tensor   — allocate tensor in shared storage mode
//
//  Build: ObjC++ (.mm) with -fobjc-arc, links Metal + libtorch
// ============================================================================

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

// NOTE: We avoid #include <ATen/native/mps/OperationUtils.h> (ARC-incompatible).
// We inline getMTLBufferStorage() and use ATen's public MPS allocator API.
#include <torch/extension.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSDevice.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <chrono>

// ============================================================================
//  Singleton: shared Metal command queue for blit operations
// ============================================================================

static id<MTLCommandQueue> getBlitCommandQueue() {
    static id<MTLCommandQueue> queue = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        queue = [device newCommandQueue];
    });
    return queue;
}

// ============================================================================
//  Helper: extract MTLBuffer from PyTorch MPS tensor
// ============================================================================

static inline id<MTLBuffer> getMTLBufferStorage(const at::TensorBase& tensor) {
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

// ============================================================================
//  Core: get CPU-readable pointer from MPS tensor
//
//  Strategy:
//    1. If buffer is SharedMode → direct [contents] access (zero copy)
//    2. If buffer is Private → blit to a shared staging buffer, read from that
// ============================================================================

/// Blit a private MTLBuffer to a shared staging buffer and return CPU pointer.
/// The staging buffer is allocated per-call (caller should cache if reusing).
static const float* blit_private_to_shared(id<MTLBuffer> src, size_t byte_offset,
                                            id<MTLBuffer> __strong* out_staging) {
    id<MTLDevice> device = [src device];
    NSUInteger len = [src length];

    // Create shared staging buffer
    id<MTLBuffer> staging = [device newBufferWithLength:len
                                               options:MTLResourceStorageModeShared];
    TORCH_CHECK(staging != nil, "Failed to allocate shared staging buffer");

    // Blit copy: private → shared (GPU-side, very fast on Apple Silicon)
    id<MTLCommandQueue> queue = getBlitCommandQueue();
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromBuffer:src sourceOffset:0
                toBuffer:staging destinationOffset:0
                    size:len];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    *out_staging = staging;  // keep alive
    return static_cast<const float*>([staging contents]) + byte_offset / sizeof(float);
}

/// Get CPU-accessible float pointer from an MPS tensor.
/// If the backing MTLBuffer is shared-mode: TRUE zero-copy.
/// If private-mode: uses Metal blit to a shared staging buffer.
///
/// staging_buf: if non-null and a blit was needed, receives the staging buffer
///              (caller must keep it alive while using the pointer).
static const float* uma_ptr_f32(const torch::Tensor& t,
                                 id<MTLBuffer> __strong* staging_buf = nullptr) {
    TORCH_CHECK(t.is_mps(), "uma_ptr_f32: tensor must be on MPS device");
    TORCH_CHECK(t.is_contiguous(), "uma_ptr_f32: tensor must be contiguous");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32,
                "uma_ptr_f32: expected float32, got ", t.scalar_type());

    id<MTLBuffer> buf = getMTLBufferStorage(t);
    TORCH_CHECK(buf != nil, "uma_ptr_f32: failed to get MTLBuffer");

    if ([buf storageMode] == MTLResourceStorageModeShared) {
        // ★ TRUE ZERO COPY — direct CPU access to GPU memory
        const float* ptr = static_cast<const float*>([buf contents]);
        TORCH_CHECK(ptr != nullptr, "uma_ptr_f32: shared buffer [contents] returned null");
        return ptr + t.storage_offset();
    } else {
        // Private storage — need Metal blit to shared staging buffer
        id<MTLBuffer> staging = nil;
        size_t byte_off = t.storage_offset() * sizeof(float);
        const float* ptr = blit_private_to_shared(buf, byte_off, &staging);
        if (staging_buf) *staging_buf = staging;
        return ptr;
    }
}

// ============================================================================
//  Create shared-mode MPS tensor (TRUE zero-copy pathway)
// ============================================================================

/// Allocate a new MPS tensor using the SHARED allocator.
/// This tensor's MTLBuffer uses MTLResourceStorageModeShared, meaning both
/// CPU and GPU can access the same physical memory — TRUE zero-copy UMA.
///
/// @param sizes  Tensor dimensions (e.g., {N, 3})
/// @return MPS tensor with shared storage mode
torch::Tensor create_shared_tensor(std::vector<int64_t> sizes) {
    // Get PyTorch's shared MPS allocator
    auto* allocator = at::mps::GetMPSAllocator(/*useSharedAllocator=*/true);
    TORCH_CHECK(allocator != nullptr, "Shared MPS allocator not available");

    // Compute total bytes
    int64_t numel = 1;
    for (auto s : sizes) numel *= s;
    size_t nbytes = numel * sizeof(float);

    // Allocate via shared allocator
    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        nbytes,
        allocator,
        /*resizable=*/false
    );

    // Create tensor from storage
    auto tensor = torch::from_blob(
        storage.mutable_data(),
        sizes,
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS)
    );

    // Verify it's actually shared
    id<MTLBuffer> buf = getMTLBufferStorage(tensor);
    TORCH_CHECK([buf storageMode] == MTLResourceStorageModeShared,
                "create_shared_tensor: resulting buffer is not shared mode!");

    return tensor;
}

/// Copy an existing MPS tensor (private) to a new shared-mode tensor.
/// The GPU data is preserved via Metal blit.
torch::Tensor to_shared(torch::Tensor src) {
    TORCH_CHECK(src.is_mps(), "to_shared: tensor must be on MPS device");
    TORCH_CHECK(src.is_contiguous(), "to_shared: tensor must be contiguous");

    auto shape = src.sizes().vec();
    auto dst = create_shared_tensor(shape);

    // Blit from source (private) to destination (shared)
    id<MTLBuffer> src_buf = getMTLBufferStorage(src);
    id<MTLBuffer> dst_buf = getMTLBufferStorage(dst);

    size_t nbytes = src.numel() * src.element_size();

    if ([src_buf storageMode] == MTLResourceStorageModeShared) {
        // Both shared — memcpy
        memcpy([dst_buf contents], [src_buf contents], nbytes);
    } else {
        // Private → shared: Metal blit
        id<MTLCommandQueue> queue = getBlitCommandQueue();
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit copyFromBuffer:src_buf sourceOffset:0
                    toBuffer:dst_buf destinationOffset:0
                        size:nbytes];
        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    return dst;
}

// ============================================================================
//  1. UMA Read Test — validate zero-copy access correctness
// ============================================================================

/// Read the first `count` floats from an MPS tensor via UMA and return as CPU tensor.
/// This is a correctness validation function for the zero-copy pathway.
torch::Tensor uma_read_test(torch::Tensor mps_tensor, int64_t count) {
    id<MTLBuffer> staging = nil;
    const float* ptr = uma_ptr_f32(mps_tensor, &staging);
    int64_t N = mps_tensor.numel();
    if (count <= 0 || count > N) count = N;

    // Read values directly from UMA — no copy of input data
    auto result = torch::empty({count}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    float* out = result.data_ptr<float>();
    for (int64_t i = 0; i < count; i++) {
        out[i] = ptr[i];
    }
    return result;  // staging released here (ARC)
}

// ============================================================================
//  2. Frustum Culling — branchy per-Gaussian visibility test
// ============================================================================

/// Compute view-frustum culling mask for N Gaussians.
///
/// For each Gaussian center P, transforms to camera space via viewmat,
/// then tests against the 6 frustum planes (near, far, left, right, top, bottom).
/// Returns a CPU bool tensor [N] where true = visible (passes all planes).
///
/// This is textbook branchy code — early-exit on first failed plane test.
/// GPUs hate this; CPUs love it (branch predictor + speculative execution).
///
/// @param means3d   [N, 3] float32, MPS device — Gaussian centers in world space
/// @param viewmat   [4, 4] float32, MPS device — camera view matrix (row-major)
/// @param near_plane  Near clipping distance
/// @param far_plane   Far clipping distance
/// @param fov_x       Horizontal half-FOV tangent (tan_fovx)
/// @param fov_y       Vertical half-FOV tangent (tan_fovy)
/// @return mask [N] bool tensor (CPU) + elapsed_ms as tuple
std::tuple<torch::Tensor, double> uma_frustum_cull(
    torch::Tensor means3d,     // [N, 3] MPS
    torch::Tensor viewmat,     // [4, 4] MPS
    float near_plane,
    float far_plane,
    float fov_x,
    float fov_y
) {
    auto t0 = std::chrono::high_resolution_clock::now();

    id<MTLBuffer> staging_m = nil, staging_v = nil;
    const float* m = uma_ptr_f32(means3d, &staging_m);
    const float* V = uma_ptr_f32(viewmat, &staging_v);

    uint32_t N = (uint32_t)means3d.size(0);

    // Allocate output mask on CPU (tiny: N bytes)
    auto mask = torch::zeros({(int64_t)N}, torch::dtype(torch::kBool).device(torch::kCPU));
    bool* mask_ptr = mask.data_ptr<bool>();

    // Extract frustum planes from row-major view matrix
    // viewmat layout (row-major):
    //   V[0]  V[1]  V[2]  V[3]     ← right (X)
    //   V[4]  V[5]  V[6]  V[7]     ← up (Y)
    //   V[8]  V[9]  V[10] V[11]    ← forward (Z)
    //   V[12] V[13] V[14] V[15]    ← translation

    // Precompute frustum tangent boundaries (padded slightly for conservative culling)
    const float pad = 1.05f;  // 5% padding to avoid popping
    const float tan_left   = -fov_x * pad;
    const float tan_right  =  fov_x * pad;
    const float tan_bottom = -fov_y * pad;
    const float tan_top    =  fov_y * pad;

    uint32_t visible_count = 0;

    for (uint32_t i = 0; i < N; i++) {
        float px = m[i * 3 + 0];
        float py = m[i * 3 + 1];
        float pz = m[i * 3 + 2];

        // Transform to camera space: p_cam = V * p_world
        // Row-major multiply: p_cam.x = V[0]*px + V[1]*py + V[2]*pz + V[3]
        float cx = V[0]*px + V[1]*py + V[2]*pz  + V[3];
        float cy = V[4]*px + V[5]*py + V[6]*pz  + V[7];
        float cz = V[8]*px + V[9]*py + V[10]*pz + V[11];

        // Depth test (near/far planes) — early exit
        if (cz < near_plane || cz > far_plane) {
            mask_ptr[i] = false;
            continue;   // <-- branch: ~50% of Gaussians culled here
        }

        // Perspective divide for frustum test
        float inv_z = 1.0f / cz;
        float ndc_x = cx * inv_z;
        float ndc_y = cy * inv_z;

        // Frustum test (left/right/top/bottom planes) — early exit
        if (ndc_x < tan_left || ndc_x > tan_right ||
            ndc_y < tan_bottom || ndc_y > tan_top) {
            mask_ptr[i] = false;
            continue;   // <-- branch: another ~20% culled here
        }

        mask_ptr[i] = true;
        visible_count++;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return std::make_tuple(mask, ms);
}

// ============================================================================
//  3. Densification Mask — gradient + scale thresholding
// ============================================================================

/// Compute densification decision masks for adaptive density control.
///
/// Given accumulated gradient magnitudes and current scales, determine which
/// Gaussians should be cloned (high gradient, small scale) or split
/// (high gradient, large scale).
///
/// This is pure branchy threshold logic — ideal for CPU, wasteful on GPU.
///
/// @param grad_magnitudes [N] float32, MPS — accumulated gradient magnitudes
/// @param scales          [N, 3] float32, MPS — log-scale parameters
/// @param grad_threshold  Clone/split threshold on gradient magnitude
/// @param scale_threshold Boundary between clone and split (in scene units)
/// @return tuple(clone_mask[N], split_mask[N], stats_dict, elapsed_ms)
std::tuple<torch::Tensor, torch::Tensor, int64_t, int64_t, double>
uma_densification_mask(
    torch::Tensor grad_magnitudes,   // [N] MPS
    torch::Tensor scales,            // [N, 3] MPS
    float grad_threshold,
    float scale_threshold
) {
    auto t0 = std::chrono::high_resolution_clock::now();

    id<MTLBuffer> staging_g = nil, staging_s = nil;
    const float* g = uma_ptr_f32(grad_magnitudes, &staging_g);
    const float* s = uma_ptr_f32(scales, &staging_s);

    uint32_t N = (uint32_t)grad_magnitudes.size(0);

    auto clone_mask = torch::zeros({(int64_t)N}, torch::dtype(torch::kBool).device(torch::kCPU));
    auto split_mask = torch::zeros({(int64_t)N}, torch::dtype(torch::kBool).device(torch::kCPU));
    bool* clone_ptr = clone_mask.data_ptr<bool>();
    bool* split_ptr = split_mask.data_ptr<bool>();

    int64_t n_clone = 0, n_split = 0;

    for (uint32_t i = 0; i < N; i++) {
        // Check gradient magnitude against threshold
        if (g[i] <= grad_threshold) {
            // Below threshold — no densification needed
            clone_ptr[i] = false;
            split_ptr[i] = false;
            continue;   // <-- early exit: ~90% of Gaussians skip here
        }

        // Above gradient threshold — check scale to decide clone vs split
        // max(exp(scales[i,0]), exp(scales[i,1]), exp(scales[i,2]))
        float s0 = std::exp(s[i * 3 + 0]);
        float s1 = std::exp(s[i * 3 + 1]);
        float s2 = std::exp(s[i * 3 + 2]);
        float max_scale = std::fmax(s0, std::fmax(s1, s2));

        if (max_scale > scale_threshold) {
            // Large Gaussian — split into two
            clone_ptr[i] = false;
            split_ptr[i] = true;
            n_split++;
        } else {
            // Small Gaussian — clone (duplicate)
            clone_ptr[i] = true;
            split_ptr[i] = false;
            n_clone++;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return std::make_tuple(clone_mask, split_mask, n_clone, n_split, ms);
}

// ============================================================================
//  4. Pruning Mask — opacity + scale thresholding
// ============================================================================

/// Compute pruning mask for removing degenerate Gaussians.
///
/// Tests three conditions:
///   1. opacity < opacity_threshold (too transparent)
///   2. max(scale) > world_size_threshold (too large)
///   3. screen_radii > screen_size_threshold (covers too many pixels)
///
/// @param opacities       [N] float32, MPS — opacity logits (pre-sigmoid)
/// @param scales          [N, 3] float32, MPS — log-scale parameters
/// @param screen_radii    [N] float32, MPS — max 2D radii from last render
/// @param opacity_threshold    Minimum opacity after sigmoid
/// @param world_size_threshold Maximum world-space scale
/// @param screen_size_threshold Maximum screen-space radius
/// @return tuple(prune_mask[N], n_pruned, elapsed_ms)
std::tuple<torch::Tensor, int64_t, double>
uma_pruning_mask(
    torch::Tensor opacities,         // [N] MPS
    torch::Tensor scales,            // [N, 3] MPS
    torch::Tensor screen_radii,      // [N] float32, MPS
    float opacity_threshold,
    float world_size_threshold,
    float screen_size_threshold
) {
    auto t0 = std::chrono::high_resolution_clock::now();

    id<MTLBuffer> staging_o = nil, staging_s = nil, staging_r = nil;
    const float* o = uma_ptr_f32(opacities, &staging_o);
    const float* s = uma_ptr_f32(scales, &staging_s);
    const float* r = uma_ptr_f32(screen_radii, &staging_r);

    uint32_t N = (uint32_t)opacities.size(0);

    auto mask = torch::zeros({(int64_t)N}, torch::dtype(torch::kBool).device(torch::kCPU));
    bool* m = mask.data_ptr<bool>();

    int64_t n_pruned = 0;

    for (uint32_t i = 0; i < N; i++) {
        // Sigmoid activation for opacity
        float sig_opacity = 1.0f / (1.0f + std::exp(-o[i]));

        // Condition 1: too transparent
        if (sig_opacity < opacity_threshold) {
            m[i] = true;
            n_pruned++;
            continue;
        }

        // Condition 2: too large in world space
        float s0 = std::exp(s[i * 3 + 0]);
        float s1 = std::exp(s[i * 3 + 1]);
        float s2 = std::exp(s[i * 3 + 2]);
        float max_scale = std::fmax(s0, std::fmax(s1, s2));
        if (max_scale > world_size_threshold) {
            m[i] = true;
            n_pruned++;
            continue;
        }

        // Condition 3: too large on screen
        if (r[i] > screen_size_threshold) {
            m[i] = true;
            n_pruned++;
            continue;
        }

        m[i] = false;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return std::make_tuple(mask, n_pruned, ms);
}

// ============================================================================
//  5. Zero-Copy MTLBuffer Wrapping — wrap MPS tensor as MTLBuffer for our
//     Metal compute pipeline without newBufferWithBytes copy
// ============================================================================

/// Demonstrate that the existing Metal dispatch can use the SAME MTLBuffer
/// that backs a PyTorch MPS tensor, avoiding the second copy in
/// newBufferWithBytes(). Returns the buffer address for validation.
std::tuple<int64_t, int64_t, bool> uma_buffer_identity_check(
    torch::Tensor mps_tensor
) {
    TORCH_CHECK(mps_tensor.is_mps(), "tensor must be on MPS device");
    TORCH_CHECK(mps_tensor.is_contiguous(), "tensor must be contiguous");

    id<MTLBuffer> buf = getMTLBufferStorage(mps_tensor);

    // Get the CPU-visible pointer from UMA
    void* contents = [buf contents];
    int64_t contents_addr = reinterpret_cast<int64_t>(contents);

    // Get the buffer's GPU address (same on UMA!)
    int64_t buf_length = (int64_t)[buf length];

    // They should point to the same physical memory on Apple Silicon
    bool is_shared = ([buf storageMode] == MTLResourceStorageModeShared);

    return std::make_tuple(contents_addr, buf_length, is_shared);
}


// ============================================================================
//  PyBind11 Module Registration
// ============================================================================

PYBIND11_MODULE(_metal_gs_uma, m) {
    m.doc() = "Metal-GS: CPU-GPU Zero-Copy Synergy via Apple Silicon UMA\n"
              "Direct CPU access to MPS tensor data through [MTLBuffer contents].";

    m.def("uma_read_test", &uma_read_test,
          py::arg("mps_tensor"),
          py::arg("count") = -1,
          "Read MPS tensor data from CPU via UMA (zero-copy). Returns CPU tensor.");

    m.def("uma_frustum_cull", &uma_frustum_cull,
          py::arg("means3d"),
          py::arg("viewmat"),
          py::arg("near_plane") = 0.01f,
          py::arg("far_plane") = 100.0f,
          py::arg("fov_x") = 1.0f,
          py::arg("fov_y") = 1.0f,
          "CPU-side frustum culling reading MPS tensor data via UMA.\n"
          "Returns (visible_mask[N], elapsed_ms).");

    m.def("uma_densification_mask", &uma_densification_mask,
          py::arg("grad_magnitudes"),
          py::arg("scales"),
          py::arg("grad_threshold") = 0.0002f,
          py::arg("scale_threshold") = 0.01f,
          "CPU-side densification mask via UMA direct access.\n"
          "Returns (clone_mask[N], split_mask[N], n_clone, n_split, elapsed_ms).");

    m.def("uma_pruning_mask", &uma_pruning_mask,
          py::arg("opacities"),
          py::arg("scales"),
          py::arg("screen_radii"),
          py::arg("opacity_threshold") = 0.005f,
          py::arg("world_size_threshold") = 1.0f,
          py::arg("screen_size_threshold") = 20.0f,
          "CPU-side pruning mask via UMA direct access.\n"
          "Returns (prune_mask[N], n_pruned, elapsed_ms).");

    m.def("uma_buffer_identity_check", &uma_buffer_identity_check,
          py::arg("mps_tensor"),
          "Inspect MTLBuffer backing an MPS tensor.\n"
          "Returns (contents_addr, buffer_length, is_shared_mode).");

    m.def("create_shared_tensor", &create_shared_tensor,
          py::arg("sizes"),
          "Create an MPS tensor with MTLResourceStorageModeShared.\n"
          "This enables TRUE zero-copy CPU↔GPU access via UMA.");

    m.def("to_shared", &to_shared,
          py::arg("src"),
          "Copy an MPS tensor to a new shared-mode tensor.\n"
          "Returns a new tensor with shared storage (CPU-accessible).");
}
