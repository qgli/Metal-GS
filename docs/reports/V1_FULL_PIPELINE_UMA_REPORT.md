# V1 Full Pipeline UMA Refactor: Post-Mortem Report

**Branch:** `exp/cpu-gpu-uma`  
**Date:** 2025-07-18  
**Hardware:** Mac Mini M4 · 10 GPU cores · 16 GB UMA  
**PyTorch:** 2.10.0 (MPS backend)  
**Last Known Working Commit:** `c909fe8`  

---

## 1. Objective

Eliminate the per-iteration `.cpu().numpy()` + `newBufferWithBytes` double-copy in
the rendering pipeline — the dominant bottleneck (~95% of per-iteration time).
The prior session showed this double-copy moves ~84 MB per frame for 165K Gaussians.

The plan was a 4-step attack:
1. Zero-copy Metal buffer extraction from MPS tensors
2. Multi-threaded CPU frustum culling (M4 10-core)
3. E2E benchmark (`METAL_GS_BF16=1 python benchmark_m4.py --cap 0 --iterations 500`)
4. Report `docs/reports/V1_FULL_PIPELINE_UMA_REPORT.md`

---

## 2. What Was Built

### 2.1 `csrc/render_bridge.mm` (~1100 lines)

A complete zero-copy render bridge providing `render_forward_zc`, `render_backward_zc`,
and `mt_frustum_cull_compact` — accepting `torch::Tensor` (MPS device) and dispatching
Metal compute kernels directly from tensor-backed `id<MTLBuffer>` objects.

**Components implemented:**
- `ZCContext` singleton: manages `id<MTLDevice>`, `id<MTLCommandQueue>`, and 13 PSOs
  loaded from the same AOT-compiled `metal_gs.metallib`
- `get_shared_buffer()`: safe MTLBuffer extraction with shared/private tensor detection
- `render_forward_zc()`: full forward pipeline (preprocess → depth sort → tile binning
  → rasterize) using direct Metal buffer access
- `render_backward_zc()`: backward pipeline using numpy inputs (same as legacy)
- `mt_frustum_cull_compact()`: multi-threaded CPU frustum culling via GCD `dispatch_apply`
- `dispatch_radix_sort()`: 6-buffer key-value radix sort (histogram → prefix scan → scatter)
- `dispatch_prefix_sum()`: multi-level inclusive prefix scan
- All struct layouts matching the Metal shader parameter structs

### 2.2 Modified Files (all restored)
- `setup.py` — added `_metal_gs_render_bridge` extension build
- `metal_gs/rasterizer.py` — dual-path forward (ZC bridge vs legacy)
- `minGS/example.py` — import path fix

### 2.3 Test Files Created
- `test_compare_fwd.py` — numerical comparison of ZC vs legacy forward outputs
- `test_buffer_debug.py` — low-level MTLBuffer contents inspection
- `test_shared_tensor.py` — verified `create_shared_tensor` / `to_shared` / gradient flow
- `test_storage_ptr.py` — `data_ptr()` vs `storage().data()` analysis
- `test_fwd_quick.py` — quick forward smoke test with real data
- `test_convergence.py` — training convergence A/B test
- `test_grad.py` — gradient flow through shared tensors

---

## 3. Root Cause Discovery: The Critical PyTorch 2.10 MPS Issue

### 3.1 The Broken Assumption

The entire approach was based on this pattern (used throughout PyTorch MPS extensions):

```objc
id<MTLBuffer> buf = __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
// → buf.contents gives CPU-readable pointer
```

**This pattern is BROKEN on PyTorch 2.10 for default (private-mode) MPS tensors.**

### 3.2 Evidence Chain

| Test | Finding |
|------|---------|
| `test_compare_fwd.py` | radii 90.29% differ, num_visible L=106,397 vs Z=165,751 (ALL visible in ZC), means2d range off by 1000× |
| `test_buffer_debug.py` | `means3d[0] = (0.0000, 0.0000, 0.0000)` from buffer but Python has `(-1.3884, 1.4960, 1.3827)` |
| ObjC class inspection | `object_getClass()` returns `AGXG16GFamilyBuffer` — appears valid but contains wrong data |
| `data_ptr()` tracking | `means3d.data_ptr()` CHANGES between measurements (0x126314730 → 0x110e504b0) — MPS allocator relocates buffers |
| Direct read | Even reading from `data_ptr()` directly gives zeros; `[buf contents]` causes segfault on some buffers |

### 3.3 Why viewmat Worked (by Accident)

`viewmat` was always processed through `.T.contiguous()` which forces a fresh tensor
allocation. This fresh allocation's `storage().data()` happened to return a valid buffer
(possibly because the copy triggers the shared allocator path internally). This was NOT
a reliable pattern.

### 3.4 The Real MPS Memory Architecture (PyTorch 2.10)

```
PyTorch MPS Default:
  nn.Parameter → MPS allocator → MTLResourceStorageModePrivate
  → storage().data() returns a GPU virtual address pointer
  → __builtin_bit_cast to id<MTLBuffer> → UNDEFINED BEHAVIOR
  → The resulting "buffer" object may contain zeros or stale data
  → [buffer contents] may segfault

PyTorch MPS Shared Allocator (at::mps::GetMPSAllocator(true)):
  create_shared_tensor() → MTLResourceStorageModeShared
  → storage().data() IS a valid id<MTLBuffer> pointer
  → __builtin_bit_cast works correctly
  → [buffer contents] returns CPU-readable data
  → isSharedBuffer() returns true
```

### 3.5 Verified Solution

The `create_shared_tensor()` / `to_shared()` functions in `cpu_gpu_synergy.mm` were
**verified working** with all tests passing:

```python
# All 3 tests PASSED:
t = uma.create_shared_tensor([10, 3])     # ✓ Creates MPS tensor with shared storage
s = uma.to_shared(private_tensor)          # ✓ Data preserved exactly (max_diff=0)
loss = (shared_tensor * 2).sum()
loss.backward()                            # ✓ Gradient = 2.0 (correct)
```

---

## 4. The Rasterize Hang (Unresolved)

After fixing buffer extraction with `get_shared_buffer()` (which falls back to
`.cpu() + newBufferWithBytes` for private tensors, matching the proven legacy path),
the pipeline progresses through Steps 1-3 successfully:

```
[ZC] Step 1: Preprocess DONE          ✓ (165,751 Gaussians)
[ZC] Step 2: Depth sort DONE          ✓ (648 blocks)
[ZC] Step 3: Tile binning DONE        ✓ (1,236,391 intersections, 4 passes)
[ZC] Step 4: Rasterize dispatched     ✗ HANGS (65×43 tiles)
```

The rasterize step dispatches 65×43 = 2,795 threadgroups of 16×16 threads and
never returns. Possible causes:
- **GPU Watchdog timeout**: some tiles may have extremely high Gaussian counts
  (the `max_gaussians_per_tile=0` parameter means unlimited processing per tile)
- **Buffer size mismatch**: the colors/opacities buffers created via `get_shared_buffer`
  may have incorrect sizes (they use `.cpu()` which should be correct, but the
  contiguous copies may not match expected layout)
- **Metal command buffer conflict**: the ZC bridge's command queue may conflict with
  PyTorch MPS's internal command queue when operating on the same device

This remained unresolved at the time of session termination.

---

## 5. Architecture Lessons Learned

### 5.1 PyTorch 2.10 MPS Internals

1. **`tensor.storage().data()` is NOT `id<MTLBuffer>`** for private-mode tensors (the
   default). This breaks the standard MPS extension pattern documented in PyTorch tutorials.

2. **`data_ptr()` and `storage().data()` return the same value** on MPS — both are
   GPU virtual addresses, not ObjC object pointers.

3. **The MPS allocator can relocate tensor storage** between Python calls. The `data_ptr()`
   of `means3d` was observed changing from `0x126314730` to `0x110e504b0` during
   normal execution.

4. **`at::mps::getIMPSAllocator()->isSharedBuffer(ptr)`** correctly identifies whether
   a pointer refers to a shared-mode allocation. This is the only reliable test.

5. **`at::mps::GetMPSAllocator(true)`** returns the shared allocator. Tensors created
   through it have valid `id<MTLBuffer>` accessible via `storage().data()`.

### 5.2 Correct Architecture for Zero-Copy

The correct approach (not fully implemented) is:

```
Phase 1: Model Allocation
  GaussianModel.__init__()
    → nn.Parameter(uma.create_shared_tensor([N, 3]))  # positions
    → nn.Parameter(uma.create_shared_tensor([N, 3]))  # scales
    → nn.Parameter(uma.create_shared_tensor([N, 4]))  # rotations
    → nn.Parameter(uma.create_shared_tensor([N, 1]))  # opacities

Phase 2: Render Pipeline
  get_shared_buffer(tensor):
    if isSharedBuffer(storage().data()):
      return __builtin_bit_cast(id<MTLBuffer>, storage().data())  # ZERO-COPY
    else:
      return tensor.cpu() + newBufferWithBytes(...)  # safe fallback

Phase 3: Training
  Gradient flow through shared tensors verified working.
  AdamW optimizer operates on shared nn.Parameters normally.
```

### 5.3 Amdahl's Law Constraint

The rendering pipeline constitutes ~95% of per-iteration time. The prior UMA
optimization of densify/prune (which fires ~2% of the time) yielded only +3.2%.
Eliminating the render double-copy is the correct target but requires solving the
buffer extraction problem fully.

---

## 6. Files Inventory

### Created (untracked, to be deleted)

| File | Purpose | Lines |
|------|---------|-------|
| `csrc/render_bridge.mm` | Zero-copy render bridge (incomplete) | ~1115 |
| `test_compare_fwd.py` | Forward numerical comparison | 124 |
| `test_buffer_debug.py` | MTLBuffer contents inspection | ~50 |
| `test_shared_tensor.py` | Shared tensor verification | ~40 |
| `test_storage_ptr.py` | data_ptr analysis | ~30 |
| `test_fwd_quick.py` | Quick forward smoke test | ~90 |
| `test_convergence.py` | Convergence A/B test | ~60 |
| `test_grad.py` | Gradient flow test | ~30 |

### Modified (restored to c909fe8)

| File | Changes Made → Restored |
|------|------------------------|
| `setup.py` | Added `_metal_gs_render_bridge` extension build → Restored |
| `metal_gs/rasterizer.py` | Dual-path forward (ZC/legacy) → Restored |
| `minGS/example.py` | Import path fix → Restored |

### Unchanged (committed at c909fe8)

| File | Status |
|------|--------|
| `csrc/cpu_gpu_synergy.mm` | Working: `create_shared_tensor`, `to_shared`, UMA ops |
| `minGS/gs/trainers/basic/helpers.py` | Working: UMA densify/prune (+3.2%) |
| `csrc/metal_wrapper.mm` | Original render pipeline (proven correct) |

---

## 7. Current State

```
Branch: exp/cpu-gpu-uma
HEAD:   c909fe8 (same as origin/exp/cpu-gpu-uma)
Status: Clean (tracked files restored, untracked test files remain)

Verified working:
  $ conda activate metal-gs
  $ python setup.py build_ext --inplace   # builds _metal_gs_core + _metal_gs_uma
  $ cd minGS && METAL_GS_BF16=1 python example.py --data_dir data/cat --iterations 500
  → 165,751 Gaussians, loss 0.195 → 0.121, ~8-10 it/s ✓
```

---

## 8. Recommendations for Future Work

### 8.1 Short-term: Fix the Rasterize Hang
- Add `max_gaussians_per_tile` cap (e.g., 4096) to prevent GPU Watchdog timeout
- Verify buffer sizes match shader expectations (colors is N×3, opacities is N×1)
- Consider splitting rasterize into smaller tile batches
- Test on a smaller image (256×256) to reduce tile count

### 8.2 Medium-term: Shared Model Parameters
- Modify `GaussianModel.__init__` to use `create_shared_tensor()` for all parameters
- This makes `isSharedBuffer()` return true → direct buffer access in `get_shared_buffer`
- Need to verify optimizer compatibility (AdamW state tensors)
- Need to handle densification (create_shared_tensor for new Gaussians after split/clone)

### 8.3 Long-term: PyTorch MPS API
- Track PyTorch issue for proper `getMTLBuffer()` API on MPS tensors
- Current `__builtin_bit_cast` pattern is an internal hack, not a public API
- Future PyTorch versions may provide `at::mps::getMTLBuffer(tensor)` directly

### 8.4 Alternative: Keep Legacy Path, Optimize Elsewhere
- The legacy `.cpu().numpy()` path is ~5.5ms per frame for 165K Gaussians
- At ~105ms/iteration, this is only ~5% overhead
- Optimizing the Metal kernels themselves (e.g., tile-based data reuse,
  threadgroup memory optimization) may yield better ROI

---

## 9. Session Timeline

| Phase | Action | Outcome |
|-------|--------|---------|
| 1 | Codebase audit (all .mm, .py, .metal files) | Identified 15 PSOs, dual pipeline architecture |
| 2 | Built `render_bridge.mm` (1100 lines) | ZCContext, 13 PSOs, forward/backward/culling |
| 3 | Sort implementation (radix sort 6-buffer KV) | Fixed gen_keys (4 buf), scatter (6 buf) |
| 4 | Tile binning (gen_isect, sort, tile_ranges) | Fixed gen_isect (8 buf), bin sort passes |
| 5 | First test: A/B benchmark | ZC 11.86 it/s vs Legacy 9.43 it/s BUT loss not converging (~0.587) |
| 6 | Isolate forward vs backward | ZC forward + legacy backward: loss 0.524 → forward is broken |
| 7 | Numerical comparison (`test_compare_fwd.py`) | radii 90.29% differ, means2d off by 1000×, all Gaussians "visible" |
| 8 | Buffer contents inspection | **SMOKING GUN:** means3d buffer contains zeros, viewmat correct |
| 9 | `data_ptr()` analysis | data_ptr CHANGES between calls, MPS allocator relocates |
| 10 | ObjC class inspection | `AGXG16GFamilyBuffer` — valid class but undefined behavior |
| 11 | **ROOT CAUSE confirmed** | `storage().data()` ≠ `id<MTLBuffer>` on PyTorch 2.10 private tensors |
| 12 | Shared tensor verification | `create_shared_tensor`, `to_shared`, gradients ALL work |
| 13 | Rewrote `get_shared_buffer()` | Shared: direct access; Private: `.cpu()` fallback |
| 14 | Clean up bridge (delete broken code) | Removed `ensure_shared`, `BlitRequest`, debug blocks |
| 15 | Test with real data | Steps 1-3 pass, Step 4 (Rasterize) hangs → Session terminated |
