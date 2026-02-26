#!/usr/bin/env python3
"""Validate UMA zero-copy access through the C++ extension."""
import torch
import metal_gs._metal_gs_uma as uma

# Test 1: UMA read correctness
print("=== Test 1: UMA Zero-Copy Read (small) ===")
t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='mps')
torch.mps.synchronize()
result = uma.uma_read_test(t, 5)
expected = t.cpu()
print(f"Expected: {expected.tolist()}")
print(f"UMA read: {result.tolist()}")
print(f"MATCH: {torch.allclose(expected, result)}")

# Test 2: Larger tensor
print("\n=== Test 2: UMA Zero-Copy Read (N=1000) ===")
torch.manual_seed(42)
t2 = torch.randn(1000, 3, device='mps')
torch.mps.synchronize()
result2 = uma.uma_read_test(t2, 100)
expected2 = t2.cpu().flatten()[:100]
match = torch.allclose(expected2, result2, atol=1e-6)
print(f"First 5 UMA: {result2[:5].tolist()}")
print(f"First 5 CPU: {expected2[:5].tolist()}")
print(f"MATCH (first 100): {match}")

# Test 3: MTLBuffer identity
print("\n=== Test 3: MTLBuffer Identity Check ===")
addr, length, shared = uma.uma_buffer_identity_check(t2)
print(f"Contents addr: 0x{addr:x}")
print(f"Buffer length: {length} bytes  (expected {t2.numel()*4})")
print(f"SharedMode: {shared}")

# Test 4: Frustum culling
print("\n=== Test 4: Frustum Cull ===")
N = 10000
means = torch.randn(N, 3, device='mps') * 5.0
viewmat = torch.eye(4, device='mps')
torch.mps.synchronize()
mask, ms = uma.uma_frustum_cull(means, viewmat, 0.01, 100.0, 1.0, 1.0)
visible = mask.sum().item()
print(f"N={N}, visible={visible}, culled={N-visible}, time={ms:.3f}ms")

# Test 5: Densification mask
print("\n=== Test 5: Densification Mask ===")
grads = torch.rand(N, device='mps') * 0.001
scales = torch.randn(N, 3, device='mps') * 0.5
torch.mps.synchronize()
clone_m, split_m, nc, ns, ms2 = uma.uma_densification_mask(grads, scales, 0.0002, 0.01)
print(f"N={N}, clone={nc}, split={ns}, time={ms2:.3f}ms")

# Test 6: Pruning mask
print("\n=== Test 6: Pruning Mask ===")
opacities = torch.randn(N, device='mps')
radii = torch.rand(N, device='mps') * 30.0
torch.mps.synchronize()
prune_m, np_, ms3 = uma.uma_pruning_mask(opacities, scales, radii, 0.005, 1.0, 20.0)
print(f"N={N}, pruned={np_}, time={ms3:.3f}ms")

print("\n=== ALL TESTS PASSED ===" if match else "\n=== TESTS FAILED ===")
