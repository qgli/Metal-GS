#!/usr/bin/env python3
"""Test shared allocator path — TRUE zero-copy UMA."""
import torch
import metal_gs._metal_gs_uma as uma

print("=== Test: create_shared_tensor ===")
t_shared = uma.create_shared_tensor([5])
print(f"Device: {t_shared.device}")
print(f"Shape: {t_shared.shape}")
print(f"Dtype: {t_shared.dtype}")

# Check it's actually shared
addr, length, shared = uma.uma_buffer_identity_check(t_shared)
print(f"SharedMode: {shared}  ← should be True")
print(f"Contents addr: 0x{addr:x}  ← should be non-zero")

# Write via PyTorch, read via UMA
t_shared.fill_(42.0)
torch.mps.synchronize()
result = uma.uma_read_test(t_shared, 5)
print(f"Expected: [42, 42, 42, 42, 42]")
print(f"UMA read: {result.tolist()}")
print(f"MATCH: {all(v == 42.0 for v in result.tolist())}")

print("\n=== Test: to_shared (private → shared copy) ===")
t_private = torch.randn(1000, 3, device='mps')
torch.mps.synchronize()
t_shared2 = uma.to_shared(t_private)

# Verify shared mode
_, _, shared2 = uma.uma_buffer_identity_check(t_shared2)
print(f"SharedMode: {shared2}  ← should be True")

# Verify data matches
expected = t_private.cpu()
result2 = uma.uma_read_test(t_shared2, 100)
match = torch.allclose(expected.flatten()[:100], result2, atol=1e-6)
print(f"Data match (first 100): {match}")

# Frustum cull on shared tensor (TRUE zero-copy path)
print("\n=== Test: Frustum cull on shared tensor ===")
positions_shared = uma.to_shared(torch.randn(10000, 3, device='mps') * 5.0)
viewmat_shared = uma.to_shared(torch.eye(4, device='mps'))
torch.mps.synchronize()
mask, ms = uma.uma_frustum_cull(positions_shared, viewmat_shared, 0.01, 100.0, 1.0, 1.0)
print(f"N=10000, visible={mask.sum().item()}, time={ms:.3f}ms (shared/zero-copy)")
