#!/usr/bin/env python3
"""Debug MPS tensor MTLBuffer properties."""
import torch
import metal_gs._metal_gs_uma as uma

print("=== Debug MTLBuffer Properties ===")

# Small tensor
t = torch.tensor([1.0, 2.0, 3.0], device='mps')
torch.mps.synchronize()

try:
    addr, length, shared = uma.uma_buffer_identity_check(t)
    print(f"contents addr: 0x{addr:x}")
    print(f"buffer length: {length}")
    print(f"is_shared: {shared}")
except Exception as e:
    print(f"Error: {e}")

# Check data_ptr from Python
print(f"\nPython data_ptr: 0x{t.data_ptr():x}")
print(f"Device: {t.device}")
print(f"is_contiguous: {t.is_contiguous()}")
print(f"storage offset: {t.storage_offset()}")
print(f"numel: {t.numel()}")
