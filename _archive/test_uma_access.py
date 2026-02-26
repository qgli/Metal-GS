#!/usr/bin/env python3
"""Test UMA direct access on MPS tensors."""
import torch
import ctypes
import time

def read_via_uma(tensor, count):
    """Read float values directly from tensor's data_ptr."""
    ptr = tensor.data_ptr()
    arr = (ctypes.c_float * count).from_address(ptr)
    return list(arr)

print("=== Test 1: CPU → MPS → UMA read ===")
t_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
t_mps = t_cpu.to('mps')
torch.mps.synchronize()
uma_vals = read_via_uma(t_mps, 5)
print(f"Expected: {t_cpu.tolist()}")
print(f"UMA read: {uma_vals}")
print(f"Match: {t_cpu.tolist() == uma_vals}")

print("\n=== Test 2: MPS-native tensor ===")
t = torch.ones(5, device='mps') * 42.0
torch.mps.synchronize()
uma_vals = read_via_uma(t, 5)
expected = t.cpu().tolist()
print(f"Expected: {expected}")
print(f"UMA read: {uma_vals}")
print(f"Match: {expected == uma_vals}")

print("\n=== Test 3: randn on MPS ===")
torch.manual_seed(42)
t3 = torch.randn(5, device='mps')
torch.mps.synchronize()
uma_vals = read_via_uma(t3, 5)
expected = t3.cpu().tolist()
print(f"Expected: {expected}")
print(f"UMA read: {uma_vals}")
print(f"Match: {all(abs(a-b) < 1e-5 for a,b in zip(expected, uma_vals))}")

print("\n=== Test 4: Large tensor N=100K ===")
N = 100_000
t4 = torch.randn(N, 3, device='mps')
torch.mps.synchronize()

# Traditional path: .cpu().numpy()
t0 = time.perf_counter()
arr_cpu = t4.cpu().numpy()
t_trad = time.perf_counter() - t0

# UMA path: direct pointer
t0 = time.perf_counter()
ptr = t4.data_ptr()
uma_arr = (ctypes.c_float * (N*3)).from_address(ptr)
t_uma = time.perf_counter() - t0

# Verify correctness
match_count = sum(1 for i in range(min(100, N*3)) if abs(arr_cpu.flat[i] - uma_arr[i]) < 1e-5)
print(f"Traditional .cpu().numpy(): {t_trad*1000:.3f} ms")
print(f"UMA data_ptr() access:      {t_uma*1000:.3f} ms")
print(f"Correctness (first 100): {match_count}/100")
print(f"Speedup: {t_trad/t_uma:.1f}x")

print("\n=== Test 5: storage_offset check ===")
print(f"storage_offset: {t4.storage_offset()}")
print(f"is_contiguous: {t4.is_contiguous()}")
print(f"data_ptr: 0x{t4.data_ptr():x}")
print(f"storage data_ptr: 0x{t4.storage().data_ptr():x}")
