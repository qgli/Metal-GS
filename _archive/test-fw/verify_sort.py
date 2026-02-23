"""
Verification script for Metal GPU radix sort.

Tests:
1. Correctness: compare Metal sorted indices with NumPy argsort
2. Performance: Metal GPU time vs NumPy CPU time
3. Edge cases: negative depths, zero depths, duplicates

Target: 1M random FP32 depths
"""

import numpy as np
import time
import sys

try:
    from metal_gs._metal_gs_core import radix_sort_by_depth
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Run 'pip install -e .' first to build the extension.")
    sys.exit(1)


def test_sort(depths, label):
    """Run Metal sort and NumPy sort, compare results."""
    N = len(depths)
    depths = depths.astype(np.float32)

    # Metal GPU sort
    metal_indices, metal_ms = radix_sort_by_depth(depths)

    # NumPy CPU sort (reference)
    t0 = time.perf_counter()
    numpy_indices = np.argsort(depths, kind='stable').astype(np.uint32)
    t1 = time.perf_counter()
    numpy_ms = (t1 - t0) * 1000.0

    # Check correctness: sorted values must match
    metal_sorted  = depths[metal_indices]
    numpy_sorted  = depths[numpy_indices]
    values_match  = np.array_equal(metal_sorted, numpy_sorted)

    # Also check if indices are identical (for stable sort)
    indices_match = np.array_equal(metal_indices, numpy_indices)

    print(f"\n{'─'*60}")
    print(f"  {label}  (N = {N:,})")
    print(f"{'─'*60}")
    print(f"  Sorted values match: {values_match}")
    print(f"  Indices match (stable): {indices_match}")
    print(f"  Metal time:      {metal_ms:8.3f} ms {'(CPU fallback)' if N <= 16384 else '(GPU)'}")
    print(f"  NumPy CPU time:  {numpy_ms:8.3f} ms")
    if numpy_ms > 0:
        print(f"  Speedup:         {numpy_ms / max(metal_ms, 0.001):8.1f}×")

    if not values_match:
        # Debug: find first mismatch
        mismatch = np.where(metal_sorted != numpy_sorted)[0]
        if len(mismatch) > 0:
            i = mismatch[0]
            print(f"\n  ⚠ First mismatch at position {i}:")
            print(f"    Metal:  depth={metal_sorted[i]:.6f} (idx={metal_indices[i]})")
            print(f"    NumPy:  depth={numpy_sorted[i]:.6f} (idx={numpy_indices[i]})")

    return values_match


def main():
    print("=" * 60)
    print("  Metal-GS Radix Sort Verification")
    print("=" * 60)

    all_pass = True

    # Test 1: Small array (sanity check)
    np.random.seed(42)
    all_pass &= test_sort(np.random.randn(100).astype(np.float32), "Small (100)")

    # Test 2: Medium array with duplicates
    depths_dup = np.random.choice([0.1, 0.5, 1.0, 2.0, 5.0], size=10000).astype(np.float32)
    all_pass &= test_sort(depths_dup, "Duplicates (10K)")

    # Test 3: Negative depths (behind camera)
    depths_neg = np.random.uniform(-10.0, 10.0, size=50000).astype(np.float32)
    all_pass &= test_sort(depths_neg, "Mixed ± depths (50K)")

    # Test 4: 100K random depths (medium benchmark)
    depths_100k = np.random.uniform(0.1, 100.0, size=100_000).astype(np.float32)
    all_pass &= test_sort(depths_100k, "Random depths (100K)")

    # Test 5: 1M random depths (performance benchmark)
    depths_1m = np.random.uniform(0.1, 100.0, size=1_000_000).astype(np.float32)
    all_pass &= test_sort(depths_1m, "★ Benchmark (1M)")

    # Test 6: Already sorted (worst case for some algorithms)
    depths_sorted = np.linspace(0.1, 50.0, 100_000).astype(np.float32)
    all_pass &= test_sort(depths_sorted, "Pre-sorted (100K)")

    # Test 7: Reverse sorted
    all_pass &= test_sort(depths_sorted[::-1].copy(), "Reverse sorted (100K)")

    # Summary
    print("\n" + "=" * 60)
    if all_pass:
        print("  🎉 ALL TESTS PASSED — Metal radix sort is correct!")
    else:
        print("  ❌ SOME TESTS FAILED — check output above")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
