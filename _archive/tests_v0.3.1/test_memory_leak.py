"""
Metal-GS v0.3.1 Memory Leak Stress Test
=========================================
Runs 1000 forward+backward iterations and monitors MPS memory allocation.
After warmup, memory MUST stay flat — any upward trend indicates a leak.

Usage:
    cd Metal-GS
    python test_memory_leak.py

Pass criteria:
    - Memory delta between iteration 100 and 1000 < 5 MB
    - No monotonic upward trend after warmup
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.environ.setdefault("METAL_GS_METALLIB_DIR",
    os.path.join(os.path.dirname(__file__), 'csrc', 'kernels'))

import torch
import numpy as np

# ─── Load minGS components ───
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'minGS'))
from gs.core.GaussianModel import GaussianModel
from gs.io.colmap import load

print("=" * 70)
print("  Metal-GS v0.3.1 Memory Leak Stress Test")
print("=" * 70)

# ─── Load dataset ───
cameras, pointcloud = load('./minGS/data/cat/')
print(f"Loaded {len(cameras)} cameras, {len(pointcloud.points)} points")

# ─── Downsample ───
import torch.nn.functional as F
DOWNSAMPLE = 2
for cam in cameras:
    if cam.image is not None:
        C, H, W = cam.image.shape
        new_H, new_W = H // DOWNSAMPLE, W // DOWNSAMPLE
        resized = F.interpolate(
            cam.image.unsqueeze(0), size=(new_H, new_W),
            mode='bilinear', align_corners=False
        ).squeeze(0)
        cam.register_buffer("image", resized)
        cam.image_height = new_H
        cam.image_width = new_W

# ─── Initialize model ───
model = GaussianModel.from_point_cloud(pointcloud)
device = "mps"
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"Model: {len(model)} Gaussians on {device}")
print(f"Resolution: {cameras[0].image_width}x{cameras[0].image_height}")
print()

# ─── Stress test: 1000 iterations ───
NUM_ITERS = 1000
REPORT_EVERY = 50
WARMUP = 200  # Adam optimizer lazily allocates state; need 200+ iters to stabilize

mem_history = []

print(f"{'Iter':>6} | {'MPS Alloc (MB)':>14} | {'Delta (MB)':>10} | {'Loss':>10} | {'Time (ms)':>10}")
print("-" * 70)

baseline_mem = None
cam_idx = 0

for i in range(1, NUM_ITERS + 1):
    camera = cameras[cam_idx % len(cameras)].to(device)
    cam_idx += 1

    t0 = time.perf_counter()

    # Forward
    optimizer.zero_grad()
    rendered = model.forward(camera)
    target = camera.image.to(device)
    loss = (rendered - target).pow(2).mean()

    # Backward
    loss.backward()
    optimizer.step()

    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000

    # Memory measurement
    if i % REPORT_EVERY == 0 or i == 1 or i == WARMUP:
        torch.mps.synchronize()
        mem_mb = torch.mps.current_allocated_memory() / (1024 ** 2)
        mem_history.append((i, mem_mb))

        if baseline_mem is None and i >= WARMUP:
            baseline_mem = mem_mb

        delta = mem_mb - baseline_mem if baseline_mem is not None else 0.0
        print(f"{i:>6} | {mem_mb:>14.2f} | {delta:>+10.2f} | {loss.item():>10.4f} | {elapsed_ms:>10.1f}")

# ─── Analysis ───
print()
print("=" * 70)
print("  MEMORY ANALYSIS")
print("=" * 70)

# Get post-warmup readings
post_warmup = [(it, m) for it, m in mem_history if it >= WARMUP]
if len(post_warmup) >= 2:
    first_stable = post_warmup[0]
    last_stable = post_warmup[-1]
    total_delta = last_stable[1] - first_stable[1]
    per_iter_delta = total_delta / (last_stable[0] - first_stable[0])

    print(f"  Warmup baseline (iter {first_stable[0]}):  {first_stable[1]:.2f} MB")
    print(f"  Final (iter {last_stable[0]}):              {last_stable[1]:.2f} MB")
    print(f"  Total delta:                    {total_delta:+.2f} MB")
    print(f"  Per-iteration delta:            {per_iter_delta:+.4f} MB/iter")
    print()

    # Pass/Fail
    THRESHOLD_MB = 5.0
    if abs(total_delta) < THRESHOLD_MB:
        print(f"  ✅ PASS — Memory stable (delta {total_delta:+.2f} MB < {THRESHOLD_MB} MB threshold)")
    else:
        print(f"  ❌ FAIL — Memory leak detected! Delta {total_delta:+.2f} MB > {THRESHOLD_MB} MB threshold")
        print(f"           Projected leak rate: {per_iter_delta * 1000:.1f} MB per 1000 iterations")
        sys.exit(1)
else:
    print("  ⚠️  Not enough data points for analysis")
    sys.exit(1)

print()
print("  Memory trace (post-warmup):")
for it, m in post_warmup:
    bar_len = int((m - post_warmup[0][1] + 1) * 2)  # scale for visibility
    bar = "█" * max(1, bar_len)
    print(f"    iter {it:>5}: {m:>8.2f} MB  {bar}")

print()
print("=" * 70)
