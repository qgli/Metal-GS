#!/usr/bin/env python3
"""
E2E Benchmark: UMA-integrated training vs PyTorch-only baseline.
Runs the same training twice with densification enabled to measure actual
UMA impact on densify/prune operations.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'minGS'))
os.environ.setdefault('METAL_GS_METALLIB_DIR',
    os.path.join(os.path.dirname(__file__), 'csrc', 'kernels'))

import torch
import torch.nn.functional as F
from gs.core.GaussianModel import GaussianModel
from gs.io.colmap import load
from gs.trainers.basic import train
import gs.trainers.basic.helpers as helpers

ITERATIONS = 300
DENSIFY_FROM = 10
DENSIFY_UNTIL = 300
DENSIFY_INTERVAL = 50
DOWNSAMPLE = 2

def prepare():
    data_dir = os.path.join(os.path.dirname(__file__), 'minGS', 'data', 'cat')
    cameras, pointcloud = load(data_dir)
    for cam in cameras:
        if cam.image is not None:
            C, H, W = cam.image.shape
            new_H, new_W = H // DOWNSAMPLE, W // DOWNSAMPLE
            resized = F.interpolate(
                cam.image.unsqueeze(0), size=(new_H, new_W),
                mode='bilinear', align_corners=False
            ).squeeze(0)
            cam.register_buffer('image', resized)
            cam.image_height = new_H
            cam.image_width = new_W
    return cameras, pointcloud

def run_one(cameras, pointcloud, use_uma, label):
    model = GaussianModel.from_point_cloud(pointcloud)
    n_init = len(model)

    # Toggle UMA
    original = helpers._UMA_AVAILABLE
    helpers._UMA_AVAILABLE = use_uma

    t0 = time.time()
    train(model, cameras, iterations=ITERATIONS, device='mps', use_viewer=False,
          densify_from_iter=DENSIFY_FROM, densify_until_iter=DENSIFY_UNTIL,
          densify_interval=DENSIFY_INTERVAL, opacity_reset_interval=100)
    t1 = time.time()

    helpers._UMA_AVAILABLE = original  # restore

    elapsed = t1 - t0
    its = ITERATIONS / elapsed

    # Final loss
    model.eval()
    with torch.no_grad():
        camera = cameras[0].to(model.positions.device)
        rendered = model.forward(camera)
        from gs.helpers.loss import mix_l1_ssim_loss
        loss = mix_l1_ssim_loss(rendered, camera.image).item()

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Gaussians: {n_init} → {len(model)}")
    print(f"  Time:      {elapsed:.2f}s")
    print(f"  Speed:     {its:.2f} it/s")
    print(f"  Loss:      {loss:.6f}")
    print(f"{'='*60}")
    return {"its": its, "loss": loss, "n_final": len(model), "elapsed": elapsed}

if __name__ == "__main__":
    cameras, pointcloud = prepare()
    print(f"Dataset: {len(cameras)} cameras, {len(pointcloud.points)} points")
    print(f"Resolution: {cameras[0].image_width}x{cameras[0].image_height}")
    print(f"Iterations: {ITERATIONS}, densify every {DENSIFY_INTERVAL} from {DENSIFY_FROM}-{DENSIFY_UNTIL}")

    print(f"\nUMA module available: {helpers._UMA_AVAILABLE}")

    # Run baseline (PyTorch only)
    print("\n>>> Running BASELINE (PyTorch-only densify/prune)...")
    baseline = run_one(cameras, pointcloud, use_uma=False, label="BASELINE (PyTorch)")

    # Run UMA
    print("\n>>> Running UMA (zero-copy densify/prune)...")
    uma_result = run_one(cameras, pointcloud, use_uma=True, label="UMA ZERO-COPY")

    # Summary
    speedup = uma_result["its"] / baseline["its"]
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline:  {baseline['its']:.2f} it/s | Loss: {baseline['loss']:.6f} | Gaussians: {baseline['n_final']}")
    print(f"  UMA:       {uma_result['its']:.2f} it/s | Loss: {uma_result['loss']:.6f} | Gaussians: {uma_result['n_final']}")
    print(f"  Speedup:   {speedup:.3f}x")
    print(f"{'='*60}")
