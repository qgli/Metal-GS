#!/usr/bin/env python3
"""
M4 Performance Benchmark for Metal-GS

Runs training with configurable precision and max_gaussians_per_tile.
Usage:
  python benchmark_m4.py                         # FP32, cap=1024
  python benchmark_m4.py --cap 2048              # FP32, cap=2048
  python benchmark_m4.py --cap 0                 # FP32, unlimited
"""
import sys
import os
import math
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add parent dir / minGS to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'minGS'))
sys.path.insert(0, os.path.dirname(__file__))

os.environ.setdefault("METAL_GS_METALLIB_DIR",
    os.path.join(os.path.dirname(__file__), 'csrc', 'kernels'))

from gs.core.GaussianModel import GaussianModel
from gs.io.colmap import load
from gs.trainers.basic import train


def run_benchmark(cap=1024, iterations=500, downsample=2, device="mps"):
    """Run training benchmark and return metrics."""
    # Load dataset
    data_dir = os.path.join(os.path.dirname(__file__), 'minGS', 'data', 'cat')
    cameras, pointcloud = load(data_dir)
    print(f"Loaded {len(cameras)} cameras, {len(pointcloud.points)} points")

    # Downsample
    for cam in cameras:
        if cam.image is not None:
            C, H, W = cam.image.shape
            new_H, new_W = H // downsample, W // downsample
            resized = F.interpolate(
                cam.image.unsqueeze(0), size=(new_H, new_W),
                mode='bilinear', align_corners=False
            ).squeeze(0)
            cam.register_buffer("image", resized)
            cam.image_height = new_H
            cam.image_width = new_W
    print(f"Downsampled {downsample}x -> {cameras[0].image_width}x{cameras[0].image_height}")

    # Patch RenderSettings default to use our cap
    from metal_gs.rasterizer import RenderSettings
    original_init = RenderSettings.__init__
    original_default = RenderSettings.max_gaussians_per_tile

    # Monkey-patch the dataclass default
    RenderSettings.max_gaussians_per_tile = cap
    print(f"max_gaussians_per_tile = {cap}")

    # Initialize model
    model = GaussianModel.from_point_cloud(pointcloud)
    print(f"GaussianModel with {len(model)} Gaussians")
    print(f"Device: {device}")

    # Actual benchmark (first few iterations serve as warmup)
    print(f"\n{'='*60}")
    print(f"  Training {iterations} iterations...")
    print(f"{'='*60}")

    t_start = time.time()
    train(model, cameras, iterations=iterations, densify_until_iter=30,
          device=device, use_viewer=False)
    t_end = time.time()

    elapsed = t_end - t_start
    its = iterations / elapsed

    # Get final loss by doing one more forward pass
    model.eval()
    with torch.no_grad():
        camera = cameras[0].to(model.positions.device)
        rendered = model.forward(camera)
        from gs.helpers.loss import mix_l1_ssim_loss
        final_loss = mix_l1_ssim_loss(rendered, camera.image).item()

    # Restore original
    RenderSettings.max_gaussians_per_tile = original_default

    print(f"\n{'='*60}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"  Precision:                {'BF16' if os.environ.get('METAL_GS_BF16') == '1' else 'FP32'}")
    print(f"  max_gaussians_per_tile:   {cap}")
    print(f"  Resolution:               {cameras[0].image_width}x{cameras[0].image_height}")
    print(f"  Iterations:               {iterations}")
    print(f"  Final Gaussians:          {len(model)}")
    print(f"  Total time:               {elapsed:.2f}s")
    print(f"  Speed:                    {its:.2f} it/s")
    print(f"  Final Loss:               {final_loss:.6f}")
    print(f"{'='*60}")

    return {
        "speed_its": its,
        "total_time": elapsed,
        "final_loss": final_loss,
        "num_gaussians": len(model),
        "cap": cap,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=1024, help="max_gaussians_per_tile (0=unlimited)")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    result = run_benchmark(
        cap=args.cap,
        iterations=args.iterations,
        downsample=args.downsample,
        device=args.device,
    )
