#!/usr/bin/env python3
"""
Metal-GS Kernel Profiling Script

Runs training for N iterations while collecting per-kernel GPU timing data.
Produces a detailed breakdown of where time is spent in the Metal rendering pipeline.

Usage:
  python profile_metal_kernels.py                          # Default: 100 iters
  python profile_metal_kernels.py --iterations 200         # More iters
  METAL_GS_BF16=1 python profile_metal_kernels.py         # With BF16

Output:
  - Terminal: Formatted timing table with percentage breakdown
  - Identifies top bottleneck kernels for optimization targeting
"""
import sys
import os
import time
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'minGS'))
sys.path.insert(0, os.path.dirname(__file__))

os.environ.setdefault("METAL_GS_METALLIB_DIR",
    os.path.join(os.path.dirname(__file__), 'csrc', 'kernels'))

from gs.core.GaussianModel import GaussianModel
from gs.io.colmap import load
from metal_gs.profiler import kernel_profiler
from metal_gs.rasterizer import RenderSettings


def profile_training(iterations=100, warmup=10, downsample=2, device="mps",
                     cap=1024, save_json=False):
    """
    Run training with kernel profiling enabled.
    
    Parameters
    ----------
    iterations : int
        Total iterations (including warmup)
    warmup : int
        Number of warmup iterations to skip in statistics
    downsample : int
        Image downsampling factor
    device : str
        PyTorch device ("mps" or "cpu")
    cap : int
        max_gaussians_per_tile (0=unlimited)
    save_json : bool
        Whether to save raw timing data to JSON
    """
    # ---- Load dataset ----
    data_dir = os.path.join(os.path.dirname(__file__), 'minGS', 'data', 'cat')
    cameras, pointcloud = load(data_dir)
    print(f"Loaded {len(cameras)} cameras, {len(pointcloud.points)} initial points")
    
    # Downsample images
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
    
    W, H = cameras[0].image_width, cameras[0].image_height
    print(f"Resolution: {W}x{H} ({downsample}x downsampled)")
    
    # ---- Patch max_gaussians_per_tile ----
    RenderSettings.max_gaussians_per_tile = cap
    print(f"max_gaussians_per_tile: {cap}")
    print(f"Device: {device}")
    precision = 'BF16' if os.environ.get('METAL_GS_BF16') == '1' else 'FP32'
    print(f"Precision: {precision}")
    
    # ---- Initialize model ----
    model = GaussianModel.from_point_cloud(pointcloud)
    model = model.to(device)
    print(f"Initial Gaussians: {len(model)}")
    
    # ---- Setup optimizer (mirrors train() in gs/trainers/basic/__init__.py) ----
    from gs.helpers.scene import estimate_scene_scale
    scene_scale = estimate_scene_scale(cameras).item()
    
    lr_groups = [
        {"params": [model.positions], "lr": 0.00016 * scene_scale, "name": "positions"},
        {"params": [model.rotations], "lr": 0.001, "name": "rotations"},
        {"params": [model.scales], "lr": 0.005, "name": "scales"},
        {"params": [model.opacities], "lr": 0.05, "name": "opacities"},
        {"params": [model.sh_coefficients_0], "lr": 0.0025, "name": "sh_coefficients_0"},
        {"params": [model.sh_coefficients_rest], "lr": 0.0025 / 20.0, "name": "sh_coefficients_rest"},
    ]
    optimizer = torch.optim.Adam(lr_groups, lr=0.0, eps=1e-15)
    
    # ---- Enable profiler ----
    kernel_profiler.enable()
    
    # ---- Training loop (simplified — no densification for clean profiling) ----
    print(f"\n{'='*60}")
    print(f"  Profiling {iterations} iterations ({warmup} warmup)...")
    print(f"{'='*60}")
    
    from gs.helpers.loss import mix_l1_ssim_loss
    
    num_cameras = len(cameras)
    losses = []
    iter_times = []
    
    for i in range(iterations):
        t_iter_start = time.perf_counter()
        
        camera = cameras[i % num_cameras].to(device)
        
        # Forward
        rendered = model.forward(camera)  # [3, H, W]
        gt_image = camera.image  # [3, H, W]
        
        # Loss
        loss = mix_l1_ssim_loss(rendered, gt_image)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        model.backprop_stats()
        
        # Step (with no_grad for optimizer)
        with torch.no_grad():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        t_iter_end = time.perf_counter()
        iter_ms = (t_iter_end - t_iter_start) * 1000.0
        iter_times.append(iter_ms)
        losses.append(loss.item())
        
        if i < 5 or (i + 1) % 20 == 0:
            print(f"  iter {i+1:>4d}/{iterations}  "
                  f"loss={loss.item():.4f}  "
                  f"iter_ms={iter_ms:.1f}  "
                  f"N={len(model)}")
    
    # ---- Print results ----
    kernel_profiler.print_summary(warmup=warmup)
    
    # Wall-clock summary
    avg_iter_ms = np.mean(iter_times[warmup:])
    std_iter_ms = np.std(iter_times[warmup:])
    print(f"  Wall-clock per iteration: {avg_iter_ms:.2f} ± {std_iter_ms:.2f} ms")
    print(f"  Wall-clock it/s: {1000.0/avg_iter_ms:.2f}")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Final Gaussians: {len(model)}")
    
    # ---- Compute overhead analysis ----
    summary = kernel_profiler.get_summary(warmup=warmup)
    fwd = summary.get("forward", {})
    bwd = summary.get("backward", {})
    
    gpu_fwd = sum(v["mean_ms"] for k, v in fwd.items() 
                  if k not in ("numpy_convert", "torch_convert"))
    gpu_bwd = sum(v["mean_ms"] for k, v in bwd.items() 
                  if k not in ("numpy_convert_bw", "torch_convert_bw"))
    cpu_overhead = sum(v["mean_ms"] for k, v in fwd.items() 
                       if k in ("numpy_convert", "torch_convert"))
    cpu_overhead += sum(v["mean_ms"] for k, v in bwd.items() 
                        if k in ("numpy_convert_bw", "torch_convert_bw"))
    
    total_profiled = gpu_fwd + gpu_bwd + cpu_overhead
    optimizer_ms = avg_iter_ms - total_profiled
    
    print(f"\n  {'─'*60}")
    print(f"  TIME BUDGET ANALYSIS (per iteration)")
    print(f"  {'─'*60}")
    print(f"    GPU Compute (Forward):   {gpu_fwd:>7.2f} ms  ({100*gpu_fwd/avg_iter_ms:.1f}%)")
    print(f"    GPU Compute (Backward):  {gpu_bwd:>7.2f} ms  ({100*gpu_bwd/avg_iter_ms:.1f}%)")
    print(f"    CPU↔GPU Transfer:        {cpu_overhead:>7.2f} ms  ({100*cpu_overhead/avg_iter_ms:.1f}%)")
    print(f"    Optimizer + Loss + Other: {optimizer_ms:>7.2f} ms  ({100*optimizer_ms/avg_iter_ms:.1f}%)")
    print(f"    {'─'*56}")
    print(f"    TOTAL Wall-clock:        {avg_iter_ms:>7.2f} ms")
    print(f"  {'─'*60}\n")
    
    # ---- Save JSON if requested ----
    if save_json:
        out_path = os.path.join(os.path.dirname(__file__), 
                                'docs', 'reports', 'kernel_profiling_data.json')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        export = {
            "config": {
                "iterations": iterations,
                "warmup": warmup,
                "downsample": downsample,
                "device": device,
                "cap": cap,
                "precision": precision,
                "resolution": f"{W}x{H}",
                "num_cameras": num_cameras,
                "final_gaussians": len(model),
            },
            "wall_clock": {
                "avg_iter_ms": float(avg_iter_ms),
                "std_iter_ms": float(std_iter_ms),
                "iter_times": [float(t) for t in iter_times],
            },
            "summary": {
                "forward": {k: {kk: float(vv) for kk, vv in v.items() if kk != "label"} 
                           for k, v in fwd.items()},
                "backward": {k: {kk: float(vv) for kk, vv in v.items() if kk != "label"} 
                            for k, v in bwd.items()},
            },
            "losses": [float(l) for l in losses],
        }
        
        with open(out_path, 'w') as f:
            json.dump(export, f, indent=2)
        print(f"  Raw data saved to: {out_path}")
    
    kernel_profiler.disable()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metal-GS Kernel Profiling")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Total iterations including warmup (default: 100)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations to skip (default: 10)")
    parser.add_argument("--downsample", type=int, default=2,
                        help="Image downsample factor (default: 2)")
    parser.add_argument("--device", type=str, default="mps",
                        help="PyTorch device (default: mps)")
    parser.add_argument("--cap", type=int, default=1024,
                        help="max_gaussians_per_tile (default: 1024)")
    parser.add_argument("--save-json", action="store_true",
                        help="Save raw timing data to JSON")
    args = parser.parse_args()
    
    profile_training(
        iterations=args.iterations,
        warmup=args.warmup,
        downsample=args.downsample,
        device=args.device,
        cap=args.cap,
        save_json=args.save_json,
    )
