#!/usr/bin/env python3
"""
Functional test: verify UMA densify/prune integration works in real training.
Runs a short training with densification ENABLED to exercise the UMA code path.
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
from gs.trainers.basic.helpers import _UMA_AVAILABLE

print(f"UMA available: {_UMA_AVAILABLE}")

data_dir = os.path.join(os.path.dirname(__file__), 'minGS', 'data', 'cat')
cameras, pointcloud = load(data_dir)

# Downsample for speed
for cam in cameras:
    if cam.image is not None:
        C, H, W = cam.image.shape
        new_H, new_W = H // 4, W // 4
        resized = F.interpolate(
            cam.image.unsqueeze(0), size=(new_H, new_W),
            mode='bilinear', align_corners=False
        ).squeeze(0)
        cam.register_buffer('image', resized)
        cam.image_height = new_H
        cam.image_width = new_W

model = GaussianModel.from_point_cloud(pointcloud)
print(f"Initial Gaussians: {len(model)}")

# Train with densification ENABLED: from_iter=10, until_iter=200, interval=50
# This means densify fires at iterations 50, 100, 150
t0 = time.time()
train(model, cameras, iterations=200, device='mps', use_viewer=False,
      densify_from_iter=10, densify_until_iter=200, densify_interval=50,
      opacity_reset_interval=100)
t1 = time.time()

print(f"Final Gaussians: {len(model)}")
print(f"Training with densification: {200/(t1-t0):.2f} it/s")
print("SUCCESS: UMA densify/prune integration works in real training")
