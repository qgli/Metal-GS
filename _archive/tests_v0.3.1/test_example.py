#!/usr/bin/env python3
"""Quick test: run minGS training for 200 iterations, save output image."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'minGS'))
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("METAL_GS_METALLIB_DIR",
    os.path.join(os.path.dirname(__file__), 'csrc', 'kernels'))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from gs.core.GaussianModel import GaussianModel
from gs.io.colmap import load
from gs.trainers.basic import train

data_dir = os.path.join(os.path.dirname(__file__), 'minGS', 'data', 'cat')
cameras, pointcloud = load(data_dir)
print(f"Loaded {len(cameras)} cameras, {len(pointcloud.points)} points")

for cam in cameras:
    if cam.image is not None:
        C, H, W = cam.image.shape
        new_H, new_W = H // 2, W // 2
        resized = F.interpolate(
            cam.image.unsqueeze(0), size=(new_H, new_W),
            mode='bilinear', align_corners=False
        ).squeeze(0)
        cam.register_buffer("image", resized)
        cam.image_height = new_H
        cam.image_width = new_W
print(f"Downsampled → {cameras[0].image_width}x{cameras[0].image_height}")

model = GaussianModel.from_point_cloud(pointcloud)
print(f"Model: {len(model)} Gaussians")

train(model, cameras, iterations=200, densify_until_iter=30,
      device='mps', use_viewer=False)

# Render
model.eval()
with torch.no_grad():
    camera = cameras[0].to(model.positions.device)
    rendered = model.forward(camera)
    img_np = rendered.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    img_pil.save("cat_v3_fixed.png")
    print(f"Saved cat_v3_fixed.png ({img_pil.size[0]}x{img_pil.size[1]})")
print("DONE")
