"""
Metal-GS + minGS integration: train 3DGS on Apple Silicon using Metal compute shaders.
No CUDA required — the Metal rasterizer handles GPU dispatch internally via unified memory.

Device flow:
  - Model parameters live on CPU (Apple unified memory = shared with GPU)
  - Metal rasterizer dispatches GPU compute kernels directly on MTLBuffers
  - CPU↔Metal transfer is zero-copy on unified memory architecture
  - Set device="mps" for Apple GPU optimizer (experimental, may stall)
"""
import sys
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add parent dir to path so metal_gs is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Ensure Metal-GS can find its metallib from any working directory
os.environ.setdefault("METAL_GS_METALLIB_DIR",
    os.path.join(os.path.dirname(__file__), '..', 'csrc', 'kernels'))

from gs.core.GaussianModel import GaussianModel
from gs.io.colmap import load
from gs.trainers.basic import train

# ---- Load COLMAP dataset (the cat) ----
cameras, pointcloud = load('./data/cat/')
print(f"Loaded {len(cameras)} cameras, {len(pointcloud.points)} points")

# ---- Downsample camera images to reduce memory pressure on 16GB M1 ----
# 2x: ~516×344, ~726 tiles — cooperative early-out in the rasterize kernel
# ensures threadgroups exit as soon as all pixels are saturated, avoiding
# the macOS GPU watchdog "Impacting Interactivity" timeout.
DOWNSAMPLE = 1
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
print(f"Downsampled images by {DOWNSAMPLE}x → {cameras[0].image_width}x{cameras[0].image_height}")

# ---- Initialize Gaussian model using Metal KNN for scale init ----
model = GaussianModel.from_point_cloud(pointcloud)
print(f"Created GaussianModel with {len(model)} Gaussians")

# ---- Device selection ----
# CPU is the default — on Apple Silicon unified memory, CPU tensors are already
# in GPU-visible memory, so the Metal rasterizer incurs zero copy overhead.
# MPS moves optimizer.step() onto the GPU (~3% speedup) but may cause stalls
# due to resource contention between PyTorch MPS and our Metal command queue.
device = "mps"
print(f"Using device: {device}  (Metal GPU rasterizer active regardless)")

# ---- Train with Metal rasterizer (viewer disabled to avoid GPU contention) ----
ITERATIONS = 500
train(model, cameras, iterations=ITERATIONS, densify_until_iter=30,
      device=device, use_viewer=False)

# ---- Render final image from first camera and save ----
print("\nRendering final image...")
model.eval()
with torch.no_grad():
    camera = cameras[0].to(model.positions.device)
    rendered = model.forward(camera)  # [3, H, W]
    # Convert to PIL and save
    img_np = rendered.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    img_pil.save("cat_mac_render.png")
    print(f"✅ Saved cat_mac_render.png ({img_pil.size[0]}x{img_pil.size[1]})")