"""Debug training loop to find where MPS hangs."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault("METAL_GS_METALLIB_DIR",
    os.path.join(os.path.dirname(__file__), '..', 'csrc', 'kernels'))

import torch
import numpy as np
from gs.core.GaussianModel import GaussianModel
from gs.io.colmap import load
from gs.helpers.loss import mix_l1_ssim_loss

# Load data
cameras, pointcloud = load('./data/cat/')
model = GaussianModel.from_point_cloud(pointcloud)
print(f"Model: {len(model)} Gaussians")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")
model.to(device)

# Simple optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-15)

for i in range(60):
    t0 = time.time()
    camera = cameras[i % len(cameras)].to(device)
    t1 = time.time()

    rendered = model.forward(camera, active_sh_degree=0)
    t2 = time.time()

    loss = mix_l1_ssim_loss(rendered, camera.image)
    t3 = time.time()

    loss.backward()
    t4 = time.time()

    with torch.no_grad():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    t5 = time.time()

    print(f"iter {i:3d} | loss={loss.item():.4f} | "
          f"cam={t1-t0:.3f}s fwd={t2-t1:.3f}s loss={t3-t2:.3f}s "
          f"bwd={t4-t3:.3f}s opt={t5-t4:.3f}s total={t5-t0:.3f}s",
          flush=True)

print("✅ 60 iterations completed")
