"""Minimal backward pass test to isolate segfault."""
import sys, os, torch
import torch.nn.functional as F

sys.path.insert(0, 'minGS')
os.environ['METAL_GS_METALLIB_DIR'] = 'csrc/kernels'

from gs.core.GaussianModel import GaussianModel
from gs.io.colmap import load

cameras, pc = load('./minGS/data/cat/')
model = GaussianModel.from_point_cloud(pc)

# Downsample like the test script
DOWNSAMPLE = 2
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

device = 'mps'
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

camera = cameras[0].to(device)

print('Forward...')
optimizer.zero_grad()
rendered = model.forward(camera)
print(f'rendered: {rendered.shape}')
target = camera.image.to(device)
print(f'target: {target.shape}')
loss = (rendered - target).pow(2).mean()
print(f'loss: {loss.item():.6f}')

print('Backward...')
loss.backward()
optimizer.step()
print('OK — iter 1 complete')

# iter 2
optimizer.zero_grad()
camera = cameras[1].to(device)
rendered = model.forward(camera)
loss = (rendered - camera.image.to(device)).pow(2).mean()
loss.backward()
optimizer.step()
print('OK — iter 2 complete')
print('Memory:', torch.mps.current_allocated_memory() / 1024**2, 'MB')
