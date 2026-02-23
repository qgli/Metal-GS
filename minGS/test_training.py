"""Quick training test to verify multi-command-buffer fix prevents GPU hang."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('METAL_GS_METALLIB_DIR',
    os.path.join(os.path.dirname(__file__), '..', 'csrc', 'kernels'))

print("[test] Loading modules...", flush=True)
from gs.core.GaussianModel import GaussianModel
from gs.io.colmap import load
from gs.trainers.basic import train

print("[test] Loading COLMAP data...", flush=True)
t0 = time.time()
cameras, pointcloud = load(os.path.join(os.path.dirname(__file__), 'data/cat/'))
print(f'[test] Loaded {len(cameras)} cameras, {len(pointcloud.points)} points in {time.time()-t0:.1f}s', flush=True)

print("[test] Creating GaussianModel (KNN)...", flush=True)
t0 = time.time()
model = GaussianModel.from_point_cloud(pointcloud)
print(f'[test] Created GaussianModel with {len(model)} Gaussians in {time.time()-t0:.1f}s', flush=True)

N_ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
print(f'[test] Training for {N_ITERS} iterations...', flush=True)
t0 = time.time()
train(model, cameras, iterations=N_ITERS, densify_until_iter=30, device='cpu', use_viewer=False)
print(f'[test] SUCCESS: {N_ITERS} iterations in {time.time()-t0:.1f}s', flush=True)
