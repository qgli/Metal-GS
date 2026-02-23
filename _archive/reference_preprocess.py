"""
NumPy reference implementation for 3DGS preprocessing (v2 — all bugs fixed).

Matches preprocess.metal math exactly:
  - FOV clamping applied to Jacobian only (screen projection uses original p_cam)
  - Eigenvalue discriminant floor = 0.1 (matches gsplat)
  - Tile bounds use floor/ceil (not integer truncation)
  - Scale multiplication: M = R * diag(s)  (column scaling, NOT element-wise)
"""

import numpy as np


def quat_to_rotmat(q):
    """Quaternion (x,y,z,w) → 3×3 rotation matrix."""
    norm = np.sqrt(np.sum(q * q))
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    q = q / norm
    x, y, z, w = q

    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
        [2*(x*y + w*z),   1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),   1 - 2*(x*x + y*y)],
    ], dtype=np.float32)
    return R


def compute_cov3d(scale, R):
    """Σ3D = R · diag(s) · diag(s)^T · R^T = M · M^T, where M = R · diag(s)."""
    M = R * scale[None, :]          # broadcast: M[i,j] = R[i,j] * s[j]
    return (M @ M.T).astype(np.float32)


def transform_point_4x3(viewmat, p):
    """Row-major 4×4 view matrix: p_cam = [R|t] · [p;1]."""
    p_hom = np.array([p[0], p[1], p[2], 1.0], dtype=np.float32)
    return np.array([np.dot(viewmat[i], p_hom) for i in range(3)], dtype=np.float32)


def preprocess_forward_numpy(
    means3d, scales, quats, viewmat,
    tan_fovx, tan_fovy, focal_x, focal_y,
    principal_x, principal_y, img_width, img_height
):
    N = means3d.shape[0]
    NEAR, BLUR, TILE = 0.01, 0.3, 16

    means2d  = np.zeros((N, 2), dtype=np.float32)
    cov2d    = np.zeros((N, 3), dtype=np.float32)
    depths   = np.zeros(N, dtype=np.float32)
    radii    = np.zeros(N, dtype=np.uint32)
    tile_min = np.zeros((N, 2), dtype=np.uint32)
    tile_max = np.zeros((N, 2), dtype=np.uint32)

    grid_x = (img_width  + TILE - 1) // TILE
    grid_y = (img_height + TILE - 1) // TILE

    for i in range(N):
        R   = quat_to_rotmat(quats[i])
        c3d = compute_cov3d(scales[i], R)

        p_cam = transform_point_4x3(viewmat, means3d[i])
        depth = float(p_cam[2])
        depths[i] = depth
        if depth <= NEAR:
            radii[i] = 0
            continue

        # --- FOV clamping (Jacobian only — do NOT touch p_cam) ---
        lx = 1.3 * tan_fovx
        ly = 1.3 * tan_fovy
        tx = np.clip(p_cam[0] / depth, -lx, lx) * depth
        ty = np.clip(p_cam[1] / depth, -ly, ly) * depth

        # --- Jacobian ---
        rz  = 1.0 / depth
        rz2 = rz * rz
        J = np.array([
            [focal_x * rz,  0.0,            -focal_x * tx * rz2],
            [0.0,           focal_y * rz,   -focal_y * ty * rz2],
        ], dtype=np.float32)

        # --- EWA splatting ---
        W = viewmat[:3, :3].astype(np.float32)
        T = J @ W                           # 2×3
        cov2d_mat = T @ c3d @ T.T           # 2×2
        cov2d_mat[0, 0] += BLUR
        cov2d_mat[1, 1] += BLUR

        a, b, c_ = cov2d_mat[0, 0], cov2d_mat[0, 1], cov2d_mat[1, 1]
        cov2d[i] = [a, b, c_]

        # --- Eigenvalue → radius ---
        det   = max(0.0, a * c_ - b * b)
        bhalf = 0.5 * (a + c_)
        disc  = max(0.1, bhalf * bhalf - det)          # 0.1 floor (gsplat)
        lam   = bhalf + np.sqrt(disc)
        rad_f = np.ceil(3.0 * np.sqrt(lam))
        rad   = int(rad_f)
        if rad == 0:
            radii[i] = 0
            continue
        radii[i] = rad

        # --- Screen projection (ORIGINAL p_cam, not clamped) ---
        u = focal_x * p_cam[0] * rz + principal_x
        v = focal_y * p_cam[1] * rz + principal_y
        means2d[i] = [u, v]

        # --- Tile bounds ---
        tmin_x = int(np.floor((u - rad_f) / TILE))
        tmin_y = int(np.floor((v - rad_f) / TILE))
        tmax_x = int(np.ceil ((u + rad_f) / TILE))
        tmax_y = int(np.ceil ((v + rad_f) / TILE))

        tmin_x = max(0, min(tmin_x, grid_x))
        tmin_y = max(0, min(tmin_y, grid_y))
        tmax_x = max(0, min(tmax_x, grid_x))
        tmax_y = max(0, min(tmax_y, grid_y))

        if tmin_x >= tmax_x or tmin_y >= tmax_y:
            radii[i] = 0
            continue

        tile_min[i] = [tmin_x, tmin_y]
        tile_max[i] = [tmax_x, tmax_y]

    return means2d, cov2d, depths, radii, tile_min, tile_max
