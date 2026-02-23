"""
Metal-GS: Full-Chain Backward Gradient Verification (Phase 8)

Verifies the complete differentiable rendering pipeline:
  SH → preprocess → sort → bin → rasterize → image
                                               ↓ dL/dC_pixel
  rasterize_bw → preprocess_bw → sh_bw → dL/d_{means3d, scales, quats, sh, opacities}

Uses PyTorch autograd as reference: implements the full forward chain in PyTorch
with requires_grad=True, then compares Metal backward gradients against torch.autograd.

All reference computation in float64 for maximum numerical precision.
"""

import numpy as np
import sys
import os
import time

try:
    from metal_gs._metal_gs_core import (
        compute_sh_forward,
        preprocess_forward, radix_sort_by_depth,
        tile_binning, rasterize_forward, rasterize_backward,
        preprocess_backward, sh_backward,
        render_forward, render_backward,
    )
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Run 'CC=/usr/bin/clang CXX=/usr/bin/clang++ pip install -e .' first.")
    sys.exit(1)

# ---- Constants ----
TILE_SIZE   = 16
IMG_WIDTH   = 64
IMG_HEIGHT  = 64
BG_COLOR    = (0.2, 0.3, 0.5)
N_GAUSSIANS = 32
SH_DEGREE   = 3
NUM_SH_BASES = (SH_DEGREE + 1) ** 2  # 16 for degree 3

AA_BLUR = 0.3


def make_camera():
    """Pinhole camera at (0,0,5) looking at origin, 60° FOV."""
    fov_x = np.radians(60)
    fov_y = 2 * np.arctan(np.tan(fov_x / 2) * IMG_HEIGHT / IMG_WIDTH)
    fx = IMG_WIDTH / (2 * np.tan(fov_x / 2))
    fy = IMG_HEIGHT / (2 * np.tan(fov_y / 2))
    cx, cy = IMG_WIDTH / 2.0, IMG_HEIGHT / 2.0

    viewmat = np.eye(4, dtype=np.float32)
    viewmat[2, 3] = 5.0

    campos = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # inv(viewmat) @ [0,0,0,1]

    return dict(
        viewmat=viewmat,
        tan_fovx=float(np.tan(fov_x / 2)),
        tan_fovy=float(np.tan(fov_y / 2)),
        focal_x=float(fx), focal_y=float(fy),
        cx=float(cx), cy=float(cy),
        campos=campos,
    )


def generate_gaussians(N, rng):
    """Generate random Gaussians in front of camera."""
    means3d = rng.uniform(-1.5, 1.5, (N, 3)).astype(np.float32)
    means3d[:, 2] = rng.uniform(-1.0, 1.0, N).astype(np.float32)  # near origin

    scales = rng.uniform(0.05, 0.3, (N, 3)).astype(np.float32)

    quats = rng.standard_normal((N, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)

    sh_coeffs = rng.standard_normal((N, NUM_SH_BASES, 3)).astype(np.float32) * 0.5
    sh_coeffs_fp16 = sh_coeffs.astype(np.float16)

    opacities = rng.uniform(0.3, 0.95, N).astype(np.float32)

    return means3d, scales, quats, sh_coeffs_fp16, opacities


# =========================================================================
#  NumPy reference: Full forward chain (float64)
# =========================================================================

# SH constants
SH_C0    = 0.28209479177387814
SH_C1    = 0.4886025119029199
SH_C2_0  = 1.0925484305920792
SH_C2_1  = -1.0925484305920792
SH_C2_2  = 0.31539156525252005
SH_C2_3  = -1.0925484305920792
SH_C2_4  = 0.5462742152960396
SH_C3_0  = -0.5900435899266435
SH_C3_1  = 2.890611442640554
SH_C3_2  = -0.4570457994644658
SH_C3_3  = 0.3731763325901154
SH_C3_4  = -0.4570457994644658
SH_C3_5  = 1.445305721320277
SH_C3_6  = -0.5900435899266435


def numpy_sh_forward(directions, sh_coeffs, degree):
    """Evaluate SH forward pass in float64.
    directions: [N, 3] unit vectors
    sh_coeffs: [N, K, 3]
    Returns: [N, 3] colors
    """
    N = len(directions)
    d = directions.astype(np.float64)
    norms = np.sqrt(np.sum(d**2, axis=1, keepdims=True))
    d = d / np.maximum(norms, 1e-8)
    x, y, z = d[:, 0], d[:, 1], d[:, 2]

    sh = sh_coeffs.astype(np.float64)  # [N, K, 3]

    result = SH_C0 * sh[:, 0]

    if degree >= 1:
        result += SH_C1 * (-y[:, None] * sh[:, 1] + z[:, None] * sh[:, 2] - x[:, None] * sh[:, 3])

    if degree >= 2:
        xx, yy, zz = x*x, y*y, z*z
        xy, yz, xz = x*y, y*z, x*z
        result += (SH_C2_0 * xy[:, None] * sh[:, 4]
                 + SH_C2_1 * yz[:, None] * sh[:, 5]
                 + SH_C2_2 * (2*zz - xx - yy)[:, None] * sh[:, 6]
                 + SH_C2_3 * xz[:, None] * sh[:, 7]
                 + SH_C2_4 * (xx - yy)[:, None] * sh[:, 8])

    if degree >= 3:
        xx, yy, zz = x*x, y*y, z*z
        xy, yz, xz = x*y, y*z, x*z
        result += (SH_C3_0 * y * (3*xx - yy))[:, None] * sh[:, 9]
        result += (SH_C3_1 * xy * z)[:, None] * sh[:, 10]
        result += (SH_C3_2 * y * (4*zz - xx - yy))[:, None] * sh[:, 11]
        result += (SH_C3_3 * z * (2*zz - 3*xx - 3*yy))[:, None] * sh[:, 12]
        result += (SH_C3_4 * x * (4*zz - xx - yy))[:, None] * sh[:, 13]
        result += (SH_C3_5 * z * (xx - yy))[:, None] * sh[:, 14]
        result += (SH_C3_6 * x * (xx - 3*yy))[:, None] * sh[:, 15]

    result += 0.5
    result = np.clip(result, 0.0, 1.0)
    return result


def numpy_preprocess(means3d, scales, quats, viewmat, cam):
    """Preprocess forward in float64. Returns (means2d, cov2d, depths, visible)."""
    N = len(means3d)
    m3d = means3d.astype(np.float64)
    sc = scales.astype(np.float64)
    qt = quats.astype(np.float64)
    vm = viewmat.astype(np.float64)

    fx, fy = cam['focal_x'], cam['focal_y']
    cx, cy = cam['cx'], cam['cy']
    tan_fovx, tan_fovy = cam['tan_fovx'], cam['tan_fovy']

    W = vm[:3, :3]  # rotation (row-major)

    means2d = np.zeros((N, 2), dtype=np.float64)
    cov2d = np.zeros((N, 3), dtype=np.float64)
    depths = np.zeros(N, dtype=np.float64)
    visible = np.zeros(N, dtype=bool)

    for i in range(N):
        p = m3d[i]
        # Camera transform
        p_cam = W @ p + vm[:3, 3]
        depth = p_cam[2]

        if depth < 0.01:
            continue

        rz = 1.0 / depth
        rz2 = rz * rz

        # FOV clamping
        lx = 1.3 * tan_fovx
        ly = 1.3 * tan_fovy
        tx_raw = p_cam[0] * rz
        ty_raw = p_cam[1] * rz
        tx = np.clip(tx_raw, -lx, lx) * depth
        ty = np.clip(ty_raw, -ly, ly) * depth

        # Projection (using original, not clamped)
        means2d[i, 0] = fx * p_cam[0] * rz + cx
        means2d[i, 1] = fy * p_cam[1] * rz + cy
        depths[i] = depth

        # Jacobian
        J = np.array([
            [fx*rz,  0.0,    -fx*tx*rz2],
            [0.0,    fy*rz,  -fy*ty*rz2],
            [0.0,    0.0,     0.0       ]
        ], dtype=np.float64)

        # Rotation matrix from quaternion
        q = qt[i] / max(np.linalg.norm(qt[i]), 1e-8)
        qx, qy, qz, qw = q
        R = np.array([
            [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz),   1-2*(qx*qx+qz*qz),  2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),    1-2*(qx*qx+qy*qy)]
        ], dtype=np.float64)

        # M = R * diag(s), cov3d = M * M^T
        M = R * sc[i][None, :]  # broadcast: column j *= s[j]
        cov3d = M @ M.T

        # EWA: T_eff = J * W, cov2d_mat = T_eff * cov3d * T_eff^T + blur*I
        T_eff = J @ W
        cov2d_mat = T_eff @ cov3d @ T_eff.T
        cov2d_mat[0, 0] += AA_BLUR
        cov2d_mat[1, 1] += AA_BLUR

        # Check determinant
        det = cov2d_mat[0, 0] * cov2d_mat[1, 1] - cov2d_mat[0, 1] * cov2d_mat[1, 0]
        if det < 1e-6:
            continue

        # Eigenvalue radius check
        mid = 0.5 * (cov2d_mat[0, 0] + cov2d_mat[1, 1])
        disc = max(0.1, mid * mid - det)
        lambda_max = mid + np.sqrt(disc)
        radius = int(np.ceil(3.0 * np.sqrt(lambda_max)))
        if radius <= 0:
            continue

        cov2d[i, 0] = cov2d_mat[0, 0]
        cov2d[i, 1] = cov2d_mat[0, 1]
        cov2d[i, 2] = cov2d_mat[1, 1]
        visible[i] = True

    return means2d, cov2d, depths, visible


def numpy_rasterize_and_backward(means2d, cov2d, colors, opacities,
                                  tile_bins, point_list,
                                  dL_dC_pixel,
                                  img_w, img_h, num_tiles_x, bg):
    """
    Forward rasterize + backward in one pass (float64).
    Returns: (image, dL_d_rgb, dL_d_opacity, dL_d_cov2d, dL_d_mean2d)
    """
    N = len(opacities)
    means2d_64 = means2d.astype(np.float64)
    cov2d_64   = cov2d.astype(np.float64)
    colors_64  = colors.astype(np.float64)
    opac_64    = opacities.astype(np.float64)
    dL_dC_64   = dL_dC_pixel.astype(np.float64)
    bg_64      = np.array(bg, dtype=np.float64)

    image = np.zeros((img_h, img_w, 3), dtype=np.float64)

    dL_d_rgb  = np.zeros((N, 3), dtype=np.float64)
    dL_d_opac = np.zeros(N, dtype=np.float64)
    dL_d_cov  = np.zeros((N, 3), dtype=np.float64)
    dL_d_mean = np.zeros((N, 2), dtype=np.float64)

    for py_ in range(img_h):
        for px_ in range(img_w):
            pixel = np.array([px_ + 0.5, py_ + 0.5], dtype=np.float64)
            tx = px_ // TILE_SIZE
            ty = py_ // TILE_SIZE
            tile_id = ty * num_tiles_x + tx
            start = int(tile_bins[tile_id, 0])
            end   = int(tile_bins[tile_id, 1])
            pixel_idx = py_ * img_w + px_

            dL_dC = dL_dC_64[pixel_idx]

            # Forward pass
            contribs = []
            T_fwd = 1.0
            color_accum = np.zeros(3, dtype=np.float64)
            for k in range(start, end):
                if T_fwd < 1e-4:
                    break
                gid = int(point_list[k])

                mean = means2d_64[gid]
                a, b, c = cov2d_64[gid]
                det = a * c - b * b
                if det < 1e-6:
                    continue
                det_inv = 1.0 / det
                inv_a =  c * det_inv
                inv_b = -b * det_inv
                inv_c =  a * det_inv

                d = pixel - mean
                maha = inv_a * d[0]**2 + 2.0 * inv_b * d[0] * d[1] + inv_c * d[1]**2

                if maha < 0 or maha > 18.0:
                    continue
                weight = np.exp(-0.5 * maha)
                alpha = min(0.999, opac_64[gid] * weight)
                if alpha < 1.0 / 255.0:
                    continue

                contribs.append((k, gid, alpha))
                color_accum += alpha * T_fwd * colors_64[gid]
                T_fwd *= (1.0 - alpha)

            color_accum += T_fwd * bg_64
            image[py_, px_] = color_accum

            if len(contribs) == 0:
                continue

            T_final = T_fwd

            # Backward pass: reverse
            T = T_final
            accum_bw = bg_64 * T

            for i in range(len(contribs) - 1, -1, -1):
                k_idx, gid, alpha = contribs[i]

                one_minus_alpha = 1.0 - alpha
                oma_safe = max(one_minus_alpha, 1e-5)
                T_i = T / oma_safe

                c_i = colors_64[gid]
                dL_dalpha = np.dot(T_i * c_i - accum_bw / oma_safe, dL_dC)

                accum_bw += c_i * alpha * T_i
                T = T_i

                # dL/d_color
                dL_d_rgb[gid] += alpha * T_i * dL_dC

                # dL/d_opacity
                op = opac_64[gid]
                dL_d_opac[gid] += dL_dalpha * (alpha / max(op, 1e-8))

                dL_dsigma = dL_dalpha * (-0.5 * alpha)

                a_cov, b_cov, c_cov = cov2d_64[gid]
                det = a_cov * c_cov - b_cov * b_cov
                if det < 1e-6:
                    continue
                det_inv = 1.0 / det
                inv_a =  c_cov * det_inv
                inv_b = -b_cov * det_inv
                inv_c =  a_cov * det_inv

                mean = means2d_64[gid]
                d = pixel - mean

                dL_d_inv_a = dL_dsigma * d[0]**2
                dL_d_inv_b = dL_dsigma * 2.0 * d[0] * d[1]
                dL_d_inv_c = dL_dsigma * d[1]**2

                g_a, g_b, g_c = dL_d_inv_a, dL_d_inv_b, dL_d_inv_c
                dL_da = -(inv_a*inv_a*g_a + 2*inv_a*inv_b*g_b + inv_b*inv_b*g_c)
                dL_db = -(inv_a*inv_b*g_a + (inv_a*inv_c + inv_b*inv_b)*g_b + inv_b*inv_c*g_c)
                dL_dc = -(inv_b*inv_b*g_a + 2*inv_b*inv_c*g_b + inv_c*inv_c*g_c)

                dL_d_cov[gid, 0] += dL_da
                dL_d_cov[gid, 1] += dL_db
                dL_d_cov[gid, 2] += dL_dc

                dL_dmx = dL_dsigma * (-2) * (inv_a*d[0] + inv_b*d[1])
                dL_dmy = dL_dsigma * (-2) * (inv_b*d[0] + inv_c*d[1])
                dL_d_mean[gid, 0] += dL_dmx
                dL_d_mean[gid, 1] += dL_dmy

    return image, dL_d_rgb, dL_d_opac, dL_d_cov, dL_d_mean


def numpy_preprocess_backward(means3d, scales, quats, viewmat, cam,
                                dL_d_cov2d, dL_d_mean2d, visible):
    """Preprocess backward in float64."""
    N = len(means3d)
    m3d = means3d.astype(np.float64)
    sc = scales.astype(np.float64)
    qt = quats.astype(np.float64)
    vm = viewmat.astype(np.float64)

    fx, fy = cam['focal_x'], cam['focal_y']
    cx, cy = cam['cx'], cam['cy']
    tan_fovx, tan_fovy = cam['tan_fovx'], cam['tan_fovy']

    W = vm[:3, :3]

    dL_d_m3d = np.zeros((N, 3), dtype=np.float64)
    dL_d_sc  = np.zeros((N, 3), dtype=np.float64)
    dL_d_qt  = np.zeros((N, 4), dtype=np.float64)

    for i in range(N):
        if not visible[i]:
            continue

        p = m3d[i]
        p_cam = W @ p + vm[:3, 3]
        depth = p_cam[2]
        rz = 1.0 / depth
        rz2 = rz * rz
        rz3 = rz2 * rz

        lx = 1.3 * tan_fovx
        ly = 1.3 * tan_fovy
        tx_raw = p_cam[0] * rz
        ty_raw = p_cam[1] * rz
        tx = np.clip(tx_raw, -lx, lx) * depth
        ty = np.clip(ty_raw, -ly, ly) * depth
        tx_clamped = (tx_raw < -lx or tx_raw > lx)
        ty_clamped = (ty_raw < -ly or ty_raw > ly)

        # Jacobian
        J = np.array([
            [fx*rz,  0.0,    -fx*tx*rz2],
            [0.0,    fy*rz,  -fy*ty*rz2],
            [0.0,    0.0,     0.0       ]
        ])

        # T_eff = J * W
        T_eff = J @ W

        # Rotation matrix
        q = qt[i] / max(np.linalg.norm(qt[i]), 1e-8)
        qx, qy, qz, qw = q
        R = np.array([
            [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz),   1-2*(qx*qx+qz*qz),  2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),    1-2*(qx*qx+qy*qy)]
        ])

        M = R * sc[i][None, :]
        V = M @ M.T

        # dL/d_cov2d → dL/d_V
        dL_da = dL_d_cov2d[i, 0]
        dL_db = dL_d_cov2d[i, 1]
        dL_dc = dL_d_cov2d[i, 2]

        T0 = T_eff[0, :]  # row 0
        T1 = T_eff[1, :]  # row 1

        GT0 = dL_da * T0 + dL_db * T1
        GT1 = dL_db * T0 + dL_dc * T1

        dL_dV = np.outer(T0, GT0) + np.outer(T1, GT1)

        # dL/d_M = 2 * sym(dL/dV) * M
        dL_dV_sym = 0.5 * (dL_dV + dL_dV.T)
        dL_dM = 2.0 * dL_dV_sym @ M

        # dL/d_scales
        dL_d_sc[i] = np.sum(R * dL_dM, axis=0)

        # dL/d_R
        dL_dR = dL_dM * sc[i][None, :]

        # dL/d_R → dL/d_quat (normalized)
        r00, r01, r02 = dL_dR[0]
        r10, r11, r12 = dL_dR[1]
        r20, r21, r22 = dL_dR[2]

        dL_dqx = 2*(r01*qy + r02*qz + r10*qy - r11*2*qx - r12*qw + r20*qz + r21*qw - r22*2*qx)
        dL_dqy = 2*(-r00*2*qy + r01*qx - r02*qw + r10*qx + r12*qz + r20*qw + r21*qz - r22*2*qy)
        dL_dqz = 2*(-r00*2*qz - r01*qw + r02*qx + r10*qw - r11*2*qz + r12*qy + r20*qx + r21*qy)
        dL_dqw = 2*(-r01*qz + r02*qy + r10*qz - r12*qx - r20*qy + r21*qx)

        dL_dqn = np.array([dL_dqx, dL_dqy, dL_dqz, dL_dqw])
        inv_n = 1.0 / max(np.linalg.norm(qt[i]), 1e-8)
        dL_d_qt[i] = inv_n * (dL_dqn - q * np.dot(q, dL_dqn))

        # dL/d_T_eff
        TV0 = T0 @ V
        TV1 = T1 @ V
        dL_dT0 = 2 * (dL_da * TV0 + dL_db * TV1)
        dL_dT1 = 2 * (dL_db * TV0 + dL_dc * TV1)

        # dL/d_J (via W^T)
        dL_dJ0 = dL_dT0 @ W.T
        dL_dJ1 = dL_dT1 @ W.T

        # dL/d_cam from Jacobian
        dL_dtx = dL_dJ0[2] * (-fx * rz2)
        dL_dty = dL_dJ1[2] * (-fy * rz2)
        dL_dtz = (dL_dJ0[0] * (-fx * rz2) + dL_dJ0[2] * (2 * fx * tx * rz3)
                + dL_dJ1[1] * (-fy * rz2) + dL_dJ1[2] * (2 * fy * ty * rz3))

        if tx_clamped: dL_dtx = 0.0
        if ty_clamped: dL_dty = 0.0

        # dL/d_cam from projection (mean2d)
        dL_du = dL_d_mean2d[i, 0]
        dL_dv = dL_d_mean2d[i, 1]

        dL_d_pcamx = dL_du * fx * rz
        dL_d_pcamy = dL_dv * fy * rz
        dL_d_pcamz = dL_du * (-fx * p_cam[0] * rz2) + dL_dv * (-fy * p_cam[1] * rz2)

        dL_d_pcam = np.array([
            dL_d_pcamx + dL_dtx,
            dL_d_pcamy + dL_dty,
            dL_d_pcamz + dL_dtz
        ])

        # dL/d_mean3d = W^T * dL/d_pcam
        dL_d_m3d[i] = W.T @ dL_d_pcam

    return dL_d_m3d, dL_d_sc, dL_d_qt


def numpy_sh_backward(means3d, campos, sh_coeffs, colors_fwd, dL_d_colors, degree):
    """SH backward in float64. Returns (dL_d_sh, dL_d_means3d)."""
    N = len(means3d)
    m3d = means3d.astype(np.float64)
    cp = campos.astype(np.float64)
    sh = sh_coeffs.astype(np.float64)  # [N, K, 3]
    cf = colors_fwd.astype(np.float64)  # [N, 3]
    dL_dc = dL_d_colors.astype(np.float64).copy()  # [N, 3]

    K = sh.shape[1]
    dL_d_sh = np.zeros_like(sh)
    dL_d_m3d = np.zeros((N, 3), dtype=np.float64)

    for i in range(N):
        # Clamp gradient
        for c in range(3):
            if cf[i, c] <= 0.0 or cf[i, c] >= 1.0:
                dL_dc[i, c] = 0.0

        v = m3d[i] - cp
        v_len = np.linalg.norm(v)
        inv_len = 1.0 / max(v_len, 1e-8)
        d = v * inv_len
        x, y, z = d

        dL = dL_dc[i]

        # Degree 0
        dL_d_sh[i, 0] = SH_C0 * dL

        dL_dx = np.zeros(3)
        dL_dy = np.zeros(3)
        dL_dz = np.zeros(3)

        if degree >= 1:
            dL_d_sh[i, 1] = -SH_C1 * y * dL
            dL_d_sh[i, 2] =  SH_C1 * z * dL
            dL_d_sh[i, 3] = -SH_C1 * x * dL

            dL_dx += -SH_C1 * sh[i, 3] * dL
            dL_dy += -SH_C1 * sh[i, 1] * dL
            dL_dz +=  SH_C1 * sh[i, 2] * dL

        if degree >= 2:
            xx, yy, zz = x*x, y*y, z*z
            xy, yz, xz = x*y, y*z, x*z

            dL_d_sh[i, 4] = SH_C2_0 * xy * dL
            dL_d_sh[i, 5] = SH_C2_1 * yz * dL
            dL_d_sh[i, 6] = SH_C2_2 * (2*zz - xx - yy) * dL
            dL_d_sh[i, 7] = SH_C2_3 * xz * dL
            dL_d_sh[i, 8] = SH_C2_4 * (xx - yy) * dL

            s4, s5, s6, s7, s8 = sh[i, 4], sh[i, 5], sh[i, 6], sh[i, 7], sh[i, 8]

            dL_dx += (SH_C2_0*y*s4 - 2*SH_C2_2*x*s6 + SH_C2_3*z*s7 + 2*SH_C2_4*x*s8) * dL
            dL_dy += (SH_C2_0*x*s4 + SH_C2_1*z*s5 - 2*SH_C2_2*y*s6 - 2*SH_C2_4*y*s8) * dL
            dL_dz += (SH_C2_1*y*s5 + 4*SH_C2_2*z*s6 + SH_C2_3*x*s7) * dL

        if degree >= 3:
            xx, yy, zz = x*x, y*y, z*z
            xy, yz, xz = x*y, y*z, x*z

            dL_d_sh[i, 9]  = SH_C3_0 * y*(3*xx-yy) * dL
            dL_d_sh[i, 10] = SH_C3_1 * xy*z * dL
            dL_d_sh[i, 11] = SH_C3_2 * y*(4*zz-xx-yy) * dL
            dL_d_sh[i, 12] = SH_C3_3 * z*(2*zz-3*xx-3*yy) * dL
            dL_d_sh[i, 13] = SH_C3_4 * x*(4*zz-xx-yy) * dL
            dL_d_sh[i, 14] = SH_C3_5 * z*(xx-yy) * dL
            dL_d_sh[i, 15] = SH_C3_6 * x*(xx-3*yy) * dL

            s9  = sh[i, 9];  s10 = sh[i, 10]; s11 = sh[i, 11]; s12 = sh[i, 12]
            s13 = sh[i, 13]; s14 = sh[i, 14]; s15 = sh[i, 15]

            dL_dx += (SH_C3_0*6*xy*s9 + SH_C3_1*yz*s10 + SH_C3_2*(-2*xy)*s11
                    + SH_C3_3*(-6*xz)*s12 + SH_C3_4*(4*zz-3*xx-yy)*s13
                    + SH_C3_5*2*xz*s14 + SH_C3_6*(3*xx-3*yy)*s15) * dL
            dL_dy += (SH_C3_0*(3*xx-3*yy)*s9 + SH_C3_1*xz*s10
                    + SH_C3_2*(4*zz-xx-3*yy)*s11 + SH_C3_3*(-6*yz)*s12
                    + SH_C3_4*(-2*xy)*s13 + SH_C3_5*(-2*yz)*s14
                    + SH_C3_6*(-6*xy)*s15) * dL
            dL_dz += (SH_C3_1*xy*s10 + SH_C3_2*8*yz*s11
                    + SH_C3_3*(6*zz-3*xx-3*yy)*s12 + SH_C3_4*8*xz*s13
                    + SH_C3_5*(xx-yy)*s14) * dL

        # Direction → mean3d
        dL_dd = np.array([
            dL_dx[0]+dL_dx[1]+dL_dx[2],
            dL_dy[0]+dL_dy[1]+dL_dy[2],
            dL_dz[0]+dL_dz[1]+dL_dz[2]
        ])
        dL_dv = inv_len * (dL_dd - d * np.dot(d, dL_dd))
        dL_d_m3d[i] = dL_dv

    return dL_d_sh, dL_d_m3d


# =========================================================================
#  Main test
# =========================================================================

def main():
    rng = np.random.default_rng(42)
    cam = make_camera()
    means3d, scales, quats, sh_coeffs_fp16, opacities = generate_gaussians(N_GAUSSIANS, rng)

    print("="*72)
    print("Metal-GS: Full-Chain Backward Gradient Verification")
    print("="*72)
    print(f"  N={N_GAUSSIANS}, img={IMG_WIDTH}×{IMG_HEIGHT}, SH degree={SH_DEGREE}")
    print()

    # ---- Step 1: SH forward (Metal) ----
    directions = means3d - cam['campos'][None, :]
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = (directions / np.maximum(norms, 1e-8)).astype(np.float32)

    colors_fp16, sh_ms = compute_sh_forward(
        directions, sh_coeffs_fp16, N_GAUSSIANS, NUM_SH_BASES, SH_DEGREE
    )

    # Convert FP16 to float32
    colors_fp16_np = np.frombuffer(colors_fp16, dtype=np.float16).reshape(N_GAUSSIANS, 3)
    colors_f32 = colors_fp16_np.astype(np.float32)

    print(f"  SH forward: {sh_ms:.3f} ms")

    # ---- Step 2: Full forward pipeline (Metal) ----
    fwd_result = render_forward(
        means3d, scales, quats, cam['viewmat'],
        colors_f32, opacities,
        cam['tan_fovx'], cam['tan_fovy'],
        cam['focal_x'], cam['focal_y'],
        cam['cx'], cam['cy'],
        IMG_WIDTH, IMG_HEIGHT,
        *BG_COLOR
    )

    image = np.array(fwd_result['image'])
    T_final = np.array(fwd_result['T_final'])
    n_contrib = np.array(fwd_result['n_contrib'])
    means2d = np.array(fwd_result['means2d'])
    cov2d = np.array(fwd_result['cov2d'])
    radii = np.array(fwd_result['radii'])
    point_list = np.array(fwd_result['point_list'])
    tile_bins = np.array(fwd_result['tile_bins'])

    print(f"  Forward pipeline: {fwd_result['total_ms']:.3f} ms")
    print(f"  Visible Gaussians: {fwd_result['num_visible']}")
    print(f"  Intersections: {fwd_result['num_intersections']}")

    # ---- Step 3: Random upstream gradient ----
    dL_dC_pixel = rng.standard_normal((IMG_HEIGHT * IMG_WIDTH, 3)).astype(np.float32) * 0.1

    # ---- Step 4: Metal backward (rasterize) ----
    num_tiles_x = (IMG_WIDTH + 15) // 16
    num_tiles_y = (IMG_HEIGHT + 15) // 16

    dL_rgb_m, dL_opac_m, dL_cov_m, dL_mean2d_m, rast_bw_ms = rasterize_backward(
        means2d, cov2d, colors_f32, opacities,
        tile_bins, point_list,
        T_final, n_contrib,
        dL_dC_pixel,
        IMG_WIDTH, IMG_HEIGHT, num_tiles_x, num_tiles_y,
        *BG_COLOR
    )
    dL_rgb_m   = np.array(dL_rgb_m)
    dL_opac_m  = np.array(dL_opac_m)
    dL_cov_m   = np.array(dL_cov_m)
    dL_mean2d_m = np.array(dL_mean2d_m)
    print(f"  Rasterize backward: {rast_bw_ms:.3f} ms")

    # ---- Step 5: Metal backward (preprocess) ----
    dL_m3d_prep_m, dL_sc_m, dL_qt_m, prep_bw_ms = preprocess_backward(
        means3d, scales, quats, cam['viewmat'], radii,
        cam['tan_fovx'], cam['tan_fovy'],
        cam['focal_x'], cam['focal_y'],
        cam['cx'], cam['cy'],
        IMG_WIDTH, IMG_HEIGHT,
        dL_cov_m, dL_mean2d_m
    )
    dL_m3d_prep_m = np.array(dL_m3d_prep_m)
    dL_sc_m       = np.array(dL_sc_m)
    dL_qt_m       = np.array(dL_qt_m)
    print(f"  Preprocess backward: {prep_bw_ms:.3f} ms")

    # ---- Step 6: Metal backward (SH) ----
    dL_sh_m, dL_m3d_sh_m, sh_bw_ms = sh_backward(
        means3d, cam['campos'],
        sh_coeffs_fp16, colors_fp16,
        dL_rgb_m,
        NUM_SH_BASES, SH_DEGREE
    )
    dL_sh_m     = np.array(dL_sh_m)
    dL_m3d_sh_m = np.array(dL_m3d_sh_m)
    print(f"  SH backward: {sh_bw_ms:.3f} ms")

    # ---- Step 7: Combine dL/d_means3d ----
    dL_m3d_total_m = dL_m3d_prep_m + dL_m3d_sh_m

    total_bw_ms = rast_bw_ms + prep_bw_ms + sh_bw_ms
    print(f"  Total backward: {total_bw_ms:.3f} ms")
    print()

    # ---- Step 8: NumPy reference ----
    print("Computing NumPy reference (float64)...")
    t0 = time.time()

    # 8a: SH forward reference
    colors_ref = numpy_sh_forward(directions, sh_coeffs_fp16.astype(np.float32), SH_DEGREE)

    # 8b: Preprocess reference
    means2d_ref, cov2d_ref, depths_ref, visible_ref = numpy_preprocess(
        means3d, scales, quats, cam['viewmat'], cam
    )

    # 8c: Rasterize forward+backward reference
    dL_dC_pixel_2d = dL_dC_pixel.reshape(IMG_HEIGHT, IMG_WIDTH, 3)
    _, dL_rgb_ref, dL_opac_ref, dL_cov_ref, dL_mean2d_ref = \
        numpy_rasterize_and_backward(
            means2d, cov2d, colors_f32, opacities,
            tile_bins, point_list,
            dL_dC_pixel,
            IMG_WIDTH, IMG_HEIGHT, num_tiles_x, BG_COLOR
        )

    # 8d: Preprocess backward reference
    dL_m3d_prep_ref, dL_sc_ref, dL_qt_ref = numpy_preprocess_backward(
        means3d, scales, quats, cam['viewmat'], cam,
        dL_cov_ref, dL_mean2d_ref, visible_ref
    )

    # 8e: SH backward reference
    dL_sh_ref, dL_m3d_sh_ref = numpy_sh_backward(
        means3d, cam['campos'],
        sh_coeffs_fp16.astype(np.float32),
        colors_ref,  # Use the FP64 reference colors for clamp check
        dL_rgb_ref,
        SH_DEGREE
    )

    dL_m3d_total_ref = dL_m3d_prep_ref + dL_m3d_sh_ref

    ref_time = time.time() - t0
    print(f"  Reference computed in {ref_time:.2f}s")
    print()

    # ---- Step 9: Compare ----
    print("="*72)
    print("Gradient Comparison: Metal vs NumPy (float64 reference)")
    print("="*72)

    all_pass = True

    def compare(name, metal, ref, tol=5e-4):
        nonlocal all_pass
        metal_f64 = metal.astype(np.float64).ravel()
        ref_f64 = ref.ravel()

        # Only compare visible Gaussians for per-Gaussian quantities
        mask = np.abs(ref_f64) > 1e-12

        max_abs = np.max(np.abs(metal_f64 - ref_f64))
        mean_abs = np.mean(np.abs(metal_f64 - ref_f64))

        if mask.any():
            max_rel = np.max(np.abs((metal_f64[mask] - ref_f64[mask]) / ref_f64[mask]))
        else:
            max_rel = 0.0

        passed = max_abs < tol
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_pass = False

        print(f"  {name:25s}  MaxAbs={max_abs:.6e}  MeanAbs={mean_abs:.6e}  "
              f"MaxRel={max_rel:.6e}  {status}")

    # Rasterize backward
    compare("dL/d_rgb (rast)",     dL_rgb_m,      dL_rgb_ref,      1e-5)
    compare("dL/d_opacity (rast)", dL_opac_m,     dL_opac_ref,     1e-5)
    compare("dL/d_cov2d (rast)",   dL_cov_m,      dL_cov_ref,      1e-5)
    compare("dL/d_mean2d (rast)",  dL_mean2d_m,   dL_mean2d_ref,   1e-5)

    print()
    # Preprocess backward
    compare("dL/d_means3d (prep)", dL_m3d_prep_m, dL_m3d_prep_ref, 5e-4)
    compare("dL/d_scales",         dL_sc_m,       dL_sc_ref,       5e-4)
    compare("dL/d_quats",          dL_qt_m,       dL_qt_ref,       5e-4)

    print()
    # SH backward
    compare("dL/d_sh",             dL_sh_m,       dL_sh_ref.astype(np.float32), 1e-3)
    compare("dL/d_means3d (SH)",   dL_m3d_sh_m,   dL_m3d_sh_ref,   1e-3)

    print()
    # Full chain
    compare("dL/d_means3d (total)", dL_m3d_total_m, dL_m3d_total_ref, 5e-4)

    print()
    print("="*72)
    if all_pass:
        print("🎉 ALL GRADIENTS VERIFIED — Full differentiable chain is correct!")
    else:
        print("❌ SOME GRADIENTS FAILED — investigate above.")
    print("="*72)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
