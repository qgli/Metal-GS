// ============================================================================
//  Metal-GS: PyBind11 bindings
//  Exposes the Metal SH kernel to Python via numpy arrays.
//
//  Key fix: allocate output in C++ and return it, avoiding pybind11
//  forcecast creating a temporary copy for mutable output buffers.
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "metal_wrapper.h"
#include <cstring>
#include <vector>

namespace py = pybind11;

static py::tuple py_compute_sh_forward(
    py::array_t<float, py::array::c_style> directions,
    py::buffer sh_coeffs_buf,
    uint32_t N,
    uint32_t K,
    uint32_t sh_degree
)
{
    // ---- Get raw pointer to directions (float32, contiguous) ----
    auto dir_info = directions.request();
    if (dir_info.ndim != 2 || dir_info.shape[1] != 3)
        throw std::runtime_error("directions must be [N, 3] float32");
    const float* dir_ptr = static_cast<const float*>(dir_info.ptr);

    // ---- Get raw pointer to SH coefficients (float16 = uint16 bits) ----
    auto sh_info = sh_coeffs_buf.request();
    const uint16_t* sh_ptr = static_cast<const uint16_t*>(sh_info.ptr);

    // ---- Allocate output as uint16 (FP16 bit-representation) ----
    auto colors_out = py::array_t<uint16_t>({(py::ssize_t)N, (py::ssize_t)3});
    auto out_info = colors_out.request();
    uint16_t* out_ptr = static_cast<uint16_t*>(out_info.ptr);
    std::memset(out_ptr, 0, N * 3 * sizeof(uint16_t));

    // ---- Call Metal kernel ----
    double elapsed_ms = metal_compute_sh_forward(dir_ptr, sh_ptr, out_ptr, N, K, sh_degree);

    return py::make_tuple(colors_out, elapsed_ms);
}

// SH forward MMA variant (V1.0 exploration — simdgroup_matrix 8×8)
static py::tuple py_compute_sh_forward_mma(
    py::array_t<float, py::array::c_style> directions,
    py::buffer sh_coeffs_buf,
    uint32_t N,
    uint32_t K,
    uint32_t sh_degree
)
{
    auto dir_info = directions.request();
    if (dir_info.ndim != 2 || dir_info.shape[1] != 3)
        throw std::runtime_error("directions must be [N, 3] float32");
    const float* dir_ptr = static_cast<const float*>(dir_info.ptr);

    auto sh_info = sh_coeffs_buf.request();
    const uint16_t* sh_ptr = static_cast<const uint16_t*>(sh_info.ptr);

    auto colors_out = py::array_t<uint16_t>({(py::ssize_t)N, (py::ssize_t)3});
    auto out_info = colors_out.request();
    uint16_t* out_ptr = static_cast<uint16_t*>(out_info.ptr);
    std::memset(out_ptr, 0, N * 3 * sizeof(uint16_t));

    double elapsed_ms = metal_compute_sh_forward_mma(dir_ptr, sh_ptr, out_ptr, N, K, sh_degree);

    return py::make_tuple(colors_out, elapsed_ms);
}

static py::tuple py_preprocess_forward(
    py::array_t<float, py::array::c_style> means3d,
    py::array_t<float, py::array::c_style> scales,
    py::array_t<float, py::array::c_style> quats,
    py::array_t<float, py::array::c_style> viewmat,
    float tan_fovx,
    float tan_fovy,
    float focal_x,
    float focal_y,
    float principal_x,
    float principal_y,
    uint32_t img_width,
    uint32_t img_height
)
{
    // ---- Validate inputs ----
    auto m3d_info = means3d.request();
    if (m3d_info.ndim != 2 || m3d_info.shape[1] != 3)
        throw std::runtime_error("means3d must be [N, 3] float32");
    uint32_t N = (uint32_t)m3d_info.shape[0];

    auto scales_info = scales.request();
    if (scales_info.ndim != 2 || scales_info.shape[0] != N || scales_info.shape[1] != 3)
        throw std::runtime_error("scales must be [N, 3] float32");

    auto quats_info = quats.request();
    if (quats_info.ndim != 2 || quats_info.shape[0] != N || quats_info.shape[1] != 4)
        throw std::runtime_error("quats must be [N, 4] float32");

    auto viewmat_info = viewmat.request();
    if (viewmat_info.ndim != 2 || viewmat_info.shape[0] != 4 || viewmat_info.shape[1] != 4)
        throw std::runtime_error("viewmat must be [4, 4] float32");

    const float* m3d_ptr     = static_cast<const float*>(m3d_info.ptr);
    const float* scales_ptr  = static_cast<const float*>(scales_info.ptr);
    const float* quats_ptr   = static_cast<const float*>(quats_info.ptr);
    const float* viewmat_ptr = static_cast<const float*>(viewmat_info.ptr);

    // ---- Allocate outputs ----
    auto means2d_out = py::array_t<float>({(py::ssize_t)N, (py::ssize_t)2});
    auto cov2d_out   = py::array_t<float>({(py::ssize_t)N, (py::ssize_t)3});  // upper tri
    auto depths_out  = py::array_t<float>((py::ssize_t)N);
    auto radii_out   = py::array_t<uint32_t>((py::ssize_t)N);
    auto tile_min_out = py::array_t<uint32_t>({(py::ssize_t)N, (py::ssize_t)2});
    auto tile_max_out = py::array_t<uint32_t>({(py::ssize_t)N, (py::ssize_t)2});

    float*    means2d_ptr  = static_cast<float*>(means2d_out.request().ptr);
    float*    cov2d_ptr    = static_cast<float*>(cov2d_out.request().ptr);
    float*    depths_ptr   = static_cast<float*>(depths_out.request().ptr);
    uint32_t* radii_ptr    = static_cast<uint32_t*>(radii_out.request().ptr);
    uint32_t* tile_min_ptr = static_cast<uint32_t*>(tile_min_out.request().ptr);
    uint32_t* tile_max_ptr = static_cast<uint32_t*>(tile_max_out.request().ptr);

    // ---- Call Metal kernel ----
    double elapsed_ms = metal_preprocess_forward(
        m3d_ptr, scales_ptr, quats_ptr, viewmat_ptr,
        tan_fovx, tan_fovy, focal_x, focal_y, principal_x, principal_y,
        img_width, img_height, N,
        means2d_ptr, cov2d_ptr, depths_ptr, radii_ptr, tile_min_ptr, tile_max_ptr
    );

    return py::make_tuple(means2d_out, cov2d_out, depths_out, radii_out,
                          tile_min_out, tile_max_out, elapsed_ms);
}

PYBIND11_MODULE(_metal_gs_core, m) {
    m.doc() = "Metal-GS: Apple Silicon-native Gaussian Splatting operators";
    
    m.def("compute_sh_forward", &py_compute_sh_forward,
          py::arg("directions"),
          py::arg("sh_coeffs"),
          py::arg("N"),
          py::arg("K"),
          py::arg("sh_degree"),
          "Compute SH → RGB on Metal GPU.\n"
          "Returns (colors_uint16[N,3], elapsed_ms).");

    m.def("compute_sh_forward_mma", &py_compute_sh_forward_mma,
          py::arg("directions"),
          py::arg("sh_coeffs"),
          py::arg("N"),
          py::arg("K"),
          py::arg("sh_degree"),
          "Compute SH → RGB using simdgroup_matrix 8×8 MMA (V1.0 exploration).\n"
          "Returns (colors_uint16[N,3], elapsed_ms).");

    m.def("preprocess_forward", &py_preprocess_forward,
          py::arg("means3d"),
          py::arg("scales"),
          py::arg("quats"),
          py::arg("viewmat"),
          py::arg("tan_fovx"),
          py::arg("tan_fovy"),
          py::arg("focal_x"),
          py::arg("focal_y"),
          py::arg("principal_x"),
          py::arg("principal_y"),
          py::arg("img_width"),
          py::arg("img_height"),
          "Preprocess 3D Gaussians to 2D screen space.\n"
          "Returns (means2d[N,2], cov2d[N,3], depths[N], radii[N], "
          "tile_min[N,2], tile_max[N,2], elapsed_ms).");

    m.def("radix_sort_by_depth",
          [](py::array_t<float, py::array::c_style> depths) -> py::tuple {
              auto info = depths.request();
              if (info.ndim != 1)
                  throw std::runtime_error("depths must be 1-D float32");
              uint32_t N = (uint32_t)info.shape[0];
              const float* d_ptr = static_cast<const float*>(info.ptr);

              auto indices_out = py::array_t<uint32_t>((py::ssize_t)N);
              uint32_t* out_ptr = static_cast<uint32_t*>(indices_out.request().ptr);

              double elapsed = metal_radix_sort_by_depth(d_ptr, out_ptr, N);
              return py::make_tuple(indices_out, elapsed);
          },
          py::arg("depths"),
          "GPU radix sort by depth.\n"
          "Returns (sorted_indices[N], elapsed_ms).");

    m.def("tile_binning",
          [](py::array_t<uint32_t, py::array::c_style> sorted_indices,
             py::array_t<uint32_t, py::array::c_style> radii,
             py::array_t<uint32_t, py::array::c_style> tile_min,
             py::array_t<uint32_t, py::array::c_style> tile_max,
             uint32_t num_tiles_x,
             uint32_t num_tiles_y) -> py::tuple {
              // ---- Validate inputs ----
              auto si_info = sorted_indices.request();
              if (si_info.ndim != 1)
                  throw std::runtime_error("sorted_indices must be 1-D uint32");
              uint32_t N = (uint32_t)si_info.shape[0];

              auto rad_info = radii.request();
              if (rad_info.ndim != 1 || rad_info.shape[0] != N)
                  throw std::runtime_error("radii must be [N] uint32");

              auto tmin_info = tile_min.request();
              if (tmin_info.ndim != 2 || tmin_info.shape[0] != N || tmin_info.shape[1] != 2)
                  throw std::runtime_error("tile_min must be [N, 2] uint32");

              auto tmax_info = tile_max.request();
              if (tmax_info.ndim != 2 || tmax_info.shape[0] != N || tmax_info.shape[1] != 2)
                  throw std::runtime_error("tile_max must be [N, 2] uint32");

              const uint32_t* si_ptr   = static_cast<const uint32_t*>(si_info.ptr);
              const uint32_t* rad_ptr  = static_cast<const uint32_t*>(rad_info.ptr);
              const uint32_t* tmin_ptr = static_cast<const uint32_t*>(tmin_info.ptr);
              const uint32_t* tmax_ptr = static_cast<const uint32_t*>(tmax_info.ptr);

              // ---- CPU-side: compute offsets + num_intersections ----
              std::vector<uint32_t> offsets(N);
              uint32_t running = 0;
              for (uint32_t i = 0; i < N; i++) {
                  offsets[i] = running;
                  uint32_t idx = si_ptr[i];
                  if (rad_ptr[idx] > 0) {
                      uint32_t w = tmax_ptr[idx * 2]     - tmin_ptr[idx * 2];
                      uint32_t h = tmax_ptr[idx * 2 + 1] - tmin_ptr[idx * 2 + 1];
                      running += w * h;
                  }
              }
              uint32_t num_isect = running;

              // ---- Allocate output numpy arrays ----
              uint32_t num_tiles = num_tiles_x * num_tiles_y;
              auto point_list_out = py::array_t<uint32_t>((py::ssize_t)num_isect);
              auto tile_bins_out  = py::array_t<uint32_t>({(py::ssize_t)num_tiles, (py::ssize_t)2});

              uint32_t* pl_ptr = static_cast<uint32_t*>(point_list_out.request().ptr);
              uint32_t* tb_ptr = static_cast<uint32_t*>(tile_bins_out.request().ptr);

              // ---- Call Metal GPU pipeline ----
              double elapsed = metal_tile_binning(
                  si_ptr, rad_ptr, tmin_ptr, tmax_ptr, offsets.data(),
                  N, num_tiles_x, num_tiles_y, num_isect,
                  pl_ptr, tb_ptr
              );

              return py::make_tuple(point_list_out, tile_bins_out,
                                    (uint32_t)num_isect, elapsed);
          },
          py::arg("sorted_indices"),
          py::arg("radii"),
          py::arg("tile_min"),
          py::arg("tile_max"),
          py::arg("num_tiles_x"),
          py::arg("num_tiles_y"),
          "Tile binning: assign depth-sorted Gaussians to screen tiles.\n"
          "Returns (point_list[num_isect], tile_bins[num_tiles,2], "
          "num_intersections, elapsed_ms).");

    m.def("rasterize_forward",
          [](py::array_t<float, py::array::c_style> means2d,
             py::array_t<float, py::array::c_style> cov2d,
             py::array_t<float, py::array::c_style> colors,
             py::array_t<float, py::array::c_style> opacities,
             py::array_t<uint32_t, py::array::c_style> tile_bins,
             py::array_t<uint32_t, py::array::c_style> point_list,
             uint32_t img_width,
             uint32_t img_height,
             uint32_t num_tiles_x,
             uint32_t num_tiles_y,
             float bg_r, float bg_g, float bg_b,
             uint32_t max_gaussians_per_tile) -> py::tuple {
              // ---- Validate inputs ----
              auto m2d_info = means2d.request();
              if (m2d_info.ndim != 2 || m2d_info.shape[1] != 2)
                  throw std::runtime_error("means2d must be [N, 2] float32");
              uint32_t N = (uint32_t)m2d_info.shape[0];

              auto cov_info = cov2d.request();
              if (cov_info.ndim != 2 || cov_info.shape[0] != N || cov_info.shape[1] != 3)
                  throw std::runtime_error("cov2d must be [N, 3] float32");

              auto col_info = colors.request();
              if (col_info.ndim != 2 || col_info.shape[0] != N || col_info.shape[1] != 3)
                  throw std::runtime_error("colors must be [N, 3] float32");

              auto op_info = opacities.request();
              if (op_info.ndim != 1 || op_info.shape[0] != N)
                  throw std::runtime_error("opacities must be [N] float32");

              auto tb_info = tile_bins.request();
              uint32_t num_tiles = num_tiles_x * num_tiles_y;
              if (tb_info.ndim != 2 || tb_info.shape[0] != num_tiles || tb_info.shape[1] != 2)
                  throw std::runtime_error("tile_bins must be [num_tiles, 2] uint32");

              auto pl_info = point_list.request();
              if (pl_info.ndim != 1)
                  throw std::runtime_error("point_list must be 1-D uint32");
              uint32_t num_isect = (uint32_t)pl_info.shape[0];

              // ---- Allocate output image ----
              auto out_img = py::array_t<float>({(py::ssize_t)img_height,
                                                  (py::ssize_t)img_width,
                                                  (py::ssize_t)3});
              auto T_final_arr = py::array_t<float>({(py::ssize_t)img_height,
                                                      (py::ssize_t)img_width});
              auto n_contrib_arr = py::array_t<uint32_t>({(py::ssize_t)img_height,
                                                           (py::ssize_t)img_width});
              float* img_ptr = static_cast<float*>(out_img.request().ptr);
              float* tf_ptr  = static_cast<float*>(T_final_arr.request().ptr);
              uint32_t* nc_ptr = static_cast<uint32_t*>(n_contrib_arr.request().ptr);

              double elapsed = metal_rasterize_forward(
                  static_cast<const float*>(m2d_info.ptr),
                  static_cast<const float*>(cov_info.ptr),
                  static_cast<const float*>(col_info.ptr),
                  static_cast<const float*>(op_info.ptr),
                  static_cast<const uint32_t*>(tb_info.ptr),
                  static_cast<const uint32_t*>(pl_info.ptr),
                  N, num_isect,
                  img_width, img_height,
                  num_tiles_x, num_tiles_y,
                  bg_r, bg_g, bg_b,
                  max_gaussians_per_tile,
                  img_ptr, tf_ptr, nc_ptr
              );

              return py::make_tuple(out_img, T_final_arr, n_contrib_arr, elapsed);
          },
          py::arg("means2d"),
          py::arg("cov2d"),
          py::arg("colors"),
          py::arg("opacities"),
          py::arg("tile_bins"),
          py::arg("point_list"),
          py::arg("img_width"),
          py::arg("img_height"),
          py::arg("num_tiles_x"),
          py::arg("num_tiles_y"),
          py::arg("bg_r") = 0.0f,
          py::arg("bg_g") = 0.0f,
          py::arg("bg_b") = 0.0f,
          py::arg("max_gaussians_per_tile") = 1024,
          "Forward rasterization: alpha-blend 2D Gaussians per tile.\n"
          "Returns (out_img[H,W,3], T_final[H,W], n_contrib[H,W], elapsed_ms).");

    m.def("rasterize_backward",
          [](py::array_t<float, py::array::c_style> means2d,
             py::array_t<float, py::array::c_style> cov2d,
             py::array_t<float, py::array::c_style> colors,
             py::array_t<float, py::array::c_style> opacities,
             py::array_t<uint32_t, py::array::c_style> tile_bins,
             py::array_t<uint32_t, py::array::c_style> point_list,
             py::array_t<float, py::array::c_style> T_final,
             py::array_t<uint32_t, py::array::c_style> n_contrib,
             py::array_t<float, py::array::c_style> dL_dC_pixel,
             uint32_t img_width,
             uint32_t img_height,
             uint32_t num_tiles_x,
             uint32_t num_tiles_y,
             float bg_r, float bg_g, float bg_b,
             uint32_t max_gaussians_per_tile) -> py::tuple {
              // ---- Validate inputs ----
              auto m2d_info = means2d.request();
              if (m2d_info.ndim != 2 || m2d_info.shape[1] != 2)
                  throw std::runtime_error("means2d must be [N, 2] float32");
              uint32_t N = (uint32_t)m2d_info.shape[0];

              auto cov_info = cov2d.request();
              auto col_info = colors.request();
              auto op_info  = opacities.request();
              auto tb_info  = tile_bins.request();
              auto pl_info  = point_list.request();
              auto tf_info  = T_final.request();
              auto nc_info  = n_contrib.request();
              auto dl_info  = dL_dC_pixel.request();

              uint32_t num_isect = (uint32_t)pl_info.shape[0];

              // ---- Allocate output gradients ----
              auto dL_rgb  = py::array_t<float>({(py::ssize_t)N, (py::ssize_t)3});
              auto dL_opac = py::array_t<float>((py::ssize_t)N);
              auto dL_cov  = py::array_t<float>({(py::ssize_t)N, (py::ssize_t)3});
              auto dL_mean = py::array_t<float>({(py::ssize_t)N, (py::ssize_t)2});

              double elapsed = metal_rasterize_backward(
                  static_cast<const float*>(m2d_info.ptr),
                  static_cast<const float*>(cov_info.ptr),
                  static_cast<const float*>(col_info.ptr),
                  static_cast<const float*>(op_info.ptr),
                  static_cast<const uint32_t*>(tb_info.ptr),
                  static_cast<const uint32_t*>(pl_info.ptr),
                  static_cast<const float*>(tf_info.ptr),
                  static_cast<const uint32_t*>(nc_info.ptr),
                  static_cast<const float*>(dl_info.ptr),
                  N, num_isect,
                  img_width, img_height,
                  num_tiles_x, num_tiles_y,
                  bg_r, bg_g, bg_b,
                  max_gaussians_per_tile,
                  static_cast<float*>(dL_rgb.request().ptr),
                  static_cast<float*>(dL_opac.request().ptr),
                  static_cast<float*>(dL_cov.request().ptr),
                  static_cast<float*>(dL_mean.request().ptr)
              );

              return py::make_tuple(dL_rgb, dL_opac, dL_cov, dL_mean, elapsed);
          },
          py::arg("means2d"),
          py::arg("cov2d"),
          py::arg("colors"),
          py::arg("opacities"),
          py::arg("tile_bins"),
          py::arg("point_list"),
          py::arg("T_final"),
          py::arg("n_contrib"),
          py::arg("dL_dC_pixel"),
          py::arg("img_width"),
          py::arg("img_height"),
          py::arg("num_tiles_x"),
          py::arg("num_tiles_y"),
          py::arg("bg_r") = 0.0f,
          py::arg("bg_g") = 0.0f,
          py::arg("bg_b") = 0.0f,
          py::arg("max_gaussians_per_tile") = 1024,
          "Backward rasterization: compute gradients for 2D Gaussian params.\n"
          "Returns (dL_rgb[N,3], dL_opacity[N], dL_cov2d[N,3], dL_mean2d[N,2], elapsed_ms).");

    m.def("preprocess_backward",
          [](py::array_t<float, py::array::c_style> means3d,
             py::array_t<float, py::array::c_style> scales,
             py::array_t<float, py::array::c_style> quats,
             py::array_t<float, py::array::c_style> viewmat,
             py::array_t<uint32_t, py::array::c_style> radii,
             float tan_fovx, float tan_fovy,
             float focal_x, float focal_y,
             float principal_x, float principal_y,
             uint32_t img_width, uint32_t img_height,
             py::array_t<float, py::array::c_style> dL_d_cov2d,
             py::array_t<float, py::array::c_style> dL_d_mean2d) -> py::tuple {
              auto m3d_info = means3d.request();
              if (m3d_info.ndim != 2 || m3d_info.shape[1] != 3)
                  throw std::runtime_error("means3d must be [N, 3]");
              uint32_t N = (uint32_t)m3d_info.shape[0];

              auto dL_m3d  = py::array_t<float>({(py::ssize_t)N, (py::ssize_t)3});
              auto dL_sc   = py::array_t<float>({(py::ssize_t)N, (py::ssize_t)3});
              auto dL_qt   = py::array_t<float>({(py::ssize_t)N, (py::ssize_t)4});

              double elapsed = metal_preprocess_backward(
                  static_cast<const float*>(m3d_info.ptr),
                  static_cast<const float*>(scales.request().ptr),
                  static_cast<const float*>(quats.request().ptr),
                  static_cast<const float*>(viewmat.request().ptr),
                  static_cast<const uint32_t*>(radii.request().ptr),
                  tan_fovx, tan_fovy, focal_x, focal_y, principal_x, principal_y,
                  img_width, img_height, N,
                  static_cast<const float*>(dL_d_cov2d.request().ptr),
                  static_cast<const float*>(dL_d_mean2d.request().ptr),
                  static_cast<float*>(dL_m3d.request().ptr),
                  static_cast<float*>(dL_sc.request().ptr),
                  static_cast<float*>(dL_qt.request().ptr)
              );

              return py::make_tuple(dL_m3d, dL_sc, dL_qt, elapsed);
          },
          py::arg("means3d"),
          py::arg("scales"),
          py::arg("quats"),
          py::arg("viewmat"),
          py::arg("radii"),
          py::arg("tan_fovx"), py::arg("tan_fovy"),
          py::arg("focal_x"), py::arg("focal_y"),
          py::arg("principal_x"), py::arg("principal_y"),
          py::arg("img_width"), py::arg("img_height"),
          py::arg("dL_d_cov2d"),
          py::arg("dL_d_mean2d"),
          "Preprocess backward (M3): dL/d_cov2d + dL/d_mean2d → dL/d_means3d, scales, quats.\n"
          "Returns (dL_means3d[N,3], dL_scales[N,3], dL_quats[N,4], elapsed_ms).");

    m.def("sh_backward",
          [](py::array_t<float, py::array::c_style> means3d,
             py::array_t<float, py::array::c_style> campos,
             py::buffer sh_coeffs,
             py::buffer colors_fwd,
             py::array_t<float, py::array::c_style> dL_d_colors,
             uint32_t K,
             uint32_t sh_degree) -> py::tuple {
              auto m3d_info = means3d.request();
              if (m3d_info.ndim != 2 || m3d_info.shape[1] != 3)
                  throw std::runtime_error("means3d must be [N, 3]");
              uint32_t N = (uint32_t)m3d_info.shape[0];

              auto cam_info = campos.request();
              auto sh_info  = sh_coeffs.request();
              auto cf_info  = colors_fwd.request();
              auto dlc_info = dL_d_colors.request();

              auto dL_sh  = py::array_t<float>({(py::ssize_t)N, (py::ssize_t)K, (py::ssize_t)3});
              auto dL_m3d = py::array_t<float>({(py::ssize_t)N, (py::ssize_t)3});

              double elapsed = metal_sh_backward(
                  static_cast<const float*>(m3d_info.ptr),
                  static_cast<const float*>(cam_info.ptr),
                  static_cast<const uint16_t*>(sh_info.ptr),
                  static_cast<const uint16_t*>(cf_info.ptr),
                  static_cast<const float*>(dlc_info.ptr),
                  N, K, sh_degree,
                  static_cast<float*>(dL_sh.request().ptr),
                  static_cast<float*>(dL_m3d.request().ptr)
              );

              return py::make_tuple(dL_sh, dL_m3d, elapsed);
          },
          py::arg("means3d"),
          py::arg("campos"),
          py::arg("sh_coeffs"),
          py::arg("colors_fwd"),
          py::arg("dL_d_colors"),
          py::arg("K"),
          py::arg("sh_degree"),
          "SH backward (M4): dL/d_colors → dL/d_sh_coeffs + dL/d_means3d.\n"
          "Returns (dL_sh[N,K,3], dL_means3d[N,3], elapsed_ms).");

    m.def("render_forward",
          [](py::array_t<float, py::array::c_style> means3d,
             py::array_t<float, py::array::c_style> scales,
             py::array_t<float, py::array::c_style> quats,
             py::array_t<float, py::array::c_style> viewmat,
             py::array_t<float, py::array::c_style> colors,
             py::array_t<float, py::array::c_style> opacities,
             float tan_fovx, float tan_fovy,
             float focal_x, float focal_y,
             float principal_x, float principal_y,
             uint32_t img_width, uint32_t img_height,
             float bg_r, float bg_g, float bg_b,
             uint32_t max_gaussians_per_tile) -> py::dict {
              // ---- Validate ----
              auto m3d_info = means3d.request();
              if (m3d_info.ndim != 2 || m3d_info.shape[1] != 3)
                  throw std::runtime_error("means3d must be [N, 3]");
              uint32_t N = (uint32_t)m3d_info.shape[0];

              uint32_t num_tiles_x = (img_width  + 15) / 16;
              uint32_t num_tiles_y = (img_height + 15) / 16;
              uint32_t num_tiles   = num_tiles_x * num_tiles_y;

              // ---- Step 1: Preprocess ----
              auto means2d_arr  = py::array_t<float>({(py::ssize_t)N, (py::ssize_t)2});
              auto cov2d_arr    = py::array_t<float>({(py::ssize_t)N, (py::ssize_t)3});
              auto depths_arr   = py::array_t<float>((py::ssize_t)N);
              auto radii_arr    = py::array_t<uint32_t>((py::ssize_t)N);
              auto tile_min_arr = py::array_t<uint32_t>({(py::ssize_t)N, (py::ssize_t)2});
              auto tile_max_arr = py::array_t<uint32_t>({(py::ssize_t)N, (py::ssize_t)2});

              double prep_ms = metal_preprocess_forward(
                  static_cast<const float*>(m3d_info.ptr),
                  static_cast<const float*>(scales.request().ptr),
                  static_cast<const float*>(quats.request().ptr),
                  static_cast<const float*>(viewmat.request().ptr),
                  tan_fovx, tan_fovy, focal_x, focal_y, principal_x, principal_y,
                  img_width, img_height, N,
                  static_cast<float*>(means2d_arr.request().ptr),
                  static_cast<float*>(cov2d_arr.request().ptr),
                  static_cast<float*>(depths_arr.request().ptr),
                  static_cast<uint32_t*>(radii_arr.request().ptr),
                  static_cast<uint32_t*>(tile_min_arr.request().ptr),
                  static_cast<uint32_t*>(tile_max_arr.request().ptr)
              );

              // ---- Step 2: Sort by depth ----
              auto sorted_idx = py::array_t<uint32_t>((py::ssize_t)N);
              double sort_ms = metal_radix_sort_by_depth(
                  static_cast<const float*>(depths_arr.request().ptr),
                  static_cast<uint32_t*>(sorted_idx.request().ptr),
                  N
              );

              // ---- Step 3: Tile binning (CPU offsets + GPU binning) ----
              const uint32_t* si_ptr   = static_cast<const uint32_t*>(sorted_idx.request().ptr);
              const uint32_t* rad_ptr  = static_cast<const uint32_t*>(radii_arr.request().ptr);
              const uint32_t* tmin_ptr = static_cast<const uint32_t*>(tile_min_arr.request().ptr);
              const uint32_t* tmax_ptr = static_cast<const uint32_t*>(tile_max_arr.request().ptr);

              std::vector<uint32_t> offsets(N);
              uint32_t running = 0;
              for (uint32_t i = 0; i < N; i++) {
                  offsets[i] = running;
                  uint32_t idx = si_ptr[i];
                  if (rad_ptr[idx] > 0) {
                      uint32_t w = tmax_ptr[idx * 2]     - tmin_ptr[idx * 2];
                      uint32_t h = tmax_ptr[idx * 2 + 1] - tmin_ptr[idx * 2 + 1];
                      running += w * h;
                  }
              }
              uint32_t num_isect = running;

              auto point_list_arr = py::array_t<uint32_t>((py::ssize_t)num_isect);
              auto tile_bins_arr  = py::array_t<uint32_t>({(py::ssize_t)num_tiles, (py::ssize_t)2});
              uint32_t* pl_ptr = static_cast<uint32_t*>(point_list_arr.request().ptr);
              uint32_t* tb_ptr = static_cast<uint32_t*>(tile_bins_arr.request().ptr);

              double bin_ms = metal_tile_binning(
                  si_ptr, rad_ptr, tmin_ptr, tmax_ptr, offsets.data(),
                  N, num_tiles_x, num_tiles_y, num_isect,
                  pl_ptr, tb_ptr
              );

              // ---- Step 4: Rasterize ----
              auto out_img = py::array_t<float>({(py::ssize_t)img_height,
                                                  (py::ssize_t)img_width,
                                                  (py::ssize_t)3});
              auto T_final_arr = py::array_t<float>({(py::ssize_t)img_height,
                                                      (py::ssize_t)img_width});
              auto n_contrib_arr = py::array_t<uint32_t>({(py::ssize_t)img_height,
                                                           (py::ssize_t)img_width});
              float* img_ptr = static_cast<float*>(out_img.request().ptr);
              float* tf_ptr  = static_cast<float*>(T_final_arr.request().ptr);
              uint32_t* nc_ptr = static_cast<uint32_t*>(n_contrib_arr.request().ptr);

              auto col_info = colors.request();
              auto op_info  = opacities.request();

              double rast_ms = metal_rasterize_forward(
                  static_cast<const float*>(means2d_arr.request().ptr),
                  static_cast<const float*>(cov2d_arr.request().ptr),
                  static_cast<const float*>(col_info.ptr),
                  static_cast<const float*>(op_info.ptr),
                  tb_ptr, pl_ptr,
                  N, num_isect,
                  img_width, img_height,
                  num_tiles_x, num_tiles_y,
                  bg_r, bg_g, bg_b,
                  max_gaussians_per_tile,
                  img_ptr, tf_ptr, nc_ptr
              );

              double total_ms = prep_ms + sort_ms + bin_ms + rast_ms;

              py::dict result;
              result["image"]          = out_img;
              result["T_final"]        = T_final_arr;
              result["n_contrib"]      = n_contrib_arr;
              result["preprocess_ms"]  = prep_ms;
              result["sort_ms"]        = sort_ms;
              result["binning_ms"]     = bin_ms;
              result["rasterize_ms"]   = rast_ms;
              result["total_ms"]       = total_ms;
              result["num_visible"]    = (uint32_t)0;
              result["num_intersections"] = num_isect;

              // Count visible
              uint32_t nvis = 0;
              for (uint32_t i = 0; i < N; i++) {
                  if (rad_ptr[i] > 0) nvis++;
              }
              result["num_visible"] = nvis;

              // Save intermediate arrays for backward pass
              result["means2d"]    = means2d_arr;
              result["cov2d"]      = cov2d_arr;
              result["radii"]      = radii_arr;
              result["point_list"] = point_list_arr;
              result["tile_bins"]  = tile_bins_arr;

              return result;
          },
          py::arg("means3d"),
          py::arg("scales"),
          py::arg("quats"),
          py::arg("viewmat"),
          py::arg("colors"),
          py::arg("opacities"),
          py::arg("tan_fovx"), py::arg("tan_fovy"),
          py::arg("focal_x"), py::arg("focal_y"),
          py::arg("principal_x"), py::arg("principal_y"),
          py::arg("img_width"), py::arg("img_height"),
          py::arg("bg_r") = 0.0f,
          py::arg("bg_g") = 0.0f,
          py::arg("bg_b") = 0.0f,
          py::arg("max_gaussians_per_tile") = 1024,
          "Full rendering pipeline: preprocess → sort → bin → rasterize.\n"
          "Returns dict with 'image'[H,W,3], per-stage timing, stats.");

    m.def("render_backward",
          [](py::array_t<float, py::array::c_style> means3d,
             py::array_t<float, py::array::c_style> scales,
             py::array_t<float, py::array::c_style> quats,
             py::array_t<float, py::array::c_style> viewmat,
             py::array_t<float, py::array::c_style> colors,
             py::array_t<float, py::array::c_style> opacities,
             py::array_t<float, py::array::c_style> campos,
             py::buffer sh_coeffs,
             py::buffer colors_fwd_fp16,
             uint32_t K, uint32_t sh_degree,
             float tan_fovx, float tan_fovy,
             float focal_x, float focal_y,
             float principal_x, float principal_y,
             uint32_t img_width, uint32_t img_height,
             float bg_r, float bg_g, float bg_b,
             uint32_t max_gaussians_per_tile,
             // Forward intermediates
             py::array_t<float, py::array::c_style> means2d,
             py::array_t<float, py::array::c_style> cov2d,
             py::array_t<uint32_t, py::array::c_style> radii,
             py::array_t<uint32_t, py::array::c_style> tile_bins,
             py::array_t<uint32_t, py::array::c_style> point_list,
             py::array_t<float, py::array::c_style> T_final,
             py::array_t<uint32_t, py::array::c_style> n_contrib,
             // Upstream gradient
             py::array_t<float, py::array::c_style> dL_dC_pixel) -> py::dict {

              auto m3d_info = means3d.request();
              uint32_t N = (uint32_t)m3d_info.shape[0];
              uint32_t num_tiles_x = (img_width  + 15) / 16;
              uint32_t num_tiles_y = (img_height + 15) / 16;
              auto pl_info = point_list.request();
              uint32_t num_isect = (uint32_t)pl_info.shape[0];

              // ---- Stage 1: Rasterize backward ----
              auto dL_rgb  = py::array_t<float>({(py::ssize_t)N, (py::ssize_t)3});
              auto dL_opac = py::array_t<float>((py::ssize_t)N);
              auto dL_cov  = py::array_t<float>({(py::ssize_t)N, (py::ssize_t)3});
              auto dL_mean = py::array_t<float>({(py::ssize_t)N, (py::ssize_t)2});

              double rast_bw_ms = metal_rasterize_backward(
                  static_cast<const float*>(means2d.request().ptr),
                  static_cast<const float*>(cov2d.request().ptr),
                  static_cast<const float*>(colors.request().ptr),
                  static_cast<const float*>(opacities.request().ptr),
                  static_cast<const uint32_t*>(tile_bins.request().ptr),
                  static_cast<const uint32_t*>(pl_info.ptr),
                  static_cast<const float*>(T_final.request().ptr),
                  static_cast<const uint32_t*>(n_contrib.request().ptr),
                  static_cast<const float*>(dL_dC_pixel.request().ptr),
                  N, num_isect,
                  img_width, img_height,
                  num_tiles_x, num_tiles_y,
                  bg_r, bg_g, bg_b,
                  max_gaussians_per_tile,
                  static_cast<float*>(dL_rgb.request().ptr),
                  static_cast<float*>(dL_opac.request().ptr),
                  static_cast<float*>(dL_cov.request().ptr),
                  static_cast<float*>(dL_mean.request().ptr)
              );

              // ---- Stage 2: Preprocess backward ----
              auto dL_m3d_prep = py::array_t<float>({(py::ssize_t)N, (py::ssize_t)3});
              auto dL_sc       = py::array_t<float>({(py::ssize_t)N, (py::ssize_t)3});
              auto dL_qt       = py::array_t<float>({(py::ssize_t)N, (py::ssize_t)4});

              double prep_bw_ms = metal_preprocess_backward(
                  static_cast<const float*>(m3d_info.ptr),
                  static_cast<const float*>(scales.request().ptr),
                  static_cast<const float*>(quats.request().ptr),
                  static_cast<const float*>(viewmat.request().ptr),
                  static_cast<const uint32_t*>(radii.request().ptr),
                  tan_fovx, tan_fovy, focal_x, focal_y, principal_x, principal_y,
                  img_width, img_height, N,
                  static_cast<const float*>(dL_cov.request().ptr),
                  static_cast<const float*>(dL_mean.request().ptr),
                  static_cast<float*>(dL_m3d_prep.request().ptr),
                  static_cast<float*>(dL_sc.request().ptr),
                  static_cast<float*>(dL_qt.request().ptr)
              );

              // ---- Stage 3: SH backward ----
              auto sh_info = sh_coeffs.request();
              auto cf_info = colors_fwd_fp16.request();

              auto dL_sh       = py::array_t<float>({(py::ssize_t)N, (py::ssize_t)K, (py::ssize_t)3});
              auto dL_m3d_sh   = py::array_t<float>({(py::ssize_t)N, (py::ssize_t)3});

              double sh_bw_ms = metal_sh_backward(
                  static_cast<const float*>(m3d_info.ptr),
                  static_cast<const float*>(campos.request().ptr),
                  static_cast<const uint16_t*>(sh_info.ptr),
                  static_cast<const uint16_t*>(cf_info.ptr),
                  static_cast<const float*>(dL_rgb.request().ptr),
                  N, K, sh_degree,
                  static_cast<float*>(dL_sh.request().ptr),
                  static_cast<float*>(dL_m3d_sh.request().ptr)
              );

              // ---- Combine: dL/d_means3d = preprocess_bw + sh_bw ----
              auto dL_m3d_total = py::array_t<float>({(py::ssize_t)N, (py::ssize_t)3});
              float* out_ptr  = static_cast<float*>(dL_m3d_total.request().ptr);
              const float* p1 = static_cast<const float*>(dL_m3d_prep.request().ptr);
              const float* p2 = static_cast<const float*>(dL_m3d_sh.request().ptr);
              for (uint32_t i = 0; i < N * 3; i++) {
                  out_ptr[i] = p1[i] + p2[i];
              }

              py::dict result;
              result["dL_d_means3d"]  = dL_m3d_total;
              result["dL_d_scales"]   = dL_sc;
              result["dL_d_quats"]    = dL_qt;
              result["dL_d_sh"]       = dL_sh;
              result["dL_d_opacities"]= dL_opac;
              result["rasterize_bw_ms"]  = rast_bw_ms;
              result["preprocess_bw_ms"] = prep_bw_ms;
              result["sh_bw_ms"]         = sh_bw_ms;
              result["total_bw_ms"]      = rast_bw_ms + prep_bw_ms + sh_bw_ms;

              return result;
          },
          py::arg("means3d"),
          py::arg("scales"),
          py::arg("quats"),
          py::arg("viewmat"),
          py::arg("colors"),
          py::arg("opacities"),
          py::arg("campos"),
          py::arg("sh_coeffs"),
          py::arg("colors_fwd_fp16"),
          py::arg("K"), py::arg("sh_degree"),
          py::arg("tan_fovx"), py::arg("tan_fovy"),
          py::arg("focal_x"), py::arg("focal_y"),
          py::arg("principal_x"), py::arg("principal_y"),
          py::arg("img_width"), py::arg("img_height"),
          py::arg("bg_r") = 0.0f,
          py::arg("bg_g") = 0.0f,
          py::arg("bg_b") = 0.0f,
          py::arg("max_gaussians_per_tile") = 1024,
          py::arg("means2d"),
          py::arg("cov2d"),
          py::arg("radii"),
          py::arg("tile_bins"),
          py::arg("point_list"),
          py::arg("T_final"),
          py::arg("n_contrib"),
          py::arg("dL_dC_pixel"),
          "Full backward pass: rasterize_bw → preprocess_bw → sh_bw.\n"
          "Returns dict with dL_d_means3d, dL_d_scales, dL_d_quats, dL_d_sh, dL_d_opacities, timing.");

    // ==================================================================
    //  Simple KNN (Morton code + radix sort)
    // ==================================================================
    m.def("simple_knn_metal",
          [](py::array_t<float, py::array::c_style> points,
             uint32_t k_neighbors,
             uint32_t search_window) -> py::tuple {
              auto info = points.request();
              if (info.ndim != 2 || info.shape[1] != 3)
                  throw std::runtime_error("points must be [N, 3] float32");
              uint32_t N = (uint32_t)info.shape[0];

              auto avg_sq_dist = py::array_t<float>((py::ssize_t)N);
              auto out_info = avg_sq_dist.request();
              float* out_ptr = static_cast<float*>(out_info.ptr);
              std::memset(out_ptr, 0, N * sizeof(float));

              double elapsed_ms = metal_simple_knn(
                  static_cast<const float*>(info.ptr),
                  out_ptr,
                  N, k_neighbors, search_window
              );

              return py::make_tuple(avg_sq_dist, elapsed_ms);
          },
          py::arg("points"),
          py::arg("k_neighbors") = 3,
          py::arg("search_window") = 32,
          "Morton-code KNN: returns (avg_sq_dist[N], elapsed_ms).\n"
          "Uses Morton-code radix sort + window search for K nearest neighbors.");
}