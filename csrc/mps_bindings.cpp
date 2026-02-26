// ============================================================================
//  Metal-GS v3: torch::Tensor-based pybind11 bindings
//
//  Replaces the numpy-based bindings.cpp.  All inputs/outputs are
//  torch::Tensor on the MPS device — zero CPU round-trips.
// ============================================================================

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

// ---- Declarations from mps_ops.mm ----
std::vector<torch::Tensor> mps_render_forward(
    const torch::Tensor& means3d,
    const torch::Tensor& scales,
    const torch::Tensor& quats,
    const torch::Tensor& sh_coeffs,
    const torch::Tensor& opacities,
    const torch::Tensor& viewmat,
    const torch::Tensor& campos,
    float tan_fovx, float tan_fovy,
    float focal_x, float focal_y,
    float principal_x, float principal_y,
    int64_t img_width, int64_t img_height,
    int64_t sh_degree,
    float bg_r, float bg_g, float bg_b,
    int64_t max_gaussians_per_tile);

std::vector<torch::Tensor> mps_render_backward(
    const torch::Tensor& means3d,
    const torch::Tensor& scales,
    const torch::Tensor& quats,
    const torch::Tensor& sh_coeffs,
    const torch::Tensor& opacities,
    const torch::Tensor& viewmat,
    const torch::Tensor& campos,
    const torch::Tensor& means2d,
    const torch::Tensor& cov2d,
    const torch::Tensor& radii,
    const torch::Tensor& colors_fp32,
    const torch::Tensor& colors_fp16,
    const torch::Tensor& tile_bins,
    const torch::Tensor& point_list,
    const torch::Tensor& T_final,
    const torch::Tensor& n_contrib,
    const torch::Tensor& dL_dC_pixel,
    float tan_fovx, float tan_fovy,
    float focal_x, float focal_y,
    float principal_x, float principal_y,
    int64_t img_width, int64_t img_height,
    int64_t sh_degree,
    float bg_r, float bg_g, float bg_b,
    int64_t max_gaussians_per_tile);

torch::Tensor mps_simple_knn(
    const torch::Tensor& points,
    int64_t k_neighbors,
    int64_t search_window);


PYBIND11_MODULE(_metal_gs_core, m) {
    m.doc() = "Metal-GS v3: zero-copy MPS custom ops";

    m.def("render_forward", &mps_render_forward,
          py::arg("means3d"),
          py::arg("scales"),
          py::arg("quats"),
          py::arg("sh_coeffs"),
          py::arg("opacities"),
          py::arg("viewmat"),
          py::arg("campos"),
          py::arg("tan_fovx"), py::arg("tan_fovy"),
          py::arg("focal_x"), py::arg("focal_y"),
          py::arg("principal_x"), py::arg("principal_y"),
          py::arg("img_width"), py::arg("img_height"),
          py::arg("sh_degree"),
          py::arg("bg_r") = 0.0f,
          py::arg("bg_g") = 0.0f,
          py::arg("bg_b") = 0.0f,
          py::arg("max_gaussians_per_tile") = 4096,
          "Full forward pipeline: SH → preprocess → sort → tile_bin → rasterize.\n"
          "All tensors must be on MPS device.\n"
          "Returns [out_img, T_final, n_contrib, means2d, cov2d, depths, radii,\n"
          "         tile_bins, point_list, sorted_indices, colors_fp16, colors_fp32, directions].");

    m.def("render_backward", &mps_render_backward,
          py::arg("means3d"),
          py::arg("scales"),
          py::arg("quats"),
          py::arg("sh_coeffs"),
          py::arg("opacities"),
          py::arg("viewmat"),
          py::arg("campos"),
          py::arg("means2d"),
          py::arg("cov2d"),
          py::arg("radii"),
          py::arg("colors_fp32"),
          py::arg("colors_fp16"),
          py::arg("tile_bins"),
          py::arg("point_list"),
          py::arg("T_final"),
          py::arg("n_contrib"),
          py::arg("dL_dC_pixel"),
          py::arg("tan_fovx"), py::arg("tan_fovy"),
          py::arg("focal_x"), py::arg("focal_y"),
          py::arg("principal_x"), py::arg("principal_y"),
          py::arg("img_width"), py::arg("img_height"),
          py::arg("sh_degree"),
          py::arg("bg_r") = 0.0f,
          py::arg("bg_g") = 0.0f,
          py::arg("bg_b") = 0.0f,
          py::arg("max_gaussians_per_tile") = 4096,
          "Full backward: rasterize_bw → preprocess_bw → sh_bw (single encoder).\n"
          "Returns [dL_m3d_prep, dL_m3d_sh, dL_scales, dL_quats, dL_sh, dL_opac].");

    m.def("simple_knn_metal", &mps_simple_knn,
          py::arg("points"),
          py::arg("k_neighbors") = 3,
          py::arg("search_window") = 32,
          "Morton-code KNN on MPS tensors.\n"
          "Returns avg_sq_dist[N] tensor on MPS.");
}
