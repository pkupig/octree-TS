#pragma once

#include <torch/extension.h>
#include <memory>
#include <algorithm>

// 声明底层的类
namespace triangulation {
    // 这样 GCC 就知道这些类的存在，而不需要去解析它们的具体实现
    class Triangulation; 
    
    // 如果你在下面的函数签名中用到了 AABB 相关的结构体，
    // 只需要在这里声明模板或类名即可
    template <typename T> struct AABB;
}

namespace triangulation_bindings {

// 导出给 Pybind11 的初始化函数
void init_triangulation_bindings(py::module &module);

// 计算 SDF 的接口
torch::Tensor compute_sdf(torch::Tensor query_points, torch::Tensor vertices, torch::Tensor faces);

// 管理类的创建函数
std::unique_ptr<triangulation::Triangulation> create_triangulation(torch::Tensor points);

// 其他工具接口
torch::Tensor get_tets(const triangulation::Triangulation &triangulation);
torch::Tensor get_point_adjacency(const triangulation::Triangulation &triangulation);
torch::Tensor get_tet_adjacency(const triangulation::Triangulation &triangulation);
torch::Tensor get_point_adjacency_offsets(const triangulation::Triangulation &triangulation);
torch::Tensor get_vert_to_tet(const triangulation::Triangulation &triangulation);
torch::Tensor permutation(const triangulation::Triangulation &triangulation);

// 新增的损失函数接口
torch::Tensor geometric_constraint_loss_forward(
    const torch::Tensor& v1, const torch::Tensor& v2, const torch::Tensor& v3,
    const torch::Tensor& anchor_mu, const torch::Tensor& anchor_sigma_inv, const torch::Tensor& anchor_normal,
    const torch::Tensor& is_active,
    float lambda_pos, float lambda_norm);

torch::Tensor laplacian_smoothness_forward(
    const torch::Tensor& v1, const torch::Tensor& v2, const torch::Tensor& v3,
    const torch::Tensor& neighbor_indices, const torch::Tensor& is_active,
    int K, float lambda_smooth);

torch::Tensor intersection_penalty_forward(
    const torch::Tensor& v1, const torch::Tensor& v2, const torch::Tensor& v3,
    const torch::Tensor& neighbor_indices, const torch::Tensor& is_active,
    int K, float lambda_intersect, float margin);

torch::Tensor build_triangle_aabb_tree(
    const torch::Tensor& triangle_centers,
    int max_triangles_per_node);

torch::Tensor intersection_penalty_forward_aabb(
    const torch::Tensor& v1, const torch::Tensor& v2, const torch::Tensor& v3,
    const torch::Tensor& triangle_centers,
    const torch::Tensor& aabb_tree_data,
    const torch::Tensor& is_active,
    float lambda_intersect, float margin);
    

// 其他辅助函数声明
bool rebuild(triangulation::Triangulation &triangulation,
             torch::Tensor points,
             bool incremental);

void init_triangulation_bindings(py::module &module);

} // namespace triangulation_bindings