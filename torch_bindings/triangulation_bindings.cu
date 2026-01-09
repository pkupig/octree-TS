#include <memory>
#include <cuda_runtime.h>
#include <stdexcept>
#include "aabb_tree/aabb_tree.h"
#include "triangulation_bindings.h"
#include "utils/cuda_helpers.h"
#include "utils/typing.h"
#include "delaunay/delaunay.h"
#include "delaunay/triangulation_ops.h"
#include "common_kernels_wrapper.h"  // 包含正确的头文件

// 实现 set_default_stream 函数
namespace triangulation {

void set_default_stream() {
    // 检查是否有可用的 CUDA 设备
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        throw std::runtime_error("No CUDA device available");
    }
    
    // 获取当前设备
    int device;
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
} // namespace triangulation

namespace triangulation_bindings {

// 1. 原始函数实现（从最开始的代码）
std::unique_ptr<triangulation::Triangulation> create_triangulation(torch::Tensor points) {
    if (points.size(-1) != 3) {
        throw std::runtime_error("points must have 3 as the last dimension");
    }
    if (points.device().type() != at::kCUDA) {
        throw std::runtime_error("points must be on CUDA device");
    }
    if (points.scalar_type() != torch::kFloat32) {
        throw std::runtime_error("points must have float32 dtype");
    }

    uint32_t num_points = points.numel() / 3;

    triangulation::set_default_stream();

    return triangulation::Triangulation::create_triangulation(points.data_ptr(), num_points);
}

bool rebuild(triangulation::Triangulation &triangulation,
             torch::Tensor points,
             bool incremental) {
    if (points.size(-1) != 3) {
        throw std::runtime_error("points must have 3 as the last dimension");
    }
    if (points.device().type() != at::kCUDA) {
        throw std::runtime_error("points must be on CUDA device");
    }
    if (points.scalar_type() != torch::kFloat32) {
        throw std::runtime_error("points must have float32 dtype");
    }

    triangulation::set_default_stream();

    return triangulation.rebuild(
        points.data_ptr(), points.numel() / 3, incremental);
}

torch::Tensor permutation(const triangulation::Triangulation &triangulation) {
    const uint32_t *permutation = triangulation.permutation();
    uint32_t num_points = triangulation.num_points();

    at::TensorOptions options =
        at::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA);

    return torch::from_blob(
        const_cast<uint32_t *>(permutation), {num_points}, options);
}

torch::Tensor get_tets(const triangulation::Triangulation &triangulation) {
    const triangulation::IndexedTet *tets = triangulation.tets();
    uint32_t num_tets = triangulation.num_tets();

    at::TensorOptions options =
        at::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA);

    return torch::from_blob(
        const_cast<triangulation::IndexedTet *>(tets), {num_tets, 4}, options);
}

torch::Tensor get_tet_adjacency(const triangulation::Triangulation &triangulation) {
    const uint32_t *tet_adjacency = triangulation.tet_adjacency();
    uint32_t num_tets = triangulation.num_tets();

    at::TensorOptions options =
        at::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA);

    return torch::from_blob(
        const_cast<uint32_t *>(tet_adjacency), {num_tets, 4}, options);
}

torch::Tensor get_point_adjacency(const triangulation::Triangulation &triangulation) {
    const uint32_t *point_adjacency = triangulation.point_adjacency();
    uint32_t point_adjacency_size = triangulation.point_adjacency_size();

    at::TensorOptions options =
        at::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA);

    return torch::from_blob(const_cast<uint32_t *>(point_adjacency),
                            {point_adjacency_size},
                            options);
}

torch::Tensor get_point_adjacency_offsets(const triangulation::Triangulation &triangulation) {
    const uint32_t *point_adjacency_offsets =
        triangulation.point_adjacency_offsets();
    uint32_t num_points = triangulation.num_points();

    at::TensorOptions options =
        at::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA);

    return torch::from_blob(const_cast<uint32_t *>(point_adjacency_offsets),
                            {num_points + 1},
                            options);
}

torch::Tensor get_vert_to_tet(const triangulation::Triangulation &triangulation) {
    const uint32_t *vert_to_tet = triangulation.vert_to_tet();
    uint32_t num_points = triangulation.num_points();

    at::TensorOptions options =
        at::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA);

    return torch::from_blob(
        const_cast<uint32_t *>(vert_to_tet), {num_points}, options);
}

// 2. 新增的损失函数实现（每个函数只实现一次）

// ** 实现 geometric_constraint_loss_forward **
torch::Tensor geometric_constraint_loss_forward(
    const torch::Tensor& v1, const torch::Tensor& v2, const torch::Tensor& v3,
    const torch::Tensor& anchor_mu, const torch::Tensor& anchor_sigma_inv, const torch::Tensor& anchor_normal,
    const torch::Tensor& is_active,
    float lambda_pos, float lambda_norm) 
{
    // 参数检查
    TORCH_CHECK(v1.dim() == 2 && v1.size(1) == 3, "v1 must be [P, 3]");
    TORCH_CHECK(v2.dim() == 2 && v2.size(1) == 3, "v2 must be [P, 3]");
    TORCH_CHECK(v3.dim() == 2 && v3.size(1) == 3, "v3 must be [P, 3]");
    TORCH_CHECK(v1.device().type() == at::kCUDA, "v1 must be on CUDA");
    
    int P = v1.size(0);
    
    // 1. 分配输出张量
    torch::Tensor L_anchor = torch::empty({P}, v1.options());
    torch::Tensor L_orient = torch::empty({P}, v1.options());
    
    // 2. 调用 triangulation 库中已有的函数（使用 CUDA 版本）
    triangulation::computeGeometricConstraintLoss(
        P, 
        (const float*)v1.data_ptr(), (const float*)v2.data_ptr(), (const float*)v3.data_ptr(),
        (const float*)anchor_mu.data_ptr(), (const float*)anchor_sigma_inv.data_ptr(), (const float*)anchor_normal.data_ptr(),
        (const float*)is_active.data_ptr(),
        (float*)L_anchor.data_ptr(), (float*)L_orient.data_ptr(),
        lambda_pos, lambda_norm
    );
    
    // 3. 返回结果
    return torch::stack({L_anchor, L_orient}, /*dim=*/1); 
}

// ** 实现 laplacian_smoothness_forward **
torch::Tensor laplacian_smoothness_forward(
    const torch::Tensor& v1, const torch::Tensor& v2, const torch::Tensor& v3,
    const torch::Tensor& neighbor_indices, const torch::Tensor& is_active,
    int K, float lambda_smooth) 
{
    TORCH_CHECK(neighbor_indices.scalar_type() == torch::kUInt32);
    TORCH_CHECK(neighbor_indices.dim() == 2 && neighbor_indices.size(1) == K);
    TORCH_CHECK(v1.device().type() == at::kCUDA, "Inputs must be on CUDA");
    
    int P = v1.size(0);
    
    // 1. 分配输出张量
    torch::Tensor L_smooth = torch::empty({P}, v1.options());
    
    // 2. 调用 triangulation 库中已有的函数（使用 CUDA 版本）
    triangulation::computeLaplacianSmoothnessLoss(
        P, 
        (const float*)v1.data_ptr(), (const float*)v2.data_ptr(), (const float*)v3.data_ptr(),
        (const uint32_t*)neighbor_indices.data_ptr(),
        (const float*)is_active.data_ptr(),
        K,
        (float*)L_smooth.data_ptr(),
        lambda_smooth
    );
    
    // 3. 返回结果
    return L_smooth; 
}


torch::Tensor intersection_penalty_forward_aabb(
    const torch::Tensor& v1, const torch::Tensor& v2, const torch::Tensor& v3,
    const torch::Tensor& triangle_centers,  // 三角形中心点，用于构建AABB树
    const torch::Tensor& aabb_tree_data,    // 预构建的AABB树数据
    const torch::Tensor& is_active,
    float lambda_intersect, float margin)
{
    TORCH_CHECK(v1.device().type() == at::kCUDA, "Inputs must be on CUDA");
    TORCH_CHECK(triangle_centers.device().type() == at::kCUDA, "Centers must be on CUDA");
    
    int P = v1.size(0);
    
    // 1. 分配输出张量
    torch::Tensor L_intersect = torch::empty({P}, v1.options());
    
    // 2. 获取AABB树数据
    // 假设aabb_tree_data是通过triangulation::build_aabb_tree构建的
    const triangulation::AABB<float>* aabb_tree = 
        static_cast<const triangulation::AABB<float>*>(aabb_tree_data.data_ptr());
    
    // 3. 调用修改后的CUDA内核
    triangulation::computeIntersectionPenaltyLoss_AABB(
        P, 
        (const float*)v1.data_ptr(), 
        (const float*)v2.data_ptr(), 
        (const float*)v3.data_ptr(),
        aabb_tree,
        P,  // 三角形数量
        (const float*)is_active.data_ptr(),
        (float*)L_intersect.data_ptr(),
        lambda_intersect, margin
    );
    
    return L_intersect;
}

// 添加构建三角形AABB树的函数
torch::Tensor build_triangle_aabb_tree(
    const torch::Tensor& triangle_centers,
    int max_triangles_per_node = 16) // 注意：底层 hardcode 了 256，这个参数目前仅作占位
{
    // 1. 严格的输入检查
    TORCH_CHECK(triangle_centers.device().type() == at::kCUDA, "Centers must be on CUDA");
    TORCH_CHECK(triangle_centers.dim() == 2 && triangle_centers.size(1) == 3, "Centers must have shape [N, 3]");
    TORCH_CHECK(triangle_centers.dtype() == torch::kFloat32, "Centers must be float32");
    
    uint32_t num_points = (uint32_t)triangle_centers.size(0);
    if (num_points == 0) return torch::empty({0, 6}, triangle_centers.options());

    // 2. 匹配底层的 pow2_round_up 逻辑
    auto pow2_round_up = [](uint32_t n) -> uint32_t {
        if (n == 0) return 0;
        n--;
        n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16;
        n++;
        return n;
    };

    // 3. 关键：根据 aabb_tree.cu 的逻辑计算总内存
    // 底层逻辑是：树的深度由 num_leaves 决定，而 num_leaves = pow2_round_up(num_points)
    // 并且每层都有 2 * (BUILD_AABB_BLOCK_SIZE - 1) 的额外偏移。
    uint32_t num_leaves = pow2_round_up(num_points);
    
    // 为了保证绝对安全，我们分配 2 * num_leaves。
    // 在 aabb_tree.cuh 的 get_node 函数中可以看到偏移逻辑：
    // level_start = aabb_tree + ((1 << tree_depth) - (1 << node_depth + 1))
    // 这意味着它需要 2 * num_leaves 的空间来存放所有层级的节点
    uint32_t total_nodes = 2 * num_leaves; 

    // 4. 分配内存 (利用 PyTorch 显存池)
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor aabb_tree_tensor = torch::empty({(int64_t)total_nodes, 6}, options);
    
    // 5. 调用底层算子
    // 注意：底层 build_aabb_tree 内部会自己处理排序，所以我们传 raw 指针即可
    try {
        triangulation::build_aabb_tree(
            triangulation::ScalarType::Float32,
            (const void*)triangle_centers.data_ptr<float>(),
            num_points,
            (void*)aabb_tree_tensor.data_ptr<float>()
        );
    } catch (const std::exception& e) {
        TORCH_CHECK(false, "Internal Error in build_aabb_tree CUDA kernel: ", e.what());
    }

    return aabb_tree_tensor;
}


// ** 实现 intersection_penalty_forward **
torch::Tensor intersection_penalty_forward(
    const torch::Tensor& v1, const torch::Tensor& v2, const torch::Tensor& v3,
    const torch::Tensor& neighbor_indices, const torch::Tensor& is_active,
    int K, float lambda_intersect, float margin) 
{
    TORCH_CHECK(neighbor_indices.scalar_type() == torch::kUInt32);
    TORCH_CHECK(neighbor_indices.dim() == 2 && neighbor_indices.size(1) == K);
    TORCH_CHECK(v1.device().type() == at::kCUDA, "Inputs must be on CUDA");
    
    int P = v1.size(0);
    
    // 1. 分配输出张量
    torch::Tensor L_intersect = torch::empty({P}, v1.options());
    
    // 2. 调用 triangulation 库中已有的函数（使用 CUDA 版本）
    triangulation::computeIntersectionPenaltyLoss(
        P, 
        (const float*)v1.data_ptr(), (const float*)v2.data_ptr(), (const float*)v3.data_ptr(),
        (const uint32_t*)neighbor_indices.data_ptr(),
        (const float*)is_active.data_ptr(),
        K,
        (float*)L_intersect.data_ptr(),
        lambda_intersect, margin
    );
    
    // 3. 返回结果
    return L_intersect; 
}

// --- SDF 计算函数 ---
torch::Tensor compute_sdf(torch::Tensor query_points, torch::Tensor vertices, torch::Tensor faces) {
    // Input checks
    if (query_points.device().type() != at::kCUDA || vertices.device().type() != at::kCUDA || faces.device().type() != at::kCUDA) {
        throw std::runtime_error("All tensors must be on CUDA device");
    }
    if (query_points.size(-1) != 3 || vertices.size(-1) != 3 || faces.size(-1) != 3) {
        throw std::runtime_error("Inputs must have last dimension 3");
    }

    int Q = query_points.size(0);
    int F = faces.size(0);
    
    auto opts_float = query_points.options().dtype(torch::kFloat32);
    torch::Tensor sdf = torch::empty({Q}, opts_float);
    torch::Tensor normal = torch::empty({Q, 3}, opts_float);
    
    // 调用 triangulation 库中已有的 SDF 函数
    triangulation::compute_sdf(
        Q, (const float*)query_points.data_ptr(),
        F, (const int32_t*)faces.data_ptr(), (const float*)vertices.data_ptr(),
        (float*)sdf.data_ptr(), (float*)normal.data_ptr()
    );

    // Return stacked: [SDF(1), Normal(3)] -> [Q, 4]
    return torch::cat({sdf.unsqueeze(1), normal}, 1);
}

// ** 统一的初始化函数 **
void init_triangulation_bindings(py::module &module) {
    printf("Binding starting...\n"); fflush(stdout);
    triangulation::global_cuda_init();

    py::register_exception<triangulation::TriangulationFailedError>(
        module, "TriangulationFailedError");

    // 注册 Triangulation 类
    py::class_<triangulation::Triangulation, std::unique_ptr<triangulation::Triangulation>>(
        module, "Triangulation")
        .def(py::init(&create_triangulation), py::arg("points"))
        .def("tets", &get_tets)
        .def("tet_adjacency", &get_tet_adjacency)
        .def("point_adjacency", &get_point_adjacency)
        .def("point_adjacency_offsets", &get_point_adjacency_offsets)
        .def("vert_to_tet", &get_vert_to_tet)
        .def("rebuild",
             &rebuild,
             py::arg("points"),
             py::arg("incremental") = false)
        .def("permutation", &permutation);

    // --- 新增: 几何约束 Loss 绑定 ---
    module.def("geometric_constraint_loss_forward", 
        &geometric_constraint_loss_forward, 
        "Computes Octree-anchored Mahalanobis Distance and Normal Orientation Loss (L_anchor + L_orient) for triangles.",
        py::arg("v1"), py::arg("v2"), py::arg("v3"),
        py::arg("anchor_mu"), py::arg("anchor_sigma_inv"), py::arg("anchor_normal"),
        py::arg("is_active"),
        py::arg("lambda_pos"), py::arg("lambda_norm"));

    module.def("laplacian_smoothness_forward", 
        &laplacian_smoothness_forward, 
        "Computes Laplacian Smoothness Loss (L_smooth) for triangle centers.",
        py::arg("v1"), py::arg("v2"), py::arg("v3"),
        py::arg("neighbor_indices"), py::arg("is_active"),
        py::arg("K"), py::arg("lambda_smooth"));

    module.def("build_triangle_aabb_tree", 
        &build_triangle_aabb_tree, 
        "Build AABB tree for triangle centers",
        py::arg("triangle_centers"),
        py::arg("max_triangles_per_node") = 16);

    module.def("intersection_penalty_forward_aabb", 
        &intersection_penalty_forward_aabb, 
        "Computes soft intersection penalty loss with AABB tree acceleration",
        py::arg("v1"), py::arg("v2"), py::arg("v3"),
        py::arg("triangle_centers"),py::arg("aabb_tree_data"), py::arg("is_active"),
        py::arg("lambda_intersect"), py::arg("margin"));
    
    module.def("intersection_penalty_forward", 
        &intersection_penalty_forward, 
        "Computes soft intersection penalty loss with KNN neighbors",
        py::arg("v1"), py::arg("v2"), py::arg("v3"),
        py::arg("neighbor_indices"), py::arg("is_active"),
        py::arg("K"), py::arg("lambda_intersect"), py::arg("margin"));

    // --- SDF 函数绑定 ---
    module.def("compute_sdf", &compute_sdf, 
               "Compute Signed Distance Field for a mesh defined by vertices and faces", 
               py::arg("query_points"), py::arg("vertices"), py::arg("faces"));
}


} // namespace triangulation_bindings

PYBIND11_MODULE(torch_bindings, module) {
    triangulation_bindings::init_triangulation_bindings(module);
}

