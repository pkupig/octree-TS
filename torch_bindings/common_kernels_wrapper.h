// torch_bindings/common_kernels_wrapper.h
#pragma once
#include <cstdint>

namespace triangulation {
    
    // 这些函数在 triangulation 库中已经存在，但可能名字不同
    // 从之前的错误信息看，它们可能是这样的：
    
    // 几何约束损失函数
    void computeGeometricConstraintLoss(
        int P,
        const float* v1, const float* v2, const float* v3,
        const float* anchor_mu, const float* anchor_sigma_inv, const float* anchor_normal,
        const float* is_active,
        float* L_anchor, float* L_orient,
        float lambda_pos, float lambda_norm);

    // Laplacian 平滑损失函数
    void computeLaplacianSmoothnessLoss(
        int P,
        const float* v1, const float* v2, const float* v3,
        const uint32_t* neighbor_indices,
        const float* is_active,
        int K,
        float* L_smooth,
        float lambda_smooth);

    // 交叉惩罚损失函数
    void computeIntersectionPenaltyLoss(
        int P,
        const float* v1, const float* v2, const float* v3,
        const uint32_t* neighbor_indices,
        const float* is_active,
        int K,
        float* L_intersect,
        float lambda_intersect, float margin);
    
    void computeIntersectionPenaltyLoss_AABB(
        int P,
        const float* v1, const float* v2, const float* v3,
        const triangulation::AABB<float>* aabb_tree,
        uint32_t num_triangles,
        const float* is_active,
        float* L_intersect,
        float lambda_intersect, float margin);    

    // SDF 计算函数
    void compute_sdf(
);
}