#include "common_kernels.cuh"
#include "../aabb_tree/aabb_tree.cuh"
#include <cuda_runtime.h>
#include "cuda_array.h"
#include <thrust/device_vector.h>
#include <stdexcept>
#include <math.h>

namespace triangulation {
// -------------------------------------------------------------------------
// 实现 OpaqueBuffer 的子类
// -------------------------------------------------------------------------
class CUDAOpaqueBufferImpl : public OpaqueBuffer {
private:
    void* ptr_;
    
public:
    CUDAOpaqueBufferImpl(size_t bytes) {
        cudaError_t err = cudaMalloc(&ptr_, bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Failed to allocate CUDA buffer: ") + cudaGetErrorString(err));
        }
    }
    
    ~CUDAOpaqueBufferImpl() override {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }
    
    void* data() override {
        return ptr_;
    }
};

// -------------------------------------------------------------------------
// 工具函数实现（从 common_kernels.cu 中移动过来）
// -------------------------------------------------------------------------

cub::CountingInputIterator<uint32_t> u32zero() {
    return cub::CountingInputIterator<uint32_t>(0);
}

cub::CountingInputIterator<uint64_t> u64zero() {
    return cub::CountingInputIterator<uint64_t>(0);
}
// -------------------------------------------------------------------------
// 内核函数实现（非模板函数）
// -------------------------------------------------------------------------
__global__ void compute_sdf_kernel(
    int num_queries,
    int num_triangles,
    const float3* __restrict__ query_pts,
    const float3* __restrict__ v1, 
    const float3* __restrict__ v2, 
    const float3* __restrict__ v3,
    float* __restrict__ out_sdf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_queries) return;

    float3 p_raw = query_pts[idx];
    Vec3f p(p_raw.x, p_raw.y, p_raw.z);
    
    float min_dist_sq = FLT_MAX;
    float sign = 1.0f;

    for (int i = 0; i < num_triangles; ++i) {
        float dist_sq = pointTriangleDistanceSq(p_raw, v1[i], v2[i], v3[i]);

        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            
            Vec3f vv1(v1[i].x, v1[i].y, v1[i].z);
            Vec3f vv2(v2[i].x, v2[i].y, v2[i].z);
            Vec3f vv3(v3[i].x, v3[i].y, v3[i].z);

            Vec3f center = (vv1 + vv2 + vv3) / 3.0f;
            Vec3f e1 = vv2 - vv1;
            Vec3f e2 = vv3 - vv1;
            Vec3f norm = e1.cross(e2); 
            
            sign = ((p - center).dot(norm) > 0) ? 1.0f : -1.0f;
        }
    }
    out_sdf[idx] = sign * sqrtf(min_dist_sq);
}

inline void compute_sdf(
    int num_queries, 
    const float3* query_points,
    int num_triangles, 
    const float3* v1,
    const float3* v2, 
    const float3* v3,
    float* out_sdf) 
{
    launch_kernel_1d<256>(
        compute_sdf_kernel,
        num_queries,
        nullptr, 
        num_queries,
        num_triangles,
        query_points,
        v1,
        v2,
        v3,
        out_sdf
    );
}

// 辅助函数：计算点到三角形的最近点和距离平方
__device__ float point_triangle_sq_dist(float3 p, float3 a, float3 b, float3 c, float3& closest_pt) {
    // 1. 将所有可能被跳过的变量提前声明
    float3 ab, ac, ap, bp, cp, diff;
    float d1, d2, d3, d4, d5, d6, vc, vb, va, v, w;

    ab = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
    ac = make_float3(c.x - a.x, c.y - a.y, c.z - a.z);
    ap = make_float3(p.x - a.x, p.y - a.y, p.z - a.z);

    d1 = ab.x*ap.x + ab.y*ap.y + ab.z*ap.z;
    d2 = ac.x*ap.x + ac.y*ap.y + ac.z*ap.z;
    
    // 情况 1: 顶点 A 区域
    if (d1 <= 0.0f && d2 <= 0.0f) { 
        closest_pt = a; 
        goto compute_dist;
    }

    bp = make_float3(p.x - b.x, p.y - b.y, p.z - b.z);
    d3 = ab.x*bp.x + ab.y*bp.y + ab.z*bp.z;
    d4 = ac.x*bp.x + ac.y*bp.y + ac.z*bp.z;
    
    // 情况 2: 顶点 B 区域
    if (d3 >= 0.0f && d4 <= d3) { 
        closest_pt = b; 
        goto compute_dist; 
    }

    vc = d1*d4 - d3*d2;
    // 情况 3: 边 AB 区域
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        v = d1 / (d1 - d3);
        closest_pt = make_float3(a.x + v * ab.x, a.y + v * ab.y, a.z + v * ab.z);
        goto compute_dist; 
    }

    cp = make_float3(p.x - c.x, p.y - c.y, p.z - c.z);
    d5 = ab.x*cp.x + ab.y*cp.y + ab.z*cp.z;
    d6 = ac.x*cp.x + ac.y*cp.y + ac.z*cp.z;
    
    // 情况 4: 顶点 C 区域
    if (d6 >= 0.0f && d5 <= d6) { 
        closest_pt = c; 
        goto compute_dist; 
    }

    vb = d5*d2 - d1*d6;
    // 情况 5: 边 AC 区域
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        w = d2 / (d2 - d6);
        closest_pt = make_float3(a.x + w * ac.x, a.y + w * ac.y, a.z + w * ac.z);
        goto compute_dist;
    }

    va = d3*d6 - d5*d4;
    // 情况 6: 边 BC 区域
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        closest_pt = make_float3(b.x + w * (c.x - b.x), b.y + w * (c.y - b.y), b.z + w * (c.z - b.z));
        goto compute_dist;
    }

    // 情况 0: 面内部区域
    {
        float denom = 1.0f / (va + vb + vc);
        v = vb * denom;
        w = vc * denom;
        closest_pt = make_float3(a.x + v * ab.x + w * ac.x, a.y + v * ab.y + w * ac.y, a.z + v * ab.z + w * ac.z);
    }

compute_dist:
    diff = make_float3(p.x - closest_pt.x, p.y - closest_pt.y, p.z - closest_pt.z);
    return diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
}


__global__ void compute_sdf_kernel_indexed(
    int Q, 
    const float3* __restrict__ queries, 
    int F, 
    const int3* __restrict__ faces, 
    const float3* __restrict__ vertices, 
    float* __restrict__ out_sdf, 
    float3* __restrict__ out_normals
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Q) return;

    float3 p = queries[idx];
    float min_dist_sq = 1e30f;
    float3 final_closest_pt = p;

    // 暴力遍历所有三角形 (注意：生产环境应使用 AABB Tree 遍历)
    for (int f = 0; f < F; ++f) {
        int3 face_idx = faces[f];
        
        // 直接通过索引读取顶点，无需中间 buffer
        float3 v1 = vertices[face_idx.x];
        float3 v2 = vertices[face_idx.y];
        float3 v3 = vertices[face_idx.z];

        float3 current_closest;
        // 如果你有现成的 triangulation::pointTriangleDistanceSq，就调用那个
        // 这里为了演示完整性，调用上面的辅助函数（你需要根据实际情况调整调用）
        // float dist_sq = pointTriangleDistanceSq(p, v1, v2, v3, current_closest); 
        float dist_sq = point_triangle_sq_dist(p, v1, v2, v3, current_closest);

        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            final_closest_pt = current_closest;
        }
    }

    // 写入 SDF (开根号)
    out_sdf[idx] = sqrtf(min_dist_sq);

    // 写入 Normal (方向向量：Query点 指向 最近点 的反方向，或者 Query点 - 最近点)
    // 通常我们希望得到 "推离表面的方向"，即 p - closest
    float3 diff = make_float3(p.x - final_closest_pt.x, p.y - final_closest_pt.y, p.z - final_closest_pt.z);
    
    // 归一化
    float len = sqrtf(min_dist_sq);
    if (len > 1e-8f) {
        diff.x /= len; diff.y /= len; diff.z /= len;
    }
    
    if (out_normals) {
        out_normals[idx] = diff;
    }
}

// Host 端实现
void compute_sdf(
    int Q, 
    const float* queries, 
    int F, 
    const int* faces, 
    const float* vertices, 
    float* sdf, 
    float* normals) 
{
    if (Q <= 0) return;

    const int block_size = 256;
    const int num_blocks = (Q + block_size - 1) / block_size;

    // 强制转换指针类型，复用 float3 的内存布局
    compute_sdf_kernel_indexed<<<num_blocks, block_size>>>(
        Q, 
        reinterpret_cast<const float3*>(queries), 
        F, 
        reinterpret_cast<const int3*>(faces), 
        reinterpret_cast<const float3*>(vertices), 
        sdf, 
        reinterpret_cast<float3*>(normals)
    );
    
    // 检查 Kernel 是否出错
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in compute_sdf: %s\n", cudaGetErrorString(err));
    }
}

// -------------------------------------------------------------------------
// 辅助函数实现
// -------------------------------------------------------------------------

template <typename T>
cub::ArgIndexInputIterator<T *> enumerate(T *begin) {
    return cub::ArgIndexInputIterator<T *>(begin);
}

template <typename T>
UnenumerateIterator<T> unenumerate(T *begin) {
    return UnenumerateIterator<T>(begin);
}

inline cub::DiscardOutputIterator<> discard() {
    return cub::DiscardOutputIterator();
}

inline RADFOAM_HD Vec3f computeTriangleNormal_Vec(const Vec3f& v1, const Vec3f& v2, const Vec3f& v3){
    Vec3f normal = (v2 - v1).cross(v3 - v2).normalized();
    return normal;
}

// -------------------------------------------------------------------------
// 损失函数内核实现
// -------------------------------------------------------------------------
__global__ void computeGeometricConstraintLossCUDA(
    int P,
    const float* __restrict__ v1_ptr,
    const float* __restrict__ v2_ptr,
    const float* __restrict__ v3_ptr,
    const float* __restrict__ anchor_mu_ptr,
    const float* __restrict__ anchor_sigma_inv_ptr,
    const float* __restrict__ anchor_normal_ptr,
    const float* __restrict__ is_active_ptr,
    float* __restrict__ out_L_anchor,
    float* __restrict__ out_L_orient,
    const float anchor_lambda_pos,
    const float anchor_lambda_norm)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    if (is_active_ptr[idx] == 0.0f) {
        out_L_anchor[idx] = 0.0f;
        out_L_orient[idx] = 0.0f;
        return;
    }
    
    Vec3f v1 = {v1_ptr[idx * 3 + 0], v1_ptr[idx * 3 + 1], v1_ptr[idx * 3 + 2]};
    Vec3f v2 = {v2_ptr[idx * 3 + 0], v2_ptr[idx * 3 + 1], v2_ptr[idx * 3 + 2]};
    Vec3f v3 = {v3_ptr[idx * 3 + 0], v3_ptr[idx * 3 + 1], v3_ptr[idx * 3 + 2]};

    Vec3f anchor_mu = {anchor_mu_ptr[idx * 3 + 0], anchor_mu_ptr[idx * 3 + 1], anchor_mu_ptr[idx * 3 + 2]};
    Vec3f anchor_normal = {anchor_normal_ptr[idx * 3 + 0], anchor_normal_ptr[idx * 3 + 1], anchor_normal_ptr[idx * 3 + 2]};

    Vec3f c_tri = (v1 + v2 + v3) / 3.0f;
    Vec3f n_tri = computeTriangleNormal_Vec(v1, v2, v3);

    Vec3f diff = c_tri - anchor_mu;
    Vec3f sigma_inv_diff;
    const float* sig_inv = &anchor_sigma_inv_ptr[idx * 6];
    
    sigma_inv_diff[0] = sig_inv[0] * diff[0] + sig_inv[1] * diff[1] + sig_inv[2] * diff[2];
    sigma_inv_diff[1] = sig_inv[1] * diff[0] + sig_inv[3] * diff[1] + sig_inv[4] * diff[2];
    sigma_inv_diff[2] = sig_inv[2] * diff[0] + sig_inv[4] * diff[1] + sig_inv[5] * diff[2];

    float L_anchor_val = diff.dot(sigma_inv_diff);
    out_L_anchor[idx] = L_anchor_val * anchor_lambda_pos; 

    float dot_product = fabsf(n_tri.dot(anchor_normal));
    float L_orient_val = 1.0f - dot_product;
    out_L_orient[idx] = L_orient_val * anchor_lambda_norm;
}

void computeGeometricConstraintLoss(
    int P, 
    const float* v1_ptr, const float* v2_ptr, const float* v3_ptr,
    const float* anchor_mu_ptr, const float* anchor_sigma_inv_ptr, const float* anchor_normal_ptr,
    const float* is_active_ptr,
    float* out_L_anchor, 
    float* out_L_orient,
    const float anchor_lambda_pos,
    const float anchor_lambda_norm) 
{
    launch_kernel_1d<256>(
        computeGeometricConstraintLossCUDA, 
        P, 
        (const void*)nullptr,
        P, 
        v1_ptr, v2_ptr, v3_ptr, 
        anchor_mu_ptr, anchor_sigma_inv_ptr, anchor_normal_ptr, 
        is_active_ptr, 
        out_L_anchor, out_L_orient,
        anchor_lambda_pos, anchor_lambda_norm
    );
}

// -------------------------------------------------------------------------
__global__ void computeLaplacianSmoothnessLossCUDA(
    int P,
    const float* __restrict__ v1_ptr,
    const float* __restrict__ v2_ptr,
    const float* __restrict__ v3_ptr,
    const uint32_t* __restrict__ neighbor_indices,
    const float* __restrict__ is_active_ptr,
    const int K,
    float* __restrict__ out_L_smooth,
    const float smooth_lambda)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    if (is_active_ptr[idx] == 0.0f) {
        out_L_smooth[idx] = 0.0f;
        return;
    }

    Vec3f v1 = {v1_ptr[idx * 3 + 0], v1_ptr[idx * 3 + 1], v1_ptr[idx * 3 + 2]};
    Vec3f v2 = {v2_ptr[idx * 3 + 0], v2_ptr[idx * 3 + 1], v2_ptr[idx * 3 + 2]};
    Vec3f v3 = {v3_ptr[idx * 3 + 0], v3_ptr[idx * 3 + 1], v3_ptr[idx * 3 + 2]};
    
    Vec3f c_i = (v1 + v2 + v3) / 3.0f;

    Vec3f c_avg = Vec3f::Zero();
    int valid_neighbors = 0;

    for (int k = 0; k < K; k++) {
        uint32_t neighbor_idx = neighbor_indices[idx * K + k];
        
        if (neighbor_idx < P) {
            Vec3f v1_j = {v1_ptr[neighbor_idx * 3 + 0], v1_ptr[neighbor_idx * 3 + 1], v1_ptr[neighbor_idx * 3 + 2]};
            Vec3f v2_j = {v2_ptr[neighbor_idx * 3 + 0], v2_ptr[neighbor_idx * 3 + 1], v2_ptr[neighbor_idx * 3 + 2]};
            Vec3f v3_j = {v3_ptr[neighbor_idx * 3 + 0], v3_ptr[neighbor_idx * 3 + 1], v3_ptr[neighbor_idx * 3 + 2]};
            
            Vec3f c_j = (v1_j + v2_j + v3_j) / 3.0f;
            
            c_avg += c_j;
            valid_neighbors++;
        }
    }

    float L_smooth_val = 0.0f;
    if (valid_neighbors > 0) {
        c_avg /= (float)valid_neighbors;
        Vec3f laplacian_vector = c_i - c_avg;
        L_smooth_val = laplacian_vector.squaredNorm();
    }

    out_L_smooth[idx] = L_smooth_val * smooth_lambda;
}

void computeLaplacianSmoothnessLoss(
    int P, 
    const float* v1_ptr, const float* v2_ptr, const float* v3_ptr,
    const uint32_t* neighbor_indices,
    const float* is_active_ptr,
    const int K,
    float* out_L_smooth,
    const float smooth_lambda) 
{
    launch_kernel_1d<256>(
        computeLaplacianSmoothnessLossCUDA, 
        P, 
        (const void*)nullptr,
        P, 
        v1_ptr, v2_ptr, v3_ptr, 
        neighbor_indices,
        is_active_ptr,
        K, 
        out_L_smooth,
        smooth_lambda
    );
}

// -------------------------------------------------------------------------
__device__ __forceinline__ float computeTriangleRadius(const Vec3f& c, const Vec3f& v1, const Vec3f& v2, const Vec3f& v3) {
    float d1 = (v1 - c).squaredNorm();
    float d2 = (v2 - c).squaredNorm();
    float d3 = (v3 - c).squaredNorm();
    return sqrtf(fmaxf(d1, fmaxf(d2, d3)));
}


__global__ void computeIntersectionPenaltyLossCUDA(
    int P,
    const float* __restrict__ v1_ptr,
    const float* __restrict__ v2_ptr,
    const float* __restrict__ v3_ptr,
    const uint32_t* __restrict__ neighbor_indices,
    const float* __restrict__ is_active_ptr,
    const int K,
    float* __restrict__ out_L_intersect,
    const float intersect_lambda,
    const float margin)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P || is_active_ptr[idx] == 0.0f) return;

    Vec3f v1 = {v1_ptr[idx * 3], v1_ptr[idx * 3 + 1], v1_ptr[idx * 3 + 2]};
    Vec3f v2 = {v2_ptr[idx * 3], v2_ptr[idx * 3 + 1], v2_ptr[idx * 3 + 2]};
    Vec3f v3 = {v3_ptr[idx * 3], v3_ptr[idx * 3 + 1], v3_ptr[idx * 3 + 2]};
    
    Vec3f c_i = (v1 + v2 + v3) / 3.0f;
    Vec3f n_i = computeTriangleNormal_Vec(v1, v2, v3);
    float r_i = computeTriangleRadius(c_i, v1, v2, v3);

    float total_penalty = 0.0f;
    int count = 0;

    for (int k = 0; k < K; k++) {
        uint32_t n_idx = neighbor_indices[idx * K + k];
        if (n_idx < P && n_idx != idx) {
            Vec3f nv1 = {v1_ptr[n_idx * 3], v1_ptr[n_idx * 3 + 1], v1_ptr[n_idx * 3 + 2]};
            Vec3f nv2 = {v2_ptr[n_idx * 3], v2_ptr[n_idx * 3 + 1], v2_ptr[n_idx * 3 + 2]};
            Vec3f nv3 = {v3_ptr[n_idx * 3], v3_ptr[n_idx * 3 + 1], v3_ptr[n_idx * 3 + 2]};
            
            Vec3f c_j = (nv1 + nv2 + nv3) / 3.0f;
            Vec3f n_j = computeTriangleNormal_Vec(nv1, nv2, nv3);
            float r_j = computeTriangleRadius(c_j, nv1, nv2, nv3);

            float dist = (c_i - c_j).norm();
            float dot = n_i.dot(n_j);
            
            float dynamic_margin = (r_i + r_j) * 0.7f;
            float effective_margin = fmaxf(margin, dynamic_margin);

            if (dist < effective_margin && fabsf(dot) < 0.8f) {
                float pen = (effective_margin - dist) / effective_margin;
                total_penalty += pen * pen * (1.0f - fabsf(dot));
                count++;
            }
        }
    }
    out_L_intersect[idx] = (count > 0 ? (total_penalty / count) : 0.0f) * intersect_lambda;
}


void computeIntersectionPenaltyLoss(
    int P, 
    const float* v1_ptr, const float* v2_ptr, const float* v3_ptr,
    const uint32_t* neighbor_indices,
    const float* is_active_ptr,
    const int K,
    float* out_L_intersect,
    const float intersect_lambda,
    const float margin) 
{
    launch_kernel_1d<256>(
        computeIntersectionPenaltyLossCUDA, 
        P, 
        (const void*)nullptr,
        P, 
        v1_ptr, v2_ptr, v3_ptr, 
        neighbor_indices, is_active_ptr, K, 
        out_L_intersect,
        intersect_lambda, margin
    );
}

__global__ void computeIntersectionPenaltyLossCUDA_AABB(
    int P,
    const float* __restrict__ v1_ptr,
    const float* __restrict__ v2_ptr,
    const float* __restrict__ v3_ptr,
    const triangulation::AABB<float>* aabb_tree,  // 使用你的AABB树结构
    uint32_t num_triangles,                       // 三角形数量（用于树遍历）
    const float* __restrict__ is_active_ptr,
    float* __restrict__ out_L_intersect,
    const float intersect_lambda,
    const float margin)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P || is_active_ptr[idx] == 0.0f) return;

    // 获取当前三角形顶点
    triangulation::Vec3f v1 = {v1_ptr[idx * 3], v1_ptr[idx * 3 + 1], v1_ptr[idx * 3 + 2]};
    triangulation::Vec3f v2 = {v2_ptr[idx * 3], v2_ptr[idx * 3 + 1], v2_ptr[idx * 3 + 2]};
    triangulation::Vec3f v3 = {v3_ptr[idx * 3], v3_ptr[idx * 3 + 1], v3_ptr[idx * 3 + 2]};
    
    // 计算三角形中心
    triangulation::Vec3f c_i = (v1 + v2 + v3) / 3.0f;
    
    // 计算三角形法线和半径
    triangulation::Vec3f n_i = triangulation::computeTriangleNormal_Vec(v1, v2, v3);
    float r_i = computeTriangleRadius(c_i, v1, v2, v3);

    // 使用你的遍历函数进行查询
    float total_penalty = 0.0f;
    int count = 0;
    
    // 定义节点遍历函数
    auto node_functor = [&](uint32_t current_depth, uint32_t current_node) {
        // 获取节点AABB
        auto node = triangulation::get_node(aabb_tree, 
                                           triangulation::log2(triangulation::pow2_round_up(num_triangles)), 
                                           current_depth, 
                                           current_node);
        
        // 检查当前三角形AABB与节点AABB是否相交
        // 为简单起见，我们使用三角形中心点进行粗略相交检测
        triangulation::Vec3f node_center = (node.min + node.max) * 0.5f;
        float node_radius = (node.max - node.min).norm() * 0.5f;
        float center_dist = (c_i - node_center).norm();
        
        if (center_dist > (r_i + node_radius + margin)) {
            return triangulation::TraversalAction::SkipSubtree;
        }
        return triangulation::TraversalAction::Continue;
    };
    
    // 定义叶子处理函数
    auto leaf_functor = [&](uint32_t leaf_idx) {
        if (leaf_idx >= P || leaf_idx == idx || is_active_ptr[leaf_idx] == 0.0f) {
            return;
        }
        
        // 获取邻居三角形的顶点
        triangulation::Vec3f nv1 = {v1_ptr[leaf_idx * 3], v1_ptr[leaf_idx * 3 + 1], v1_ptr[leaf_idx * 3 + 2]};
        triangulation::Vec3f nv2 = {v2_ptr[leaf_idx * 3], v2_ptr[leaf_idx * 3 + 1], v2_ptr[leaf_idx * 3 + 2]};
        triangulation::Vec3f nv3 = {v3_ptr[leaf_idx * 3], v3_ptr[leaf_idx * 3 + 1], v3_ptr[leaf_idx * 3 + 2]};
        
        // 精确相交检测
        triangulation::Vec3f c_j = (nv1 + nv2 + nv3) / 3.0f;
        triangulation::Vec3f n_j = triangulation::computeTriangleNormal_Vec(nv1, nv2, nv3);
        float r_j = computeTriangleRadius(c_j, nv1, nv2, nv3);

        float dist = (c_i - c_j).norm();
        float dot_val = n_i.dot(n_j);
        
        float dynamic_margin = (r_i + r_j) * 0.7f;
        float effective_margin = fmaxf(margin, dynamic_margin);

        if (dist < effective_margin && fabsf(dot_val) < 0.8f) {
            float pen = (effective_margin - dist) / effective_margin;
            total_penalty += pen * pen * (1.0f - fabsf(dot_val));
            count++;
        }
    };
    
    // 使用你的traverse函数遍历AABB树
    triangulation::traverse(num_triangles, 
                          triangulation::log2(triangulation::pow2_round_up(num_triangles)), 
                          [&](uint32_t depth, uint32_t node) {
        if (depth == triangulation::log2(triangulation::pow2_round_up(num_triangles))) {
            leaf_functor(node);
            return triangulation::TraversalAction::Continue;
        } else {
            return node_functor(depth, node);
        }
    });
    
    out_L_intersect[idx] = (count > 0 ? (total_penalty / count) : 0.0f) * intersect_lambda;
}

void computeIntersectionPenaltyLoss_AABB(
    int P, 
    const float* v1_ptr, const float* v2_ptr, const float* v3_ptr,
    const triangulation::AABB<float>* aabb_tree,
    uint32_t num_triangles,
    const float* is_active_ptr,
    float* out_L_intersect,
    const float intersect_lambda,
    const float margin) 
{
    launch_kernel_1d<256>(
        computeIntersectionPenaltyLossCUDA_AABB, 
        P, 
        (const void*)nullptr,
        P, 
        v1_ptr, v2_ptr, v3_ptr, 
        aabb_tree, num_triangles, is_active_ptr,
        out_L_intersect,
        intersect_lambda, margin
    );
}


template cub::ArgIndexInputIterator<float *> enumerate<float>(float *begin);
template cub::ArgIndexInputIterator<float3 *> enumerate<float3>(float3 *begin);
template cub::ArgIndexInputIterator<IndexedTet *> enumerate<IndexedTet>(IndexedTet *begin);
template cub::ArgIndexInputIterator<IndexedTriangle *> enumerate<IndexedTriangle>(IndexedTriangle *begin);
template cub::ArgIndexInputIterator<uint32_t *> enumerate<uint32_t>(uint32_t *begin);  

template UnenumerateIterator<float> unenumerate<float>(float *begin);
template UnenumerateIterator<float3> unenumerate<float3>(float3 *begin);
template UnenumerateIterator<IndexedTet> unenumerate<IndexedTet>(IndexedTet *begin);
template UnenumerateIterator<IndexedTriangle> unenumerate<IndexedTriangle>(IndexedTriangle *begin);
template UnenumerateIterator<uint32_t> unenumerate<uint32_t>(uint32_t *begin); 
} // namespace triangulation