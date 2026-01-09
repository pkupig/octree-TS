#pragma once

#include <iostream>
#include "geometry.h"
#include "cuda_helpers.h"
#include <cub/iterator/arg_index_input_iterator.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/discard_output_iterator.cuh>
#include "unenumerate_iterator.cuh"
#include "../aabb_tree/aabb_tree.cuh" 



namespace triangulation {
// -------------------------------------------------------------------------
// 模板内核定义（放在头文件中，因为模板必须在编译时看到完整定义）
// -------------------------------------------------------------------------
template <typename InputIterator, typename UnaryFunction>
__global__ void for_n_kernel(InputIterator begin, size_t n, UnaryFunction f) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    while (i < n) {
        f(begin[i]);
        i += stride;
    }
}

template <int block_size, typename Kernel, typename... Args>
void launch_kernel_1d(Kernel kernel,
                      size_t n,
                      const void *stream,
                      Args... args) {
    if (n == 0) {
        return;
    }
    size_t num_blocks = (n + block_size - 1) / block_size;
    if (stream) {
        cudaStream_t s = *reinterpret_cast<const cudaStream_t *>(stream);
        kernel<<<num_blocks, block_size, 0, s>>>(args...);
    } else {
        kernel<<<num_blocks, block_size>>>(args...);
    }
    cuda_check(cudaGetLastError());
}

template <int block_size, typename InputIterator, typename UnaryFunction>
void for_n_b(InputIterator begin,
             size_t n,
             UnaryFunction f,
             bool strided = false,
             const void *stream = nullptr) {
    size_t num_threads = n;
    if (strided) {
        int mpc;
        cuda_check(
            cudaDeviceGetAttribute(&mpc, cudaDevAttrMultiProcessorCount, 0));
        num_threads = block_size * mpc;
    }

    launch_kernel_1d<block_size>(for_n_kernel<InputIterator, UnaryFunction>,
                                 num_threads,
                                 stream,
                                 begin,
                                 n,
                                 f);
}

template <typename InputIterator, typename UnaryFunction>
void for_n(InputIterator begin,
           size_t n,
           UnaryFunction f,
           bool strided = false,
           const void *stream = nullptr) {
    for_n_b<256>(begin, n, f, strided, stream);
}

template <typename InputIterator, typename UnaryFunction>
void for_range(InputIterator begin,
               InputIterator end,
               UnaryFunction f,
               bool strided = false,
               const void *stream = nullptr) {
    size_t n = end - begin;
    for_n(begin, n, f, strided, stream);
}

template <typename InputIterator,
          typename OutputIterator,
          typename UnaryFunction>
struct TransformFunctor {
    InputIterator begin;
    OutputIterator result;
    UnaryFunction f;

    __device__ void operator()(decltype(*begin) x) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        result[i] = f(x);
    }
};

template <typename InputIterator,
          typename OutputIterator,
          typename UnaryFunction>
void transform_range(InputIterator begin,
                     InputIterator end,
                     OutputIterator result,
                     UnaryFunction f,
                     bool strided = false,
                     const void *stream = nullptr) {
    size_t n = end - begin;
    TransformFunctor<InputIterator, OutputIterator, UnaryFunction> func = {
        begin, result, f};
    for_n(begin, n, func, strided, stream);
}

template <typename InputIterator, typename OutputIterator>
void copy_range(InputIterator begin,
                InputIterator end,
                OutputIterator result,
                bool strided = false,
                const void *stream = nullptr) {
    transform_range(
        begin,
        end,
        result,
        [] __device__(auto x) { return x; },
        strided,
        stream);
}

// -------------------------------------------------------------------------
// 内联设备函数声明
// -------------------------------------------------------------------------
__device__ __forceinline__ float3 operator-(float3 a, float3 b);
__device__ __forceinline__ float3 operator+(float3 a, float3 b);
__device__ __forceinline__ float3 operator*(float a, float3 b);
__device__ __forceinline__ float dot(float3 a, float3 b);
__device__ __forceinline__ float3 cross(float3 a, float3 b);
__device__ __forceinline__ float3 operator*(float3 b, float a);

// 工具函数
cub::CountingInputIterator<uint32_t> u32zero();
cub::CountingInputIterator<uint64_t> u64zero();
inline void compute_sdf(
    int num_queries, 
    const float3* query_points,
    int num_triangles, 
    const float3* v1,
    const float3* v2, 
    const float3* v3,
    float* out_sdf);

void compute_sdf(
    int Q, 
    const float* queries,
    int F, 
    const int32_t* faces,
    const float* verts,
    float* sdf,
    float* normals);

// 主要内核函数声明（在 .cu 文件中定义）
extern __global__ void compute_sdf_kernel(
    int num_queries,
    int num_triangles,
    const float3* __restrict__ query_pts,
    const float3* __restrict__ v1, 
    const float3* __restrict__ v2, 
    const float3* __restrict__ v3,
    float* __restrict__ out_sdf);


// 辅助迭代器函数
template <typename T>
cub::ArgIndexInputIterator<T *> enumerate(T *begin);
template <typename T>
UnenumerateIterator<T> unenumerate(T *begin);
inline cub::DiscardOutputIterator<> discard();

// 几何函数
inline RADFOAM_HD Vec3f computeTriangleNormal_Vec(const Vec3f& v1, const Vec3f& v2, const Vec3f& v3);

// 损失函数内核声明
extern __global__ void computeGeometricConstraintLossCUDA(
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
    const float anchor_lambda_norm);

extern void computeGeometricConstraintLoss(
    int P, 
    const float* v1_ptr, const float* v2_ptr, const float* v3_ptr,
    const float* anchor_mu_ptr, const float* anchor_sigma_inv_ptr, const float* anchor_normal_ptr,
    const float* is_active_ptr,
    float* out_L_anchor, 
    float* out_L_orient,
    const float anchor_lambda_pos,
    const float anchor_lambda_norm);

extern __global__ void computeLaplacianSmoothnessLossCUDA(
    int P,
    const float* __restrict__ v1_ptr,
    const float* __restrict__ v2_ptr,
    const float* __restrict__ v3_ptr,
    const uint32_t* __restrict__ neighbor_indices,
    const float* __restrict__ is_active_ptr,
    const int K,
    float* __restrict__ out_L_smooth,
    const float smooth_lambda);

extern void computeLaplacianSmoothnessLoss(
    int P, 
    const float* v1_ptr, const float* v2_ptr, const float* v3_ptr,
    const uint32_t* neighbor_indices,
    const float* is_active_ptr,
    const int K,
    float* out_L_smooth,
    const float smooth_lambda);

extern __global__ void computeIntersectionPenaltyLossCUDA(
    int P,
    const float* __restrict__ v1_ptr,
    const float* __restrict__ v2_ptr,
    const float* __restrict__ v3_ptr,
    const uint32_t* __restrict__ neighbor_indices,
    const float* __restrict__ is_active_ptr,
    const int K,
    float* __restrict__ out_L_intersect,
    const float intersect_lambda,
    const float margin);

extern void computeIntersectionPenaltyLoss(
    int P, 
    const float* v1_ptr, const float* v2_ptr, const float* v3_ptr,
    const uint32_t* neighbor_indices,
    const float* is_active_ptr,
    const int K,
    float* out_L_intersect,
    const float intersect_lambda,
    const float margin);

// AABB树加速的相交损失函数
extern __global__ void computeIntersectionPenaltyLossCUDA_AABB(
    int P,
    const float* __restrict__ v1_ptr,
    const float* __restrict__ v2_ptr,
    const float* __restrict__ v3_ptr,
    const triangulation::AABB<float>* aabb_tree,
    uint32_t num_triangles,
    const float* __restrict__ is_active_ptr,
    float* __restrict__ out_L_intersect,
    const float intersect_lambda,
    const float margin);

extern void computeIntersectionPenaltyLoss_AABB(
    int P, 
    const float* v1_ptr, const float* v2_ptr, const float* v3_ptr,
    const triangulation::AABB<float>* aabb_tree,
    uint32_t num_triangles,
    const float* is_active_ptr,
    float* out_L_intersect,
    const float intersect_lambda,
    const float margin);

// 设备函数声明
__device__ __forceinline__ float computeTriangleRadius(const Vec3f& c, const Vec3f& v1, const Vec3f& v2, const Vec3f& v3);

extern template cub::ArgIndexInputIterator<float *> enumerate<float>(float *begin);
extern template cub::ArgIndexInputIterator<float3 *> enumerate<float3>(float3 *begin);
extern template cub::ArgIndexInputIterator<IndexedTet *> enumerate<IndexedTet>(IndexedTet *begin);
extern template cub::ArgIndexInputIterator<IndexedTriangle*> enumerate<IndexedTriangle>(IndexedTriangle* begin);
extern template cub::ArgIndexInputIterator<uint32_t *> enumerate<uint32_t>(uint32_t *begin);

extern template UnenumerateIterator<float> unenumerate<float>(float *begin);
extern template UnenumerateIterator<float3> unenumerate<float3>(float3 *begin);
extern template UnenumerateIterator<IndexedTet> unenumerate<IndexedTet>(IndexedTet *begin);
extern template UnenumerateIterator<IndexedTriangle> unenumerate<IndexedTriangle>(IndexedTriangle* begin);
extern template UnenumerateIterator<uint32_t> unenumerate<uint32_t>(uint32_t *begin);

// -------------------------------------------------------------------------
// 内联设备函数定义（放在头文件末尾）
// -------------------------------------------------------------------------
__device__ __forceinline__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator*(float a, float3 b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ __forceinline__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ __forceinline__ float3 operator*(float3 b, float a) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

inline __device__ float point_triangle_distance_squared(
    const triangulation::Vec3f& p, 
    const triangulation::Vec3f& v1, const triangulation::Vec3f& v2, const triangulation::Vec3f& v3,
    triangulation::Vec3f& closest_point) 
{
    triangulation::Vec3f ab = v2 - v1;
    triangulation::Vec3f ac = v3 - v1;
    triangulation::Vec3f ap = p - v1;
    
    float d1 = ab.dot(ap);
    float d2 = ac.dot(ap);
    
    if (d1 <= 0.0f && d2 <= 0.0f) { closest_point = v1; return (p - v1).squaredNorm(); }
    
    triangulation::Vec3f bp = p - v2;
    float d3 = ab.dot(bp);
    float d4 = ac.dot(bp);
    if (d3 >= 0.0f && d4 <= d3) { closest_point = v2; return (p - v2).squaredNorm(); }
    
    float vc = d1*d4 - d3*d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        closest_point = v1 + ab * v;
        return (p - closest_point).squaredNorm();
    }
    
    triangulation::Vec3f cp = p - v3;
    float d5 = ab.dot(cp);
    float d6 = ac.dot(cp);
    if (d6 >= 0.0f && d5 <= d6) { closest_point = v3; return (p - v3).squaredNorm(); }
    
    float vb = d5*d2 - d1*d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        closest_point = v1 + ac * w;
        return (p - closest_point).squaredNorm();
    }
    
    float va = d3*d6 - d5*d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        closest_point = v2 + (v3 - v2) * w;
        return (p - closest_point).squaredNorm();
    }
    
    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    closest_point = v1 + ab * v + ac * w;
    return (p - closest_point).squaredNorm();
}

__device__ __forceinline__ float pointTriangleDistanceSq(float3 p, float3 a, float3 b, float3 c) {
    float3 ab = b - a;
    float3 ac = c - a;
    float3 ap = p - a;
    
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return dot(ap, ap);

    float3 bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return dot(bp, bp);

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        float3 target = a + v * ab;
        float3 diff = p - target;
        return dot(diff, diff);
    }

    float3 cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return dot(cp, cp);

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        float3 target = a + w * ac;
        float3 diff = p - target;
        return dot(diff, diff);
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        float3 target = b + w * (c - b);
        float3 diff = p - target;
        return dot(diff, diff);
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    float3 target = a + ab * v + ac * w;
    float3 diff = p - target;
    return dot(diff, diff);
}

void build_aabb_tree(
    ScalarType scalar_type,
    const void* points,
    uint32_t num_points,
    void* aabb_tree);



} // namespace triangulation