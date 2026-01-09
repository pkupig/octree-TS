#

#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud
import math
from pytorch3d.ops import knn_points
from skimage.measure import marching_cubes
import triangulation
import torch.nn.functional as F
import time
from functools import reduce
from torch_scatter import scatter_max
from einops import repeat


class TriangleModel(nn.Module):

    def setup_functions(self):
        """self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid"""

        self.eps = 1e-6
        self.opacity_floor = 0.0
        self.opacity_activation = lambda x: self.opacity_floor + (1.0 - self.opacity_floor) * torch.sigmoid(x)
        # Matching inverse for any y in [m, 1): logit( (y - m)/(1 - m) )
        self.inverse_opacity_activation = lambda y: inverse_sigmoid(
            ((y.clamp(self.opacity_floor + self.eps, 1.0 - self.eps) - self.opacity_floor) /
            (1.0 - self.opacity_floor + self.eps))
        )

        self.exponential_activation = lambda x:math.exp(x)
        self.inverse_exponential_activation = lambda y: math.log(y)

    def __init__(self, sh_degree : int):
        super().__init__()
        self._triangles = torch.empty(0) # can be deleted eventually

        self.size_probs_zero = 0.0
        self.size_probs_zero_image_space = 0.0
        self.vertices = torch.empty(0)
        self._triangle_indices = torch.empty(0)
        self.vertex_weight = torch.empty(0)

        self._sigma = 0
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self.optimizer = None
        self.image_size = torch.empty(0)
        self.importance_score = torch.empty(0)
        self.add_percentage = 1.0

        self.scaling = 1

        # --- C1: Geometry Anchoring Parameters (NEW!) ---
        # 将 anchor_points 改为 buffer，不参与优化
        self.register_buffer('anchor_points', torch.empty(0, 3, dtype=torch.float32))
        self.anchor_feature_index = torch.empty(0, dtype=torch.long).cuda()    
        self.triangulation_obj = None # Hold C++ Triangulation object for SDF
        
        # --- 新增：八叉树层次结构（借鉴Octree-GS）---
        self._level = torch.empty(0, dtype=torch.int32)  # 每个顶点/三角形的层次
        self.levels = 0  # 总层次数
        self.base_layer = -1  # 基础层次（自动计算）
        self.voxel_size = 0.0  # 体素大小
        self.init_pos = torch.zeros(3, device="cuda")  # 八叉树起始点
        self.standard_dist = 1.0  # 标准距离
        self.fork = 2  # 八叉树分支因子
        self.extend = 1.1  # 边界扩展系数
        # --- 新增：八叉树锚定参数 (Octree-GS) ---
        # 这些是从八叉树预计算的，不参与训练
        self.register_buffer('octree_res', torch.empty(0, dtype=torch.float32))
        self.register_buffer('anchor_mu', torch.empty(0, 3, dtype=torch.float32))
        self.register_buffer('anchor_sigma_inv', torch.empty(0, 6, dtype=torch.float32))
        self.register_buffer('anchor_normal', torch.empty(0, 3, dtype=torch.float32))
        
        # 邻域索引（拓扑信息）
        self.register_buffer('neighbor_indices', torch.empty(0, dtype=torch.int32))

        # 渐进式训练参数
        self.progressive = True
        self.coarse_intervals = []
        self.init_level = 0  # 初始训练层次
        
        # 可见性筛选
        self._triangle_mask = torch.empty(0, dtype=torch.bool)  # 三角形可见性掩码
        self._triangle_centers = None  # 缓存三角形中心
        self._vertex_levels = None  # 顶点层次（从三角形层次推导）
        
        # 训练统计（类似Octree-GS的统计机制）
        self.opacity_accum = torch.empty(0)
        self.triangle_denom = torch.empty(0)
        self.triangle_gradient_accum = torch.empty(0)
        self.triangle_gradient_denom = torch.empty(0)
        
        # 相机信息缓存（用于可见性计算）
        self.cam_infos = torch.empty(0, 4).float().cuda()  # [cam_center_x, cam_center_y, cam_center_z, scale]
        
        self.setup_functions()

    def save_parameters(self, path):

        mkdir_p(path)

        point_cloud_state_dict = {}

        point_cloud_state_dict["triangles_points"] = self.vertices
        point_cloud_state_dict["_triangle_indices"] = self._triangle_indices
        point_cloud_state_dict["vertex_weight"] = self.vertex_weight
        point_cloud_state_dict["sigma"] = self._sigma
        point_cloud_state_dict["active_sh_degree"] = self.active_sh_degree
        point_cloud_state_dict["features_dc"] = self._features_dc
        point_cloud_state_dict["features_rest"] = self._features_rest
        point_cloud_state_dict["importance_score"] = self.importance_score
        point_cloud_state_dict["image_size"] = self.image_size        
        
        # 保存新增的八叉树参数
        point_cloud_state_dict["vertex_levels"] = self._level
        point_cloud_state_dict["voxel_size"] = self.voxel_size
        point_cloud_state_dict["init_pos"] = self.init_pos
        point_cloud_state_dict["standard_dist"] = self.standard_dist
        point_cloud_state_dict["levels"] = self.levels
        point_cloud_state_dict["base_layer"] = self.base_layer
        point_cloud_state_dict["fork"] = self.fork
        point_cloud_state_dict["anchor_mu"] = self.anchor_mu
        point_cloud_state_dict["anchor_sigma_inv"] = self.anchor_sigma_inv
        point_cloud_state_dict["anchor_normal"] = self.anchor_normal       
        if hasattr(self, 'neighbor_indices'):
            point_cloud_state_dict["neighbor_indices"] = self.neighbor_indices

        torch.save(point_cloud_state_dict, os.path.join(path, 'point_cloud_state_dict.pt'))

    def load_parameters(self, path, device="cuda", segment=False, ratio_threshold = 0.25):
        # 1. Load the dict you saved
        state = torch.load(os.path.join(path, "point_cloud_state_dict.pt"), map_location=device)

        # 2. Restore everything you put in there (one line each)
        self.vertices            = state["triangles_points"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        self._triangle_indices   = state["_triangle_indices"].to(device).to(torch.int32)
        self.vertex_weight       = state["vertex_weight"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        self._sigma              = state["sigma"]
        self.active_sh_degree    = state["active_sh_degree"]
        self._features_dc        = state["features_dc"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        self._features_rest      = state["features_rest"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        self.importance_score = state["importance_score"].to(device).to(torch.float32).detach().clone().requires_grad_(True)

        # 加载八叉树参数
        if "vertex_levels" in state:
            self._level = state["vertex_levels"].to(device).to(torch.int32)
            self.voxel_size = state["voxel_size"]
            self.init_pos = state["init_pos"].to(device)
            self.standard_dist = state["standard_dist"]
            self.levels = state["levels"]
            self.base_layer = state["base_layer"]
            self.fork = state.get("fork", 2)
        else:
            # 向后兼容：如果没有保存八叉树参数，则重新计算
            print("Warning: No octree parameters found in checkpoint, computing from scratch...")
            self.create_octree_from_points(self.vertices.detach())

        # For object extraction
        if segment:
            base = os.path.dirname(os.path.dirname(path))
            triangle_hits = torch.load(os.path.join(base, 'segmentation/triangle_hits_mask.pt'))
            triangle_hits_total = torch.load(os.path.join(base, 'segmentation/triangle_hits_total.pt'))

            min_hits = 1  

            # Handle division by zero - triangles with no renders get ratio 0
            triangle_ratio = torch.zeros_like(triangle_hits, dtype=torch.float32)
            valid_mask = triangle_hits_total > 0
            triangle_ratio[valid_mask] = triangle_hits[valid_mask].float() / triangle_hits_total[valid_mask].float()

            # Create the keep mask: triangles must meet both ratio and minimum hits criteria
            keep_mask = (triangle_ratio >= ratio_threshold) & (triangle_hits >= min_hits)
            #keep_mask = ~keep_mask

            with torch.no_grad():
                self._triangle_indices = self._triangle_indices[keep_mask]

        ################################################################


        self.opacity_floor = 0.9999

        # 3. (Re)compute any derived quantities

        self._triangle_indices = self._triangle_indices.to(torch.int32)


        param_groups = [
            {'params': [self._features_dc], 'lr': 0.0, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': 0.0 / 20.0, "name": "f_rest"},
            {'params': [self.vertices], 'lr': 0.0, "name": "vertices"},
            {'params': [self.vertex_weight], 'lr': 0.0, "name": "vertex_weight"},
            {'params': [self.anchor_points], 'lr': 0.0, "name": "anchor_points"}
        ]

        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        
        # 初始化训练统计
        self.opacity_accum = torch.zeros((self._triangle_indices.shape[0], 1), device="cuda")
        self.triangle_denom = torch.zeros((self._triangle_indices.shape[0], 1), device="cuda")
        self.triangle_gradient_accum = torch.zeros((self._triangle_indices.shape[0], 1), device="cuda")
        self.triangle_gradient_denom = torch.zeros((self._triangle_indices.shape[0], 1), device="cuda")
        
        # 初始化可见性掩码
        self._triangle_mask = torch.ones((self._triangle_indices.shape[0]), dtype=torch.bool, device="cuda")


    def capture(self):
        return (
            self.active_sh_degree,
            self._features_dc,
            self._features_rest,
            self.optimizer.state_dict(),
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._features_dc, 
        self._features_rest,
        opt_dict) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    @property
    def get_triangle_indices(self):
        return self._triangle_indices

    @property
    def get_triangles_points_flatten(self):
        return self._triangles.flatten(0)
  
    @property
    def get_triangles_points(self):
        return self._triangles

    @property
    def get_vertices(self):
        return self.vertices
    
    @property
    def get_sigma(self):
        return self.exponential_activation(self._sigma)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_vertex_weight(self):
        return self.opacity_activation(self.vertex_weight)
    
    @property
    def get_vertex_levels(self):
        return self._level
    
    @property
    def get_triangle_levels(self):
        """计算三角形的平均层次"""
        if self._level.numel() == 0 or self._triangle_indices.numel() == 0:
            return torch.empty(0, dtype=torch.int32, device="cuda")
        
        # 三角形的层次是其三个顶点层次的平均值
        triangle_levels = self._level[self._triangle_indices].float().mean(dim=1).int()
        return triangle_levels

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_octree_from_points(self, points, visible_threshold=-1):
        """从点云构建八叉树层次结构（类似Octree-GS）"""
        print("Building octree structure for triangles...")
        
        # 计算场景边界
        box_min = torch.min(points, dim=0)[0] * self.extend
        box_max = torch.max(points, dim=0)[0] * self.extend
        box_d = box_max - box_min
        
        # 自动计算基础层
        if self.base_layer < 0:
            default_voxel_size = 0.02
            avg_box_d = torch.mean(box_d)  # 或者使用 torch.max(box_d)
            self.base_layer = torch.round(torch.log2(avg_box_d / default_voxel_size) / math.log2(self.fork)).int().item() - (self.levels // 2 if self.levels > 0 else 0) + 1        
        self.voxel_size = torch.mean(box_d) / (float(self.fork) ** self.base_layer)
        self.init_pos = torch.tensor([box_min[0], box_min[1], box_min[2]], device="cuda")
        
        torch.cuda.synchronize()
        t0 = time.time()
        
        # 分层采样构建八叉树
        positions = torch.empty(0, 3, device="cuda")
        levels = torch.empty(0, dtype=torch.int32, device="cuda")
        
        for cur_level in range(self.levels):
            cur_size = self.voxel_size / (float(self.fork) ** cur_level)
            new_positions = torch.unique(torch.round((points - self.init_pos) / cur_size), dim=0) * cur_size + self.init_pos
            new_levels = torch.ones(new_positions.shape[0], dtype=torch.int32, device="cuda") * cur_level
            positions = torch.cat([positions, new_positions], dim=0)
            levels = torch.cat([levels, new_levels], dim=0)
        
        # 为每个顶点分配最近的八叉树节点层次
        if positions.shape[0] > 0:
            # 使用KNN找到每个点最近的八叉树节点
            knn_result = knn_points(points.unsqueeze(0), positions.unsqueeze(0), K=1)
            nearest_indices = knn_result.idx.squeeze()
            self._level = levels[nearest_indices]
        else:
            self._level = torch.zeros(points.shape[0], dtype=torch.int32, device="cuda")
        
        torch.cuda.synchronize()
        t1 = time.time()
        time_diff = t1 - t0
        
        print(f"Octree building time: {int(time_diff // 60)} min {time_diff % 60:.2f} sec")
        print(f"Levels: {self.levels}, Base layer: {self.base_layer}")
        print(f"Voxel size range: {self.voxel_size/(2.0**(self.levels-1)):.6f} to {self.voxel_size:.6f}")
        print(f"Vertex levels distribution: {torch.unique(self._level, return_counts=True)}")
        
        return self._level

    def create_from_pcd(self, pcd : BasicPointCloud, opacity : float, set_sigma : float):

        # we remove all points that are too close to each other. Otherwise, this somehow gives an oom
        pcd_points = np.asarray(pcd.points)
        pcd_points = np.round(pcd_points, decimals=6)
        _, unique_indices = np.unique(pcd_points, axis=0, return_index=True)
        pcd_points = pcd_points[np.sort(unique_indices)]
        _points = torch.tensor(pcd_points).float().cuda()

        n = _points.shape[0]
        
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)[:n]).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        ################################
        # RUN DELAUNAY TRIANGULATION   #
        ################################
        max_init_points = 150000  # 建议先限制在 10 万点以内，4090 应该能跑
        if _points.shape[0] > max_init_points:
            print(f"[Warning] Initial points ({_points.shape[0]}) too many, subsampling to {max_init_points}")
            indices = torch.randperm(_points.shape[0])[:max_init_points]
            _points = _points[indices]
        torch.cuda.empty_cache()
        dt = triangulation.Triangulation(_points)
        perm = dt.permutation().to(torch.long)
        _points = _points[perm]
        features = features[perm]
        
        tets = dt.tets()     
        tets = dt.tets().to(torch.int64)

        faces = torch.cat([
            tets[:, [0, 1, 2]],
            tets[:, [0, 1, 3]],
            tets[:, [0, 2, 3]],
            tets[:, [1, 2, 3]],
        ], dim=0)  # [4 * num_tets, 3]

        
        # Step 3: Sort to ignore winding order
        faces, _ = torch.sort(faces, dim=1)
        faces = torch.unique(faces, dim=0)

        # finally stash on your module
        self.vertices = nn.Parameter(_points.requires_grad_(True))
        self._triangle_indices = faces.to(torch.int32)                         # [T,3]
        vert_weight = inverse_sigmoid(opacity * torch.ones((self.vertices.shape[0], 1), dtype=torch.float, device="cuda")) 
        self.vertex_weight = nn.Parameter(vert_weight.requires_grad_(True))

        # Sigma should be very low such that the triangles are solid. No need for soft triangles.
        self._sigma = self.inverse_exponential_activation(set_sigma)


        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))

        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        
        # --- C1: Initialize Anchors and Topology (NEW!) ---
        # 锚点使用初始点云位置，但不参与优化
        _anchor_points = _points.clone().detach()
        self.anchor_points = _anchor_points  # 直接赋值给 buffer

        # 锚点索引与三角形索引同步
        self.anchor_feature_index = faces.clone().detach().to(torch.long).cuda()

        # 初始化八叉树锚定参数（这里用简化版本，实际应从八叉树计算）
        num_triangles = faces.shape[0]
        self.anchor_mu = _anchor_points[faces].mean(dim=1)  # 三角形中心作为初始mu
        self.anchor_sigma_inv = torch.ones((num_triangles, 6), device="cuda") * 0.01  # 简化的协方差逆
        self.anchor_normal = torch.linalg.cross(
            _anchor_points[faces[:, 1]] - _anchor_points[faces[:, 0]],
            _anchor_points[faces[:, 2]] - _anchor_points[faces[:, 0]],
            dim = -1
        ).detach()
        self.anchor_normal = F.normalize(self.anchor_normal, p=2, dim=1)

        if hasattr(self, '_level') and self._level.numel() > 0:
            # 根据层次计算边长：基础边长 / 2^level
            base_size = self.voxel_size  # 基础体素大小
            triangle_levels = self.get_triangle_levels()
            self.octree_res = base_size / (2.0 ** triangle_levels.float())
        else:
            # 使用默认值
            self.octree_res = torch.ones((num_triangles,), device="cuda") * 0.05

        # 初始化邻域索引（这里用简化版本，实际应从八叉树计算）
        K = 5  # 邻居数量
        # 计算三角形中心
        tri_centers = self.anchor_mu
        # 使用KNN找到邻居（这里简化处理，实际应该使用八叉树）
        if num_triangles > 0:
            try:
                from pytorch3d.ops import knn_points
                # 转换为合适的形状进行KNN
                tri_centers_unsqueezed = tri_centers.unsqueeze(0)  # [1, N, 3]
                knn_result = knn_points(tri_centers_unsqueezed, tri_centers_unsqueezed, K=K+1)  # +1 因为包含自己
                neighbor_idx = knn_result.idx.squeeze(0)[:, 1:]  # 去掉自己，得到[N, K]
                self.neighbor_indices = neighbor_idx.reshape(-1).to(torch.int32)
            except ImportError:
                print("Warning: PyTorch3D not available, using random neighbor indices")
                # 生成随机邻居索引作为占位符
                random_neighbors = torch.randint(0, num_triangles, (num_triangles * K,), device="cuda")
                self.neighbor_indices = random_neighbors.to(torch.int32)

        # 初始化 is_active 张量
        self._is_active_tensor = torch.ones((num_triangles, 1), device="cuda", dtype=torch.float32)        
        # 初始化 C++ Triangulation 对象（用于未来的拓扑查询和 SDF）
        try:
            # 这是一个可选步骤，如果不需要在训练时动态重建拓扑，可以省略
            self.triangulation_obj = triangulation.Triangulation(_points) 
        except Exception as e:
            print(f"Warning: Triangulation object creation failed: {e}. SDF extraction may fail.")
        
        # --- 新增：初始化八叉树层次结构 ---
        # 首先计算场景的层次信息
        self.set_level(_points, [], [], dist_ratio=0.95, init_level=-1, levels=-1)
        # 然后构建八叉树
        self.create_octree_from_points(_points)
        
        # --- 新增：初始化训练统计 ---
        self.opacity_accum = torch.zeros((self._triangle_indices.shape[0], 1), device="cuda")
        self.triangle_denom = torch.zeros((self._triangle_indices.shape[0], 1), device="cuda")
        self.triangle_gradient_accum = torch.zeros((self._triangle_indices.shape[0], 1), device="cuda")
        self.triangle_gradient_denom = torch.zeros((self._triangle_indices.shape[0], 1), device="cuda")
        self._triangle_mask = torch.ones((self._triangle_indices.shape[0]), dtype=torch.bool, device="cuda")
        
        print(f"Triangle model created: {self.vertices.shape[0]} vertices, {self._triangle_indices.shape[0]} triangles")
        print(f"Octree initialized: {self.levels} levels, base layer {self.base_layer}")


    def set_level(self, points, cameras, scales, dist_ratio=0.95, init_level=-1, levels=-1):
        """设置层次参数（类似Octree-GS）"""
        all_dist = torch.tensor([], device="cuda")
        self.cam_infos = torch.empty(0, 4).float().cuda()
        
        # 收集相机信息
        if len(cameras) > 0:
            for scale in scales:
                for cam in cameras[scale]:
                    cam_center = cam.camera_center
                    cam_info = torch.tensor([cam_center[0], cam_center[1], cam_center[2], scale]).float().cuda()
                    self.cam_infos = torch.cat((self.cam_infos, cam_info.unsqueeze(dim=0)), dim=0)
                    dist = torch.sqrt(torch.sum((points - cam_center)**2, dim=1))
                    dist_max = torch.quantile(dist, dist_ratio)
                    dist_min = torch.quantile(dist, 1 - dist_ratio)
                    new_dist = torch.tensor([dist_min, dist_max]).float().cuda()
                    new_dist = new_dist * scale
                    all_dist = torch.cat((all_dist, new_dist), dim=0)
        
        if all_dist.numel() > 0:
            dist_max = torch.quantile(all_dist, dist_ratio)
            dist_min = torch.quantile(all_dist, 1 - dist_ratio)
        else:
            # 如果没有相机信息，使用基于点云分布的估计
            scene_center = points.mean(dim=0)
            dist = torch.norm(points - scene_center, dim=1)
            dist_max = torch.quantile(dist, 0.95)
            dist_min = torch.quantile(dist, 0.05)
        
        self.standard_dist = dist_max
        
        if levels == -1:
            self.levels = torch.round(torch.log2(dist_max / dist_min) / math.log2(self.fork)).int().item() + 1
        else:
            self.levels = levels
            
        if init_level == -1:
            self.init_level = int(self.levels / 2)
        else:
            self.init_level = init_level
            
        print(f"Level settings: total levels={self.levels}, init_level={self.init_level}, standard_dist={self.standard_dist:.4f}")

    def set_coarse_interval(self, coarse_iter, coarse_factor):
        """设置从粗到细的训练间隔（类似Octree-GS）"""
        self.coarse_intervals = []
        num_level = self.levels - 1 - self.init_level
        if num_level > 0:
            q = 1 / coarse_factor
            a1 = coarse_iter * (1 - q) / (1 - q ** num_level)
            temp_interval = 0
            for i in range(num_level):
                interval = a1 * q ** i + temp_interval
                temp_interval = interval
                self.coarse_intervals.append(interval)
        print(f"Coarse intervals: {self.coarse_intervals}")

    def set_triangle_mask(self, cam_center, iteration, resolution_scale=1.0):
        """根据相机位置筛选可见三角形（类似Octree-GS的set_anchor_mask）"""
        # 如果没有三角形，返回空的掩码
        if self._triangle_indices.numel() == 0:
            self._triangle_mask = torch.empty(0, dtype=torch.bool, device="cuda")
            return self._triangle_mask
        
        # 计算三角形中心并缓存
        if self._triangle_centers is None or self._triangle_centers.shape[0] != self._triangle_indices.shape[0]:
            try:
                tri_pts = self.vertices[self._triangle_indices]
                self._triangle_centers = tri_pts.mean(dim=1)
            except Exception as e:
                print(f"[Error] Failed to compute triangle centers: {e}")
                print(f"  vertices shape: {self.vertices.shape}")
                print(f"  triangle_indices shape: {self._triangle_indices.shape}")
                if self._triangle_indices.numel() > 0:
                    print(f"  triangle_indices max: {self._triangle_indices.max()}, vertices max index: {self.vertices.shape[0]-1}")
                self._triangle_mask = torch.zeros(self._triangle_indices.shape[0], dtype=torch.bool, device="cuda")
                return self._triangle_mask
        
        if self._triangle_centers.numel() == 0:
            self._triangle_mask = torch.empty(0, dtype=torch.bool, device="cuda")
            return self._triangle_mask
        
        # 计算相机到三角形中心的距离
        try:
            dist = torch.norm(self._triangle_centers - cam_center, dim=1) * resolution_scale
        except Exception as e:
            print(f"[Error] Failed to compute distances: {e}")
            self._triangle_mask = torch.zeros(self._triangle_centers.shape[0], dtype=torch.bool, device="cuda")
            return self._triangle_mask
        
        # 根据距离预测层次
        # 防止除零错误
        safe_dist = torch.clamp(dist, min=1e-6)
        pred_level = torch.log2(self.standard_dist / safe_dist) / math.log2(self.fork)
        
        # 根据训练阶段决定最大层次
        is_training = hasattr(self, 'optimizer') and self.optimizer is not None
        if self.progressive and is_training and len(self.coarse_intervals) > 0:
            coarse_index = min(np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level, self.levels - 1)
        else:
            coarse_index = self.levels - 1
        
        # 映射到整数层次
        pred_level = torch.clamp(pred_level, 0, coarse_index)
        int_level = torch.round(pred_level).int()
        
        # 获取三角形层次（三角形三个顶点层次的平均值）
        if self._level.numel() == 0:
            triangle_levels = torch.zeros(self._triangle_indices.shape[0], dtype=torch.int32, device="cuda")
        else:
            try:
                # 确保索引在有效范围内
                if self._triangle_indices.max() < self._level.shape[0]:
                    triangle_levels = self._level[self._triangle_indices].float().mean(dim=1).int()
                else:
                    print(f"[Warning] Triangle indices out of range for level: {self._triangle_indices.max()} >= {self._level.shape[0]}")
                    triangle_levels = torch.zeros(self._triangle_indices.shape[0], dtype=torch.int32, device="cuda")
            except Exception as e:
                print(f"[Error] Failed to compute triangle levels: {e}")
                triangle_levels = torch.zeros(self._triangle_indices.shape[0], dtype=torch.int32, device="cuda")
        
        # 筛选可见三角形（三角形层次 <= 预测层次）
        try:
            self._triangle_mask = (triangle_levels <= int_level)
        except Exception as e:
            print(f"[Error] Failed to create triangle mask: {e}")
            print(f"  triangle_levels shape: {triangle_levels.shape}")
            print(f"  int_level shape: {int_level.shape}")
            self._triangle_mask = torch.zeros(self._triangle_indices.shape[0], dtype=torch.bool, device="cuda")
        
        # 可选：添加基于距离的进一步筛选（避免太远的三角形）
        try:
            max_dist = self.standard_dist * 2.0  # 可调整的参数
            distance_mask = dist < max_dist
            self._triangle_mask = self._triangle_mask & distance_mask
        except Exception as e:
            print(f"[Warning] Failed to apply distance mask: {e}")
        
        # 确保掩码形状正确
        if self._triangle_mask.shape[0] != self._triangle_indices.shape[0]:
            print(f"[Error] Triangle mask shape mismatch: {self._triangle_mask.shape[0]} != {self._triangle_indices.shape[0]}")
            self._triangle_mask = torch.zeros(self._triangle_indices.shape[0], dtype=torch.bool, device="cuda")
        
        return self._triangle_mask


    def training_setup(self, training_args, lr_features, weight_lr, lr_triangles_init):

        l = [
            {'params': [self._features_dc], 'lr': lr_features, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': lr_features / 20.0, "name": "f_rest"},
            {'params': [self.vertices], 'lr': lr_triangles_init, "name": "vertices"},
            {'params': [self.vertex_weight], 'lr': weight_lr, "name": "vertex_weight"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.triangle_scheduler_args = get_expon_lr_func(lr_init=lr_triangles_init,
                                                        lr_final=lr_triangles_init/100,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)

    def set_sigma(self, sigma):
        self._sigma = self.inverse_exponential_activation(sigma)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "vertices":
                    if iteration < 1000:
                        lr = 0
                    else:
                        lr = self.triangle_scheduler_args(iteration)
                    param_group['lr'] = lr
                    return lr

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            group_name = group["name"]
            
            if group_name not in tensors_dict:
                continue
                
            extension_tensor = tensors_dict[group_name]
            
            # 确保 extension_tensor 是有效的张量
            if extension_tensor is None:
                print(f"[Error] extension_tensor is None for group {group_name}, skipping")
                continue
            
            if not isinstance(extension_tensor, torch.Tensor):
                print(f"[Error] extension_tensor is not a torch.Tensor for group {group_name}, type: {type(extension_tensor)}")
                continue
            
            p = group["params"][0]
            stored_state = self.optimizer.state.get(p, None)
            
            if stored_state is not None:
                # 确保状态张量存在
                if "exp_avg" not in stored_state or stored_state["exp_avg"] is None:
                    stored_state["exp_avg"] = torch.zeros_like(p)
                if "exp_avg_sq" not in stored_state or stored_state["exp_avg_sq"] is None:
                    stored_state["exp_avg_sq"] = torch.zeros_like(p)
                
                # 检查形状是否匹配
                try:
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                    
                    del self.optimizer.state[p]
                    group["params"][0] = nn.Parameter(torch.cat((p, extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group_name] = group["params"][0]
                except Exception as e:
                    print(f"[Error] Group {group_name} failed during cat: {e}")
                    print(f"  p shape: {p.shape}, extension shape: {extension_tensor.shape}")
                    print(f"  exp_avg shape: {stored_state['exp_avg'].shape if 'exp_avg' in stored_state else 'N/A'}")
                    raise e
            else:
                group["params"][0] = nn.Parameter(torch.cat((p, extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group_name] = group["params"][0]
        
        return optimizable_tensors
    

    def densification_postfix(self, new_vertices, new_vertex_weight, new_features_dc, new_features_rest, new_triangles):
        # 防御性检查
        if new_vertices is None or new_vertex_weight is None or new_features_dc is None or new_features_rest is None:
            print("[Error] One or more input tensors are None in densification_postfix")
            return
        
        # 创建字典（只包含可训练参数）
        d = {
            "vertices": new_vertices,
            "vertex_weight": new_vertex_weight,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
        }

        # 合并到优化器（只合并可训练参数）
        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        # 更新模型参数
        self.vertices = optimizable_tensors.get("vertices", self.vertices)
        self.vertex_weight = optimizable_tensors.get("vertex_weight", self.vertex_weight)
        self._features_dc = optimizable_tensors.get("f_dc", self._features_dc)
        self._features_rest = optimizable_tensors.get("f_rest", self._features_rest)

        # 更新锚点（buffer，不参与优化）
        if hasattr(self, 'new_anchor_points') and self.new_anchor_points.numel() > 0:
            # 合并到 anchor_points buffer
            self.anchor_points = torch.cat([self.anchor_points, self.new_anchor_points], dim=0)
            
            # 为新顶点创建层次信息
            if hasattr(self, '_level') and self._level.numel() > 0:
                # 获取父顶点的层次
                new_vertex_base = self.vertices.shape[0] - new_vertices.shape[0]
                parent_levels = self._level[new_triangles[:, 0]]  # 使用第一个顶点的父层次
                new_levels = torch.clamp(parent_levels + 1, 0, max(self.levels - 1, 0))
                self._level = torch.cat([self._level, new_levels], dim=0)
            
            # 清理临时存储
            delattr(self, 'new_anchor_points')     

        # 更新三角形索引
        self._triangle_indices = torch.cat([
            self._triangle_indices, 
            new_triangles
        ], dim=0)

        # 为新三角形创建锚点索引
        new_triangle_count = new_triangles.shape[0]
        if self.anchor_feature_index.numel() > 0:
            # 简单复制新三角形的索引（实际可能需要更复杂的映射）
            self.anchor_feature_index = torch.cat([
                self.anchor_feature_index,
                new_triangles.clone().detach().to(torch.long)
            ], dim=0)
        else:
            self.anchor_feature_index = new_triangles.clone().detach().to(torch.long)

        # 为新三角形初始化八叉树锚定参数
        if hasattr(self, 'anchor_mu') and self.anchor_mu.numel() > 0:
            # 计算新三角形的中心
            new_tri_verts = self.vertices[new_triangles]
            new_mu = new_tri_verts.mean(dim=1)
            
            # 扩展锚定参数
            self.anchor_mu = torch.cat([self.anchor_mu, new_mu], dim=0)
            
            # 扩展协方差逆（使用父节点的平均值）
            if new_triangle_count > 0:
                parent_sigma_inv = self.anchor_sigma_inv[selected_indices].mean(dim=0, keepdim=True)
                new_sigma_inv = parent_sigma_inv.repeat(new_triangle_count, 1)
                self.anchor_sigma_inv = torch.cat([self.anchor_sigma_inv, new_sigma_inv], dim=0)
                
                # 扩展法线（计算新三角形的法线）
                new_normal = torch.cross(
                    new_tri_verts[:, 1] - new_tri_verts[:, 0],
                    new_tri_verts[:, 2] - new_tri_verts[:, 0]
                )
                new_normal = F.normalize(new_normal, p=2, dim=1)
                self.anchor_normal = torch.cat([self.anchor_normal, new_normal], dim=0)

        # 更新 is_active 张量
        new_is_active = torch.ones((new_triangle_count, 1), device="cuda", dtype=torch.float32)
        self._is_active_tensor = torch.cat([self._is_active_tensor, new_is_active], dim=0)

        # 更新邻居索引（这里简化处理，实际需要重新计算）
        if hasattr(self, 'neighbor_indices') and self.neighbor_indices.numel() > 0:
            # 暂时清空，等待下次重新计算
            self.neighbor_indices = torch.empty(0, dtype=torch.int32, device="cuda")
            print("Warning: Neighbor indices cleared after densification, will be recomputed later")

        # 重置统计信息
        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        
        # 更新八叉树层次
        if hasattr(self, '_level') and self._level.numel() > 0:
            # 为新顶点分配层次（父顶点层次+1）
            if new_triangles.numel() > 0:
                parent_levels = self._level[new_triangles[:, 0]]
                new_levels = torch.clamp(parent_levels + 1, 0, self.levels - 1)
                self._level = torch.cat([self._level, new_levels], dim=0)
        
        # 扩展训练统计
        new_opacity_accum = torch.zeros((new_triangles.shape[0], 1), device="cuda")
        new_triangle_denom = torch.zeros((new_triangles.shape[0], 1), device="cuda")
        new_gradient_accum = torch.zeros((new_triangles.shape[0], 1), device="cuda")
        new_gradient_denom = torch.zeros((new_triangles.shape[0], 1), device="cuda")
        
        self.opacity_accum = torch.cat([self.opacity_accum, new_opacity_accum], dim=0)
        self.triangle_denom = torch.cat([self.triangle_denom, new_triangle_denom], dim=0)
        self.triangle_gradient_accum = torch.cat([self.triangle_gradient_accum, new_gradient_accum], dim=0)
        self.triangle_gradient_denom = torch.cat([self.triangle_gradient_denom, new_gradient_denom], dim=0)
        
        # 更新可见性掩码
        self._triangle_mask = torch.ones((self._triangle_indices.shape[0]), dtype=torch.bool, device="cuda")
        
        # 重置缓存
        self._triangle_centers = None
        
        # 清理临时存储
        if hasattr(self, 'new_anchor_points'):
            delattr(self, 'new_anchor_points')



    def _update_params_fast(self, selected_indices, iteration):
        # 首先确保 selected_indices 在 CPU 上处理，避免 GPU 异步错误
        if selected_indices.device.type == 'cuda':
            selected_indices_cpu = selected_indices.cpu()
        else:
            selected_indices_cpu = selected_indices
        
        # 获取当前三角形数量
        current_num_triangles = self._triangle_indices.shape[0]
        
        # 如果没有三角形或者没有选中的索引，返回空结果
        if current_num_triangles == 0 or selected_indices_cpu.numel() == 0:
            return (
                torch.empty((0, 3), device=self._triangle_indices.device),
                torch.empty((0, 1), device=self._triangle_indices.device),
                torch.empty((0, self._features_dc.shape[1], self._features_dc.shape[2]), 
                        device=self._triangle_indices.device),
                torch.empty((0, self._features_rest.shape[1], self._features_rest.shape[2]), 
                        device=self._triangle_indices.device),
                torch.empty((0, 3), dtype=torch.int32, device=self._triangle_indices.device)
            )
        
        # 在 CPU 上过滤索引，避免 GPU 断言错误
        valid_mask = selected_indices_cpu < current_num_triangles
        selected_indices_cpu = selected_indices_cpu[valid_mask]
        
        if selected_indices_cpu.numel() == 0:
            return (
                torch.empty((0, 3), device=self._triangle_indices.device),
                torch.empty((0, 1), device=self._triangle_indices.device),
                torch.empty((0, self._features_dc.shape[1], self._features_dc.shape[2]), 
                        device=self._triangle_indices.device),
                torch.empty((0, self._features_rest.shape[1], self._features_rest.shape[2]), 
                        device=self._triangle_indices.device),
                torch.empty((0, 3), dtype=torch.int32, device=self._triangle_indices.device)
            )
        
        # 将过滤后的索引移回 GPU
        selected_indices = selected_indices_cpu.to(device=self._triangle_indices.device)
        selected_indices = torch.unique(selected_indices)
        
        # 后续代码保持不变...
        selected_triangles_indices = self._triangle_indices[selected_indices]  # [S, 3]
        S = selected_triangles_indices.shape[0]
        
        # 如果 S 为 0，返回空结果
        if S == 0:
            return (
                torch.empty((0, 3), device=self._triangle_indices.device),
                torch.empty((0, 1), device=self._triangle_indices.device),
                torch.empty((0, self._features_dc.shape[1], self._features_dc.shape[2]), 
                        device=self._triangle_indices.device),
                torch.empty((0, self._features_rest.shape[1], self._features_rest.shape[2]), 
                        device=self._triangle_indices.device),
                torch.empty((0, 3), dtype=torch.int32, device=self._triangle_indices.device)
            )
        
        edges = torch.cat([
            selected_triangles_indices[:, [0, 1]],
            selected_triangles_indices[:, [0, 2]],
            selected_triangles_indices[:, [1, 2]]
        ], dim=0) 
        edges_sorted, _ = torch.sort(edges, dim=1)
        
        unique_edges_tensor, unique_indices = torch.unique(
            edges_sorted, return_inverse=True, dim=0
        )  
        M = unique_edges_tensor.shape[0]
        
        v0 = self.vertices[unique_edges_tensor[:, 0]]
        v1 = self.vertices[unique_edges_tensor[:, 1]]
        new_vertices = (v0 + v1) / 2.0
        
        new_vertex_base = self.vertices.shape[0]
        
        unique_edges_cpu = unique_edges_tensor.cpu()
        edge_to_midpoint = {}
        for i in range(M):
            edge_tuple = (unique_edges_cpu[i, 0].item(), unique_edges_cpu[i, 1].item())
            edge_to_midpoint[edge_tuple] = new_vertex_base + i

        new_triangles_list = []
        selected_triangles_cpu = selected_triangles_indices.cpu()
        
        for i in range(S):
            tri = selected_triangles_cpu[i]
            a, b, c = tri[0].item(), tri[1].item(), tri[2].item()
            
            ab = (min(a, b), max(a, b))
            ac = (min(a, c), max(a, c))
            bc = (min(b, c), max(b, c))
            
            m_ab = edge_to_midpoint[ab]
            m_ac = edge_to_midpoint[ac]
            m_bc = edge_to_midpoint[bc]

            new_triangles_list.append([a, m_ab, m_ac])
            new_triangles_list.append([b, m_ab, m_bc])
            new_triangles_list.append([c, m_ac, m_bc])
            new_triangles_list.append([m_ab, m_bc, m_ac])
        
        subdivided_triangles = torch.tensor(
            new_triangles_list, 
            dtype=torch.int32, 
            device=self._triangle_indices.device
        )

        u, v = unique_edges_tensor[:, 0], unique_edges_tensor[:, 1]

        # 1. 颜色特征继承（简单平均）
        new_features_dc = (self._features_dc[u] + self._features_dc[v]) / 2.0
        new_features_rest = (self._features_rest[u] + self._features_rest[v]) / 2.0

        # 2. 不透明度继承
        opacity_u = self.opacity_activation(self.vertex_weight[u])
        opacity_v = self.opacity_activation(self.vertex_weight[v])
        avg_opacity = (opacity_u + opacity_v) / 2.0
        avg_opacity = torch.clamp(avg_opacity, self.opacity_floor + self.eps, 1 - self.eps)
        new_vertex_weight = self.inverse_opacity_activation(avg_opacity)

        # 3. 应用渐进式属性继承
        parent_indices = torch.stack([u, v], dim=1)  # [M, 2]
        new_vertex_weight, new_features_dc, new_features_rest = self.inherit_attributes_from_parents(
            parent_indices, new_vertex_weight, new_features_dc, new_features_rest, iteration
        )

        new_triangles = subdivided_triangles
        
        return (
            new_vertices,
            new_vertex_weight,
            new_features_dc,
            new_features_rest,
            new_triangles
        )


    def _prune_vertex_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["vertices", "vertex_offsets", "anchor_points", "offset", "anchor_opacity", "anchor_scaling"]:
                # --- 新增：防御性检查，防止 mask 长度与参数不符 ---
                p = group["params"][0]
                if p.shape[0] != mask.shape[0]:
                    print(f"[Warning] Mask size {mask.shape[0]} doesn't match param {group['name']} size {p.shape[0]}. Skipping.")
                    continue
                # ----------------------------------------------

                stored_state = self.optimizer.state.get(p, None)
                if stored_state is not None:
                    # 同样需要检查 exp_avg 是否存在
                    if "exp_avg" in stored_state and stored_state["exp_avg"] is not None:
                        stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                        stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]     

                    del self.optimizer.state[group['params'][0]]
                    # Update parameter
                    group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group['params'][0]
                else:
                    group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group['params'][0]
        
        # Update model parameters
        for name, tensor in optimizable_tensors.items():
            if name == "vertices":
                self.vertices = tensor
            elif name == "vertex_weight":
                self.vertex_weight = tensor
            elif name == "f_dc":
                self._features_dc = tensor
            elif name == "f_rest":
                self._features_rest = tensor
            elif name == "anchor_points":
                self.anchor_points = tensor
        


    def _prune_vertices(self, vertex_mask: torch.Tensor):
        device = vertex_mask.device
        oldV = vertex_mask.numel()

        # Create mapping from old vertex IDs to new IDs (-1 for removed vertices)
        new_id = torch.full((oldV,), -1, dtype=torch.long, device=device)
        kept = torch.nonzero(vertex_mask, as_tuple=True)[0]
        new_id[kept] = torch.arange(kept.numel(), device=device, dtype=torch.long)

        # Remap triangle indices and drop triangles with removed vertices
        if self._triangle_indices.numel() > 0:
            remapped = new_id[self._triangle_indices.long()]
            valid_tris = (remapped >= 0).all(dim=1)
            remapped = remapped[valid_tris]
            self._triangle_indices = remapped.to(torch.int32).contiguous()

            if isinstance(self.image_size, torch.Tensor) and self.image_size.numel() > 0:
                self.image_size = self.image_size[valid_tris]
            if isinstance(self.importance_score, torch.Tensor) and self.importance_score.numel() > 0:
                self.importance_score = self.importance_score[valid_tris]
                
            # 同时更新训练统计和可见性掩码
            if hasattr(self, 'opacity_accum') and self.opacity_accum.shape[0] == valid_tris.shape[0]:
                self.opacity_accum = self.opacity_accum[valid_tris]
            if hasattr(self, 'triangle_denom') and self.triangle_denom.shape[0] == valid_tris.shape[0]:
                self.triangle_denom = self.triangle_denom[valid_tris]
            if hasattr(self, 'triangle_gradient_accum') and self.triangle_gradient_accum.shape[0] == valid_tris.shape[0]:
                self.triangle_gradient_accum = self.triangle_gradient_accum[valid_tris]
            if hasattr(self, 'triangle_gradient_denom') and self.triangle_gradient_denom.shape[0] == valid_tris.shape[0]:
                self.triangle_gradient_denom = self.triangle_gradient_denom[valid_tris]
            if hasattr(self, '_triangle_mask') and self._triangle_mask.shape[0] == valid_tris.shape[0]:
                self._triangle_mask = self._triangle_mask[valid_tris]

        # Prune vertex-related parameters using the initial mask
        self._prune_vertex_optimizer(vertex_mask)
        
        # 同时剪裁层次信息
        if hasattr(self, '_level') and self._level.numel() == oldV:
            self._level = self._level[vertex_mask]

        # After initial pruning, check for unreferenced vertices
        current_vertex_count = self.vertices.shape[0]
        if current_vertex_count > 0:
            # Identify vertices still referenced by triangles
            if self._triangle_indices.numel() > 0:
                referenced_vertices = torch.unique(self._triangle_indices)
                mask_referenced = torch.zeros(current_vertex_count, dtype=torch.bool, device=device)
                mask_referenced[referenced_vertices] = True
            else:
                mask_referenced = torch.zeros(current_vertex_count, dtype=torch.bool, device=device)

            # Remove unreferenced vertices
            if not mask_referenced.all():
                # Prune vertex parameters
                self._prune_vertex_optimizer(mask_referenced)
                
                # 剪裁层次信息
                if hasattr(self, '_level') and self._level.numel() == current_vertex_count:
                    self._level = self._level[mask_referenced]

                # Remap triangle indices if triangles exist
                if self._triangle_indices.numel() > 0:
                    new_id2 = torch.full((current_vertex_count,), -1, dtype=torch.long, device=device)
                    kept2 = torch.nonzero(mask_referenced, as_tuple=True)[0]
                    new_id2[kept2] = torch.arange(kept2.numel(), device=device, dtype=torch.long)
                    self._triangle_indices = new_id2[self._triangle_indices.long()].to(torch.int32).contiguous()



    def prune_triangles(self, mask):
        if mask.shape[0] != self._triangle_indices.shape[0]:
            print(f"[Error] Prune mask shape mismatch: {mask.shape[0]} != {self._triangle_indices.shape[0]}")
            return
        
        try:
            # 应用掩码
            self._triangle_indices = self._triangle_indices[mask]
            self._triangle_indices = self._triangle_indices.to(torch.int32)

            if self.image_size.numel() > 0 and self.image_size.shape[0] == mask.shape[0]:
                self.image_size = self.image_size[mask]
            
            if self.importance_score.numel() > 0 and self.importance_score.shape[0] == mask.shape[0]:
                self.importance_score = self.importance_score[mask]
            
            if hasattr(self, 'anchor_mu') and self.anchor_mu.shape[0] == mask.shape[0]:
                self.anchor_mu = self.anchor_mu[mask].detach()
            
            if hasattr(self, 'anchor_sigma_inv') and self.anchor_sigma_inv.shape[0] == mask.shape[0]:
                self.anchor_sigma_inv = self.anchor_sigma_inv[mask].detach()
                
            if hasattr(self, 'anchor_normal') and self.anchor_normal.shape[0] == mask.shape[0]:
                self.anchor_normal = self.anchor_normal[mask].detach()
                
            if hasattr(self, 'octree_res') and self.octree_res.shape[0] == mask.shape[0]:
                self.octree_res = self.octree_res[mask].detach()

            # 3. [CRITICAL FIX] 邻居索引失效处理
            # 剪枝后，索引发生变化，旧的邻居索引表 (neighbor_indices) 彻底失效
            # 必须清空，并在下一次训练循环中重新计算 (或由 C++ 动态查找)
            if hasattr(self, 'neighbor_indices'):
                self.neighbor_indices = torch.empty(0, dtype=torch.int32, device="cuda")
            
            # 4. 更新 is_active 张量
            if hasattr(self, '_is_active_tensor') and self._is_active_tensor.shape[0] == mask.shape[0]:
                self._is_active_tensor = self._is_active_tensor[mask]
            else:
                # 如果形状不匹配，重新创建掩码
                self._triangle_mask = torch.ones(self._triangle_indices.shape[0], dtype=torch.bool, device="cuda")
            
            # 重置三角形中心缓存
            self._triangle_centers = None
            
            # 更新锚点索引
            if hasattr(self, 'anchor_feature_index') and self.anchor_feature_index is not None:
                if self.anchor_feature_index.shape[0] == mask.shape[0]:
                    try:
                        self.anchor_feature_index = self.anchor_feature_index[mask].detach()
                    except Exception as e:
                        print(f"[Warning] Failed to prune anchor_feature_index: {e}")
                else:
                    # 极端情况下的保命对齐
                    try:
                        self.anchor_feature_index = self._triangle_indices.clone().detach().to(torch.long)
                    except Exception as e:
                        print(f"[Warning] Failed to reinitialize anchor_feature_index: {e}")
        except Exception as e:
            print(f"[Error] Failed during triangle pruning: {e}")
            # 尝试恢复最小状态
            if self._triangle_indices.numel() > 0:
                self._triangle_centers = None
                self._triangle_mask = torch.ones(self._triangle_indices.shape[0], dtype=torch.bool, device="cuda")        

    def _sample_alives(self, probs, num, alive_indices=None):
        # 确保输入有效
        if probs.numel() == 0 or num <= 0:
            return torch.tensor([], dtype=torch.long, device=probs.device)
        
        # 确保概率和为正值
        prob_sum = probs.sum()
        if prob_sum <= 0:
            return torch.tensor([], dtype=torch.long, device=probs.device)
        
        # 确保采样数量不超过非零元素数量
        num_nonzero = (probs > 0).sum().item()
        num = min(num, num_nonzero)
        if num == 0:
            return torch.tensor([], dtype=torch.long, device=probs.device)
        
        torch.manual_seed(1)  # always same "random" indices
        probs = probs / prob_sum
        
        try:
            sampled_idxs = torch.multinomial(probs, num, replacement=False)
        except RuntimeError:
            # 如果采样失败，返回空
            return torch.tensor([], dtype=torch.long, device=probs.device)
        
        if alive_indices is not None and alive_indices.numel() > 0:
            # 确保 sampled_idxs 在 alive_indices 范围内
            if sampled_idxs.numel() > 0 and sampled_idxs.max() < alive_indices.shape[0]:
                sampled_idxs = alive_indices[sampled_idxs]
            else:
                sampled_idxs = torch.tensor([], dtype=torch.long, device=probs.device)
        
        return sampled_idxs 

    def add_new_gs(self, iteration, cap_max, splitt_large_triangles, probs_opacity=True):
        current_num_triangles = self._triangle_indices.shape[0]
        
        if current_num_triangles == 0:
            return 0
        
        target_num = min(cap_max, int(self.add_percentage * current_num_triangles))
        num_new_requested = max(0, target_num - current_num_triangles)

        if num_new_requested <= 0:
            return 0

        # --- 考虑三角形层次进行采样 ---
        if self.progressive and len(self.coarse_intervals) > 0:
            coarse_index = np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level
            coarse_index = min(coarse_index, self.levels - 1)
            
            triangle_levels = self.get_triangle_levels
            
            allowed_mask = (triangle_levels <= coarse_index)
            allowed_probs = self.importance_score * allowed_mask.float()
        else:
            allowed_probs = self.importance_score
        
        # 采样逻辑
        probs = allowed_probs.squeeze()
        areas = self.triangle_areas().squeeze()
        
        if probs.sum() == 0 or areas.numel() == 0:
            return 0
        
        # 采样随机索引
        num_to_sample = min(num_new_requested, (probs > 0).sum().item())
        if num_to_sample > 0:
            rand_idx = self._sample_alives(probs=probs, num=num_to_sample)
        else:
            rand_idx = torch.tensor([], dtype=torch.long, device=self._triangle_indices.device)
        
        # 选出面积最大的进行分裂
        k = min(splitt_large_triangles, areas.numel())  
        if k > 0:
            _, top_idx = torch.topk(areas, k, largest=True, sorted=False)
        else:
            top_idx = torch.tensor([], dtype=torch.long, device=self._triangle_indices.device)
        
        # 合并索引
        all_indices = torch.cat([rand_idx, top_idx])
        if all_indices.numel() == 0:
            return 0
        
        # 去重并确保索引有效
        add_idx = torch.unique(all_indices)
        add_idx = add_idx[add_idx < current_num_triangles]
        
        if add_idx.numel() == 0:
            return 0
        
        num_to_split = add_idx.shape[0]
        
        # 1. 提取锚点遗传信息
        if hasattr(self, 'anchor_points') and self.anchor_points is not None and self.anchor_points.numel() > 0:
            selected_triangle_indices = self._triangle_indices[add_idx]  # [S, 3]
            
            # 为每个新顶点（边中点）创建锚点
            num_new_vertices_per_triangle = 3
            repeated_anchors = torch.empty((selected_triangle_indices.shape[0] * num_new_vertices_per_triangle, 3), 
                                        device=self._triangle_indices.device)
            
            # 为每条边计算中点锚点
            for edge_idx in range(3):
                v0_idx = edge_idx
                v1_idx = (edge_idx + 1) % 3
                
                anchor0 = self.anchor_points[selected_triangle_indices[:, v0_idx]]
                anchor1 = self.anchor_points[selected_triangle_indices[:, v1_idx]]
                
                edge_mid_anchor = (anchor0 + anchor1) / 2.0
                
                start_idx = edge_idx * selected_triangle_indices.shape[0]
                end_idx = (edge_idx + 1) * selected_triangle_indices.shape[0]
                repeated_anchors[start_idx:end_idx] = edge_mid_anchor
        else:
            repeated_anchors = torch.empty((0, 3), device=self._triangle_indices.device)

        # 2. 执行顶点分裂
        result = self._update_params_fast(add_idx, iteration)
        if result is None:
            return 0
        
        (new_vertices, new_vertex_weight, new_features_dc, new_features_rest, new_triangles) = result
        
        if new_vertices.shape[0] == 0:
            return 0
        
        # 3. 存储新的锚点
        if hasattr(self, 'anchor_points') and repeated_anchors.shape[0] > 0:
            if self.anchor_points.numel() > 0 and self.anchor_points is not None:
                self.new_anchor_points = torch.cat([self.anchor_points.detach(), repeated_anchors], dim=0)
            else:
                self.new_anchor_points = repeated_anchors
        else:
            self.new_anchor_points = torch.empty((0, 3), device=self._triangle_indices.device)

        # 4. 合并参数
        self.densification_postfix(new_vertices, new_vertex_weight, new_features_dc, new_features_rest, new_triangles)
        
        # 5. 更新 _level 属性
        # 获取最新的顶点总数
        total_vertex_count = self.vertices.shape[0]
        current_level_count = self._level.shape[0] if hasattr(self, '_level') else 0
        
        if total_vertex_count > current_level_count:
            # 计算需要补齐的差额
            added_v_count = total_vertex_count - current_level_count
            
            # 为新生成的顶点分配层级
            if hasattr(self, '_level'):
                # 为新顶点分配层级（通常设为当前最高精度层级）
                new_v_levels = torch.ones(added_v_count, dtype=torch.int32, device=self._level.device) * (self.levels - 1)
                
                # 拼接，保证 _level 的长度等于 vertices 的长度
                self._level = torch.cat([self._level, new_v_levels], dim=0)
            else:
                # 如果 _level 不存在，创建新的
                self._level = torch.ones(total_vertex_count, dtype=torch.int32, device=self.vertices.device) * (self.levels - 1)
            
            # 重置相关的 mask 缓存
            self._triangle_mask = torch.ones(self._triangle_indices.shape[0], dtype=torch.bool, device="cuda")
            self._triangle_centers = None
        
        # 6. 更新索引表
        if hasattr(self, 'anchor_feature_index'):
            if self.anchor_feature_index.numel() > 0:
                # 为新三角形创建锚点索引
                num_new_triangles = new_triangles.shape[0]
                new_anchor_indices = torch.arange(
                    self.vertices.shape[0] - new_vertices.shape[0],  # 新顶点的起始索引
                    self.vertices.shape[0],
                    device=self._triangle_indices.device
                ).reshape(-1, 3)[:num_new_triangles]  # 确保形状匹配
                
                self.anchor_feature_index = torch.cat([self.anchor_feature_index, new_anchor_indices], dim=0)
            else:
                self.anchor_feature_index = new_triangles.clone().detach().to(torch.long)

        # 7. 剪枝：删除那批已经被裂变的"老父亲"三角形
        mask = torch.ones(self._triangle_indices.shape[0], dtype=torch.bool, device="cuda")
        
        # 确保 add_idx 中的索引在当前三角形范围内
        valid_indices = add_idx[add_idx < mask.shape[0]]
        
        if valid_indices.numel() > 0:
            mask[valid_indices] = False
        else:
            print(f"[Warning] No valid indices to prune: add_idx max={add_idx.max() if add_idx.numel()>0 else 'empty'}, mask shape={mask.shape}")
        
        self.prune_triangles(mask)

        return num_to_split * 4


    def update_min_weight(self, new_min_weight: float, preserve_outputs: bool = True):
        new_m = float(max(0.0, min(new_min_weight, 1.0 - 1e-4)))

        # 1) grab the current realized opacities y (under the old floor)
        with torch.no_grad():
            y = self.get_vertex_weight.detach()
            y = y.clamp(new_m + self.eps, 1.0 - self.eps)   # clamp to the *new* floor
        self.opacity_floor = new_m
        new_logits = self.inverse_opacity_activation(y)
        with torch.no_grad():
            self.vertex_weight.data.copy_(new_logits)


    def triangle_areas(self):
        tri = self.vertices[self._triangle_indices]                    # [T, 3, 3]
        AB  = tri[:, 1] - tri[:, 0]                                    # [T, 3]
        AC  = tri[:, 2] - tri[:, 0]                                    # [T, 3]
        cross_prod = torch.cross(AB, AC, dim=1)                        # [T, 3]
        areas = 0.5 * torch.linalg.norm(cross_prod, dim=1)             # [T]
        areas = torch.nan_to_num(areas, nan=0.0, posinf=0.0, neginf=0.0)
        return areas

    def update_training_stats(self, opacity, gradient_norm=None, triangle_mask=None):
        """
        更新训练统计信息
        """
        if triangle_mask is None:
            triangle_mask = self._triangle_mask
        
        if triangle_mask.sum() == 0:
            return
        
        # 更新不透明度累积
        visible_opacity = opacity[triangle_mask].view(-1, 1)
        self.opacity_accum[triangle_mask] += visible_opacity
        
        # 更新访问统计
        self.triangle_denom[triangle_mask] += 1
        
        # 更新梯度统计（如果提供了梯度信息）
        if gradient_norm is not None:
            visible_gradients = gradient_norm[triangle_mask].view(-1, 1)
            self.triangle_gradient_accum[triangle_mask] += visible_gradients
            self.triangle_gradient_denom[triangle_mask] += 1

    def adjust_triangles(self, iteration, check_interval=100, success_threshold=0.8, 
                        min_opacity=0.005, grad_threshold=0.0002):
        """
        动态调整三角形：基于训练统计进行生长和剪枝（类似Octree-GS的adjust_anchor）
        """
        num_pruned = 0
        
        # 1. 剪枝：移除不重要的三角形
        if self.triangle_denom.sum() > 0:
            # 计算平均不透明度
            avg_opacity = self.opacity_accum / (self.triangle_denom + 1e-6)
            
            # 筛选条件：访问次数足够但平均不透明度太低
            visited_enough = (self.triangle_denom > check_interval * success_threshold).squeeze(1)
            low_opacity = (avg_opacity < min_opacity).squeeze(1)
            
            prune_mask = visited_enough & low_opacity
            
            # 可选：基于梯度信息进一步筛选
            if self.triangle_gradient_denom.sum() > 0:
                avg_gradient = self.triangle_gradient_accum / (self.triangle_gradient_denom + 1e-6)
                low_gradient = (avg_gradient < grad_threshold).squeeze(1)
                prune_mask = prune_mask & low_gradient
            
            num_pruned = prune_mask.sum().item()
            
            if num_pruned > 0:
                keep_mask = ~prune_mask
                self.prune_triangles(keep_mask)
                
                # 重置被剪枝三角形的统计信息
                self.opacity_accum = self.opacity_accum[keep_mask]
                self.triangle_denom = self.triangle_denom[keep_mask]
                self.triangle_gradient_accum = self.triangle_gradient_accum[keep_mask]
                self.triangle_gradient_denom = self.triangle_gradient_denom[keep_mask]
        
        # 2. 重置访问次数过多的三角形的统计（防止统计值过大）
        if self.triangle_denom.sum() > 0:
            reset_mask = (self.triangle_denom > check_interval * 2).squeeze(1)
            if reset_mask.sum() > 0:
                self.opacity_accum[reset_mask] = 0
                self.triangle_denom[reset_mask] = 0
                self.triangle_gradient_accum[reset_mask] = 0
                self.triangle_gradient_denom[reset_mask] = 0
        
        return num_pruned

    def get_triangles_by_level(self, max_level=None):
        """根据层次获取三角形掩码"""
        if not hasattr(self, '_level') or self._level.numel() == 0:
            return torch.ones((self._triangle_indices.shape[0]), dtype=torch.bool, device="cuda")
        
        # 计算每个三角形的平均层次
        triangle_levels = self.get_triangle_levels()
        
        if max_level is None:
            max_level = self.levels - 1
        
        return triangle_levels <= max_level

    def get_triangle_centers(self):
        """获取或计算三角形中心（带缓存）"""
        if self._triangle_centers is None or self._triangle_centers.shape[0] != self._triangle_indices.shape[0]:
            tri_pts = self.vertices[self._triangle_indices]
            self._triangle_centers = tri_pts.mean(dim=1)
        return self._triangle_centers

    # --- Challenge 3: SG-DC Mesh Extraction (NEW METHOD!) ---
    @torch.no_grad()
    def extract_mesh(self, resolution=128, threshold=0.0):
        """
        Extracts a mesh using Splat-Guided Dual Contouring (via SDF & Marching Cubes).
        Requires 'scikit-image' and the 'triangulation.compute_sdf' C++ binding.
        """

        # 1. 准备查询网格
        grid_pts = self.generate_grid(resolution) # [M, 3]
        
        # 2. [NEW] 预计算三角形包围盒加速 CUDA
        v = self.get_vertices # [V, 3] (Property access, no parens)
        tri_v = v[self._triangle_indices.long()] # [N, 3, 3]
        tri_min = tri_v.min(dim=1)[0].contiguous() # [N, 3]
        tri_max = tri_v.max(dim=1)[0].contiguous() # [N, 3]
        
        v1, v2, v3 = tri_v[:,0], tri_v[:,1], tri_v[:,2]
        
        # 3. 调用优化后的接口
        # 确保传入了 tri_min 和 tri_max
        sdf_values = triangulation.compute_sdf(
            grid_pts.contiguous(),
            self.vertices.contiguous(),
            self._triangle_indices.int().contiguous()
        )
        sdf_values = sdf_values[:, 0]
        # 3. Marching Cubes
        try:
            # Applies Marching Cubes on the SDF field at the given threshold
            verts_mc, faces_mc, _, _ = marching_cubes(sdf_values, level=threshold)
        except RuntimeError:
            print("Marching Cubes failed (no surface found).")
            return None

        # 4. Map back to world coordinates
        verts_mc = torch.tensor(verts_mc, dtype=torch.float32)
        
        # Calculate the scaling and offset
        scale = (max_bound.cpu() - min_bound.cpu()) / (resolution - 1)
        
        # Map [0, resolution-1] grid coordinates back to [min_bound, max_bound] world coordinates
        verts_mc = verts_mc * scale + min_bound.cpu()
        
        print(f"Mesh extracted: {verts_mc.shape[0]} vertices, {faces_mc.shape[0]} faces.")
        return verts_mc.numpy(), faces_mc

    def initialize_anchors_and_topology(self, anchor_data):
        """
        初始化 Octree-GS 锚定参数和拓扑邻域信息 (C1 & C2)。
        """
        device = self.get_device()
        
        # --- Challenge 1: 几何锚定参数 (detach 确保不参与高频优化) ---
        if 'anchor_mu' in anchor_data:
            self.anchor_mu = anchor_data['anchor_mu'].to(device).detach()
        
        if 'anchor_sigma_inv' in anchor_data:
            self.anchor_sigma_inv = anchor_data['anchor_sigma_inv'].to(device).detach()
        
        if 'anchor_normal' in anchor_data:
            self.anchor_normal = anchor_data['anchor_normal'].to(device).detach()
        
        # --- Challenge 2: 拓扑信息 ---
        if 'neighbor_indices' in anchor_data:
            # 必须是 torch.int32/uint32 类型
            neighbor_indices = anchor_data['neighbor_indices']
            if neighbor_indices.dtype != torch.int32:
                neighbor_indices = neighbor_indices.to(torch.int32)
            self.neighbor_indices = neighbor_indices.to(device).detach()
        
        # is_active 状态标志 (用于 Loss Kernel 中的条件判断)
        N = self._triangle_indices.shape[0] if hasattr(self, '_triangle_indices') else 0
        self._is_active_tensor = torch.ones(
            (N, 1), 
            device=device, 
            dtype=torch.float32).detach()
        
        print(f"Initialized anchors: anchor_mu={hasattr(self, 'anchor_mu')}, "
            f"neighbor_indices={hasattr(self, 'neighbor_indices')}")

    def update_anchor_optimizer(self, optimizer, iteration):
        """在 densification 后，将更新后的 anchor_points 重新加入优化器"""
        for group in optimizer.param_groups:
            if group["name"] == "anchor_points":
                # 1. 提取旧的状态（如果有必要的话，比如 Adam 的 momentum）
                # 这里简单处理：直接替换参数，并重置该组的优化状态
                stored_state = optimizer.state.get(group['params'][0], None)
                
                # 2. 从优化器中移除旧的参数
                del optimizer.state[group['params'][0]]
                group['params'][0] = self.anchor_points
                
                # 3. 重新初始化该参数的状态
                # 注意：由于维度变了，旧的 momentum 无法直接对齐，通常这里选择重置
                if stored_state is not None:
                    optimizer.state[group['params'][0]] = {} 
                    
                print(f"[Octree-GS] Optimizer updated for anchor_points. New shape: {self.anchor_points.shape[0]}")
        torch.cuda.empty_cache()

    @property
    def is_active(self):
        """返回当前的激活状态张量，用于 CUDA Loss Kernel"""
        if not hasattr(self, '_is_active_tensor'):
            # 兼容性处理
            N = self._triangle_indices.shape[0] if hasattr(self, '_triangle_indices') else 0
            return torch.ones(
                (N, 1), 
                device=self.get_device(), 
                dtype=torch.float32)
        return self._is_active_tensor
    
    # 确保 loss_utils.py 使用的接口存在
    def get_v1(self):
        return self.vertices[self._triangle_indices[:, 0], :]

    def get_v2(self):
        return self.vertices[self._triangle_indices[:, 1], :]

    def get_v3(self):
        return self.vertices[self._triangle_indices[:, 2], :]
    
    def get_device(self):
        """获取模型所在的设备"""
        if self.vertices.numel() > 0:
            return self.vertices.device
        elif self._triangle_indices.numel() > 0:
            return self._triangle_indices.device
        else:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def plot_level_distribution(self):
        """打印层次分布统计"""
        if not hasattr(self, '_level') or self._level.numel() == 0:
            print("No level information available.")
            return
        
        unique_levels, counts = torch.unique(self._level, return_counts=True)
        print("Vertex level distribution:")
        for level, count in zip(unique_levels.cpu().numpy(), counts.cpu().numpy()):
            print(f"  Level {level}: {count} vertices ({count/self._level.shape[0]*100:.1f}%)")
        
        # 三角形层次分布
        if self._triangle_indices.numel() > 0:
            triangle_levels = self.get_triangle_levels()
            unique_tri_levels, tri_counts = torch.unique(triangle_levels, return_counts=True)
            print("\nTriangle level distribution:")
            for level, count in zip(unique_tri_levels.cpu().numpy(), tri_counts.cpu().numpy()):
                print(f"  Level {level}: {count} triangles ({count/triangle_levels.shape[0]*100:.1f}%)")


    def inherit_attributes_from_parents(self, parent_indices, new_vertex_weight, new_features_dc, new_features_rest, iteration=0):
        """
        根据父节点层次渐进式继承属性
        parent_indices: 父顶点的索引 [M, 2]
        iteration: 当前迭代次数，用于控制渐进性
        """
        # 获取父顶点的层次
        if hasattr(self, '_level') and self._level.numel() > 0:
            parent_levels = self._level[parent_indices.flatten()].reshape(-1, 2)  # [M, 2]
            
            # 计算平均层次
            avg_levels = parent_levels.float().mean(dim=1)  # [M]
            
            # 层次比例因子：层次越高（越精细），初始不透明度越低
            # 避免除零
            if self.levels > 0:
                # 归一化层次因子
                normalized_level = avg_levels / max(self.levels - 1, 1)
                # 高层级（细节）初始透明度更高（不透明度更低）
                level_factor = 0.3 + 0.7 * (1.0 - normalized_level)  # [M]
            else:
                level_factor = torch.ones_like(avg_levels)
        else:
            level_factor = torch.ones(parent_indices.shape[0], device=parent_indices.device)
        
        # 训练进度因子：随着训练进行，逐渐增加新顶点的权重
        # 假设在5000次迭代内逐渐增加
        train_progress = min(1.0, iteration / 5000.0) if iteration > 0 else 0.0
        progress_factor = 0.2 + 0.8 * train_progress  # 从0.2到1.0
        
        # 最终因子
        final_factor = level_factor * progress_factor
        
        # 调整新顶点的不透明度
        # new_vertex_weight 已经是 inverse_sigmoid 的结果
        # 我们需要将其调整到更低的初始值
        new_vertex_weight = new_vertex_weight * final_factor.unsqueeze(1)
        
        # 调整颜色特征：添加基于层次的小扰动，增加细节多样性
        if new_features_dc.numel() > 0:
            # 扰动因子：层次越高，扰动越小（细节越精确）
            noise_scale = 0.1 * (1.0 - final_factor).unsqueeze(1).unsqueeze(2)  # [M, 1, 1]
            noise = torch.randn_like(new_features_dc) * noise_scale
            new_features_dc = new_features_dc + noise
        
        if new_features_rest.numel() > 0:
            noise_scale = 0.05 * (1.0 - final_factor).unsqueeze(1).unsqueeze(2)
            noise = torch.randn_like(new_features_rest) * noise_scale
            new_features_rest = new_features_rest + noise
        
        return new_vertex_weight, new_features_dc, new_features_rest

    def gradually_increase_opacity(self, iteration, total_iterations=5000):
        """
        在训练过程中逐步增加新生成顶点的不透明度
        iteration: 当前迭代次数
        total_iterations: 完全增加所需的总迭代次数
        """
        if iteration < 1000:  # 前1000次迭代不调整
            return
        
        # 找到最近生成的顶点（通过顶点数量变化来估计）
        if not hasattr(self, '_last_vertex_count'):
            self._last_vertex_count = self.vertices.shape[0]
            return
        
        current_vertex_count = self.vertices.shape[0]
        if current_vertex_count <= self._last_vertex_count:
            self._last_vertex_count = current_vertex_count
            return
        
        # 新顶点的数量
        new_vertex_count = current_vertex_count - self._last_vertex_count
        if new_vertex_count <= 0:
            return
        
        # 计算渐进因子
        progress = min(1.0, (iteration - 1000) / total_iterations)
        
        # 新顶点的起始索引
        new_vertex_start = self._last_vertex_count
        
        # 获取新顶点的当前不透明度
        new_opacities = self.opacity_activation(self.vertex_weight[new_vertex_start:])
        
        # 目标不透明度：从初始值渐进到完整值
        # 初始不透明度较低（例如0.3），逐步增加到1.0
        target_min = 0.3 + 0.7 * progress
        
        # 确保不透明度不低于目标最小值
        needs_update = new_opacities < target_min
        if needs_update.any():
            # 设置目标不透明度
            target_opacities = torch.where(
                new_opacities < target_min,
                torch.full_like(new_opacities, target_min),
                new_opacities
            )
            
            # 转换回 logits
            target_logits = self.inverse_opacity_activation(target_opacities)
            
            # 更新权重
            with torch.no_grad():
                self.vertex_weight[new_vertex_start:] = torch.where(
                    needs_update,
                    target_logits,
                    self.vertex_weight[new_vertex_start:]
                )
        
        # 更新记录
        self._last_vertex_count = current_vertex_count

    def update_topology(self, K=5):
        """
        重新计算三角形之间的邻接关系。
        在 Pruning 或 Densification 导致三角形索引发生变化后必须调用。
        """
        if self._triangle_indices.shape[0] == 0:
            return

        # 1. 获取三角形中心点作为 KNN 的输入
        # centers 形状为 [N, 3]
        centers = self.get_triangle_centers().detach()
        
        # 2. 使用 pytorch3d 进行高效 KNN 搜索
        # 如果没有安装 pytorch3d，请确保环境中有，它是处理这种拓扑关系最快的方案
        from pytorch3d.ops import knn_points
        
        # p_cloud 形状需为 [Batch=1, N, 3]
        p_cloud = centers.unsqueeze(0)
        
        # 搜索 K+1 个邻居（因为第 0 个通常是三角形自己）
        # 返回 idx 形状为 [1, N, K+1]
        try:
            knn = knn_points(p_cloud, p_cloud, K=K+1)
            # 移除自身索引，取剩余 K 个邻居
            # [N, K] -> 转换为 int32 供 CUDA 使用
            neighbor_idx = knn.idx.squeeze(0)[:, 1:].to(torch.int32).contiguous()
            
            # 展平存储，匹配 loss_utils.py 中的读取逻辑
            self.neighbor_indices = neighbor_idx.reshape(-1)
            
            # 更新活跃状态标记（同步 Buffer）
            self._is_active_tensor = torch.ones((self._triangle_indices.shape[0], 1), 
                                              device="cuda", dtype=torch.float32)
        except Exception as e:
            print(f"Topology update failed: {e}")

    def update_aabb_tree(self, rebuild_interval=10):
        """
        更新三角形中心点的AABB树
        """
        if not hasattr(self, '_aabb_tree_iteration'):
            self._aabb_tree_iteration = -rebuild_interval
        
        # 检查是否需要重建
        current_iteration = getattr(self, '_current_iteration', 0)
        if current_iteration - self._aabb_tree_iteration < rebuild_interval:
            return
        
        print(f"Building AABB tree for {self._triangle_indices.shape[0]} triangles...")
        
        try:
            # 计算三角形中心点
            # 修正：get_vertices 是 property，不能加括号
            v = self.get_vertices 
            tri_verts = v[self._triangle_indices.long()]
            triangle_centers = tri_verts.mean(dim=1).contiguous()
            
            # 调用C++函数构建AABB树
            self._aabb_tree = triangulation.build_triangle_aabb_tree(
                triangle_centers,
                max_triangles_per_node=16
            )
            
            self._aabb_tree_iteration = current_iteration
            print(f"AABB tree built with {self._aabb_tree.shape[0] // 6} nodes")
            
        except Exception as e:
            print(f"Warning: Failed to build AABB tree: {e}")
            self._aabb_tree = None

    def set_current_iteration(self, iteration):
        """设置当前迭代次数，用于控制AABB树重建频率"""
        self._current_iteration = iteration
        # 每10次迭代重建一次AABB树
        if iteration % 10 == 0:
            self.update_aabb_tree()



