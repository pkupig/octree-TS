##########################################################################################################
#
#
#
#
#
#
#
##########################################################################################################


import torch
import torch.nn.functional as F
from torch.autograd import Variable
import triangulation as tb
from math import exp


def triangle_probability_loss(vertices, _triangle_indices, _closest_vertices, theta=10.0):
    """
    Computes the probability loss for triangles to maximize their separation from nearby points.
    
    For each triangle:
    1. Computes circumcenter and circumradius
    2. Finds closest point among K neighbors
    3. Calculates signed distance (closest_distance - circumradius)
    4. Computes probability = sigmoid(signed_distance * theta)
    
    Returns the negative mean probability (loss to minimize)
    """
    # Get triangle vertices [T, 3, 3]
    tri_pts = vertices[_triangle_indices]
    
    # Compute circumcenters and circumradii
    A, B, C = tri_pts.unbind(dim=1)
    AB = B - A
    AC = C - A
    n = torch.cross(AB, AC, dim=1)
    
    # Vector magnitudes
    AB2 = (AB * AB).sum(dim=1, keepdim=True)
    AC2 = (AC * AC).sum(dim=1, keepdim=True)
    
    # Denominator (2 * |n|^2)
    den = 2.0 * (n * n).sum(dim=1, keepdim=True)
    
    # Identify degenerate triangles (den near zero)
    eps = 1e-12
    degenerate = (den.abs() < eps).squeeze(1)
    
    # Compute circumcenters (use centroid for degenerate triangles)
    term1 = AB2 * torch.cross(AC, n, dim=1)
    term2 = AC2 * torch.cross(n, AB, dim=1)
    circumcenters = A + (term1 + term2) / den
    
    # For degenerate triangles, use centroid instead
    centroid = tri_pts.mean(dim=1)
    circumcenters = torch.where(degenerate.unsqueeze(1), centroid, circumcenters)
    
    # Compute circumradii (distance from circumcenter to vertices)
    dists_to_vertices = torch.norm(tri_pts - circumcenters.unsqueeze(1), dim=2)
    circumradii = dists_to_vertices.max(dim=1).values
    
    # Get closest points for each triangle [T, K, 3]
    closest_points = vertices[_closest_vertices]
    
    # Compute distances to closest points
    dists_to_neighbors = torch.norm(
        closest_points - circumcenters.unsqueeze(1),
        dim=2
    )
    
    # Find minimum distance for each triangle
    min_dists, _ = torch.min(dists_to_neighbors, dim=1)
    
    # Signed distance = (closest distance) - circumradius
    signed_dist = min_dists - circumradii
    
    # For degenerate triangles, set signed_dist to large negative value
    #signed_dist = torch.where(degenerate, -1e6 * torch.ones_like(signed_dist), signed_dist)
    
    # Compute probability using sigmoid
    probability = torch.sigmoid(-theta * signed_dist)
    
    # Loss is negative mean probability (to maximize probability)
    return torch.mean(probability)


def u_shaped_opacity_loss(x, center=0.1, width=0.03):
    # Normalized distance to center (e.g., 0.1)
    penalty = torch.exp(-((x - center) ** 2) / (2 * width ** 2))
    return penalty.mean()

def binarization_loss(x, eps=1e-6):
    x = torch.clamp(x, eps, 1 - eps)  # avoid log(0)
    return -x * torch.log(x) - (1 - x) * torch.log(1 - x)

def equilateral_regularizer(triangles):

    nan_mask = torch.isnan(triangles).any(dim=(1, 2))
    if nan_mask.any():
        print("NaN detected in triangle(s):")

    v0 = triangles[:, 1, :] - triangles[:, 0, :]
    v1 = triangles[:, 2, :] - triangles[:, 0, :]
    cross = torch.cross(v0, v1, dim=1)
    area = 0.5 * torch.norm(cross, dim=1)

    return area


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def lp_loss(pred, target, p=0.7, eps=1e-6):
    """
    Computes Lp loss with 0 < p < 1.
    Args:
        pred: (N, C, H, W) predicted image
        target: (N, C, H, W) groundtruth image
        p: norm degree < 1
        eps: small constant for numerical stability
    """
    diff = torch.abs(pred - target) + eps
    loss = torch.pow(diff, p).mean()
    return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def geometric_constraint_loss(
    triangle_model, 
    lambda_pos=0.1, 
    lambda_norm=0.05
):
    """
    计算 Octree-GS 锚定约束 (L_anchor) 和法线方向约束 (L_orient)。
    L_total = lambda_pos * L_anchor + lambda_norm * L_orient
    """
    # 假设 triangle_model 已经通过 Octree 生成并存储了这些属性
    if not hasattr(triangle_model, 'anchor_mu'):
        return torch.tensor(0.0, device=triangle_model.get_device())

    # --- 1. 准备输入数据 ---
    # 假设模型提供了以下接口来获取连续张量
    v1 = triangle_model.get_v1().contiguous() # [N, 3] 顶点 V1
    v2 = triangle_model.get_v2().contiguous() # [N, 3] 顶点 V2
    v3 = triangle_model.get_v3().contiguous() # [N, 3] 顶点 V3
    
    # 锚定参数 (从模型中获取)
    anchor_mu = triangle_model.anchor_mu.contiguous() 
    anchor_sigma_inv = triangle_model.anchor_sigma_inv.contiguous() 
    anchor_normal = triangle_model.anchor_normal.contiguous()
    # 确保 is_active 存在且维度匹配
    if not hasattr(triangle_model, '_is_active_tensor'):
        # 兜底：全 1
        is_active = torch.ones((v1.shape[0], 1), device=v1.device, dtype=torch.float32)
    else:
        is_active = triangle_model.is_active.contiguous()

    # 再次检查维度一致性 (Pipe Guardrail)
    if anchor_mu.shape[0] != v1.shape[0]:
        print(f"Pipe Error: Shape mismatch anchor {anchor_mu.shape[0]} vs tris {v1.shape[0]}")
        return torch.tensor(0.0, device=v1.device)    

    # --- 2. 调用 C++ / CUDA Kernel ---
    L_combined_per_tri = tb.geometric_constraint_loss_forward(
        v1, v2, v3, anchor_mu, anchor_sigma_inv, anchor_normal, is_active,
        lambda_pos, lambda_norm
    )
    
    # --- 3. 计算最终的总 Loss ---
    # Loss 在 Cuda Kernel 内部已经被 λ 缩放
    # L_combined_per_tri 维度为 [N, 2]，求所有 Triangle 的平均 Loss
    L_total = L_combined_per_tri.sum() / N
    
    return L_total

def laplacian_smoothness_loss(
    triangle_model, 
    K=5, 
    lambda_smooth=0.001
):
    """
    计算拉普拉斯平滑损失 L_smooth: 约束每个 Triangle 中心靠近其 K-NN 邻域的均值。
    """
    # 假设模型已经提供了预计算的邻域索引张量
    if not hasattr(triangle_model, 'neighbor_indices'):
        return torch.tensor(0.0, device=triangle_model.get_device())

    # --- 1. 准备输入数据 ---
    v1 = triangle_model.get_v1().contiguous() # [N, 3] 顶点 V1
    v2 = triangle_model.get_v2().contiguous() # [N, 3] 顶点 V2
    v3 = triangle_model.get_v3().contiguous() # [N, 3] 顶点 V3
    is_active = triangle_model.is_active.contiguous()
    
    # 邻域索引必须是 uint32_t 格式
    neighbor_indices = triangle_model.neighbor_indices.contiguous().to(torch.int32) 
    
    N = v1.shape[0]

    # --- 2. 调用 C++ / CUDA Kernel ---
    # 返回一个 [N] 的张量，包含每个 Triangle 的 L_smooth (已乘权重)
    L_smooth_per_tri = tb.laplacian_smoothness_forward(
        v1, v2, v3, 
        neighbor_indices, is_active, K,
        lambda_smooth
    )
    
    # --- 3. 计算最终的总 Loss ---
    L_smooth_total = L_smooth_per_tri.sum() / N
    
    return L_smooth_total

def binary_opacity_loss(
    triangle_model, 
    lambda_binary=0.01
):
    """
    计算不透明度二值化损失 L_binary = lambda_binary * mean(|alpha - 0.5|)
    此项惩罚 alpha 接近 0.5，推动其趋向 0 或 1。
    """
    # 假设模型提供了 get_opacity() 接口
    if not hasattr(triangle_model, 'get_opacity'):
        return torch.tensor(0.0, device=triangle_model.get_device())

    # 获取不透明度张量 (alpha)。它通常是一个 sigmoid 后的值 [0, 1]。
    opacity = triangle_model.get_opacity().contiguous() # [N, 1] 或 [N]
    
    # 计算 |alpha - 0.5|
    # 当 alpha = 0.5 时，Loss 最大为 0.5。当 alpha = 0 或 1 时，Loss 最小为 0.5。
    # 我们希望最小化这个损失。
    L_binary_per_tri = torch.abs(opacity - 0.5)
    
    # 最终的 Loss = mean(L_binary_per_tri) * lambda_binary
    L_binary_total = L_binary_per_tri.mean() * lambda_binary
    
    return L_binary_total


r'''
def intersection_penalty_loss(
    triangle_model, 
    K=5, 
    lambda_intersect=0.05,
    margin=1e-4
):
    """
    计算互斥性惩罚损失 L_intersect: 惩罚相邻三角形之间的相交或距离过近。
    """
    # 假设模型已经提供了预计算的邻域索引张量
    if not hasattr(triangle_model, 'neighbor_indices'):
        return torch.tensor(0.0, device=triangle_model.get_device())

    # --- 1. 准备输入数据 ---
    v1 = triangle_model.get_v1().contiguous() 
    v2 = triangle_model.get_v2().contiguous() 
    v3 = triangle_model.get_v3().contiguous() 
    is_active = triangle_model.is_active.contiguous()
    
    # 邻域索引必须是 uint32_t 格式
    neighbor_indices = triangle_model.neighbor_indices.contiguous().to(torch.int32) 
    
    N = v1.shape[0]

    # --- 2. 调用 C++ / CUDA Kernel ---
    L_intersect_per_tri = tb.intersection_penalty_forward(
        v1, v2, v3, 
        neighbor_indices, is_active, K,
        lambda_intersect, margin
    )
    
    # --- 3. 计算最终的总 Loss ---
    L_intersect_total = L_intersect_per_tri.sum() / N
    
    return L_intersect_total
'''

def l_anchor_loss(vertices, anchor_feature_index, triangle_mask=None, 
                  anchor_mu=None, anchor_sigma_inv=None, octree_res=None):
    if anchor_feature_index.numel() == 0 or anchor_mu is None:
        return torch.tensor(0.0, device=vertices.device)
    
    if triangle_mask is not None:
        anchor_feature_index = anchor_feature_index[triangle_mask]
        anchor_mu = anchor_mu[triangle_mask]
        if anchor_sigma_inv is not None: anchor_sigma_inv = anchor_sigma_inv[triangle_mask]
        if octree_res is not None: octree_res = octree_res[triangle_mask]

    tri_pts = vertices[anchor_feature_index] # [N, 3, 3]
    tri_centers = tri_pts.mean(dim=1)         # [N, 3]
    diff = tri_centers - anchor_mu           # [N, 3]

    if anchor_sigma_inv is None or anchor_sigma_inv.numel() == 0:
        pos_loss = (diff * diff).sum(dim=1).mean()
    else:
        x, y, z = diff[:, 0], diff[:, 1], diff[:, 2]
        # 提取 6 个分量
        s_xx, s_xy, s_xz, s_yy, s_yz, s_zz = [anchor_sigma_inv[:, i] for i in range(6)]
        
        # 马氏距离计算（全矩阵展开）
        mahalanobis_dist = (x*x*s_xx + y*y*s_yy + z*z*s_zz + 
                            2*(x*y*s_xy + x*z*s_xz + y*z*s_yz))
        pos_loss = F.relu(mahalanobis_dist).mean() # 确保非负

    # 盒子限制
    if octree_res is not None:
        limit = octree_res * 0.5
    else:
        # 计算三角形各边长
        edges = torch.norm(tri_pts[:, [1, 2, 0]] - tri_pts[:, [0, 1, 2]], dim=-1) # [N, 3]
        limit = edges.mean(dim=1) * 0.5
    
    # 对齐维度: [N, 1, 1] 用于广播到 [N, 3, 3]
    v_offsets = tri_pts - anchor_mu.unsqueeze(1)
    box_violations = torch.abs(v_offsets) - limit.view(-1, 1, 1)
    box_loss = torch.mean(F.relu(box_violations)**2)
    
    return pos_loss + 1.0 * box_loss

def l_orient_loss(vertices, anchor_feature_index, triangle_mask=None, 
                  anchor_normal=None):
    if anchor_feature_index.numel() == 0 or anchor_normal is None:
        return torch.tensor(0.0, device=vertices.device)
    
    if triangle_mask is not None:
        anchor_feature_index = anchor_feature_index[triangle_mask]
        anchor_normal = anchor_normal[triangle_mask]
    
    tri_pts = vertices[anchor_feature_index]
    v1, v2, v3 = tri_pts[:, 0], tri_pts[:, 1], tri_pts[:, 2]
    
    # 鲁棒的法线计算
    tri_normal = torch.cross(v2 - v1, v3 - v1, dim=-1)
    tri_normal = F.normalize(tri_normal, p=2, dim=-1, eps=1e-8)
    
    # 目标法线也需要标准化
    anchor_normal = F.normalize(anchor_normal, p=2, dim=-1, eps=1e-8)
    
    # 1 - |cos(theta)|
    dot_product = torch.sum(tri_normal * anchor_normal, dim=-1).abs()
    return (1.0 - dot_product).mean()

def l_smooth_loss(vertices, _triangle_indices, neighbor_indices=None, K=5, triangle_mask=None):
    """
    向量化的平滑损失函数
    """
    if _triangle_indices.numel() == 0:
        return torch.tensor(0.0, device=vertices.device)
    
    tri_pts = vertices[_triangle_indices]  # [N, 3, 3]
    
    # 计算三角形中心
    centers = tri_pts.mean(dim=1)  # [N, 3]
    
    # 如果提供了三角形掩码，筛选数据
    if triangle_mask is not None and triangle_mask.any():
        # 筛选可见三角形
        centers = centers[triangle_mask]
        N_visible = centers.shape[0]
        
        # 如果有邻居索引，需要筛选对应的邻居
        if neighbor_indices is not None and neighbor_indices.numel() > 0:
            # 获取可见三角形的原始索引
            original_indices = torch.where(triangle_mask)[0]
            
            # 创建一个映射：原始索引 -> 可见索引位置
            index_map = torch.full((_triangle_indices.shape[0],), -1, dtype=torch.long, device=vertices.device)
            index_map[original_indices] = torch.arange(N_visible, device=vertices.device)
            
            # 获取可见三角形的邻居索引
            neighbor_indices_visible = neighbor_indices.view(-1, K)[triangle_mask]  # [N_visible, K]
            
            # 将邻居索引映射到可见索引
            neighbor_indices_visible = index_map[neighbor_indices_visible]
            
            # 处理无效映射（-1表示邻居不可见）
            valid_mask = neighbor_indices_visible >= 0
            
            # 计算有效邻居的中心
            neighbor_centers = torch.zeros(N_visible, K, 3, device=vertices.device)
            for i in range(N_visible):
                valid_neighbors = valid_mask[i]
                if valid_neighbors.any():
                    neighbor_centers[i, valid_neighbors] = centers[neighbor_indices_visible[i, valid_neighbors]]
            
            # 计算每个三角形的有效邻居数量
            valid_count = valid_mask.sum(dim=1, keepdim=True).float()  # [N_visible, 1]
            valid_count = torch.clamp(valid_count, min=1e-6)  # 避免除零
            
            # 计算邻居中心均值
            neighbor_mean = neighbor_centers.sum(dim=1) / valid_count  # [N_visible, 3]
            
            # 计算拉普拉斯项
            diff = centers - neighbor_mean  # [N_visible, 3]
            smooth_loss = (diff ** 2).sum(dim=1)  # [N_visible]
            
            # 只对有有效邻居的三角形计算损失
            has_valid_neighbors = valid_count.squeeze() > 0.5  # [N_visible]
            if has_valid_neighbors.any():
                smooth_loss = smooth_loss[has_valid_neighbors].mean()
            else:
                smooth_loss = torch.tensor(0.0, device=vertices.device)
        else:
            # 如果没有邻居索引，使用形状约束
            edge_a = torch.norm(tri_pts[triangle_mask, 1] - tri_pts[triangle_mask, 0], dim=-1)
            edge_b = torch.norm(tri_pts[triangle_mask, 2] - tri_pts[triangle_mask, 1], dim=-1)
            edge_c = torch.norm(tri_pts[triangle_mask, 0] - tri_pts[triangle_mask, 2], dim=-1)
            
            edges = torch.stack([edge_a, edge_b, edge_c], dim=0)  # [3, N_visible]
            shape_loss = torch.var(edges, dim=0).mean()
            
            # 法向平滑
            normals = torch.cross(
                tri_pts[triangle_mask, 1] - tri_pts[triangle_mask, 0], 
                tri_pts[triangle_mask, 2] - tri_pts[triangle_mask, 0], 
                dim=-1
            )
            normals = F.normalize(normals, p=2, dim=-1, eps=1e-8)
            normal_loss = torch.var(normals, dim=0).mean()
            
            smooth_loss = 1.0 * shape_loss + 0.5 * normal_loss
    else:
        # 如果没有可见三角形，返回0
        smooth_loss = torch.tensor(0.0, device=vertices.device)
    
    return smooth_loss



def l_intersect_loss(
    triangle_model, 
    lambda_intersect=0.05,
    margin=1e-4,
    use_aabb=True
):
    """
    计算互斥性惩罚损失 L_intersect，使用AABB树加速
    """
    if not use_aabb:
        # 回退到KNN版本
        return l_intersect_loss_knn(triangle_model, lambda_intersect, margin)
    
    # 1. 获取三角形顶点
    v1 = triangle_model.get_v1().contiguous()
    v2 = triangle_model.get_v2().contiguous()
    v3 = triangle_model.get_v3().contiguous()
    is_active = triangle_model.is_active.contiguous()
    
    # 2. 计算三角形中心点
    triangle_centers = (v1 + v2 + v3) / 3.0
    
    # 3. 检查或构建AABB树
    if not hasattr(triangle_model, '_aabb_tree') or \
        triangle_model._aabb_tree is None or \
        triangle_model._aabb_tree.shape[0] == 0:
        
        print("Building AABB tree for intersection detection...")
        triangle_model._aabb_tree = tb.build_triangle_aabb_tree(
            triangle_centers,
            max_triangles_per_node=16
        )
    
    # 4. 调用AABB树加速版本
    try:
        L_intersect_per_tri = tb.intersection_penalty_forward_aabb(
            v1, v2, v3,
            triangle_centers,           # 三角形中心点
            triangle_model._aabb_tree,  # 预构建的AABB树
            is_active,
            lambda_intersect, margin
        )
    except Exception as e:
        print(f"Warning: AABB intersection loss failed: {e}")
        print("Falling back to KNN version")
        return l_intersect_loss_knn(triangle_model, lambda_intersect, margin)
    
    # 5. 计算总损失
    N = v1.shape[0]
    L_intersect_total = L_intersect_per_tri.sum() / N if N > 0 else 0.0
    
    return L_intersect_total

def l_intersect_loss_knn(triangle_model, lambda_intersect=0.05, margin=1e-4):
    """回退到KNN版本的相交损失"""
    if not hasattr(triangle_model, 'neighbor_indices') or \
       triangle_model.neighbor_indices is None or \
       triangle_model.neighbor_indices.numel() == 0:
        return torch.tensor(0.0, device=triangle_model.get_device())

    v1 = triangle_model.get_v1().contiguous() 
    v2 = triangle_model.get_v2().contiguous() 
    v3 = triangle_model.get_v3().contiguous() 
    is_active = triangle_model.is_active.contiguous()
    neighbor_indices = triangle_model.neighbor_indices.to(torch.int32).contiguous()
    
    N = v1.shape[0]
    
    # 推断K值
    total_elements = neighbor_indices.numel()
    if total_elements % N != 0:
        raise ValueError(f"neighbor_indices size {total_elements} is not divisible by N={N}")
    K = total_elements // N
    
    # 重塑为[N, K]
    neighbor_indices_reshaped = neighbor_indices.view(N, K)
    
    L_intersect_per_tri = tb.intersection_penalty_forward(
        v1, v2, v3, 
        neighbor_indices_reshaped,  # 使用二维张量
        is_active,
        K=K,  # 使用推断的K
        lambda_intersect=lambda_intersect, 
        margin=margin
    )
    
    L_intersect_total = L_intersect_per_tri.sum() / N if N > 0 else 0.0
    
    return L_intersect_total



    