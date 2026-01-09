#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2025, University of Liege
# TELIM research group, http://www.telecom.ulg.ac.be/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l_anchor_loss, l_orient_loss, l_smooth_loss, l_intersect_loss
from triangle_renderer import render
import sys
from scene import Scene, TriangleModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, update_indoor
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import lpips
import torch.nn.functional as F


# These weights need tuning, but provide a starting point for automatic optimization
DEFAULT_LAMBDA_ANCHOR = 0.001
DEFAULT_LAMBDA_ORIENT = 0.005 
DEFAULT_LAMBDA_SMOOTH = 0.001 
DEFAULT_LAMBDA_INTERSECT = 0.01
DEFAULT_USE_AABB = True  # 默认使用AABB树加速
DEFAULT_AABB_REBUILD_INTERVAL = 10  # AABB树重建间隔

def training(
        dataset,   
        opt, 
        pipe,
        testing_iterations,
        checkpoint, 
        debug_from,
        ):
    
    first_iter = 0      
    tb_writer = prepare_output_and_logger(dataset)

    # Load parameters, triangles and scene
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma)
    triangles.training_setup(opt, opt.feature_lr, opt.weight_lr, opt.lr_triangles_points_init) 
    triangles.add_percentage = opt.add_percentage
    
    # --- 新增：设置渐进式训练参数 ---
    if hasattr(opt, 'coarse_iter') and hasattr(opt, 'coarse_factor'):
        triangles.set_coarse_interval(opt.coarse_iter, opt.coarse_factor)
    else:
        # 默认值
        triangles.set_coarse_interval(5000, 2.0)
    if hasattr(triangles, 'update_aabb_tree'):
        triangles.update_aabb_tree(rebuild_interval=1)
        
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        triangles.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # 初始化损失权重
    lambda_anchor = getattr(opt, 'lambda_anchor', DEFAULT_LAMBDA_ANCHOR)
    lambda_orient = getattr(opt, 'lambda_orient', DEFAULT_LAMBDA_ORIENT)
    lambda_smooth = getattr(opt, 'lambda_smooth', DEFAULT_LAMBDA_SMOOTH)
    lambda_intersect = getattr(opt, 'lambda_intersect', DEFAULT_LAMBDA_INTERSECT)
    lambda_binary = getattr(opt, 'lambda_binary', 0.01)  # 不透明度二值化权重

    use_aabb = getattr(opt, 'use_aabb', DEFAULT_USE_AABB)
    aabb_rebuild_interval = getattr(opt, 'aabb_rebuild_interval', DEFAULT_AABB_REBUILD_INTERVAL)   
    initial_sigma = opt.set_sigma
    final_sigma = 0.0001
    sigma_start = opt.sigma_start
    total_iters = opt.sigma_until

    init_opacity = 0.1
    final_opacity = .9999
    total_iters_opacity = opt.final_opacity_iter

    lambda_weight = opt.lambda_weight
    prune_triangles = opt.prune_triangles_threshold
    prune_size = opt.prune_size
    start_upsampling = opt.start_upsampling
    splitt_large_triangles = opt.splitt_large_triangles
    triangles.size_probs_zero = opt.size_probs_zero
    triangles.size_probs_zero_image_space = opt.size_probs_zero_image_space

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # DEBUG
        # if iteration == first_iter: print("Step 1: Update LR & Sigma")
        # 1. 动态学习率更新
        triangles.update_learning_rate(iteration)

        # 2. 动态 Sigma 调度
        if iteration < sigma_start:
            current_sigma = initial_sigma
        else:
            progress = min((iteration - sigma_start) / (total_iters - sigma_start), 1.0)
            current_sigma = initial_sigma - (initial_sigma - final_sigma) * progress
        triangles.set_sigma(current_sigma)

        # 3. 属性阶段性升级
        if iteration % 1000 == 0:
            triangles.oneupSHdegree()
        
        # 4. 渐进式不透明度增加
        if iteration > 1000 and iteration % 100 == 0:
            triangles.gradually_increase_opacity(iteration, total_iterations=5000)

        # 更新AABB树（按间隔重建）
        if use_aabb and iteration % aabb_rebuild_interval == 0:
            triangles.set_current_iteration(iteration)

        # 5. 相机/视角加载
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # DEBUG
        # if iteration == first_iter: print("Step 2: Set Triangle Mask")
        # 6. 背景色与可见性掩码处理
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        cam_center = viewpoint_cam.camera_center
        res_scale = triangles.scaling if hasattr(triangles, 'scaling') else 1.0
        
        # 设置三角形可见性掩码
        triangle_mask = triangles.set_triangle_mask(
            cam_center, 
            iteration, 
            resolution_scale=res_scale
        )

        # DEBUG
        # if iteration == first_iter: print("Step 3: Rendering")
        # 7. 渲染管道
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # 动态调整渲染分辨率
        if iteration < 5000:  # 早期训练使用低分辨率
            current_resolution_scale = 0.5
        elif iteration < 15000:  # 中期训练使用中等分辨率
            current_resolution_scale = 0.75
        else:  # 后期训练使用全分辨率
            current_resolution_scale = 1.0

        # 创建临时pipe用于训练渲染
        train_pipe = type(pipe)()
        for attr in dir(pipe):
            if not attr.startswith('_'):
                setattr(train_pipe, attr, getattr(pipe, attr))

        # 设置分辨率缩放
        train_pipe.img_resolution_factor = current_resolution_scale

        # 使用调整后的pipe渲染
        render_pkg = render(viewpoint_cam, triangles, train_pipe, bg, triangle_mask=triangle_mask)
        image = render_pkg["render"]

        # 如果渲染图像与gt图像尺寸不匹配，调整gt图像
        gt_image_original = viewpoint_cam.original_image.cuda()
        if gt_image_original.shape[1:] != image.shape[1:]:
            gt_image = F.interpolate(
                gt_image_original.unsqueeze(0), 
                size=image.shape[1:], 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        else:
            gt_image = gt_image_original

        # DEBUG
        # if iteration == first_iter: print("Step 4: Loss Calculation (Image)")
        # 8. 图像损失计算
        gt_image = viewpoint_cam.original_image.cuda()
        pixel_loss = l1_loss(image, gt_image)
        loss_image = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # 9. 统计量异步更新
        with torch.no_grad():
            # current_visibility_mask 长度为 1694561 (三角形总数)
            current_visibility_mask = render_pkg["visibility_filter"]
            
            # --- 核心修正部分 ---
            if current_visibility_mask is not None and current_visibility_mask.any():
                
                # 更新 image_size
                if "scaling" in render_pkg:
                    # 渲染器返回的可见三角形的缩放数据
                    scales = render_pkg["scaling"].detach() 
                    
                    # 关键点：确保 scales 也是一维的，以匹配你 self.image_size 的定义
                    if scales.dim() > 1:
                        scales = scales.squeeze() # [131, 1] -> [131]
                    
                    # 执行更新
                    # 左边: triangles.image_size[mask] 选出 131 个位置
                    # 右边: scales 提供 131 个数值
                    triangles.image_size[current_visibility_mask] = torch.max(
                        triangles.image_size[current_visibility_mask], 
                        scales[current_visibility_mask]  # <--- 这里加上了索引
                    )
                
                # 更新 importance_score
                if "max_blending" in render_pkg:
                    blends = render_pkg["max_blending"].detach()
                    if blends.dim() > 1:
                        blends = blends.squeeze()
                        
                    triangles.importance_score[current_visibility_mask] = torch.max(
                        triangles.importance_score[current_visibility_mask],
                        blends[current_visibility_mask]
                    )

            # --- 处理 update_training_stats (八叉树逻辑) ---
            vertex_opacity = triangles.get_vertex_weight.detach()
            tri_indices = triangles._triangle_indices.long()
            # 将顶点透明度映射到三角形：[N, 3] -> [N]
            triangle_opacity = vertex_opacity[tri_indices].mean(dim=1).squeeze()

            if hasattr(triangles, 'update_training_stats'):
                triangles.update_training_stats(
                    opacity=triangle_opacity,
                    gradient_norm=None, 
                    triangle_mask=current_visibility_mask
                )

        # 10. 法线损失
        loss_normal = torch.tensor(0.0, device="cuda")
        if iteration > opt.iteration_mesh and hasattr(viewpoint_cam, 'normal_map'):
            gt_normal = viewpoint_cam.normal_map.cuda()
            if gt_normal.shape[1:] != image.shape[1:]:
                gt_normal = F.interpolate(gt_normal.unsqueeze(0), size=image.shape[1:], mode="area").squeeze(0)
            
            rend_normal = render_pkg.get('rend_normal', None)
            if rend_normal is not None:
                normal_error = (1.0 - (rend_normal * gt_normal).sum(dim=0))
                loss_normal = getattr(opt, 'lambda_normals', 0.01) * normal_error.mean()

        # debug
        # if iteration == first_iter: 
            # print("Step 5: Loss Calculation (Geometric Reg)")
            # print(f"  Total triangles: {triangles._triangle_indices.shape[0]}")
            
        # 11. 几何正则化损失 (Octree-GS + 三角形溅射)
        loss_reg = torch.tensor(0.0, device="cuda")

        # 确保三角形存在且拓扑关系正确
        if hasattr(triangles, '_triangle_indices') and triangles._triangle_indices.numel() > 0:
            
            # 检查拓扑关系是否有效
            N_triangles = triangles._triangle_indices.shape[0]
            
            # 检查邻居索引是否有效
            if hasattr(triangles, 'neighbor_indices') and triangles.neighbor_indices is not None:
                if triangles.neighbor_indices.numel() > 0:
                    # 确保邻居索引不越界
                    if triangles.neighbor_indices.max() >= N_triangles:
                        print(f"[Iter {iteration}] Warning: neighbor_indices out of bounds before regularization, skipping smooth loss.")
                        # 暂时禁用平滑损失
                        compute_smooth_loss = False
                    else:
                        compute_smooth_loss = True
                else:
                    compute_smooth_loss = False
            else:
                compute_smooth_loss = False
            
            # 使用 current_visibility_mask 筛选可见三角形
            visible_mask = current_visibility_mask if current_visibility_mask is not None else None

            # 只对可见三角形计算几何正则化
            if visible_mask is not None and visible_mask.any():
                visible_count = visible_mask.sum().item()
                
                if iteration % 500 == 0:
                    print(f"  Visible triangles: {visible_count}/{triangles._triangle_indices.shape[0]}")
                
                # A. 锚定损失和方向损失
                if hasattr(triangles, 'anchor_mu') and triangles.anchor_mu is not None:
                    # 计算锚定损失 (使用可见三角形掩码)
                    if hasattr(triangles, 'anchor_feature_index'):
                        loss_anchor = l_anchor_loss(
                            vertices=triangles.vertices,
                            anchor_feature_index=triangles.anchor_feature_index,
                            triangle_mask=visible_mask,  # 使用可见性掩码
                            anchor_mu=triangles.anchor_mu,
                            anchor_sigma_inv=triangles.anchor_sigma_inv if hasattr(triangles, 'anchor_sigma_inv') else None,
                            octree_res=triangles.octree_res if hasattr(triangles, 'octree_res') else None
                        )
                        loss_reg += lambda_anchor * loss_anchor
                        if iteration % 500 == 0:
                            print(f"  Anchor loss: {loss_anchor.item():.6f}")
                    
                    # 计算方向损失 (使用可见三角形掩码)
                    if hasattr(triangles, 'anchor_normal') and triangles.anchor_normal is not None:
                        loss_orient = l_orient_loss(
                            vertices=triangles.vertices,
                            anchor_feature_index=triangles.anchor_feature_index,
                            triangle_mask=visible_mask,  # 使用可见性掩码
                            anchor_normal=triangles.anchor_normal
                        )
                        loss_reg += lambda_orient * loss_orient
                        if iteration % 500 == 0:
                            print(f"  Orientation loss: {loss_orient.item():.6f}")
                
                # B. 平滑损失 (低频计算，只对可见三角形)
                if iteration % 20 == 0:
                    neighbor_indices = triangles.neighbor_indices if hasattr(triangles, 'neighbor_indices') else None
                    if neighbor_indices is not None:
                        # 只对可见三角形计算平滑损失
                        loss_smooth = l_smooth_loss(
                            vertices=triangles.vertices,
                            _triangle_indices=triangles._triangle_indices,
                            neighbor_indices=neighbor_indices,
                            triangle_mask=visible_mask,  # 使用可见性掩码
                            K=5
                        )
                        loss_reg += lambda_smooth * loss_smooth
                        if iteration % 500 == 0:
                            print(f"  Smooth loss: {loss_smooth.item():.6f}")
                
                # C. 交叉惩罚损失 (使用AABB加速版本)
                # 动态调整策略：早期少计算，后期多计算
                compute_intersection = False
                intersect_weight = 1.0
                
                if iteration < 1000:
                    # 前1000次迭代：完全禁用相交损失
                    compute_intersection = False
                elif iteration < 5000:
                    # 1000-5000次：每100次迭代计算一次
                    compute_intersection = (iteration % 100 == 0)
                    intersect_weight = 0.1 * min(1.0, (iteration - 1000) / 4000.0)
                elif iteration < 15000:
                    # 5000-15000次：每50次迭代计算一次
                    compute_intersection = (iteration % 50 == 0)
                    intersect_weight = 0.5
                else:
                    # 15000次以后：每20次迭代计算一次
                    compute_intersection = (iteration % 20 == 0)
                    intersect_weight = 1.0
                
                if compute_intersection and iteration > 100:
                    # 关键优化：只在有足够多可见三角形时计算相交损失
                    if visible_count > 100:
                        try:
                            # 使用AABB树加速版本，但控制频率
                            use_aabb = getattr(opt, 'use_aabb', DEFAULT_USE_AABB)
                            
                            # 只在需要时重建AABB树
                            if use_aabb and iteration % 50 == 0:
                                triangles.set_current_iteration(iteration)
                            
                            # 只对可见三角形计算相交损失
                            loss_intersect = l_intersect_loss(
                                triangle_model=triangles,
                                lambda_intersect=lambda_intersect * intersect_weight,
                                margin=1e-4,
                                use_aabb=use_aabb
                            )
                            
                            # 渐进式增加相交损失权重
                            if iteration < 5000:
                                intersect_weight = min(1.0, (iteration - 100) / 4900.0)
                                loss_intersect = loss_intersect * intersect_weight
                            
                            loss_reg += loss_intersect
                            
                            if iteration % 500 == 0:
                                print(f"  Intersection loss: {loss_intersect.item():.6f}")
                                
                        except Exception as e:
                            print(f"[Iter {iteration}] Intersection loss failed: {e}")
                            # 降级到KNN版本
                            try:
                                if hasattr(triangles, 'neighbor_indices') and triangles.neighbor_indices.numel() > 0:
                                    loss_intersect = l_intersect_loss(
                                        triangle_model=triangles,
                                        lambda_intersect=lambda_intersect * intersect_weight,
                                        margin=1e-4,
                                        use_aabb=False
                                    )
                                    loss_reg += loss_intersect
                            except:
                                pass
                
                # D. 不透明度二值化损失 (只对可见三角形)
                if hasattr(triangles, 'get_opacity'):
                    # 只获取可见三角形的不透明度
                    visible_opacity = triangles.get_opacity()[visible_mask] if visible_mask is not None else triangles.get_opacity()
                    loss_binary = torch.abs(visible_opacity - 0.5).mean() * lambda_binary
                    loss_reg += loss_binary

        # 12. 顶点权重损失 (早期训练阶段)
        if iteration < opt.start_opacity_floor:
            tri_indices = triangles._triangle_indices.long()
            if tri_indices.numel() > 0:
                loss_weight = triangles.get_vertex_weight[tri_indices].mean() * lambda_weight
                loss_reg += loss_weight

        # 13. 总损失
        total_loss = loss_image + loss_normal + loss_reg

        # debug
        # if iteration == first_iter: print("Step 6: Backward")
        # 14. 反向传播
        triangles.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        # --- Debug Print 7 ---
        # if iteration == first_iter: print("Step 7: Optimizer Step")

        # 更新优化器
        if iteration < opt.iterations:
            triangles.optimizer.step()

        iter_end.record()
        torch.cuda.synchronize()

        # 15. 更新进度条
        with torch.no_grad():
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                visible_count = triangle_mask.sum().item() if triangle_mask is not None else 0
                total_count = triangles._triangle_indices.shape[0] if hasattr(triangles, '_triangle_indices') else 0
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Visible": f"{visible_count}/{total_count}",
                    "Image": f"{loss_image.item():.4f}",
                    "Reg": f"{loss_reg.item():.4f}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

        # 16. 定期操作：剪枝、增密等
        if iteration % 500 == 0:
            # 剪枝逻辑
            if iteration > opt.start_pruning:
                try:
                    # 获取三角形顶点权重
                    tri_verts = triangles.vertices[triangles._triangle_indices]
                    tri_opacities = triangles.get_vertex_weight[triangles._triangle_indices]
                    
                    # 计算每个三角形的最小不透明度
                    min_weights = tri_opacities.min(dim=1).values
                    
                    # 构建删除掩码
                    mask_opacity = (min_weights <= prune_triangles)
                    mask_importance = (triangles.importance_score <= prune_triangles)
                    mask_size = (triangles.image_size > prune_size)
                    
                    delete_mask = mask_opacity | mask_importance | mask_size
                    keep_mask = ~delete_mask
                    
                    if delete_mask.sum() > 0:
                        triangles.prune_triangles(keep_mask)
                        # 更新拓扑关系
                        triangles.update_topology(K=5)
                        
                        # 剪枝后立即重建AABB树
                        if hasattr(triangles, 'update_aabb_tree'):
                            triangles.update_aabb_tree(rebuild_interval=1)  # 强制重建
                            print(f"[Iter {iteration}] AABB tree rebuilt after pruning")

                        
                except Exception as e:
                    print(f"[Warning] Pruning failed at iteration {iteration}: {e}")
            
            # 
            needs_densification = False
            '''
            needs_densification = (iteration < opt.densify_until_iter and 
                                 iteration % opt.densification_interval == 0 and 
                                 iteration > opt.densify_from_iter)
                                 '''
            
            if needs_densification:
                try:
                    probs_opacity = (iteration < opt.start_opacity_floor) or (iteration % 1000 == 0)
                    
                    # 核心检查：打印当前三角形数量，排查是否因为数量激增导致显存溢出或索引溢出
                    # print(f"Before densify: {triangles._triangle_indices.shape[0]}")
                    
                    num_added = triangles.add_new_gs(
                        iteration, 
                        opt.max_points, 
                        splitt_large_triangles, 
                        probs_opacity
                    )
                    
                    if num_added > 0:
                        # 必须立即同步顶点和拓扑关系
                        triangles.update_topology(K=5)
                        if hasattr(triangles, 'neighbor_indices') and triangles.neighbor_indices is not None:
                            N = triangles._triangle_indices.shape[0]
                            if triangles.neighbor_indices.max() >= N:
                                print(f"[Iter {iteration}] Warning: neighbor_indices out of bounds after densification, clipping.")
                                triangles.neighbor_indices = torch.clamp(triangles.neighbor_indices, 0, N-1)
                        if hasattr(triangles, 'update_aabb_tree'):
                            triangles.update_aabb_tree(rebuild_interval=1)
                except Exception as e:
                    print(f"[Warning] Densification failed at iteration {iteration}: {e}")

            # 动态调整三角形
            if iteration % 1000 == 0 and iteration > 5000:
                try:
                    num_pruned = triangles.adjust_triangles(iteration)
                    if num_pruned > 0:
                        print(f"[Iter {iteration}] Pruned {num_pruned} low-quality triangles")
                except Exception as e:
                    print(f"[Warning] Triangle adjustment failed at iteration {iteration}: {e}")
            
            # 更新不透明度地板值
            if iteration > opt.start_opacity_floor:
                start_iter = opt.start_opacity_floor
                end_iter = total_iters_opacity
                a = min(1.0, max(0.0, (iteration - start_iter) / max(1, end_iter - start_iter)))
                current_opacity = init_opacity + (final_opacity - init_opacity) * a
                current_opacity = min(current_opacity, final_opacity)
                triangles.update_min_weight(current_opacity)
                
                if iteration > opt.start_opacity_floor + 1000:
                    prune_triangles = min(prune_triangles + 0.01, 0.5)

        # 17. 训练报告和保存
        if iteration % 1000 == 0:
            training_report(tb_writer, iteration, pixel_loss, total_loss, l1_loss, 
                          iter_start.elapsed_time(iter_end), testing_iterations, 
                          scene, render, (pipe, background))
            if hasattr(triangles, '_aabb_tree') and triangles._aabb_tree is not None:
                print(f"[Iter {iteration}] AABB tree: {triangles._aabb_tree.shape}")        
        if iteration in testing_iterations or iteration in getattr(opt, 'save_iterations', []):
            scene.save(iteration)

    # 18. 训练结束后的清理
    viewpoint_stack = scene.getTrainCameras().copy()
    triangles.importance_score = torch.zeros((triangles._triangle_indices.shape[0]), 
                                             dtype=torch.float, device="cuda")
    while viewpoint_stack:
        viewpoint_cam = viewpoint_stack.pop(0)
        render_pkg = render(viewpoint_cam, triangles, pipe, background)
        importance_score = render_pkg["max_blending"].detach()
        mask = importance_score > triangles.importance_score
        triangles.importance_score[mask] = importance_score[mask]
    
    # 最终剪枝
    mask_importance = (triangles.importance_score <= 0.5).squeeze()
    triangles.prune_triangles(~mask_importance)
    
    # 保存最终模型
    scene.save(iteration)
    print("Training complete!")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, pixel_loss, loss, loss_fn, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/pixel_loss', pixel_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # 报告测试和样本训练集
    if iteration % 1000 == 0:
        torch.cuda.empty_cache()
        
        # 临时存储原始pipe设置
        original_pipe = renderArgs[0]
        
        # 创建低分辨率版本的pipe用于验证
        lowres_pipe = type(original_pipe)()
        for attr in dir(original_pipe):
            if not attr.startswith('_'):
                setattr(lowres_pipe, attr, getattr(original_pipe, attr))
        
        # 降低验证图像分辨率
        lowres_pipe.img_resolution_factor = 0.5  # 降低到50%分辨率
        
        # 更新renderArgs
        lowres_renderArgs = (lowres_pipe, renderArgs[1])
        
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()[:1]},  # 只取1个测试相机
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 6)]})  # 只取1个训练相机

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                pixel_loss_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                total_time = 0.0
                
                for idx, viewpoint in enumerate(config['cameras']):
                    try:
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                        
                        # 使用低分辨率pipe渲染
                        image = torch.clamp(renderFunc(viewpoint, scene.triangles, *lowres_renderArgs)["render"], 0.0, 1.0)
                        
                        end_event.record()
                        torch.cuda.synchronize()
                        runtime = start_event.elapsed_time(end_event)
                        total_time += runtime

                        # 同样降低gt图像分辨率以匹配
                        gt_image_original = viewpoint.original_image.to("cuda")
                        
                        # 如果原图太大，进行下采样
                        if gt_image_original.shape[1] > 512 or gt_image_original.shape[2] > 512:
                            scale_factor = min(512/gt_image_original.shape[1], 512/gt_image_original.shape[2])
                            new_h = int(gt_image_original.shape[1] * scale_factor)
                            new_w = int(gt_image_original.shape[2] * scale_factor)
                            gt_image = F.interpolate(gt_image_original.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
                            
                            # 同样下采样渲染图像
                            image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
                        else:
                            gt_image = gt_image_original
                        
                        gt_image = torch.clamp(gt_image, 0.0, 1.0)
                        
                        if tb_writer and (idx < 1):  # 只保存第一个样本
                            tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        
                        pixel_loss_test += loss_fn(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                        ssim_test += ssim(image, gt_image).mean().double()
                        
                        # 及时清理显存
                        del image, gt_image, gt_image_original
                        torch.cuda.empty_cache()
                        
                    except torch.OutOfMemoryError as e:
                        print(f"[Warning] OOM during validation for {config['name']} camera {idx}: {e}")
                        # 尝试更激进的显存清理
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                
                if len(config['cameras']) > 0:
                    psnr_test /= len(config['cameras'])
                    pixel_loss_test /= len(config['cameras'])       
                    ssim_test /= len(config['cameras'])
                    total_time /= len(config['cameras'])
                    fps = 1000.0 / total_time if total_time > 0 else 0
                    
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {}".format(
                        iteration, config['name'], pixel_loss_test, psnr_test, ssim_test))

                    if tb_writer:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', pixel_loss_test, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--indoor", action="store_true", default=False)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    lpips_fn = lpips.LPIPS(net='vgg').to(device="cuda")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    lps = lp.extract(args)
    ops = op.extract(args)
    pps = pp.extract(args)

    if args.indoor:
        ops = update_indoor(ops)

    # Configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lps,
             ops,
             pps,
             args.test_iterations,
             args.start_checkpoint,
             args.debug_from,
             )
    
    # All done
    print("\nTraining complete.")