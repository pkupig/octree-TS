import triangulation as tb

# 测试函数是否存在
test_functions = [
    'geometric_constraint_loss_forward',
    'laplacian_smoothness_forward',
    'intersection_penalty_forward',
    'intersection_penalty_forward_aabb',
    'build_triangle_aabb_tree',
    'compute_sdf'
]

print("Checking available functions in triangulation module:")
for func_name in test_functions:
    if hasattr(tb, func_name):
        print(f"  ✓ {func_name}")
    else:
        print(f"  ✗ {func_name} - NOT FOUND")

# 测试函数调用
print("\nTesting function calls:")
try:
    import torch
    # 创建测试数据
    v1 = torch.randn(10, 3, device='cuda')
    v2 = torch.randn(10, 3, device='cuda')
    v3 = torch.randn(10, 3, device='cuda')
    
    print("Testing build_triangle_aabb_tree...")
    centers = (v1 + v2 + v3) / 3.0
    aabb_tree = tb.build_triangle_aabb_tree(centers)
    print(f"  AABB tree shape: {aabb_tree.shape}")
    
    print("\nAll tests passed!")
except Exception as e:
    print(f"Test failed: {e}")