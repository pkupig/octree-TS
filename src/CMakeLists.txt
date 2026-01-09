find_package(CUDAToolkit REQUIRED)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -D_USE_MATH_DEFINES)

# 设置 CUDA 标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 添加包含路径
list(
  APPEND
  TRIANGULATION_INCLUDES
  ${CMAKE_SOURCE_DIR}/submodules/eigen
  ${CMAKE_SOURCE_DIR}/src  # 添加 src 目录，确保头文件可以被其他模块找到
)

set(TRIANGULATION_CXX_SOURCES)

# 只列出 triangulation 库的 CUDA 源文件
list(
  APPEND
  TRIANGULATION_CUDA_SOURCES
  aabb_tree/aabb_tree.cu
  delaunay/delaunay.cu
  delaunay/sample_initial_tets.cu
  delaunay/growth_iteration.cu
  delaunay/delete_violations.cu
  delaunay/triangulation_ops.cu
  utils/common_kernels.cu)

# 创建 triangulation 静态库
add_library(triangulation STATIC ${TRIANGULATION_CXX_SOURCES} ${TRIANGULATION_CUDA_SOURCES} )

# 设置 triangulation 库的公共头文件路径
target_include_directories(triangulation PUBLIC 
  ${TRIANGULATION_INCLUDES}
  ${CMAKE_CURRENT_SOURCE_DIR}  # 添加当前目录，确保相对路径包含正确
)

# 链接依赖库
target_link_libraries(triangulation PUBLIC 
  CUDA::cudart 
  CUDA::cuda_driver 
  ${TBB_IMPORTED_TARGETS})

# 编译选项
target_compile_options(triangulation PRIVATE 
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -O3 -Wall -Wno-unknown-pragmas>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>)

# 设置 CUDA 属性
set_target_properties(triangulation PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
  INTERPROCEDURAL_OPTIMIZATION OFF
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

# 安装 triangulation 库（可选，用于调试）
install(TARGETS triangulation
  EXPORT triangulation-targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

# 安装头文件（可选）
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/triangulation
  FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)