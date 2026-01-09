# 设置 CUDA NVCC 标志
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --relocatable-device-code=false")

# 关闭过程间优化
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# 获取 PyTorch 版本信息
execute_process(
  COMMAND python ${CMAKE_SOURCE_DIR}/scripts/torch_info.py torch
  OUTPUT_VARIABLE TORCH_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND python ${CMAKE_SOURCE_DIR}/scripts/torch_info.py cuda
  OUTPUT_VARIABLE TORCH_CUDA_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# 检查 CUDA 版本兼容性
if(TORCH_CUDA_VERSION VERSION_EQUAL CUDA_VERSION_STRING)
  message(STATUS "CUDA version is: ${CUDA_VERSION_STRING}")
else()
  message(
    FATAL_ERROR
      "CUDA version found (${CUDA_VERSION_STRING}) does not match the version required by PyTorch (${TORCH_CUDA_VERSION})."
  )
endif()

# 启用 CUDA 语言支持
enable_language(CUDA)

# 创建 pybind11 模块 - 只包含 bindings 文件
pybind11_add_module(torch_bindings MODULE torch_bindings.cpp
                    triangulation_bindings.cu)  # 只保留 bindings 文件
                   

# 设置模块属性
set_target_properties(torch_bindings PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    INTERPROCEDURAL_OPTIMIZATION OFF
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# 添加 triangulation 库的头文件路径
target_include_directories(torch_bindings PUBLIC 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/submodules/eigen
    ${CMAKE_CURRENT_SOURCE_DIR}  # 当前目录的头文件
)

# 链接 triangulation 库和其他依赖
target_link_libraries(torch_bindings PRIVATE 
    torch 
    ${TORCH_PYTHON_LIBRARY}
    triangulation  # 链接 triangulation 库
    CUDA::cudart
    CUDA::cuda_driver  # 如果需要
)

# 添加编译选项
target_compile_options(torch_bindings PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -O3 -Wall -Wno-unknown-pragmas>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>)

# 添加 triangulation 库的预编译宏（如果需要）
target_compile_definitions(torch_bindings PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:_GLIBCXX_USE_CXX11_ABI=0>
)

message(STATUS "TRIANGULATION_INSTALL_PREFIX: ${TRIANGULATION_INSTALL_PREFIX}")

# 安装目标
install(
  TARGETS torch_bindings
  COMPONENT torch_bindings
  LIBRARY DESTINATION ${TRIANGULATION_INSTALL_PREFIX}
  ARCHIVE DESTINATION ${TRIANGULATION_INSTALL_PREFIX} 
  RUNTIME DESTINATION ${TRIANGULATION_INSTALL_PREFIX})

# 生成并安装 __init__.py
configure_file(${CMAKE_SOURCE_DIR}/torch_bindings/triangulation/__init__.py.in
               ${CMAKE_BINARY_DIR}/__init__.py @ONLY)

install(FILES ${CMAKE_BINARY_DIR}/__init__.py
        DESTINATION ${TRIANGULATION_INSTALL_PREFIX})

