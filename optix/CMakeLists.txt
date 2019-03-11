find_package(CUDA 10 REQUIRED)
find_package(OptiX 6 REQUIRED)

include(cmake/Bin2C.cmake)
add_ptx_embed_library(optix_kernels render_optix.cu
	INCLUDE_DIRECTORIES ${OptiX_INCLUDE}
	COMPILE_DEFINITIONS --use_fast_math)

add_library(render_optix render_optix.cpp)

set_target_properties(render_optix PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

target_include_directories(render_optix PUBLIC
	$<BUILD_INTERFACE:${OptiX_INCLUDE}>
	$<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>)

target_link_libraries(render_optix PUBLIC
	optix_kernels optix util ${CUDA_LIBRARIES})

