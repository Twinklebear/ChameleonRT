set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(CUDA 10 REQUIRED)
find_package(OptiX 7 REQUIRED)

include(cmake/Bin2C.cmake)

set(CUDA_COMPILE_DEFNS "-O3;--use_fast_math")
if (REPORT_RAY_STATS)
    set(CUDA_COMPILE_DEFNS "${CUDA_COMPILE_DEFNS};-DREPORT_RAY_STATS=1")
endif()

add_ptx_embed_library(optix_kernels render_optix.cu
    INCLUDE_DIRECTORIES
        ${OptiX_INCLUDE_DIR} 
        ${PROJECT_SOURCE_DIR}
    COMPILE_DEFINITIONS
        ${CUDA_COMPILE_DEFNS}
        # These options don't seem to do print any warnings?
        -Xptxas --warn-on-double-precision-use
        -Xptxas -Werror)

add_library(render_optix render_optix.cpp optix_utils.cpp)

set_target_properties(render_optix PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)

if (REPORT_RAY_STATS)
    target_compile_options(render_optix PUBLIC
        -DREPORT_RAY_STATS=1)
endif()

target_include_directories(render_optix PUBLIC
    $<BUILD_INTERFACE:${OptiX_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>)

target_link_libraries(render_optix PUBLIC
    optix_kernels
    util
    glad
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY})

