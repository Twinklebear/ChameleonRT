cmake_dependent_option(ENABLE_OPTIX "Build the OptiX rendering backend. Requires OptiX and CUDA" OFF
    "NOT APPLE" OFF)

if (NOT ENABLE_OPTIX)
    return()
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(CUDA 11 REQUIRED)
find_package(OptiX 7 REQUIRED)

include(cmake/Bin2C.cmake)

set(CUDA_COMPILE_DEFNS "-lineinfo;-O3;--use_fast_math")
if (REPORT_RAY_STATS)
    set(CUDA_COMPILE_DEFNS "${CUDA_COMPILE_DEFNS};-DREPORT_RAY_STATS=1")
endif()

add_ptx_embed_library(optix_kernels_pt render_optix.cu
    INCLUDE_DIRECTORIES
        ${OptiX_INCLUDE_DIR} 
        ${PROJECT_SOURCE_DIR}
    COMPILE_DEFINITIONS
        ${CUDA_COMPILE_DEFNS}
        # These options don't seem to do print any warnings?
        -Xptxas --warn-on-double-precision-use
        -Xptxas -Werror)

add_ptx_embed_library(optix_kernels_barycentrics render_optix_barycentrics.cu
    INCLUDE_DIRECTORIES
        ${OptiX_INCLUDE_DIR} 
        ${PROJECT_SOURCE_DIR}
    COMPILE_DEFINITIONS
        ${CUDA_COMPILE_DEFNS}
        # These options don't seem to do print any warnings?
        -Xptxas --warn-on-double-precision-use
        -Xptxas -Werror)

add_ptx_embed_library(optix_kernels_ao render_optix_ao.cu
    INCLUDE_DIRECTORIES
        ${OptiX_INCLUDE_DIR} 
        ${PROJECT_SOURCE_DIR}
    COMPILE_DEFINITIONS
        ${CUDA_COMPILE_DEFNS}
        # These options don't seem to do print any warnings?
        -Xptxas --warn-on-double-precision-use
        -Xptxas -Werror)

# Build the different benchmark versions of the backend:
# pathtracing (no suffix, default)
# barycentrics
# ao
add_library(crt_optix MODULE
    render_optix_plugin.cpp
    render_optix.cpp
    optix_utils.cpp)

set_target_properties(crt_optix PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)

if (REPORT_RAY_STATS)
    target_compile_options(crt_optix PUBLIC
        -DREPORT_RAY_STATS=1)
endif()

target_include_directories(crt_optix PUBLIC
    $<BUILD_INTERFACE:${OptiX_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>)

target_compile_definitions(crt_optix PRIVATE
    -DINCLUDE_PTX_KERNELS=\"render_optix_embedded_ptx.h\"
    -DINCLUDE_PTX_VAR=render_optix_ptx
)

target_link_libraries(crt_optix PUBLIC
    optix_kernels_pt
    util
    glad
    display
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY})

add_library(crt_optix_barycentrics MODULE
    render_optix_plugin.cpp
    render_optix.cpp
    optix_utils.cpp)

set_target_properties(crt_optix_barycentrics PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)

if (REPORT_RAY_STATS)
    target_compile_options(crt_optix_barycentrics PUBLIC
        -DREPORT_RAY_STATS=1)
endif()

target_include_directories(crt_optix_barycentrics PUBLIC
    $<BUILD_INTERFACE:${OptiX_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>)

target_compile_definitions(crt_optix_barycentrics PRIVATE
    -DINCLUDE_PTX_KERNELS=\"render_optix_barycentrics_embedded_ptx.h\"
    -DINCLUDE_PTX_VAR=render_optix_barycentrics_ptx
)

target_link_libraries(crt_optix_barycentrics PUBLIC
    optix_kernels_barycentrics
    util
    glad
    display
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY})

add_library(crt_optix_ao MODULE
    render_optix_plugin.cpp
    render_optix.cpp
    optix_utils.cpp)

set_target_properties(crt_optix_ao PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)

if (REPORT_RAY_STATS)
    target_compile_options(crt_optix_ao PUBLIC
        -DREPORT_RAY_STATS=1)
endif()

target_include_directories(crt_optix_ao PUBLIC
    $<BUILD_INTERFACE:${OptiX_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>)

target_compile_definitions(crt_optix_ao PRIVATE
    -DINCLUDE_PTX_KERNELS=\"render_optix_ao_embedded_ptx.h\"
    -DINCLUDE_PTX_VAR=render_optix_ao_ptx
)

target_link_libraries(crt_optix_ao PUBLIC
    optix_kernels_ao
    util
    glad
    display
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY})

install(TARGETS crt_optix crt_optix_barycentrics crt_optix_ao
    LIBRARY DESTINATION bin)

