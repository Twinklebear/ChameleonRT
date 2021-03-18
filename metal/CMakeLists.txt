
include(${CMAKE_CURRENT_LIST_DIR}/cmake/MetalShader.cmake)

find_library(QUARTZ_CORE QuartzCore)
if (NOT QUARTZ_CORE)
    message(FATAL_ERROR "QuartzCore not found")
endif()

find_library(METAL_LIB Metal)
if (NOT METAL_LIB)
    message(FATAL_ERROR "Metal not found")
endif()

find_library(COCOA_LIB Cocoa)
if (NOT COCOA_LIB)
    message(FATAL_ERROR "Cocoa not found")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=objc-method-access -fobjc-arc -fobjc-weak")

set(METAL_COMPILE_DEFNS "")
if (REPORT_RAY_STATS)
    set(METAL_COMPILE_DEFNS "REPORT_RAY_STATS=1")
endif()

add_metal_embed_library(metal_shaders
    render_metal.metal
    disney_bsdf.metal
    lcg_rng.metal
    lights.metal
    util.metal
    shader_types.h
    INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_LIST_DIR}
    ${PROJECT_SOURCE_DIR}
    COMPILE_DEFINITIONS
    ${METAL_COMPILE_DEFNS}
    COMPILE_OPTIONS
    -O3 -ffast-math)

add_metal_embed_library(metaldisplay_shaders
    metaldisplay.metal)

add_library(render_metal
    render_metal.mm
    metaldisplay.mm
    metalrt_utils.mm
    imgui_impl_metal.mm)

set_target_properties(render_metal PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)

target_link_libraries(render_metal
    PUBLIC
    util
    ${QUARTZ_CORE}
    ${METAL_LIB}
    ${COCOA_LIB}
    metal_shaders
    metaldisplay_shaders)

if (REPORT_RAY_STATS)
	target_compile_options(render_metal PUBLIC
		-DREPORT_RAY_STATS=1)
endif()

