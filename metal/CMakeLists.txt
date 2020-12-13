
include(${CMAKE_CURRENT_LIST_DIR}/cmake/MetalShader.cmake)

find_library(QUARTZ_CORE QuartzCore)
if (NOT QUARTZ_CORE)
    message(FATAL_ERROR "QuartzCore not found")
endif()

find_library(METAL_LIB Metal)
if (NOT METAL_LIB)
    message(FATAL_ERROR "Metal not found")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=objc-method-access")

add_metal_embed_library(metal_shaders render_metal.metal shader_types.h
    INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_LIST_DIR}
        ${PROJECT_SOURCE_DIR})

add_library(render_metal
    render_metal.mm)

set_target_properties(render_metal PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)

target_link_libraries(render_metal
    PUBLIC
    util
    ${QUARTZ_CORE}
    ${METAL_LIB}
    metal_shaders)

