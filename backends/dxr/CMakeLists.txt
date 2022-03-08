cmake_dependent_option(ENABLE_DXR "Build the DXR rendering backend" OFF
    "WIN32" OFF)

if (NOT ENABLE_DXR)
    return()
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(D3D12 REQUIRED)

set(HLSL_COMPILE_DEFNS "")
if (REPORT_RAY_STATS)
	set(HLSL_COMPILE_DEFNS "REPORT_RAY_STATS=1")
endif()

set(DXR_VARIANT_SUFFIXES
    ""
    "_ng1"
    "_ng2"
    "_ng4"
    "_ng8"
    "_ng16"
    "_ng32"
    "_ng64"
    "_ao1"
    "_ao2"
    "_ao4"
    "_ao8"
    "_ao16"
    "_ao32"
    "_ao64"
)

set(DXR_SHADER_VARIANT_SUFFIXES
    ""
    "_ng"
    "_ng"
    "_ng"
    "_ng"
    "_ng"
    "_ng"
    "_ng"
    "_ao"
    "_ao"
    "_ao"
    "_ao"
    "_ao"
    "_ao"
    "_ao"
)

set(DXR_VARIANT_DEFINES
    ""
    "-DDXR_NG"
    "-DDXR_NG"
    "-DDXR_NG"
    "-DDXR_NG"
    "-DDXR_NG"
    "-DDXR_NG"
    "-DDXR_NG"
    "-DDXR_AO"
    "-DDXR_AO"
    "-DDXR_AO"
    "-DDXR_AO"
    "-DDXR_AO"
    "-DDXR_AO"
    "-DDXR_AO"
)

set(DXR_SHADER_VARIANT_DEFINES
    ""
    "NUM_PIXEL_SAMPLES=1"
    "NUM_PIXEL_SAMPLES=2"
    "NUM_PIXEL_SAMPLES=4"
    "NUM_PIXEL_SAMPLES=8"
    "NUM_PIXEL_SAMPLES=16"
    "NUM_PIXEL_SAMPLES=32"
    "NUM_PIXEL_SAMPLES=64"
    "NUM_AO_SAMPLES=1"
    "NUM_AO_SAMPLES=2"
    "NUM_AO_SAMPLES=4"
    "NUM_AO_SAMPLES=8"
    "NUM_AO_SAMPLES=16"
    "NUM_AO_SAMPLES=32"
    "NUM_AO_SAMPLES=64"
)

foreach (VARIANT_SUFFIX SHADER_VARIANT_SUFFIX VARIANT_DEFINE SHADER_VARIANT_DEFINE
        IN ZIP_LISTS DXR_VARIANT_SUFFIXES DXR_SHADER_VARIANT_SUFFIXES DXR_VARIANT_DEFINES DXR_SHADER_VARIANT_DEFINES)

    file(COPY_FILE ${CMAKE_CURRENT_SOURCE_DIR}/render_dxr${SHADER_VARIANT_SUFFIX}.hlsl
        ${CMAKE_CURRENT_BINARY_DIR}/render_dxr${VARIANT_SUFFIX}.hlsl)

    add_dxil_embed_library(dxr_shaders${VARIANT_SUFFIX}
        ${CMAKE_CURRENT_BINARY_DIR}/render_dxr${VARIANT_SUFFIX}.hlsl
        util.hlsl
        disney_bsdf.hlsl
        lcg_rng.hlsl
        COMPILE_OPTIONS -O3
        COMPILE_DEFINITIONS
            ${HLSL_COMPILE_DEFNS}
            ${SHADER_VARIANT_DEFINE}
        INCLUDE_DIRECTORIES
            ${CMAKE_CURRENT_LIST_DIR}
            ${PROJECT_SOURCE_DIR})


    add_library(crt_dxr${VARIANT_SUFFIX} MODULE
        render_dxr_plugin.cpp
        render_dxr.cpp
        dxdisplay.cpp
        dx12_utils.cpp
        dxr_utils.cpp
        imgui_impl_dx12.cpp)

    set_target_properties(crt_dxr${VARIANT_SUFFIX} PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON)

    if (REPORT_RAY_STATS)
        target_compile_options(crt_dxr${VARIANT_SUFFIX} PUBLIC
            -DREPORT_RAY_STATS=1)
    endif()

    target_compile_definitions(crt_dxr${VARIANT_SUFFIX} PUBLIC
        ${VARIANT_DEFINE}
        -D${SHADER_VARIANT_DEFINE}
        -DDXIL_EMBED_INCLUDE="render_dxr${VARIANT_SUFFIX}_embedded_dxil.h"
        -DSHADER_BYTECODE_NAME=render_dxr${VARIANT_SUFFIX}_dxil)

    target_include_directories(crt_dxr${VARIANT_SUFFIX} PUBLIC
        $<BUILD_INTERFACE:${D3D12_INCLUDE_DIRS}>)

    target_link_libraries(crt_dxr${VARIANT_SUFFIX} PUBLIC
        dxr_shaders${VARIANT_SUFFIX}
        util
        display
        ${D3D12_LIBRARIES})

    install(TARGETS crt_dxr${VARIANT_SUFFIX}
        LIBRARY DESTINATION bin)
endforeach()

