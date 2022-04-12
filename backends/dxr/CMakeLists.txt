cmake_dependent_option(ENABLE_DXR "Build the DXR rendering backend" OFF
    "WIN32" OFF)

if (NOT ENABLE_DXR)
    return()
endif()

cmake_dependent_option(ENABLE_PIX_RUNTIME
    "Build support for WinPixEventRuntime tracing. Set WinPixEventRuntime_DIR" OFF
    "ENABLE_DXR" OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(D3D12 REQUIRED)

if (ENABLE_PIX_RUNTIME)
    find_package(WinPixEventRuntime REQUIRED)
endif()

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
    "_ao1"
    "_ao2"
    "_ao4"
    "_ao8"
    "_ao16"
    "_ao32"
    "_ao_tailrec1"
    "_ao_tailrec2"
    "_ao_tailrec4"
    "_ao_tailrec8"
    "_ao_tailrec16"
    "_ao_tailrec32"
)

set(DXR_SHADER_VARIANT_SUFFIXES
    ""
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
    "_ao_tailrec"
    "_ao_tailrec"
    "_ao_tailrec"
    "_ao_tailrec"
    "_ao_tailrec"
    "_ao_tailrec"
)

set(DXR_VARIANT_DEFINES
    ""
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
    "-DDXR_AO_TAILREC"
    "-DDXR_AO_TAILREC"
    "-DDXR_AO_TAILREC"
    "-DDXR_AO_TAILREC"
    "-DDXR_AO_TAILREC"
    "-DDXR_AO_TAILREC"
)

set(DXR_SHADER_VARIANT_DEFINES
    ""
    "NUM_SAMPLES=1"
    "NUM_SAMPLES=2"
    "NUM_SAMPLES=4"
    "NUM_SAMPLES=8"
    "NUM_SAMPLES=16"
    "NUM_SAMPLES=32"
    "NUM_SAMPLES=1"
    "NUM_SAMPLES=2"
    "NUM_SAMPLES=4"
    "NUM_SAMPLES=8"
    "NUM_SAMPLES=16"
    "NUM_SAMPLES=32"
    "NUM_SAMPLES=1"
    "NUM_SAMPLES=2"
    "NUM_SAMPLES=4"
    "NUM_SAMPLES=8"
    "NUM_SAMPLES=16"
    "NUM_SAMPLES=32"
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

    if (ENABLE_PIX_RUNTIME)
        target_compile_options(crt_dxr${VARIANT_SUFFIX} PUBLIC
            -DUSE_PIX=1
            -DENABLE_PIX_RUNTIME=1)

        target_link_libraries(crt_dxr${VARIANT_SUFFIX} PUBLIC
            PIX::WinPixEventRuntime)
    endif()

    install(TARGETS crt_dxr${VARIANT_SUFFIX}
        LIBRARY DESTINATION bin)
endforeach()

