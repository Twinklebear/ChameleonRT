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

set(DXR_VARIATION_SUFFIXES "" "_ng" "_ao")
set(DXR_VARIATION_DEFINES "" "-DDXR_NG" "-DDXR_AO")

foreach (VARIANT_SUFFIX VARIANT_DEFINE IN ZIP_LISTS DXR_VARIATION_SUFFIXES DXR_VARIATION_DEFINES)
    add_dxil_embed_library(dxr_shaders${VARIANT_SUFFIX}
        render_dxr${VARIANT_SUFFIX}.hlsl
        util.hlsl
        disney_bsdf.hlsl
        lcg_rng.hlsl
        COMPILE_OPTIONS -O3
        COMPILE_DEFINITIONS
            ${HLSL_COMPILE_DEFNS}
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
        ${VARIANT_DEFINE})

    target_include_directories(crt_dxr${VARIANT_SUFFIX} PUBLIC
        $<BUILD_INTERFACE:${D3D12_INCLUDE_DIRS}>)

    target_link_libraries(crt_dxr${VARIANT_SUFFIX} PUBLIC
        dxr_shaders
        util
        display
        ${D3D12_LIBRARIES})

    install(TARGETS crt_dxr${VARIANT_SUFFIX}
        LIBRARY DESTINATION bin)
endforeach()

