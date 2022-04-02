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

add_dxil_embed_library(dxr_shaders render_dxr.hlsl util.hlsl
	disney_bsdf.hlsl lcg_rng.hlsl
	COMPILE_OPTIONS -O3
	COMPILE_DEFINITIONS
        ${HLSL_COMPILE_DEFNS}
	INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_LIST_DIR}
        ${PROJECT_SOURCE_DIR})

add_library(crt_dxr MODULE
    render_dxr_plugin.cpp
    render_dxr.cpp
    dxdisplay.cpp
    dx12_utils.cpp
    dxr_utils.cpp
    imgui_impl_dx12.cpp)

set_target_properties(crt_dxr PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

if (REPORT_RAY_STATS)
	target_compile_options(crt_dxr PUBLIC
		-DREPORT_RAY_STATS=1)
endif()

target_include_directories(crt_dxr PUBLIC
	$<BUILD_INTERFACE:${D3D12_INCLUDE_DIRS}>)

target_link_libraries(crt_dxr PUBLIC
	dxr_shaders
    util
    display
    ${D3D12_LIBRARIES})

if (ENABLE_PIX_RUNTIME)
    target_compile_options(crt_dxr PUBLIC
        -DUSE_PIX=1
        -DENABLE_PIX_RUNTIME=1)

    target_link_libraries(crt_dxr PUBLIC
        PIX::WinPixEventRuntime)
endif()

install(TARGETS crt_dxr
    LIBRARY DESTINATION bin)

