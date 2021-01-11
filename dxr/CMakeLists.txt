set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(D3D12 REQUIRED)

option(AO_RENDERER "Build the AO renderer instead of path tracer" OFF)
option(OGCH_SHADOWS "Use the Occlusion Group with Closest Hit Method" OFF)
option(OGAH_SHADOWS "Use the Occlusion Group with Any Hit + Skip CH Method" OFF)

if (OGCH_SHADOWS AND OGAH_SHADOWS)
    message(ERROR "OGCH and OGAH are mutually exclusive")
endif()

set(HLSL_COMPILE_DEFNS "")
if (REPORT_RAY_STATS)
	set(HLSL_COMPILE_DEFNS "REPORT_RAY_STATS=1")
endif()

if (OGCH_SHADOWS)
    set(HLSL_COMPILE_DEFNS "${HLSL_COMPILE_DEFNS};OGCH_SHADOWS=1")
endif()

if (OGAH_SHADOWS)
    set(HLSL_COMPILE_DEFNS "${HLSL_COMPILE_DEFNS};OGAH_SHADOWS=1")
endif()

add_dxil_embed_library(dxr_shaders render_dxr.hlsl util.hlsl
	disney_bsdf.hlsl lcg_rng.hlsl
	COMPILE_OPTIONS -O3
	COMPILE_DEFINITIONS
        ${HLSL_COMPILE_DEFNS}
	INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_LIST_DIR}
        ${PROJECT_SOURCE_DIR})

add_library(render_dxr
    render_dxr.cpp
    dxdisplay.cpp
    dx12_utils.cpp
    dxr_utils.cpp
    imgui_impl_dx12.cpp)

set_target_properties(render_dxr PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

if (REPORT_RAY_STATS)
	target_compile_options(render_dxr PUBLIC
		-DREPORT_RAY_STATS=1)
endif()

if (AO_RENDERER)
	target_compile_options(render_dxr PUBLIC
        -DAO_RENDERER=1)
endif()

if (OGCH_SHADOWS)
    target_compile_options(render_dxr PUBLIC
        -DOGCH_SHADOWS=1)
endif()

if (OGAH_SHADOWS)
    target_compile_options(render_dxr PUBLIC
        -DOGAH_SHADOWS=1)
endif()

target_include_directories(render_dxr PUBLIC
	$<BUILD_INTERFACE:${D3D12_INCLUDE_DIRS}>)

target_link_libraries(render_dxr PUBLIC
	dxr_shaders util ${D3D12_LIBRARIES})

