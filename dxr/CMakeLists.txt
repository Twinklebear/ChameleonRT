set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(D3D12 REQUIRED)

add_dxil_embed_library(shaders render_dxr.hlsl
	COMPILE_OPTIONS -Zi)

add_library(render_dxr render_dxr.cpp dx12_utils.cpp dxr_utils.cpp)

set_target_properties(render_dxr PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

target_include_directories(render_dxr PUBLIC
	$<BUILD_INTERFACE:${D3D12_INCLUDE_PATH}>)

target_link_libraries(render_dxr PUBLIC
	shaders util ${D3D12_LIBRARIES})

