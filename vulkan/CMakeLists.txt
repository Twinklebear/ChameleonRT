set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(Vulkan REQUIRED)

add_spirv_embed_library(spv_shaders raygen.rgen miss.rmiss hit.rchit
	COMPILE_OPTIONS -g)

add_library(render_vulkan render_vulkan.cpp)

set_target_properties(render_vulkan PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

if (REPORT_RAY_STATS)
	target_compile_options(render_vulkan PUBLIC
		-DREPORT_RAY_STATS=1)
endif()

target_link_libraries(render_vulkan PUBLIC
	spv_shaders util Vulkan::Vulkan)

