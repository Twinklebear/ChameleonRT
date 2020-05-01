set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(Vulkan REQUIRED)

set(GLSL_COMPILE_DEFNS "")
if (REPORT_RAY_STATS)
	set(GLSL_COMPILE_DEFNS "REPORT_RAY_STATS=1")
endif()

add_spirv_embed_library(spv_shaders raygen.rgen miss.rmiss hit.rchit
    occlusion_miss.rmiss
	COMPILE_OPTIONS -O
    INCLUDE_DIRECTORIES
        ${PROJECT_SOURCE_DIR}
    COMPILE_DEFINITIONS
        ${GLSL_COMPILE_DEFNS})

add_library(render_vulkan
    render_vulkan.cpp
    vulkan_utils.cpp
    vulkanrt_utils.cpp
    vkdisplay.cpp
    imgui_impl_vulkan.cpp)

set_target_properties(render_vulkan PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

if (REPORT_RAY_STATS)
	target_compile_options(render_vulkan PUBLIC
		-DREPORT_RAY_STATS=1)
endif()

target_compile_options(render_vulkan PUBLIC
    -DVK_ENABLE_BETA_EXTENSIONS=1)

target_link_libraries(render_vulkan PUBLIC
	spv_shaders util Vulkan::Vulkan)

