cmake_dependent_option(ENABLE_VULKAN "Build the Vulkan rendering backend. Requires Vulkan" OFF
    "NOT APPLE" OFF)

if (NOT ENABLE_VULKAN)
    return()
endif()

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

add_library(crt_vulkan MODULE
    render_vulkan_plugin.cpp
    render_vulkan.cpp
    vulkan_utils.cpp
    vulkanrt_utils.cpp
    vkdisplay.cpp
    imgui_impl_vulkan.cpp)

set_target_properties(crt_vulkan PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

if (REPORT_RAY_STATS)
	target_compile_options(crt_vulkan PUBLIC
		-DREPORT_RAY_STATS=1)
endif()

target_compile_options(crt_vulkan PUBLIC
    -DVK_ENABLE_BETA_EXTENSIONS=1)

target_link_libraries(crt_vulkan PUBLIC
	spv_shaders
    util
    display
    Vulkan::Vulkan)

install(TARGETS crt_vulkan
    LIBRARY DESTINATION bin)

