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

add_spirv_embed_library(spv_shaders_barycentrics
    raygen_barycentrics.rgen
    miss_barycentrics.rmiss
    hit_barycentrics.rchit
    occlusion_miss.rmiss
	COMPILE_OPTIONS -O
    INCLUDE_DIRECTORIES
        ${PROJECT_SOURCE_DIR}
    COMPILE_DEFINITIONS
        ${GLSL_COMPILE_DEFNS})

# Build the different benchmark backends
# defalt: pt (no suffix
# barycentrics: primary rays w/ barycentrics (_barycentrics)
# ao: ao w/ 4 AO samples per primary ray(_ao)
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

target_compile_definitions(crt_vulkan PRIVATE
    -DINCLUDE_SPV_SHADERS=\"spv_shaders_embedded_spv.h\"
    -DRAYGEN_SPV_NAME=raygen_spv
    -DMISS_SPV_NAME=miss_spv
    -DOCCLUSION_MISS_SPV_NAME=occlusion_miss_spv
    -DHIT_SPV_NAME=hit_spv
)

target_compile_options(crt_vulkan PUBLIC
    -DVK_ENABLE_BETA_EXTENSIONS=1)

target_link_libraries(crt_vulkan PUBLIC
	spv_shaders
    util
    display
    Vulkan::Vulkan)

# barycentrics
add_library(crt_vulkan_barycentrics MODULE
    render_vulkan_plugin.cpp
    render_vulkan.cpp
    vulkan_utils.cpp
    vulkanrt_utils.cpp
    vkdisplay.cpp
    imgui_impl_vulkan.cpp)

set_target_properties(crt_vulkan_barycentrics PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

if (REPORT_RAY_STATS)
	target_compile_options(crt_vulkan_barycentrics PUBLIC
		-DREPORT_RAY_STATS=1)
endif()

target_compile_definitions(crt_vulkan_barycentrics PRIVATE
    -DINCLUDE_SPV_SHADERS=\"spv_shaders_barycentrics_embedded_spv.h\"
    -DRAYGEN_SPV_NAME=raygen_barycentrics_spv
    -DMISS_SPV_NAME=miss_barycentrics_spv
    -DOCCLUSION_MISS_SPV_NAME=occlusion_miss_spv
    -DHIT_SPV_NAME=hit_barycentrics_spv
)

target_compile_options(crt_vulkan_barycentrics PUBLIC
    -DVK_ENABLE_BETA_EXTENSIONS=1)

target_link_libraries(crt_vulkan_barycentrics PUBLIC
	spv_shaders_barycentrics
    util
    display
    Vulkan::Vulkan)

install(TARGETS crt_vulkan crt_vulkan_barycentrics
    LIBRARY DESTINATION bin)

