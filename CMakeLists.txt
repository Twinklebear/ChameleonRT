cmake_minimum_required(VERSION 3.5)
project(chameleonrt)

if (NOT WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
endif()

add_definitions(-DNOMINMAX -DGLM_ENABLE_EXPERIMENTAL -DSDL_MAIN_HANDLED
    -DWIN32_LEAN_AND_MEAN)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(Threads REQUIRED)
find_package(glm REQUIRED)
find_package(SDL2 CONFIG REQUIRED)

add_subdirectory(imgui)
add_subdirectory(util)

option(ENABLE_OSPRAY "Build the OSPRay rendering backend. Requires OSPRay" OFF)
option(ENABLE_EMBREE "Build the Embree + TBB + ISPC rendering backend. Requires Embree, TBB and ISPC" OFF)

include(CMakeDependentOption)

cmake_dependent_option(ENABLE_OPTIX "Build the OptiX rendering backend. Requires OptiX and CUDA" OFF
    "NOT APPLE" OFF)

cmake_dependent_option(ENABLE_VULKAN "Build the Vulkan rendering backend. Requires Vulkan" OFF
    "NOT APPLE" OFF)

cmake_dependent_option(ENABLE_DXR "Build the DXR rendering backend" OFF
    "WIN32" OFF)

cmake_dependent_option(ENABLE_METAL "Build the Metal rendering backend" OFF
    "APPLE" OFF)

option(REPORT_RAY_STATS "Track and report rays/second. May incur a slight rendering performance penalty" OFF)

if (ENABLE_OSPRAY)
	add_definitions(-DENABLE_OSPRAY)
	add_subdirectory(ospray)
endif()

if (ENABLE_EMBREE)
	add_definitions(-DENABLE_EMBREE)
	add_subdirectory(embree)
endif()

if (ENABLE_OPTIX)
    add_definitions(-DENABLE_OPTIX)
    add_subdirectory(optix)
endif()

if (ENABLE_VULKAN)
    add_definitions(-DENABLE_VULKAN)
    add_subdirectory(vulkan)
endif()

if (ENABLE_DXR)
    add_definitions(-DENABLE_DXR)
    add_subdirectory(dxr)
endif()

if (ENABLE_METAL)
    add_definitions(-DENABLE_METAL)
    add_subdirectory(metal)
endif()

if (NOT ENABLE_OSPRAY AND NOT ENABLE_EMBREE AND NOT ENABLE_OPTIX
        AND NOT ENABLE_VULKAN AND NOT ENABLE_DXR AND NOT ENABLE_METAL)
    message(FATAL_ERROR
        "At least one backend (OSPRay, Embree, OptiX, Vulkan DXR, Metal) must be selected\n"
        "Enable at least one of: ENABLE_OSPRAY, ENABLE_EMBREE, ENABLE_OPTIX, ENABLE_VULKAN, ENABLE_DXR, ENABLE_METAL") 
endif()

add_executable(chameleonrt main.cpp)

set_target_properties(chameleonrt PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

target_link_libraries(chameleonrt PUBLIC
    util
    display)

if (ENABLE_OSPRAY)
	target_link_libraries(chameleonrt PUBLIC render_ospray)
endif()

if (ENABLE_EMBREE)
	target_link_libraries(chameleonrt PUBLIC render_embree)
endif()

if (ENABLE_OPTIX)
	target_link_libraries(chameleonrt PUBLIC render_optix)
endif()

if (ENABLE_VULKAN)
	target_link_libraries(chameleonrt PUBLIC render_vulkan)
endif()

if (ENABLE_DXR)
	target_link_libraries(chameleonrt PUBLIC render_dxr)
endif()

if (ENABLE_METAL)
    target_link_libraries(chameleonrt PUBLIC render_metal)
endif()

