cmake_minimum_required(VERSION 3.5)
project(rtobj)

if (NOT WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
endif()

add_definitions(-DNOMINMAX -DGLM_ENABLE_EXPERIMENTAL -DSDL_MAIN_HANDLED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(Threads REQUIRED)
find_package(glm REQUIRED)
find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)

add_subdirectory(imgui)
add_subdirectory(util)

if (${ENABLE_OSPRAY})
	add_definitions(-DENABLE_OSPRAY)
	add_subdirectory(ospray)
endif()

if (${ENABLE_EMBREE})
	add_definitions(-DENABLE_EMBREE)
	add_subdirectory(embree)
endif()

if (${ENABLE_OPTIX})
	add_definitions(-DENABLE_OPTIX)
	add_subdirectory(optix)
endif()

if (WIN32 AND ${ENABLE_DXR})
	add_definitions(-DENABLE_DXR)
	add_subdirectory(dxr)
endif()

if (NOT ENABLE_OSPRAY AND NOT ENABLE_EMBREE AND NOT ENABLE_OPTIX)
	message(FATAL_ERROR "At least one rendering backend (OSPRAY, EMBREE, OPTIX) must be selected\n"
		"Enable at least one of: ENABLE_OSPRAY, ENABLE_EMBREE, ENABLE_OPTIX") 
endif()

add_executable(rtobj main.cpp)

set_target_properties(rtobj PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

target_link_libraries(rtobj PUBLIC util)

if (${ENABLE_OSPRAY})
	target_link_libraries(rtobj PUBLIC render_ospray)
endif()

if (${ENABLE_EMBREE})
	target_link_libraries(rtobj PUBLIC render_embree)
endif()

if (${ENABLE_OPTIX})
	target_link_libraries(rtobj PUBLIC render_optix)
endif()

if (WIN32 AND ${ENABLE_DXR})
	target_link_libraries(rtobj PUBLIC render_dxr)
endif()

