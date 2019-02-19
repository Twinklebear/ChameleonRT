cmake_minimum_required(VERSION 3.5)
project(rtobj)

if (NOT WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
else()
	message(FATAL_ERROR "no windows for now")
endif()

add_definitions(-DNOMINMAX -DGLM_ENABLE_EXPERIMENTAL)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(Threads REQUIRED)
find_package(glm REQUIRED)
find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(ospray REQUIRED)

add_subdirectory(imgui)

add_executable(rtobj main.cpp
	render_ospray.cpp
	arcball_camera.cpp
	imgui_impl_sdl.cpp
	imgui_impl_opengl3.cpp
	shader.cpp
	gl_core_4_5.c)

set_target_properties(rtobj PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

target_include_directories(rtobj PUBLIC
	$<BUILD_INTERFACE:${OSPRAY_INCLUDE_DIR}>
	$<BUILD_INTERFACE:${SDL2_INCLUDE_DIR}>
	$<BUILD_INTERFACE:${OPENGL_INCLUDE_DIR}>)

target_link_libraries(rtobj PUBLIC
	imgui ${OSPRAY_LIBRARIES} ${SDL2_LIBRARY} ${OPENGL_LIBRARIES})

