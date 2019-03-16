add_library(util
	arcball_camera.cpp
	imgui_impl_sdl.cpp
	imgui_impl_opengl3.cpp
	shader.cpp
	util.cpp
	gl_core_4_5.c)

set_target_properties(util PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

target_include_directories(util PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
	$<BUILD_INTERFACE:${SDL2_INCLUDE_DIR}>
	$<BUILD_INTERFACE:${OPENGL_INCLUDE_DIR}>
	$<BUILD_INTERFACE:${GLM_INCLUDE_DIRS}>)

target_link_libraries(util PUBLIC
	imgui
	${SDL2_LIBRARY}
	${OPENGL_LIBRARIES})

