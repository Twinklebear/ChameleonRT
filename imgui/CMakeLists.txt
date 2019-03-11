add_library(imgui imgui.cpp imgui_demo.cpp imgui_draw.cpp
	imgui_widgets.cpp)

target_include_directories(imgui PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)

set_target_properties(imgui PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

