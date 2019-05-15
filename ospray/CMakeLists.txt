find_package(ospray REQUIRED)

# TODO WILL: Once OSPRay 2.0 is released we'll only support 2+
# so it's fine to ignore the minor & patch versions here
if ("${OSPRAY_VERSION}" STREQUAL "2.0.0")
	add_definitions(-DOSPRAY_VERSION_MAJOR=2)
else()
	add_definitions(-DOSPRAY_VERSION_MAJOR=1)
endif()

add_library(render_ospray render_ospray.cpp)

set_target_properties(render_ospray PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

target_link_libraries(render_ospray util ospray::ospray)

