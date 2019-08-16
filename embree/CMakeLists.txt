find_package(embree 3 REQUIRED)
find_package(TBB REQUIRED)

include(cmake/ISPC.cmake)
add_ispc_library(ispc_kernels render_embree.ispc
	INCLUDE_DIRECTORIES ${EMBREE_INCLUDE_DIRS} ${CMAKE_CURRENT_LIST_DIR}
	COMPILE_DEFINITIONS -g -O0)

add_library(render_embree render_embree.cpp embree_utils.cpp)

set_target_properties(render_embree PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

target_include_directories(render_embree PUBLIC
	$<BUILD_INTERFACE:${EMBREE_INCLUDE_DIRS}>)

target_link_libraries(render_embree PUBLIC
	ispc_kernels util TBB::tbb ${EMBREE_LIBRARIES})

