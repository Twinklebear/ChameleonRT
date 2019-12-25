find_package(embree 3 REQUIRED)
find_package(TBB REQUIRED)

include(cmake/ISPC.cmake)

set(ISPC_COMPILE_DEFNS "-O3;--opt=fast-math")
if (REPORT_RAY_STATS)
	set(ISPC_COMPILE_DEFNS "${ISPC_COMPILE_DEFNS};-DREPORT_RAY_STATS=1")
endif()

add_ispc_library(ispc_kernels render_embree.ispc
	INCLUDE_DIRECTORIES
        ${EMBREE_INCLUDE_DIRS}
        ${CMAKE_CURRENT_LIST_DIR}
        ${PROJECT_SOURCE_DIR}
	COMPILE_DEFINITIONS
        ${ISPC_COMPILE_DEFNS})

add_library(render_embree render_embree.cpp embree_utils.cpp)

set_target_properties(render_embree PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

if (REPORT_RAY_STATS)
	target_compile_options(render_embree PUBLIC
		-DREPORT_RAY_STATS=1)
endif()

target_include_directories(render_embree PUBLIC
	$<BUILD_INTERFACE:${EMBREE_INCLUDE_DIRS}>)

target_link_libraries(render_embree PUBLIC
	ispc_kernels util TBB::tbb ${EMBREE_LIBRARIES})

