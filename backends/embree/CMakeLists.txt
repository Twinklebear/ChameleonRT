option(ENABLE_EMBREE "Build the Embree + TBB + ISPC rendering backend. Requires Embree, TBB and ISPC" OFF)

if (NOT ENABLE_EMBREE)
    return()
endif()

find_package(embree 4 REQUIRED)
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
	COMPILE_DEFINITIONS
        ${ISPC_COMPILE_DEFNS})

add_library(crt_embree MODULE
    render_embree_plugin.cpp
    render_embree.cpp
    embree_utils.cpp)

set_target_properties(crt_embree PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

if (REPORT_RAY_STATS)
	target_compile_options(crt_embree PUBLIC
		-DREPORT_RAY_STATS=1)
endif()

target_link_libraries(crt_embree PUBLIC
	ispc_kernels
    util
    display
    TBB::tbb
    embree)

install(TARGETS crt_embree
    LIBRARY DESTINATION bin)

crt_add_packaged_dependency(embree)
crt_add_packaged_dependency(TBB::tbb)

