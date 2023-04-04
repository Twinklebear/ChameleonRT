option(ENABLE_EMBREE_SYCL
    "Build the Embree4 + SYCL rendering backend. Requires Embree4, SYCL and TBB" OFF)

if (NOT ENABLE_EMBREE_SYCL)
    return()
endif()

find_package(embree 4 REQUIRED)
find_package(TBB REQUIRED)

# Build benchmark versions for the regular path tracer,
# barycentrics and AO versions

# regular path tracer
add_library(crt_embree_sycl MODULE
    render_embree_plugin.cpp
    render_embree.cpp
    embree_utils.cpp
    sycl_utils.cpp)

set_target_properties(crt_embree_sycl PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON)

if (REPORT_RAY_STATS)
	target_compile_options(crt_embree_sycl PUBLIC
		-DREPORT_RAY_STATS=1)
endif()

target_compile_options(crt_embree_sycl PRIVATE
    -fsycl
    -fsycl-unnamed-lambda
    -Xclang=-fsycl-allow-func-ptr
    -fsycl-targets=spir64)

target_link_libraries(crt_embree_sycl PUBLIC
    util
    display
    TBB::tbb
    embree)

target_link_options(crt_embree_sycl PRIVATE
    -fsycl
    -fsycl-targets=spir64
    -Xsycl-target-backend=spir64
    -cl-intel-greater-than-4GB-buffer-required
    "-options -cl-intel-force-global-mem-allocation -cl-intel-no-local-to-generic -O2")

install(TARGETS crt_embree_sycl
    LIBRARY DESTINATION bin)

crt_add_packaged_dependency(embree)

