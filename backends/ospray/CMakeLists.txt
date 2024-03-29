option(ENABLE_OSPRAY "Build the OSPRay rendering backend. Requires OSPRay" OFF)

if (NOT ENABLE_OSPRAY)
    return()
endif()

find_package(ospray REQUIRED)
find_package(TBB REQUIRED)

add_library(crt_ospray MODULE
    render_ospray_plugin.cpp
    render_ospray.cpp)

set_target_properties(crt_ospray PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)

target_link_libraries(crt_ospray PUBLIC
    util
    display
    ospray::ospray
    TBB::tbb)

install(TARGETS crt_ospray
    LIBRARY DESTINATION bin)

crt_add_packaged_dependency(ospray::ospray)
crt_add_packaged_dependency(ospray::ospray_module_ispc)
crt_add_packaged_dependency(openvkl::openvkl)
crt_add_packaged_dependency(rkcommon::rkcommon)
crt_add_packaged_dependency(embree)
crt_add_packaged_dependency(TBB::tbb)

