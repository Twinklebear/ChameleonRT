find_package(ospray REQUIRED)
find_package(TBB REQUIRED)

add_library(render_ospray MODULE
    render_ospray_plugin.cpp
    render_ospray.cpp)

set_target_properties(render_ospray PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)

target_link_libraries(render_ospray PUBLIC
    util
    display
    ospray::ospray
    TBB::tbb)

