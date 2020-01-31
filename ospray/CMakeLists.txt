find_package(ospray REQUIRED)
find_package(TBB REQUIRED)

add_library(render_ospray render_ospray.cpp)

set_target_properties(render_ospray PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)

target_link_libraries(render_ospray PUBLIC
    util ospray::ospray TBB::tbb)

option(ENABLE_OSPRAY_DEVICE "Build the OSPRay API device module for Chameleon backend. \
Requires OSPRay, ospcommon and one other backend" OFF)

if (ENABLE_OSPRAY_DEVICE)
    add_subdirectory(device)
endif()

