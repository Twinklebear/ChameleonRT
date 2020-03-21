add_subdirectory(display)

add_library(util
    arcball_camera.cpp
    util.cpp
    material.cpp
    mesh.cpp
    scene.cpp
    buffer_view.cpp
    gltf_types.cpp
    flatten_gltf.cpp
    file_mapping.cpp)

set_target_properties(util PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)

target_include_directories(util PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/parallel_hashmap>
    $<BUILD_INTERFACE:${SDL2_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${GLM_INCLUDE_DIRS}>)

target_link_libraries(util PUBLIC
    imgui
    ${SDL2_LIBRARY})

find_package(pbrtParser)
if (${pbrtParser_FOUND})
    target_link_libraries(util PUBLIC pbrtParser)
    target_compile_definitions(util PUBLIC PBRT_PARSER_ENABLED)
endif()

