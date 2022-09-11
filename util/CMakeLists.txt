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
    file_mapping.cpp
    render_plugin.cpp)

set_target_properties(util PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)

target_include_directories(util PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/parallel_hashmap>)

target_link_libraries(util PUBLIC imgui glm)

if (NOT TARGET SDL2::SDL2)
    # Assume SDL2 is in the default library path and create
    # imported targets for it, we re-find the library since
    # SDL2_LIBRARIES might just be -lSDL2
    find_library(SDL2_SHARED_LIBRARY SDL2 REQUIRED)

    add_library(SDL2::SDL2 SHARED IMPORTED)
    set_target_properties(SDL2::SDL2 PROPERTIES
        IMPORTED_LOCATION ${SDL2_SHARED_LIBRARY})
    target_include_directories(SDL2::SDL2 INTERFACE
        ${SDL2_INCLUDE_DIRS})

    if (WIN32)
        # Same for SDL2main
        find_library(SDL2MAIN_SHARED_LIBRARY SDL2main REQUIRED)
        add_library(SDL2::SDL2main SHARED IMPORTED)
        set_target_properties(SDL2::SDL2main PROPERTIES
            IMPORTED_LOCATION ${SDL2MAIN_SHARED_LIBRARY})
    endif()
endif()

target_link_libraries(util PUBLIC
    SDL2::SDL2)

if (WIN32)
    target_link_libraries(util PUBLIC
        SDL2::SDL2main)
endif()

install(IMPORTED_RUNTIME_ARTIFACTS SDL2::SDL2
    DESTINATION bin)

crt_add_packaged_dependency(SDL2::SDL2)

option(CHAMELEONRT_PBRT_SUPPORT "Enable support for loading pbrtv3 files using Ingo Wald's PBRT parser" OFF)
if (CHAMELEONRT_PBRT_SUPPORT)
    find_package(pbrtParser REQUIRED)
    target_link_libraries(util PUBLIC pbrtParser)
    target_compile_definitions(util PUBLIC PBRT_PARSER_ENABLED)
endif()

