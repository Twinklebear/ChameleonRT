find_package(OpenGL REQUIRED)

add_library(display imgui_impl_sdl.cpp
    gldisplay.cpp
    shader.cpp
    imgui_impl_opengl3.cpp
    gl_core_4_5.c)

set_target_properties(display PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)

target_include_directories(display PUBLIC
    $<BUILD_INTERFACE:${OPENGL_INCLUDE_DIR}>)

target_link_libraries(display PUBLIC
    util
    ${OPENGL_LIBRARIES})

