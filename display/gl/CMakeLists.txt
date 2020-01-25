find_package(OpenGL REQUIRED)

add_library(gldisplay gldisplay.cpp
    shader.cpp
    imgui_impl_opengl3.cpp
    gl_core_4_5.c)

set_target_properties(gldisplay PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)

target_include_directories(gldisplay PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
    $<BUILD_INTERFACE:${OPENGL_INCLUDE_DIR}>)

target_link_libraries(gldisplay PUBLIC
    display
    util
    ${OPENGL_LIBRARIES})
