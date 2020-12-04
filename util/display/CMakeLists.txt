find_package(OpenGL REQUIRED)

add_subdirectory(glad)

add_library(display imgui_impl_sdl.cpp
    gldisplay.cpp
    shader.cpp
    imgui_impl_opengl3.cpp)

set_target_properties(display PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)

target_link_libraries(display PUBLIC util glad)

