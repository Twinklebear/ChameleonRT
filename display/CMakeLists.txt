add_subdirectory(gl)
#add_subdirectory(vk)
if (WIN32)
    add_subdirectory(dx)
endif()

add_library(display imgui_impl_sdl.cpp)

set_target_properties(display PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)

target_include_directories(gldisplay PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)

target_link_libraries(display PUBLIC util)

