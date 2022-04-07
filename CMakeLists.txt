cmake_minimum_required(VERSION 3.18)
project(chameleonrt)

if (NOT WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
endif()

add_definitions(-DNOMINMAX -DGLM_ENABLE_EXPERIMENTAL -DSDL_MAIN_HANDLED
    -DWIN32_LEAN_AND_MEAN)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")
# Output all render plugins to the root binary dir where the executable is
# so setting runtime paths is easier
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

find_package(Threads REQUIRED)
find_package(SDL2 CONFIG REQUIRED)
# Include glm as an external project
include(cmake/glm.cmake)
include(cmake/package.cmake)

include(CMakeDependentOption)

add_subdirectory(imgui)
add_subdirectory(util)
add_subdirectory(backends)

option(REPORT_RAY_STATS "Track and report rays/second. May incur a slight rendering performance penalty" OFF)

add_executable(chameleonrt main.cpp)

set_target_properties(chameleonrt PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

target_link_libraries(chameleonrt PUBLIC
    util
    display)

install(TARGETS chameleonrt
        RUNTIME DESTINATION bin)

