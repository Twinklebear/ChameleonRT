# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#.rst:
# FindVulkan
# ----------
#
# Try to find Vulkan
#
# IMPORTED Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines :prop_tgt:`IMPORTED` target ``Vulkan::Vulkan``, if
# Vulkan has been found.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables::
#
#   Vulkan_FOUND          - True if Vulkan was found
#   Vulkan_INCLUDE_DIRS   - include directories for Vulkan
#   Vulkan_LIBRARIES      - link against this library to use Vulkan
#
# The module will also define two cache variables::
#
#   Vulkan_INCLUDE_DIR    - the Vulkan include directory
#   Vulkan_LIBRARY        - the path to the Vulkan library
#

if(WIN32)
    find_path(Vulkan_INCLUDE_DIR
        NAMES vulkan/vulkan.h
        PATHS
        ${VULKAN_SDK}/Include
        $ENV{VULKAN_SDK}/Include
    )

    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        find_library(Vulkan_LIBRARY
            NAMES vulkan-1
            PATHS
            ${VULKAN_SDK}/Lib
            ${VULKAN_SDK}/Bin
            $ENV{VULKAN_SDK}/Lib
            $ENV{VULKAN_SDK}/Bin
        )
        find_program(SPIRV_COMPILER
            NAMES glslc
            PATHS
            ${VULKAN_SDK}/Bin
            $ENV{VULKAN_SDK}/Bin
        )
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        find_library(Vulkan_LIBRARY
            NAMES vulkan-1
            PATHS
            ${VULKAN_SDK}/Lib32
            ${VULKAN_SDK}/Bin32
            $ENV{VULKAN_SDK}/Lib32
            $ENV{VULKAN_SDK}/Bin32
            NO_SYSTEM_ENVIRONMENT_PATH
        )
        find_program(SPIRV_COMPILER
            NAMES glslc
            PATHS
            ${VULKAN_SDK}/Bin32
            $ENV{VULKAN_SDK}/Bin32
        )
    endif()
else()
    find_path(Vulkan_INCLUDE_DIR
        NAMES vulkan/vulkan.h
        PATHS
        ${VULKAN_SDK}/x86_64/include
        $ENV{VULKAN_SDK}/x86_64/include
    )
    find_library(Vulkan_LIBRARY
        NAMES vulkan
        PATHS
        ${VULKAN_SDK}/x86_64/lib
        $ENV{VULKAN_SDK}/x86_64/lib
    )
    find_program(SPIRV_COMPILER
        NAMES glslc
        PATHS
        ${VULKAN_SDK}/x86_64/bin
        $ENV{VULKAN_SDK}/x86_64/bin
    )
endif()

find_file(SPIRV2C NAME SPIRV2C.cmake
    PATHS ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR})

# Note that the include paths and defines should not have
# the -I or -D prefix, respectively
function(add_spirv_embed_library)
    set(options INCLUDE_DIRECTORIES COMPILE_DEFINITIONS COMPILE_OPTIONS)
    cmake_parse_arguments(PARSE_ARGV 1 SPIRV "" "" "${options}")

    set(GLSL_INCLUDE_DIRECTORIES "")
    foreach (inc ${SPIRV_INCLUDE_DIRECTORIES})
        file(TO_NATIVE_PATH "${inc}" native_path)
        list(APPEND GLSL_INCLUDE_DIRECTORIES "-I${native_path}")
    endforeach()

    set(GLSL_COMPILE_DEFNS "")
    foreach (def ${SPIRV_COMPILE_DEFINITIONS})
        list(APPEND GLSL_COMPILE_DEFNS "-D${def}")
    endforeach()

    # Compile each GLSL file to embedded SPIRV bytecode
    set(SPIRV_LIB ${ARGV0})
    set(SPIRV_BINARIES "")
    foreach (shader ${SPIRV_UNPARSED_ARGUMENTS})
        get_filename_component(FNAME ${shader} NAME_WE)
        set(SPV_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${FNAME}.spv)
        list(APPEND SPIRV_BINARIES ${SPV_OUTPUT})

        # Determine the dependencies for the shader and track them
        execute_process(
            COMMAND ${SPIRV_COMPILER} ${CMAKE_CURRENT_LIST_DIR}/${shader}
            ${GLSL_INCLUDE_DIRECTORIES} ${GLSL_COMPILE_DEFNS}
            --target-env=vulkan1.2 ${SPIRV_COMPILE_OPTIONS} -MM
            OUTPUT_VARIABLE SPV_DEPS_STRING)

        # The first item is the spv file name formatted as <shader>.spv:, so remove that
        string(REPLACE " " ";" SPV_DEPS_LIST "${SPV_DEPS_STRING}")
        list(REMOVE_AT SPV_DEPS_LIST 0)

        add_custom_command(OUTPUT ${SPV_OUTPUT}
            COMMAND ${SPIRV_COMPILER} ${CMAKE_CURRENT_LIST_DIR}/${shader}
            ${GLSL_INCLUDE_DIRECTORIES} ${GLSL_COMPILE_DEFNS}
            --target-env=vulkan1.2 -mfmt=c -o ${SPV_OUTPUT} ${SPIRV_COMPILE_OPTIONS}
            DEPENDS ${SPV_DEPS_LIST}
            COMMENT "Compiling ${CMAKE_CURRENT_LIST_DIR}/${shader} to ${SPV_OUTPUT}")
    endforeach()

    set(SPIRV_EMBED_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SPIRV_LIB}_embedded_spv.h")
    add_custom_command(OUTPUT ${SPIRV_EMBED_FILE}
        COMMAND ${CMAKE_COMMAND} 
        -DSPIRV_EMBED_FILE=${SPIRV_EMBED_FILE} -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}
        -P ${SPIRV2C}
        DEPENDS ${SPIRV_BINARIES}
        COMMENT "Embedding SPIRV bytecode into ${SPIRV_EMBED_FILE}")

    set(SPIRV_CMAKE_CUSTOM_WRAPPER ${SPIRV_LIB}_custom_target)
    add_custom_target(${SPIRV_CMAKE_CUSTOM_WRAPPER} ALL DEPENDS ${SPIRV_EMBED_FILE})

    add_library(${SPIRV_LIB} INTERFACE)
    add_dependencies(${SPIRV_LIB} ${SPIRV_CMAKE_CUSTOM_WRAPPER})
    target_include_directories(${SPIRV_LIB} INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
endfunction()

set(Vulkan_LIBRARIES ${Vulkan_LIBRARY})
set(Vulkan_INCLUDE_DIRS ${Vulkan_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Vulkan
    DEFAULT_MSG
    Vulkan_LIBRARY Vulkan_INCLUDE_DIR)

mark_as_advanced(Vulkan_INCLUDE_DIR Vulkan_LIBRARY)

if(Vulkan_FOUND AND NOT TARGET Vulkan::Vulkan)
    add_library(Vulkan::Vulkan UNKNOWN IMPORTED)
    set_target_properties(Vulkan::Vulkan PROPERTIES
        IMPORTED_LOCATION "${Vulkan_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${Vulkan_INCLUDE_DIRS}")
endif()

