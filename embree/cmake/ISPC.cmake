find_program(ispc NAMES ispc PATHS ${ISPC_DIR})
if (NOT ispc)
	message(FATAL_ERROR "Failed to find ispc, please set ISPC_DIR")
endif()

function(add_ispc_library)
    set(options INCLUDE_DIRECTORIES COMPILE_DEFINITIONS) 
    cmake_parse_arguments(PARSE_ARGV 1 ISPC "" "" "${options}")

    set(ISPC_INCLUDES "")
    foreach (inc ${ISPC_INCLUDE_DIRECTORIES})
        list(APPEND ISPC_INCLUDES "-I${inc}")
    endforeach()

    set(ISPC_LIB ${ARGV0})
    set(ISPC_SRCS ${ISPC_UNPARSED_ARGUMENTS})

    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "AMD64" OR ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        set(ISPC_ARCH "x86-64")
    elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64")
        set(ISPC_TARGET_ARG "--target=neon-i32x4")
        set(ISPC_ARCH "aarch64")
    else ()
        message(FATAL_ERROR "Unrecognized system processor ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    if (UNIX)
        set(ISPC_PIC "--pic")
    endif()

    set(ISPC_OBJS "")
    foreach (SRC ${ISPC_SRCS})
        # First build the list of dependencies of the ISPC file to
        # populate its actual dependencies list
        get_filename_component(FNAME ${SRC} NAME_WE)
        list(APPEND ISPC_OBJS ${CMAKE_CURRENT_BINARY_DIR}/${FNAME}.o)

        message("Writing ISPC dependency list for ${SRC} to ${CMAKE_CURRENT_BINARY_DIR}/${FNAME}.idep")
        execute_process(
            COMMAND ${ispc} ${CMAKE_CURRENT_LIST_DIR}/${SRC}
            -MMM ${CMAKE_CURRENT_BINARY_DIR}/${FNAME}.idep
            ${ISPC_INCLUDES}
            --arch=${ISPC_ARCH}
            ${ISPC_COMPILE_DEFINITIONS}
            ${ISPC_PIC}
            --quiet)

        set(DEPS "${CMAKE_CURRENT_LIST_DIR}/${SRC}")
        if (EXISTS ${CMAKE_CURRENT_BINARY_DIR}/${FNAME}.idep)
            file(READ ${CMAKE_CURRENT_BINARY_DIR}/${FNAME}.idep DEPS_CONTENT)
            string(REPLACE "\n" ";" DEPS_CONTENT "${DEPS_CONTENT}")
            foreach (d ${DEPS_CONTENT})
                string(REPLACE "\\\\" "/" d "${d}")
                if (EXISTS ${d})
                    list(APPEND DEPS ${d})
                endif()
            endforeach()
        endif()

        add_custom_command(OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/${FNAME}.o
            ${CMAKE_CURRENT_BINARY_DIR}/${FNAME}_ispc.h
            COMMAND ${ispc} ${CMAKE_CURRENT_LIST_DIR}/${SRC}
            -o ${CMAKE_CURRENT_BINARY_DIR}/${FNAME}.o
            -h ${CMAKE_CURRENT_BINARY_DIR}/${FNAME}_ispc.h
            ${ISPC_INCLUDES}
            --arch=${ISPC_ARCH}
            ${ISPC_COMPILE_DEFINITIONS}
            ${ISPC_PIC}
            ${ISPC_TARGET_ARG}            
            DEPENDS ${DEPS}
            COMMENT "Compiling ISPC file ${CMAKE_CURRENT_LIST_DIR}/${SRC}")
    endforeach()

    add_library(${ISPC_LIB} ${ISPC_OBJS})
    target_include_directories(${ISPC_LIB} PUBLIC
        $<BUILD_INTERFACE:${ISPC_INCLUDE_DIRECTORIES}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
    set_target_properties(${ISPC_LIB} PROPERTIES LINKER_LANGUAGE C)
endfunction()

