find_file(BINTOH NAME BinToH.cmake
    PATHS ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR})

function(add_metal_embed_library)
	set(options INCLUDE_DIRECTORIES COMPILE_DEFINITIONS COMPILE_OPTIONS)
    cmake_parse_arguments(PARSE_ARGV 1 METAL "" "" "${options}")

    set(METAL_INCLUDE_DIRS "")
    foreach (inc ${METAL_INCLUDE_DIRECTORIES})
		file(TO_NATIVE_PATH "${inc}" native_path)
        list(APPEND METAL_INCLUDE_DIRS "-I${native_path}")
	endforeach()

    set(METAL_COMPILE_DEFNS "")
    foreach (def ${METAL_COMPILE_DEFINITIONS})
        list(APPEND METAL_COMPILE_DEFNS "-D${def}")
	endforeach()

    set(METAL_LIB ${ARGV0})
    set(METAL_SRCS "")
    foreach (shader ${METAL_UNPARSED_ARGUMENTS})
        list(APPEND METAL_SRCS "${CMAKE_CURRENT_LIST_DIR}/${shader}")
	endforeach()
    list(GET METAL_SRCS 0 MAIN_SHADER)

	# We only compile the main shader, but use the rest to
	# set the target dependencies properly
    # TODO: Use Metal compiler -H flag to get header dependencies
	get_filename_component(FNAME ${MAIN_SHADER} NAME_WE)
    set(METALAIR_FILE "${CMAKE_CURRENT_BINARY_DIR}/${FNAME}.air")
    set(METALLIB_FILE "${CMAKE_CURRENT_BINARY_DIR}/${FNAME}.metallib")
    set(METAL_EMBED_FILE "${CMAKE_CURRENT_BINARY_DIR}/${FNAME}_embedded_metallib.h")
    # TODO later: compiling and linking multiple air files to one lib?
    add_custom_command(OUTPUT ${METALLIB_FILE}
        COMMAND xcrun -sdk macosx metal
            -c ${MAIN_SHADER}
            ${METAL_INCLUDE_DIRS}
            ${METAL_COMPILE_DEFNS}
            ${METAL_COMPILE_OPTIONS}
            -o ${METALAIR_FILE}
        COMMAND xcrun -sdk macosx metallib ${METALAIR_FILE} -o ${METALLIB_FILE}
        DEPENDS ${METAL_SRCS}
        BYPRODUCTS ${METALAIR_FILE})

    add_custom_command(OUTPUT ${METAL_EMBED_FILE}
        COMMAND ${CMAKE_COMMAND}
            -DBIN_TO_H_BINARY_FILE=${METALLIB_FILE}
            -DBIN_TO_H_OUTPUT=${METAL_EMBED_FILE}
            -DBIN_TO_H_VAR_NAME=${FNAME}_metallib
            -P ${BINTOH}
        DEPENDS ${METALLIB_FILE}
        COMMENT "Embedding ${METALLIB_FILE} as ${FNAME}_metallib in ${FNAME}_embedded_metallib.h")

    set(METAL_CMAKE_CUSTOM_WRAPPER ${METAL_LIB}_custom_target)
    add_custom_target(${METAL_CMAKE_CUSTOM_WRAPPER} ALL DEPENDS ${METAL_EMBED_FILE})

    add_library(${METAL_LIB} INTERFACE)
    add_dependencies(${METAL_LIB} ${METAL_CMAKE_CUSTOM_WRAPPER})
    target_include_directories(${METAL_LIB} INTERFACE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
endfunction()

