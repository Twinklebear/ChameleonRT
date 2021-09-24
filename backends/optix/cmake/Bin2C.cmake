# Based on https://github.com/robertmaynard/code-samples/tree/master/posts/cmake_ptx

get_filename_component(CUDA_COMPILER_BIN "${CUDA_NVCC_EXECUTABLE}" DIRECTORY)

find_program(bin2c NAMES bin2c
    PATHS
    ${CUDA_SDK_ROOT_DIR}
    ${CUDA_COMPILER_BIN})

if (NOT bin2c)
    message(FATAL_ERROR "Failed to find bin2c, searched ${CUDA_SDK_ROOT_DIR} and ${CUDA_COMPILER_BIN}")
endif()

function(add_ptx_embed_library)
	set(options INCLUDE_DIRECTORIES COMPILE_DEFINITIONS) 
	cmake_parse_arguments(PARSE_ARGV 1 PTX "" "" "${options}")

	cuda_include_directories(${PTX_INCLUDE_DIRECTORIES})

	set(PTX_INCLUDES "")
    foreach (inc ${PTX_INCLUDE_DIRECTORIES})
		list(APPEND PTX_INCLUDES "-I${inc}")
	endforeach()

	set(PTX_LIB ${ARGV0})
	set(CUDA_SRCS ${PTX_UNPARSED_ARGUMENTS})

	set(PTX_SRCS "")
	foreach (SRC ${CUDA_SRCS})
		get_filename_component(FNAME ${SRC} NAME_WE)
        set(CU_FILE ${CMAKE_CURRENT_LIST_DIR}/${SRC})

        # Note: on linux we can pass -MM to not getsystem deps
        # Compute dependencies for this file
        execute_process(COMMAND ${CUDA_NVCC_EXECUTABLE} -ptx ${CU_FILE}
            ${PTX_INCLUDES} ${PTX_COMPILE_DEFINITIONS} -M
            -ccbin ${CMAKE_CXX_COMPILER}
            OUTPUT_VARIABLE PTX_DEPS_STRING)

        set(PTX_DEPENDENCIES "")
        string(REPLACE "\\\n" ";" PTX_DEPS_LIST ${PTX_DEPS_STRING})
        foreach (dep ${PTX_DEPS_LIST})
            string(STRIP ${dep} dep)
            string(FIND ${dep} ${CMAKE_CURRENT_LIST_DIR} FND)
            if (${FND} EQUAL "0")
                list(APPEND PTX_DEPENDENCIES ${dep})
            endif()
        endforeach()

        set(PTX_FILE "${CMAKE_CURRENT_BINARY_DIR}/${FNAME}.ptx")
        add_custom_command(OUTPUT ${PTX_FILE}
            COMMAND ${CUDA_NVCC_EXECUTABLE} -ptx ${CMAKE_CURRENT_LIST_DIR}/${SRC}
            ${PTX_INCLUDES} ${PTX_COMPILE_DEFINITIONS} -o ${PTX_FILE}
            DEPENDS ${CU_FILE} ${PTX_DEPENDENCIES}
            COMMENT "Compiling ${PTX_FILE}")

		set(PTX_EMBED_FILE "${CMAKE_CURRENT_BINARY_DIR}/${FNAME}_embedded_ptx.h")
		add_custom_command(OUTPUT ${PTX_EMBED_FILE}
			COMMAND ${bin2c} -c --padd 0 --type char --name "${FNAME}_ptx" ${PTX_FILE} > ${PTX_EMBED_FILE}
			DEPENDS ${PTX_FILE}
			COMMENT "Compiling and embedding ${SRC} as ${FNAME}_ptx in ${PTX_EMBED_FILE}")

		list(APPEND PTX_SRCS ${PTX_EMBED_FILE})
	endforeach()

	set(PTX_CMAKE_CUSTOM_WRAPPER ${PTX_LIB}_custom_target)
	add_custom_target(${PTX_CMAKE_CUSTOM_WRAPPER} ALL DEPENDS ${PTX_SRCS})

	add_library(${PTX_LIB} INTERFACE)
	add_dependencies(${PTX_LIB} ${PTX_CMAKE_CUSTOM_WRAPPER})
	target_include_directories(${PTX_LIB} INTERFACE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
endfunction()

