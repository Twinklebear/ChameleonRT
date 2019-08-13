# Based on https://github.com/robertmaynard/code-samples/tree/master/posts/cmake_ptx

get_filename_component(CUDA_COMPILER_BIN "${CUDA_NVCC_EXECUTABLE}" DIRECTORY)

find_program(bin2c NAMES bin2c PATHS ${CUDA_SDK_ROOT_DIR})
if (NOT bin2c)
	message(FATAL_ERROR "Failed to find bin2c, searched ${CUDA_COMPILER_BIN}")
endif()

function(add_ptx_embed_library)
	set(options INCLUDE_DIRECTORIES COMPILE_DEFINITIONS) 
	cmake_parse_arguments(PARSE_ARGV 1 PTX "" "" "${options}")

	cuda_include_directories(${PTX_INCLUDE_DIRECTORIES})

	set(PTX_LIB ${ARGV0})
	set(CUDA_SRCS ${PTX_UNPARSED_ARGUMENTS})

	set(PTX_SRCS "")
	foreach (SRC ${CUDA_SRCS})
		get_filename_component(FNAME ${SRC} NAME_WE)
		cuda_compile_ptx(ptx_file ${CMAKE_CURRENT_LIST_DIR}/${SRC}
			OPTIONS ${PTX_COMPILE_DEFINITIONS})

		set(PTX_EMBED_FILE "${CMAKE_CURRENT_BINARY_DIR}/${FNAME}_embedded_ptx.h")
		add_custom_command(OUTPUT ${PTX_EMBED_FILE}
			COMMAND ${bin2c} -c --padd 0 --type char --name "${FNAME}_ptx" ${ptx_file} > ${PTX_EMBED_FILE}
			DEPENDS ${ptx_file}
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

