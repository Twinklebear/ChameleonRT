# Based on https://github.com/robertmaynard/code-samples/tree/master/posts/cmake_ptx

get_filename_component(CUDA_COMPILER_BIN "${CUDA_NVCC_EXECUTABLE}" DIRECTORY)

find_program(bin2c NAMES bin2c PATHS ${CUDA_SDK_ROOT_DIR})
if (NOT bin2c)
	message(FATAL_ERROR "Failed to find bin2c, searched ${CUDA_COMPILER_BIN}")
endif()

#macro(cuda_embed_ptx embedded_file cuda_files)
#	cuda_compile_ptx(ptx_files ${cuda_files})
#	foreach (ptx ${ptx_files})
#		get_filename_component(ptx_name ${ptx} NAME_WE)
#		get_filename_component(ptx_dir ${ptx} DIRECTORY)
#		add_custom_command(OUTPUT ${embedded_file}
#			COMMAND ${bin2c} -c --padd 0 --type char --name 
#	endforeach()
#endmacro()

