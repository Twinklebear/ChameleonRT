find_program(ispc NAMES ispc PATHS ${ISPC_DIR})
if (NOT ispc)
	message(FATAL_ERROR "Failed to find ispc, please set ISPC_DIR")
endif()

