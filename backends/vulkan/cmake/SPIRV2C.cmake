function(spirv2c)
	file(GLOB SPIRV_BINARIES *.spv)

	file(WRITE ${SPIRV_EMBED_FILE} "#pragma once\n")
	foreach (spv ${SPIRV_BINARIES})
		get_filename_component(FNAME ${spv} NAME_WE)
		file(READ ${spv} SPV_CONTENT)
		file(APPEND ${SPIRV_EMBED_FILE} "const uint32_t ${FNAME}_spv[] =\n${SPV_CONTENT};\n")
	endforeach()
endfunction()

spirv2c(${SPIRV_EMBED_FILE} ${OUTPUT_DIR})

