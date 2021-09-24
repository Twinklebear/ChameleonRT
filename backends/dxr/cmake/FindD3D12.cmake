# modified from https://github.com/Microsoft/DirectXShaderCompiler/blob/master/cmake/modules/FindD3D12.cmake
# Find the win10 SDK path.
if ("$ENV{WIN10_SDK_PATH}$ENV{WIN10_SDK_VERSION}" STREQUAL "" )
	get_filename_component(WIN10_SDK_PATH "[HKEY_LOCAL_MACHINE\\SOFTWARE\\WOW6432Node\\Microsoft\\Microsoft SDKs\\Windows\\v10.0;InstallationFolder]" ABSOLUTE CACHE)
	get_filename_component(TEMP_WIN10_SDK_VERSION "[HKEY_LOCAL_MACHINE\\SOFTWARE\\WOW6432Node\\Microsoft\\Microsoft SDKs\\Windows\\v10.0;ProductVersion]" ABSOLUTE CACHE)
	get_filename_component(WIN10_SDK_VERSION ${TEMP_WIN10_SDK_VERSION} NAME)
elseif(TRUE)
	set (WIN10_SDK_PATH $ENV{WIN10_SDK_PATH})
	set (WIN10_SDK_VERSION $ENV{WIN10_SDK_VERSION})
endif ("$ENV{WIN10_SDK_PATH}$ENV{WIN10_SDK_VERSION}" STREQUAL "" )

# WIN10_SDK_PATH will be something like C:\Program Files (x86)\Windows Kits\10
# WIN10_SDK_VERSION will be something like 10.0.14393 or 10.0.14393.0; we need the
# one that matches the directory name.

if (IS_DIRECTORY "${WIN10_SDK_PATH}/Include/${WIN10_SDK_VERSION}.0")
	set(WIN10_SDK_VERSION "${WIN10_SDK_VERSION}.0")
endif (IS_DIRECTORY "${WIN10_SDK_PATH}/Include/${WIN10_SDK_VERSION}.0")


# Find the d3d12 and dxgi include path, it will typically look something like this.
# C:\Program Files (x86)\Windows Kits\10\Include\10.0.10586.0\um\d3d12.h
# C:\Program Files (x86)\Windows Kits\10\Include\10.0.10586.0\shared\dxgi1_4.h
find_path(D3D12_INCLUDE_DIR    # Set variable D3D12_INCLUDE_DIR
	d3d12.h                # Find a path with d3d12.h
	HINTS "${WIN10_SDK_PATH}/Include/${WIN10_SDK_VERSION}/um"
	DOC "path to WIN10 SDK header files")

find_path(DXGI_INCLUDE_DIR    # Set variable DXGI_INCLUDE_DIR
	dxgi1_4.h           # Find a path with dxgi1_4.h
	HINTS "${WIN10_SDK_PATH}/Include/${WIN10_SDK_VERSION}/shared"
	DOC "path to WIN10 SDK header files")

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
	find_library(D3D12_LIBRARY NAMES d3d12.lib
		HINTS ${WIN10_SDK_PATH}/Lib/${WIN10_SDK_VERSION}/um/x64)
	find_library(D3D12_COMPILER_LIBRARY NAMES d3dcompiler.lib
		HINTS ${WIN10_SDK_PATH}/Lib/${WIN10_SDK_VERSION}/um/x64)
	find_library(DXGI_LIBRARY NAMES dxgi.lib
		HINTS ${WIN10_SDK_PATH}/Lib/${WIN10_SDK_VERSION}/um/x64)
	find_program(D3D12_SHADER_COMPILER NAMES dxc
		PATHS ${WIN10_SDK_PATH}/bin/${WIN10_SDK_VERSION}/x64)
else()
	find_library(D3D12_LIBRARY NAMES d3d12.lib
		HINTS ${WIN10_SDK_PATH}/Lib/${WIN10_SDK_VERSION}/um/x86)
	find_library(D3D12_COMPILER_LIBRARY NAMES d3dcompiler.lib
		HINTS ${WIN10_SDK_PATH}/Lib/${WIN10_SDK_VERSION}/um/x86)
	find_library(DXGI_LIBRARY NAMES dxgi.lib
		HINTS ${WIN10_SDK_PATH}/Lib/${WIN10_SDK_VERSION}/um/x86)
	find_program(D3D12_SHADER_COMPILER NAMES dxc
		PATHS ${WIN10_SDK_PATH}/bin/${WIN10_SDK_VERSION}/x86)
endif()

# The first shader in the list should be the main shader, while
# others contain dependencies and additional required functions.
# Note that the include paths and defines should not have
# the -I or -D prefix, respectively
function(add_dxil_embed_library)
	set(options INCLUDE_DIRECTORIES COMPILE_DEFINITIONS COMPILE_OPTIONS)
	cmake_parse_arguments(PARSE_ARGV 1 DXIL "" "" "${options}")

	set(HLSL_INCLUDE_DIRS "")
	foreach (inc ${DXIL_INCLUDE_DIRECTORIES})
		file(TO_NATIVE_PATH "${inc}" native_path)
		list(APPEND HLSL_INCLUDE_DIRS "-I${native_path}")
	endforeach()

	set(HLSL_COMPILE_DEFNS "")
	foreach (def ${DXIL_COMPILE_DEFINITIONS})
		list(APPEND HLSL_COMPILE_DEFNS "-D${def}")
	endforeach()

	set(DXIL_LIB ${ARGV0})
	set(HLSL_SRCS "")
	foreach (shader ${DXIL_UNPARSED_ARGUMENTS})
		list(APPEND HLSL_SRCS "${CMAKE_CURRENT_LIST_DIR}/${shader}")
	endforeach()
	list(GET DXIL_UNPARSED_ARGUMENTS 0 MAIN_SHADER)

	# We only compile the main shader with dxc, but use the rest to
	# set the target dependencies properly
	get_filename_component(FNAME ${MAIN_SHADER} NAME_WE)
	set(DXIL_EMBED_FILE "${CMAKE_CURRENT_BINARY_DIR}/${FNAME}_embedded_dxil.h")
	add_custom_command(OUTPUT ${DXIL_EMBED_FILE}
		COMMAND ${D3D12_SHADER_COMPILER} ${CMAKE_CURRENT_LIST_DIR}/${MAIN_SHADER}
		-T lib_6_3 -Fh ${DXIL_EMBED_FILE} -Vn ${FNAME}_dxil
		${HLSL_INCLUDE_DIRS} ${HLSL_COMPILE_DEFNS} ${DXIL_COMPILE_OPTIONS}
		DEPENDS ${HLSL_SRCS}
		COMMENT "Compiling and embedding ${MAIN_SHADER} as ${FNAME}_dxil")

	# This is needed for some reason to get CMake to generate the file properly
	# and not screw up the build, because the original approach of just
	# setting target_sources on ${DXIL_LIB} stopped working once this got put
	# in some subdirectory. CMake ¯\_(ツ)_/¯
	set(DXIL_CMAKE_CUSTOM_WRAPPER ${DXIL_LIB}_custom_target)
	add_custom_target(${DXIL_CMAKE_CUSTOM_WRAPPER} ALL DEPENDS ${DXIL_EMBED_FILE})

	add_library(${DXIL_LIB} INTERFACE)
	add_dependencies(${DXIL_LIB} ${DXIL_CMAKE_CUSTOM_WRAPPER})
	target_include_directories(${DXIL_LIB} INTERFACE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
endfunction()

set(D3D12_LIBRARIES ${D3D12_LIBRARY} ${D3D12_COMPILER_LIBRARY} ${DXGI_LIBRARY})
set(D3D12_INCLUDE_DIRS ${D3D12_INCLUDE_DIR} ${DXGI_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set D3D12_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(D3D12 DEFAULT_MSG
	D3D12_INCLUDE_DIRS D3D12_LIBRARIES D3D12_SHADER_COMPILER)

mark_as_advanced(D3D12_INCLUDE_DIRS D3D12_LIBRARIES D3D12_SHADER_COMPILER)

