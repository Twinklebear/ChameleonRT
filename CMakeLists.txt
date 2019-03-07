cmake_minimum_required(VERSION 3.5)
project(rtobj)

if (NOT WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
endif()

add_definitions(-DNOMINMAX -DGLM_ENABLE_EXPERIMENTAL -DSDL_MAIN_HANDLED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(Threads REQUIRED)
find_package(glm REQUIRED)
find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)

set(RENDER_BACKEND_SRCS "") 

if (${ENABLE_OSPRAY})
	find_package(ospray REQUIRED)
	add_definitions(-DENABLE_OSPRAY)
	# TODO WILL: Once OSPRay 2.0 is released we'll only support 2+
	# so it's fine to ignore the minor & patch versions here
	if ("${OSPRAY_VERSION}" STREQUAL "2.0.0")
		add_definitions(-DOSPRAY_VERSION_MAJOR=2)
	else()
		add_definitions(-DOSPRAY_VERSION_MAJOR=1)
	endif()
	list(APPEND RENDER_BACKEND_SRCS render_ospray.cpp)
endif()

if (${ENABLE_EMBREE})
	find_package(embree 3 REQUIRED)
	find_package(TBB REQUIRED)
	add_definitions(-DENABLE_EMBREE)
	list(APPEND RENDER_BACKEND_SRCS render_embree.cpp)

	set(ISPC_ARCH "x86")
	if (CMAKE_SIZEOF_VOID_P EQUAL 8)
		set(ISPC_ARCH "x86-64")
	endif()

	if (UNIX)
		set(ISPC_PIC "--pic")
	endif()

	include(cmake/ISPC.cmake)
	add_custom_command(OUTPUT
		${CMAKE_CURRENT_BINARY_DIR}/render_embree_ispc.o
		${CMAKE_CURRENT_BINARY_DIR}/render_embree_ispc.h
		COMMAND ${ispc} ${CMAKE_CURRENT_LIST_DIR}/render_embree.ispc
		-o ${CMAKE_CURRENT_BINARY_DIR}/render_embree_ispc.o
		-h ${CMAKE_CURRENT_BINARY_DIR}/render_embree_ispc.h
		-I ${EMBREE_INCLUDE_DIRS}
		--arch=${ISPC_ARCH}
		--opt=fast-math
		-O2
		${ISPC_PIC}
		DEPENDS ${CMAKE_CURRENT_LIST_DIR}/render_embree.ispc
		COMMENT "Compiling Embree ISPC kernels")

	add_library(ispc_kernels ${CMAKE_CURRENT_BINARY_DIR}/render_embree_ispc.o)
	target_include_directories(ispc_kernels PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
	set_target_properties(ispc_kernels PROPERTIES LINKER_LANGUAGE C)
endif()

if (${ENABLE_OPTIX})
	find_package(CUDA 10 REQUIRED)
	find_package(OptiX 6 REQUIRED)
	include(cmake/Bin2C.cmake)
	add_definitions(-DENABLE_OPTIX)

	include_directories(${OptiX_INCLUDE})
	cuda_compile_ptx(ptx_file render_optix.cu)
	set(optix_embed_output ${CMAKE_CURRENT_BINARY_DIR}/render_optix_embedded_ptx.c)
	add_custom_command(OUTPUT ${optix_embed_output}
		COMMAND ${bin2c} -c --padd 0 --type char --name "render_optix_programs" ${ptx_file} > ${optix_embed_output}
		DEPENDS ${ptx_file}
		COMMENT "Compiling and embedding PTX code")

	list(APPEND RENDER_BACKEND_SRCS render_optix.cpp ${optix_embed_output})
endif()

if (RENDER_BACKEND_SRCS STREQUAL "")
	message(FATAL_ERROR "At least one rendering backend (OSPRAY, EMBREE, OPTIX) must be selected\n"
		"Enable at least one of: ENABLE_OSPRAY, ENABLE_EMBREE, ENABLE_OPTIX") 
endif()

add_subdirectory(imgui)

add_executable(rtobj main.cpp
	${RENDER_BACKEND_SRCS}
	arcball_camera.cpp
	imgui_impl_sdl.cpp
	imgui_impl_opengl3.cpp
	shader.cpp
	gl_core_4_5.c)

set_target_properties(rtobj PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

target_include_directories(rtobj PUBLIC
	$<BUILD_INTERFACE:${SDL2_INCLUDE_DIR}>
	$<BUILD_INTERFACE:${OPENGL_INCLUDE_DIR}>
	$<BUILD_INTERFACE:${GLM_INCLUDE_DIRS}>)

target_link_libraries(rtobj PUBLIC
	imgui
	${SDL2_LIBRARY}
	${OPENGL_LIBRARIES})

if (${ENABLE_OSPRAY})
	target_include_directories(rtobj PUBLIC
		$<BUILD_INTERFACE:${OSPRAY_INCLUDE_DIR}>)

	target_link_libraries(rtobj PUBLIC ${OSPRAY_LIBRARIES})
endif()

if (${ENABLE_EMBREE})
	target_include_directories(rtobj PUBLIC
		$<BUILD_INTERFACE:${TBB_INCLUDE_DIR}>
		$<BUILD_INTERFACE:${EMBREE_INCLUDE_DIRS}>)

	target_link_libraries(rtobj PUBLIC ispc_kernels ${EMBREE_LIBRARIES} ${TBB_LIBRARY})
endif()

if (${ENABLE_OPTIX})
	target_include_directories(rtobj PUBLIC
		$<BUILD_INTERFACE:${OptiX_INCLUDE}>
		$<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>)

	target_link_libraries(rtobj PUBLIC optix ${CUDA_LIBRARIES})
endif()

