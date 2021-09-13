# Based on OSPRay's dependency redistribution code 
# https://github.com/ospray/ospray/blob/master/cmake/ospray_redistribute_deps.cmake 

macro(crt_install_namelink NAME)
    get_filename_component(TARGET_NAME ${NAME} NAME)
    set(LIB_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})

    # Create patch version suffixed namelink
    if (APPLE)
        set(LIBREGEX "(.+)[.]([0-9]+)([.][0-9]+[.][0-9]+)${LIB_SUFFIX}")
    else()
        set(LIBREGEX "(.+)${LIB_SUFFIX}[.]([0-9]+)([.][0-9]+[.][0-9]+)")
    endif()
    string(REGEX REPLACE ${LIBREGEX} "\\1" BASE_LIB_NAME ${TARGET_NAME})
    if (CMAKE_MATCH_COUNT GREATER 2)
        if (APPLE)
            set(SYMLINK ${BASE_LIB_NAME}.${CMAKE_MATCH_2}${LIB_SUFFIX})
        else()
            set(SYMLINK ${BASE_LIB_NAME}${LIB_SUFFIX}.${CMAKE_MATCH_2})
        endif()
        execute_process(COMMAND "${CMAKE_COMMAND}" -E
            create_symlink ${TARGET_NAME} ${PROJECT_BINARY_DIR}/${SYMLINK})
        install(PROGRAMS ${PROJECT_BINARY_DIR}/${SYMLINK}
            DESTINATION bin)
        set(TARGET_NAME ${SYMLINK})
    endif()

    # Create minor version suffixed namelink
    if (APPLE)
        set(LIBREGEX "(.+)[.]([0-9]+)([.][0-9]+)${LIB_SUFFIX}")
    else()
        set(LIBREGEX "(.+)${LIB_SUFFIX}[.]([0-9]+)([.][0-9]+)")
    endif()
    string(REGEX REPLACE ${LIBREGEX} "\\1" BASE_LIB_NAME ${TARGET_NAME})
    if (CMAKE_MATCH_COUNT GREATER 2)
        if (APPLE)
            set(SYMLINK ${BASE_LIB_NAME}.${CMAKE_MATCH_2}${LIB_SUFFIX})
        else()
            set(SYMLINK ${BASE_LIB_NAME}${LIB_SUFFIX}.${CMAKE_MATCH_2})
        endif()
        execute_process(COMMAND "${CMAKE_COMMAND}" -E
            create_symlink ${TARGET_NAME} ${PROJECT_BINARY_DIR}/${SYMLINK})
        install(PROGRAMS ${PROJECT_BINARY_DIR}/${SYMLINK}
            DESTINATION bin)
        set(TARGET_NAME ${SYMLINK})
    endif()

    # Create major version suffixed namelink
    if (APPLE)
        set(LIBREGEX "(.+)[.]([0-9]+)${LIB_SUFFIX}")
    else()
        set(LIBREGEX "(.+)${LIB_SUFFIX}[.]([0-9]+)")
    endif()
    string(REGEX REPLACE ${LIBREGEX} "\\1" BASE_LIB_NAME ${TARGET_NAME})
    if (CMAKE_MATCH_COUNT)
        set(SYMLINK ${PROJECT_BINARY_DIR}/${BASE_LIB_NAME}${LIB_SUFFIX})
        execute_process(COMMAND "${CMAKE_COMMAND}" -E
            create_symlink ${TARGET_NAME} ${SYMLINK})
        install(PROGRAMS ${SYMLINK}
            DESTINATION bin)
    endif()
endmacro()

macro(crt_add_packaged_dependency TARGET_NAME)
    get_target_property(CONFIGURATIONS ${TARGET_NAME} IMPORTED_CONFIGURATIONS)
    list(GET CONFIGURATIONS 0 CONFIGURATION)
    if ("${CONFIGURATION}" STREQUAL "CONFIGURATIONS-NOTFOUND")
        get_target_property(LIBRARY ${TARGET_NAME} IMPORTED_LOCATION)
    else()
        get_target_property(LIBRARY ${TARGET_NAME} IMPORTED_LOCATION_${CONFIGURATION})
    endif()
    # Resolve symlinks in the library name we're given
    file(REAL_PATH ${LIBRARY} LIBRARY)

    crt_install_namelink(${LIBRARY})
    install(PROGRAMS ${LIBRARY}
        DESTINATION bin)
endmacro()

