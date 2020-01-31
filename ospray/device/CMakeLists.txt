# TODO: What about finding ospcommon against an OSPRay install?
# Right now I have issues with the libimf dependency leaking in the
# releases so can't test
find_package(ospcommon REQUIRED)

set(MODULE_SRC
    chameleon_device.cpp
    data.cpp
    objects.cpp)

# We need gl_core_4_5 for OptiX since it has some GL calls in there
# (which won't be called in the module code path)
if (ENABLE_OPTIX)
    set(MODULE_SRC ${MODULE_SRC} ${PROJECT_SOURCE_DIR}/util/display/gl_core_4_5.c)
endif()

add_library(ospray_module_chameleon SHARED ${MODULE_SRC})

set_target_properties(ospray_module_chameleon PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)

target_include_directories(ospray_module_chameleon PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(ospray_module_chameleon PRIVATE
    util
    ospray::ospray
    ospcommon::ospcommon)

if (ENABLE_EMBREE)
    target_compile_definitions(ospray_module_chameleon PRIVATE -DENABLE_EMBREE)
    target_link_libraries(ospray_module_chameleon PRIVATE render_embree)
endif()

if (ENABLE_OPTIX)
    target_compile_definitions(ospray_module_chameleon PRIVATE -DENABLE_OPTIX)
    target_link_libraries(ospray_module_chameleon PRIVATE render_optix)
endif()

if (ENABLE_DXR)
    target_compile_definitions(ospray_module_chameleon PRIVATE -DENABLE_DXR)
    target_link_libraries(ospray_module_chameleon PRIVATE render_dxr)
endif()

if (ENABLE_VULKAN)
    target_compile_definitions(ospray_module_chameleon PRIVATE -DENABLE_VULKAN)
    target_link_libraries(ospray_module_chameleon PRIVATE render_vulkan)
endif()

include(GenerateExportHeader)
generate_export_header(ospray_module_chameleon)

