# TODO: What about finding ospcommon against an OSPRay install?
# Right now I have issues with the libimf dependency leaking in the
# releases so can't test
find_package(ospcommon REQUIRED)

add_library(ospray_module_chameleon SHARED
  chameleon_device.cpp)

set_target_properties(ospray_module_chameleon PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON)

target_link_libraries(ospray_module_chameleon
  util
  ospray::ospray
  ospcommon::ospcommon)

include(GenerateExportHeader)
generate_export_header(ospray_module_chameleon)

