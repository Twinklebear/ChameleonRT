# Finds the WinPixEventRuntime
# HINTS
# ^^^^^^^^^^^^^^^^
# Will look for the WinPixEventRuntime under WinPixEventRuntime_DIR
# https://devblogs.microsoft.com/pix/winpixeventruntime/
#
# IMPORTED Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines :prop_tgt:`IMPORTED` target ``PIX::WinPixEventRuntime``, if
# the WinPixEventRuntime has been found.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables::
#
#   WinPixEventRuntime_FOUND          - True if WinPixEventRuntime was found
#   WinPixEventRuntime_INCLUDE_DIR    - include directory for WinPixEventRuntime
#   WinPixEventRuntime_LIBRARY        - link against this library to use WinPixEventRuntime
#

find_path(WinPixEventRuntime_INCLUDE_DIR
    NAMES pix3.h
    PATHS
    ${WinPixEventRuntime_DIR}/Include/WinPixEventRuntime
    $ENV{WinPixEventRuntime_DIR}/Include/WinPixEventRuntime
)

find_library(WinPixEventRuntime_LIBRARY
    NAMES WinPixEventRuntime
    PATHS
    ${WinPixEventRuntime_DIR}/bin/x64
    $ENV{WinPixEventRuntime_DIR}/bin/x64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(WinPixEventRuntime
    DEFAULT_MSG WinPixEventRuntime_LIBRARY WinPixEventRuntime_INCLUDE_DIR)

if (WinPixEventRuntime_FOUND AND NOT TARGET PIX::WinPixEventRuntime)
    add_library(PIX::WinPixEventRuntime UNKNOWN IMPORTED)
    set_target_properties(PIX::WinPixEventRuntime PROPERTIES
        IMPORTED_LOCATION "${WinPixEventRuntime_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${WinPixEventRuntime_INCLUDE_DIR}")
endif()

