ExternalProject_Add(glm_ext
    PREFIX glm
    DOWNLOAD_DIR glm
    STAMP_DIR glm/stamp
    SOURCE_DIR glm/src
    BINARY_DIR glm
    URL "https://github.com/g-truc/glm/releases/download/0.9.9.8/glm-0.9.9.8.zip"
    URL_HASH "SHA256=37e2a3d62ea3322e43593c34bae29f57e3e251ea89f4067506c94043769ade4c"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_ALWAYS OFF
)

set(GLM_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/glm/src)

add_library(glm INTERFACE)

add_dependencies(glm glm_ext)

target_include_directories(glm INTERFACE
    ${GLM_INCLUDE_DIRS})


