#pragma once

#include <glm/glm.hpp>

struct Camera {
    glm::vec3 position, center, up;
    float fov_y;
};
