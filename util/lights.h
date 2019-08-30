#pragma once

#include <glm/glm.hpp>

// Quad-shaped light source
struct QuadLight {
    glm::vec4 emission;
    glm::vec4 position;
    glm::vec4 normal;

    // x and y vectors spanning the quad, with
    // the half-width and height
    glm::vec3 v_x;
    float width;

    glm::vec3 v_y;
    float height;
};
