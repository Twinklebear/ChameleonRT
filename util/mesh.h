#pragma once

#include <vector>
#include <glm/glm.hpp>

struct Mesh {
    std::vector<glm::vec3> vertices, normals;
    std::vector<glm::vec2> uvs;
    std::vector<glm::uvec3> indices;
    // TODO Later: a concept of instancing each mesh, for OBJ it's trivially
    // a single instance of each mesh with an identity transform. For PBRT/GLTF
    // it may be more complicated
    uint32_t material_id = -1;

    uint32_t num_tris() const;
};
