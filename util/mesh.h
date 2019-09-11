#pragma once

#include <vector>
#include <glm/glm.hpp>

struct Geometry {
    std::vector<glm::vec3> vertices, normals;
    std::vector<glm::vec2> uvs;
    std::vector<glm::uvec3> indices;
    // TODO Later: a concept of instancing each mesh, for OBJ it's trivially
    // a single instance of each mesh with an identity transform. For PBRT/GLTF
    // it may be more complicated
    uint32_t material_id = -1;

    size_t num_tris() const;
};

struct Mesh {
    std::vector<Geometry> geometries;

    Mesh(const std::vector<Geometry> &geometries);

    Mesh() = default;

    size_t num_tris() const;
};

struct Instance {
    glm::mat4 transform;
    size_t mesh_id;

    Instance(const glm::mat4 &transform, size_t mesh_id);

    Instance() = default;
};
