#pragma once

#include <vector>
#include <glm/glm.hpp>

struct Geometry {
    std::vector<glm::vec3> vertices, normals;
    std::vector<glm::vec2> uvs;
    std::vector<glm::uvec3> indices;

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
    // Material IDs for the geometry in this instance's mesh
    std::vector<uint32_t> material_ids;

    Instance(const glm::mat4 &transform,
             size_t mesh_id,
             const std::vector<uint32_t> &material_ids);

    Instance() = default;
};
