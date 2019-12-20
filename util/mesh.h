#pragma once

#include <vector>
#include <glm/glm.hpp>

struct Geometry {
    std::vector<glm::vec3> vertices, normals;
    std::vector<glm::vec2> uvs;
    std::vector<glm::uvec3> indices;

    // TODO: should separate the material ID from the geometric data itself,
    // so one geometry can be re-used in different instances w/ different materials
    // TODO: This should be removed but need to update the GPU backends to use the
    // new per-instance materials method
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
    // Material IDs for the geometry in this instance's mesh
    std::vector<uint32_t> material_ids;

    Instance(const glm::mat4 &transform,
             size_t mesh_id,
             const std::vector<uint32_t> &material_ids);

    Instance() = default;
};
