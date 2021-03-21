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

/* A parameterized mesh is a combination of a mesh containing the geometries
 * with a set of material parameters to set the appearance information for those
 * geometries.
 */
struct ParameterizedMesh {
    size_t mesh_id;
    // Material IDs for the geometry to parameterize this mesh with
    std::vector<uint32_t> material_ids;

    ParameterizedMesh(size_t mesh_id, const std::vector<uint32_t> &material_ids);

    ParameterizedMesh() = default;
};

/* An instance places a parameterized mesh at some location in the scene
 */
struct Instance {
    glm::mat4 transform;
    size_t parameterized_mesh_id;

    Instance(const glm::mat4 &transform, size_t parameterized_mesh_id);

    Instance() = default;
};
