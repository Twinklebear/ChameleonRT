#include "mesh.h"
#include <algorithm>
#include <numeric>

size_t Geometry::num_tris() const
{
    return indices.size();
}

Mesh::Mesh(const std::vector<Geometry> &geometries) : geometries(geometries) {}

size_t Mesh::num_tris() const
{
    return std::accumulate(
        geometries.begin(), geometries.end(), 0, [](const size_t &n, const Geometry &g) {
            return n + g.num_tris();
        });
}

ParameterizedMesh::ParameterizedMesh(size_t mesh_id, const std::vector<uint32_t> &material_ids)
    : mesh_id(mesh_id), material_ids(material_ids)
{
}

Instance::Instance(const glm::mat4 &transform, size_t parameterized_mesh_id)
    : transform(transform), parameterized_mesh_id(parameterized_mesh_id)
{
}
