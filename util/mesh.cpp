#include "mesh.h"

uint32_t Geometry::num_tris() const
{
    return indices.size();
}

Mesh::Mesh(const std::vector<Geometry> &geometries) : geometries(geometries) {}

Instance::Instance(const glm::mat4 &transform, size_t mesh_id)
    : transform(transform), mesh_id(mesh_id)
{
}
