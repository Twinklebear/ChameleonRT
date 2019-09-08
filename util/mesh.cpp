#include "mesh.h"

uint32_t Geometry::num_tris() const
{
    return indices.size();
}
