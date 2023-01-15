#version 460

#include "util.glsl"

layout(location = PRIMARY_RAY) rayPayloadInEXT RayPayload payload;

hitAttributeEXT vec3 attrib;

layout(buffer_reference, scalar) buffer VertexBuffer {
    vec3 v[];
};

layout(buffer_reference, scalar) buffer IndexBuffer {
    uvec3 i[];
};

layout(buffer_reference, scalar) buffer NormalBuffer {
    vec3 n[];
};

layout(buffer_reference, buffer_reference_align=8) buffer UVBuffer {
    vec2 uv;
};

layout(shaderRecordEXT, std430) buffer SBT {
    VertexBuffer verts;
    IndexBuffer indices;
    NormalBuffer normals;
    UVBuffer uvs;
    uint32_t num_normals;
    uint32_t num_uvs;
    uint32_t material_id;
};

void main() {
    const uvec3 idx = indices.i[gl_PrimitiveID];
    const vec3 va = verts.v[idx.x];
    const vec3 vb = verts.v[idx.y];
    const vec3 vc = verts.v[idx.z];
    const vec3 n = normalize(cross(vb - va, vc - va));

    mat3 inv_transp = transpose(mat3(gl_WorldToObjectEXT));
    payload.normal = normalize(inv_transp * n);
    payload.dist = gl_RayTmaxEXT;
}

