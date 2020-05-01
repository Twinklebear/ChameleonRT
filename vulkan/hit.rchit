#version 460

#include "util.glsl"

layout(location = PRIMARY_RAY) rayPayloadInEXT RayPayload payload;

hitAttributeEXT vec3 attrib;

layout(buffer_reference, buffer_reference_align=8, scalar) buffer VertexBuffer {
    vec3 v[];
};

layout(buffer_reference, buffer_reference_align=8, scalar) buffer IndexBuffer {
    uvec3 i[];
};

layout(buffer_reference, buffer_reference_align=8, scalar) buffer NormalBuffer {
    vec3 n[];
};

layout(buffer_reference, buffer_reference_align=8, scalar) buffer UVBuffer {
    vec2 uv[];
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

    vec2 uv = vec2(0);
    if (num_uvs > 0) {
        const vec2 uva = uvs.uv[idx.x];
        const vec2 uvb = uvs.uv[idx.y];
        const vec2 uvc = uvs.uv[idx.z];
        uv = (1.f - attrib.x - attrib.y) * uva
            + attrib.x * uvb + attrib.y * uvc;
    }

    mat3 inv_transp = transpose(mat3(gl_WorldToObjectEXT));
    payload.normal = normalize(inv_transp * n);
    payload.dist = gl_RayTmaxEXT;
    payload.uv = uv;
    payload.material_id = material_id;
}

