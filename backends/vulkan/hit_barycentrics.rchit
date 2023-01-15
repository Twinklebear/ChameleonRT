#version 460

#include "util.glsl"

layout(location = PRIMARY_RAY) rayPayloadInEXT BarycentricsPayload payload;

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
    payload.bary = vec2(attrib.x, attrib.y);
    payload.dist = gl_RayTmaxEXT;
}

