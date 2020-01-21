#version 460

#include "util.glsl"

layout(location = PRIMARY_RAY) rayPayloadInNV RayPayload payload;

hitAttributeNV vec3 attrib;

layout(binding = 0, set = 1, std430) buffer IndexBuffers {
    pack_uint3 i[];
} index_buffers[];

layout(binding = 0, set = 2, std430) buffer VertexBuffers {
    pack_float3 v[];
} vertex_buffers[];

layout(binding = 0, set = 3, std430) buffer NormalBuffers {
    pack_float3 n[];
} normal_buffers[];

layout(binding = 0, set = 4, std430) buffer UVBuffers {
    vec2 uv[];
} uv_buffers[];

layout(shaderRecordNV) buffer SBT {
    uint32_t vert_buf;
    uint32_t idx_buf;
    uint32_t normal_buf;
    uint32_t uv_buf;
    uint32_t material_id;
};

void main() {
    const uvec3 idx = unpack_uint3(index_buffers[nonuniformEXT(idx_buf)].i[gl_PrimitiveID]);
    const vec3 va = unpack_float3(vertex_buffers[nonuniformEXT(vert_buf)].v[idx.x]);
    const vec3 vb = unpack_float3(vertex_buffers[nonuniformEXT(vert_buf)].v[idx.y]);
    const vec3 vc = unpack_float3(vertex_buffers[nonuniformEXT(vert_buf)].v[idx.z]);
    const vec3 n = normalize(cross(vb - va, vc - va));

    vec2 uv = vec2(0);
    if (uv_buf != uint32_t(-1)) {
        vec2 uva = uv_buffers[nonuniformEXT(uv_buf)].uv[idx.x];
        vec2 uvb = uv_buffers[nonuniformEXT(uv_buf)].uv[idx.y];
        vec2 uvc = uv_buffers[nonuniformEXT(uv_buf)].uv[idx.z];
        uv = (1.f - attrib.x - attrib.y) * uva
            + attrib.x * uvb + attrib.y * uvc;
    }

    mat3 inv_transp = transpose(mat3(gl_WorldToObjectNV));
    payload.normal = normalize(inv_transp * n);
    payload.dist = gl_RayTmaxNV;
    payload.uv = uv;
    payload.material_id = material_id;
}

