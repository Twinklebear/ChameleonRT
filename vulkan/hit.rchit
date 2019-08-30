#version 460
#extension GL_NV_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

// No typedefs GLSL?
#define uint32_t uint

layout(location = 0) rayPayloadInNV vec4 hitColor;

hitAttributeNV vec3 attrib;

struct uint3 {
	uint x, y, z; 
};

struct float3 {
	float x, y, z;
};

struct float2 {
	float x, y;
};

layout(binding = 0, set = 1, std430) buffer IndexBuffers {
	uint3 i[];
} index_buffers[];

layout(binding = 0, set = 2, std430) buffer VertexBuffers {
	float3 v[];
} vertex_buffers[];

layout(binding = 0, set = 3, std430) buffer NormalBuffers {
	float3 n[];
} normal_buffers[];

layout(binding = 0, set = 4, std430) buffer UVBuffers {
	float2 uv[];
} uv_buffers[];

layout(shaderRecordNV) buffer SBT {
    uint32_t normal_buf;
    uint32_t num_normals;

    uint32_t uv_buf;
    uint32_t num_uvs;
};

void main() {
	const uint3 idx = index_buffers[nonuniformEXT(gl_InstanceCustomIndexNV)].i[gl_PrimitiveID];
	const float3 va = vertex_buffers[nonuniformEXT(gl_InstanceCustomIndexNV)].v[idx.x];
	const float3 vb = vertex_buffers[nonuniformEXT(gl_InstanceCustomIndexNV)].v[idx.y];
	const float3 vc = vertex_buffers[nonuniformEXT(gl_InstanceCustomIndexNV)].v[idx.z];
	const vec3 n = normalize(cross(vec3(vb.x, vb.y, vb.z) - vec3(va.x, va.y, va.z),
				vec3(vc.x, vc.y, vc.z) - vec3(va.x, va.y, va.z))); 

	vec2 uv = vec2(0);
	if (num_uvs > 0) {
		float2 uva = uv_buffers[nonuniformEXT(uv_buf)].uv[idx.x];
		float2 uvb = uv_buffers[nonuniformEXT(uv_buf)].uv[idx.y];
		float2 uvc = uv_buffers[nonuniformEXT(uv_buf)].uv[idx.z];
		uv = (1.f - attrib.x - attrib.y) * vec2(uva.x, uva.y)
			+ attrib.x * vec2(uvb.x, uvb.y) + attrib.y * vec2(uvc.x, uvc.y);
        hitColor = vec4(uv.x, uv.y, 0.f, 1.f);
	} else {
        hitColor = vec4((n + vec3(1)) * 0.5f, 1.f);
    }
    //hitColor = vec4(1.f - attrib.x - attrib.y, attrib.x, attrib.y, 1.f);
}

