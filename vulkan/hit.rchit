#version 460
#extension GL_NV_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) rayPayloadInNV vec4 hitColor;

struct uint3 {
	uint x, y, z; 
};

struct float3 {
	float x, y, z;
};

layout(binding = 3, set = 0, std430) buffer IndexBuffers {
	uint3 indices[];
} index_buffers[];

layout(binding = 4, set = 0, std430) buffer VertexBuffers {
	float3 data[];
} vertex_buffers[]; // this can be unsized? who knows the size? how's it set?

layout(shaderRecordNV) buffer SBT {
	float blueness;
};

void main() {
	const uint3 indices = index_buffers[nonuniformEXT(gl_InstanceID)].indices[gl_PrimitiveID];
	const float3 va = vertex_buffers[nonuniformEXT(gl_InstanceID)].data[indices.x];
	const float3 vb = vertex_buffers[nonuniformEXT(gl_InstanceID)].data[indices.y];
	const float3 vc = vertex_buffers[nonuniformEXT(gl_InstanceID)].data[indices.z];
	const vec3 n = normalize(cross(vec3(vb.x, vb.y, vb.z) - vec3(va.x, va.y, va.z),
				vec3(vc.x, vc.y, vc.z) - vec3(va.x, va.y, va.z))); 

	hitColor = vec4((n + vec3(1)) * 0.5f, 1.f);
}

