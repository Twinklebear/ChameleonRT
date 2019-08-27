#version 460
#extension GL_NV_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) rayPayloadInNV vec4 hitColor;

layout(binding = 3, set = 0, std430) buffer IndexBuffers {
	uvec3 indices[];
} index_buffers[];

layout(binding = 4, set = 0, std430) buffer VertexBuffers {
	vec3 data[];
} vertex_buffers[]; // this can be unsized? who knows the size? how's it set?

layout(shaderRecordNV) buffer SBT {
	float blueness;
};

void main() {
	// TODO: nonuniform qualifier? (do i also need in dxr?)
	const uvec3 indices = index_buffers[nonuniformEXT(gl_InstanceID)].indices[gl_PrimitiveID];
	const vec3 va = vertex_buffers[nonuniformEXT(gl_InstanceID)].data[indices.x];
	const vec3 vb = vertex_buffers[nonuniformEXT(gl_InstanceID)].data[indices.y];
	const vec3 vc = vertex_buffers[nonuniformEXT(gl_InstanceID)].data[indices.z];
	const vec3 n = normalize(cross(vb - va, vc - va)); 

	hitColor = vec4((n + vec3(1)) * 0.5f, 1.f);
}

