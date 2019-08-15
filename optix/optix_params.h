#pragma once

// TODO: This can be made to match the host-side struct nicer since we
// won't need to worry about any layout/padding weirdness
struct MaterialParams {
#ifdef __CUDA_ARCH__
	float3 base_color;
#else
	glm::vec3 base_color;
#endif
	float metallic;

	float specular;
	float roughness;
	float specular_tint;
	float anisotropy;

	float sheen;
	float sheen_tint;
	float clearcoat;
	float clearcoat_gloss;

	float ior;
	float specular_transmission;
	float pad1, pad2;

	float pad3,pad4,pad5,pad6;
	//int32_t color_tex_id;
	//glm::ivec3 tex_pad = glm::ivec3(-1);
};

struct LaunchParams {
#ifdef __CUDA_ARCH__
	float4 cam_pos;
	float4 cam_du;
	float4 cam_dv;
	float4 cam_dir_top_left;
#else
	glm::vec4 cam_pos;
	glm::vec4 cam_du;
	glm::vec4 cam_dv;
	glm::vec4 cam_dir_top_left;
#endif

	uint32_t frame_id;

#ifdef __CUDA_ARCH__
	uchar4 *framebuffer;
	float4 *accum_buffer;
#else
	CUdeviceptr framebuffer;
	CUdeviceptr accum_buffer;
#endif

	OptixTraversableHandle scene;
};

struct RayGenParams {
#ifdef __CUDA_ARCH__
	MaterialParams *mat_params;
#else
	CUdeviceptr mat_params;
#endif
};

struct HitGroupParams {
#ifdef __CUDA_ARCH__
	float3 *vertex_buffer;
	uint3 *index_buffer;
#else
	CUdeviceptr vertex_buffer;
	CUdeviceptr index_buffer;
#endif
};

