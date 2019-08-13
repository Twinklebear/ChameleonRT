#pragma once

// TODO: something to better share these structs between cuda and C++ side?
struct LaunchParams {
	glm::vec4 cam_pos;
	glm::vec4 cam_du;
	glm::vec4 cam_dv;
	glm::vec4 cam_dir_top_left;

	uint32_t frame_id;

	CUdeviceptr framebuffer;
	CUdeviceptr accum_buffer;

	OptixTraversableHandle scene;
};

struct RayGenRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	CUdeviceptr mat_params;
};

struct MissRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct HitGroupRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	CUdeviceptr vertex_buffer;
	CUdeviceptr index_buffer;
};

