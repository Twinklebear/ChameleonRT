#pragma once

#include <optix.h>
#include "render_backend.h"

struct RenderOptiX : RenderBackend {
	CUcontext cuda_context;
	CUstream cuda_stream;

	OptixDeviceContext optix_context;

	// TODO: Technically these should all actually be CUdeviceptr
	void *framebuffer = nullptr;
	void *accum_buffer = nullptr;
	void *launch_params = nullptr;
	void *mat_params = nullptr;

	void *vertices = nullptr;
	void *indices = nullptr;

	void *blas_buffer = nullptr;
	OptixTraversableHandle blas_handle = 0;

	void *instance_buffer = nullptr;
	void *tlas_buffer = nullptr;
	OptixTraversableHandle tlas_handle = 0;

	OptixModule module;
	OptixPipeline pipeline;

	void *shader_table_data = nullptr;
	OptixShaderBindingTable shader_table;

	int width, height;
	uint32_t frame_id = 0;

	RenderOptiX();
	~RenderOptiX();

	void initialize(const int fb_width, const int fb_height) override;
	void set_scene(const Scene &scene) override;
	double render(const glm::vec3 &pos, const glm::vec3 &dir,
			const glm::vec3 &up, const float fovy, const bool camera_changed) override;

private:
	void build_raytracing_pipeline();
	void update_view_parameters(const glm::vec3 &pos, const glm::vec3 &dir,
		const glm::vec3 &up, const float fovy);
	void sync_gpu();
};

