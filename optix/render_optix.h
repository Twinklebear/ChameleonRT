#pragma once

#include <optix.h>
#include "render_backend.h"
#include "optix_utils.h"

struct RenderOptiX : RenderBackend {
	CUcontext cuda_context;
	CUstream cuda_stream;

	OptixDeviceContext optix_context;

	optix::Buffer framebuffer, accum_buffer,
		launch_params, mat_params;

	optix::Buffer vertices, indices;

	optix::Buffer blas_buffer;
	OptixTraversableHandle blas_handle = 0;

	optix::Buffer instance_buffer;
	optix::Buffer tlas_buffer;
	OptixTraversableHandle tlas_handle = 0;

	OptixPipeline pipeline;

	optix::Buffer shader_table_data;
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

