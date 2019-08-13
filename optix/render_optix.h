#pragma once

#include <optix.h>
#include "render_backend.h"

struct RenderOptiX : RenderBackend {
	CUstream cuda_stream;
	CUcontext cuda_context;

	OptixDeviceContext optix_context;

	void *fb = nullptr;
	void *accum_buffer = nullptr;
	void *view_params = nullptr;
	void *mat_params = nullptr;

	int width, height;
	uint32_t frame_id = 0;

	RenderOptiX();

	void initialize(const int fb_width, const int fb_height) override;
	void set_scene(const Scene &scene) override;
	double render(const glm::vec3 &pos, const glm::vec3 &dir,
			const glm::vec3 &up, const float fovy, const bool camera_changed) override;
};

