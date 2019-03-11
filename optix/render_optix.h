#pragma once

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>
#include "render_backend.h"

struct RenderOptiX : RenderBackend {
	optix::Context context;
	optix::Buffer fb;
	int width, height;

	RenderOptiX();

	void initialize(const int fb_width, const int fb_height) override;
	void set_mesh(const std::vector<float> &verts,
			const std::vector<uint32_t> &indices) override;
	void render(const glm::vec3 &pos, const glm::vec3 &dir,
			const glm::vec3 &up, const float fovy) override;
};

