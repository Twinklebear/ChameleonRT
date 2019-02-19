#pragma once

#include "render_backend.h"

struct RenderOptiX : RenderBackend {

	RenderOptiX();

	void initialize(const float fovy,
			const int fb_width, const int fb_height) override;
	void set_mesh(const std::vector<float> &verts,
			const std::vector<int32_t> &indices) override;
	void render(const glm::vec3 &pos, const glm::vec3 &dir,
			const glm::vec3 &up) override;
};

