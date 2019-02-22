#pragma once

#include <embree3/rtcore.h>
#include "render_backend.h"

struct RenderEmbree : RenderBackend {

	RenderEmbree();

	void initialize(const int fb_width, const int fb_height) override;
	void set_mesh(const std::vector<float> &verts,
			const std::vector<uint32_t> &indices) override;
	void render(const glm::vec3 &pos, const glm::vec3 &dir,
			const glm::vec3 &up, const float fovy) override;
};

