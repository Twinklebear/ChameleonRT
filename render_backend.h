#pragma once

#include <vector>
#include <glm/glm.hpp>

struct RenderBackend {
	std::vector<uint32_t> img;

	virtual void initialize(const float fovy,
			const int fb_width, const int fb_height) = 0;
	virtual void set_mesh(const std::vector<float> &verts,
			const std::vector<int32_t> &indices) = 0;
	virtual void render(const glm::vec3 &pos, const glm::vec3 &dir,
			const glm::vec3 &up) = 0;
};

