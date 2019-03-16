#pragma once

#include <vector>
#include <glm/glm.hpp>

struct RenderBackend {
	std::vector<uint32_t> img;

	virtual void initialize(const int fb_width, const int fb_height) = 0;
	virtual void set_mesh(const std::vector<float> &verts,
			const std::vector<uint32_t> &indices) = 0;
	// Returns the rays per-second achieved, or -1 if this is not tracked
	virtual double render(const glm::vec3 &pos, const glm::vec3 &dir,
			const glm::vec3 &up, const float fovy) = 0;
};

