#pragma once

#include <ospray/ospray.h>
#include "render_backend.h"

struct RenderOSPRay : RenderBackend {
	OSPModel world;
	OSPCamera camera;
	OSPRenderer renderer;
	OSPFrameBuffer fb;

	RenderOSPRay();

	void initialize(const int fb_width, const int fb_height) override;
	void set_mesh(const std::vector<float> &verts,
			const std::vector<uint32_t> &indices) override;
	double render(const glm::vec3 &pos, const glm::vec3 &dir,
			const glm::vec3 &up, const float fovy, const bool camera_changed) override;
};

