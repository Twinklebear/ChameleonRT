#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "scene.h"

struct RenderBackend {
	std::vector<uint32_t> img;

	virtual void initialize(const int fb_width, const int fb_height) = 0;
	virtual void set_mesh(const std::vector<float> &verts,
			const std::vector<uint32_t> &indices) = 0;
	// TODO: Just a temp thing to learn about setting this up in DXR
	// kind of tuned to be a bit hacky for tinyobj style of loading the obj files
	// all in one big buffer
	virtual void set_scene(const Scene &scene) {}
	virtual void set_material(const DisneyMaterial &m){}
	// Returns the rays per-second achieved, or -1 if this is not tracked
	virtual double render(const glm::vec3 &pos, const glm::vec3 &dir,
			const glm::vec3 &up, const float fovy, const bool camera_changed) = 0;
};

