#pragma once

#include <vector>
#include <glm/glm.hpp>

struct DisneyMaterial {
	glm::vec3 base_color = glm::vec3(0.9);
	float metallic = 0;
	float specular = 0;
	float roughness = 1;
	float specular_tint = 0;
	float anisotropy = 0;
	float sheen = 0;
	float sheen_tint = 0;
	float clearcoat = 0;
	float clearcoat_gloss = 0;
};

struct RenderBackend {
	std::vector<uint32_t> img;

	virtual void initialize(const int fb_width, const int fb_height) = 0;
	virtual void set_mesh(const std::vector<float> &verts,
			const std::vector<uint32_t> &indices) = 0;
	virtual void set_material(const DisneyMaterial &m) {}
	// Returns the rays per-second achieved, or -1 if this is not tracked
	virtual double render(const glm::vec3 &pos, const glm::vec3 &dir,
			const glm::vec3 &up, const float fovy, const bool camera_changed) = 0;
};

