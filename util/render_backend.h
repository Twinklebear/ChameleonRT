#pragma once

#include <vector>
#include <glm/glm.hpp>

struct DisneyMaterial {
	glm::vec3 base_color = glm::vec3(0.9f);
	float metallic = 0;

	float specular = 0;
	float roughness = 1;
	float specular_tint = 0;
	float anisotropy = 0;

	float sheen = 0;
	float sheen_tint = 0;
	float clearcoat = 0;
	float clearcoat_gloss = 0;

	float ior = 1.5;
	float specular_transmission = 0;
	glm::vec2 pad = glm::vec2(0.f);
};

struct RenderBackend {
	std::vector<uint32_t> img;

	virtual void initialize(const int fb_width, const int fb_height) = 0;
	virtual void set_mesh(const std::vector<float> &verts,
			const std::vector<uint32_t> &indices) = 0;
	// TODO: Just a temp thing to learn about setting this up in DXR
	// kind of tuned to be a bit hacky for tinyobj style of loading the obj files
	// all in one big buffer
	virtual void set_scene(const std::vector<float> &verts,
			const std::vector<std::vector<uint32_t>> &indices,
			const std::vector<uint32_t> &material_ids){}
	virtual void set_material(const DisneyMaterial &m) {}
	virtual void set_materials(const std::vector<DisneyMaterial> &materials){}
	// Returns the rays per-second achieved, or -1 if this is not tracked
	virtual double render(const glm::vec3 &pos, const glm::vec3 &dir,
			const glm::vec3 &up, const float fovy, const bool camera_changed) = 0;
};

