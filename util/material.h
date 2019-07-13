#pragma once

#include <string>
#include <vector>
#include <memory>
#include <glm/glm.hpp>

struct Image {
	std::string name;
	int width = -1;
	int height = -1;
	int channels = -1;
	std::vector<uint8_t> img;

	Image(const std::string &file, const std::string &name);
	Image() = default;
};

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

	float ior = 0;
	float specular_transmission = 0;

	std::shared_ptr<Image> color_texture = nullptr;
};

// The poinerless GPU version of the material
struct GPUDisneyMaterial {
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

	float ior = 0;
	float specular_transmission = 0;
	glm::vec2 pad = glm::vec2(0);

	glm::ivec4 tex_ids = glm::ivec4(-1);
	
	GPUDisneyMaterial() = default;

	// Note: these do not set the color tex id, as this dependents on
	// the backend's ordering of textures in memory
	GPUDisneyMaterial(const DisneyMaterial &d);
	GPUDisneyMaterial& operator=(const DisneyMaterial &d);
};

