#pragma once

#include <string>
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

struct Image {
	int width = -1;
	int height = -1;
	int channels = -1;
	std::vector<uint8_t> img;

	Image(const std::string &file);
	Image() = default;
};

