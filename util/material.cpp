#include <iostream>
#include "material.h"
#include "stb_image.h"

Image::Image(const std::string &file, const std::string &name) : name(name) {
	stbi_set_flip_vertically_on_load(1);
	uint8_t *data = stbi_load(file.c_str(), &width, &height, &channels, 4);
	channels = 4;
	if (!data) {
		throw std::runtime_error("Failed to load " + file);
	}
	img = std::vector<uint8_t>(data, data + width * height * channels);
	stbi_image_free(data);

	std::cout << "Img: " << file << ": " << width << "x" << height
		<< ", " << img.size() << "b\n";
}

GPUDisneyMaterial::GPUDisneyMaterial(const DisneyMaterial &d)
	: base_color(d.base_color),
	metallic(d.metallic),
	specular(d.specular),
	roughness(d.roughness),
	specular_tint(d.specular_tint),
	anisotropy(d.anisotropy),
	sheen(d.sheen),
	sheen_tint(d.sheen_tint),
	clearcoat(d.clearcoat),
	clearcoat_gloss(d.clearcoat_gloss),
	ior(d.ior),
	specular_transmission(d.specular_transmission)
{}

GPUDisneyMaterial& GPUDisneyMaterial::operator=(const DisneyMaterial &d) {
	base_color = d.base_color;
	metallic = d.metallic;

	specular = d.specular;
	roughness = d.roughness;
	specular_tint = d.specular_tint;
	anisotropy = d.anisotropy;

	sheen = d.sheen;
	sheen_tint = d.sheen_tint;
	clearcoat = d.clearcoat;
	clearcoat_gloss = d.clearcoat_gloss;

	ior = d.ior;
	specular_transmission = d.specular_transmission;

	tex_ids = glm::ivec4(-1);
	return *this;
}

