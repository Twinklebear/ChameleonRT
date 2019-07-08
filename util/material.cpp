#include <iostream>
#include "material.h"
#include "stb_image.h"

Image::Image(const std::string &file) {
	uint8_t *data = stbi_load(file.c_str(), &width, &height, &channels, 0);
	if (!data) {
		throw std::runtime_error("Failed to load " + file);
	}
	img = std::vector<uint8_t>(data, data + width * height * channels);
	stbi_image_free(data);

	std::cout << "Img: " << file << ": " << width << "x" << height
		<< ", " << img.size() "b\n";
}

