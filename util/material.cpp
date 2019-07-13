#include <iostream>
#include "material.h"
#include "stb_image.h"

Image::Image(const std::string &file) {
	// TODO WILL: For now we force 4 channels for dx12 to have RGBA, but
	// does it have some built in conversion support? Or I should write a converson
	// step before uploading
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

