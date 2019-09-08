#include "material.h"
#include <stdexcept>
#include "stb_image.h"

Image::Image(const std::string &file, const std::string &name, ColorSpace color_space)
    : name(name), color_space(color_space)
{
    stbi_set_flip_vertically_on_load(1);
    uint8_t *data = stbi_load(file.c_str(), &width, &height, &channels, 4);
    channels = 4;
    if (!data) {
        throw std::runtime_error("Failed to load " + file);
    }
    img = std::vector<uint8_t>(data, data + width * height * channels);
    stbi_image_free(data);
}

