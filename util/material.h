#pragma once

#include <memory>
#include <string>
#include <vector>
#include <glm/glm.hpp>

enum ColorSpace { LINEAR, SRGB };

struct Image {
    std::string name;
    int width = -1;
    int height = -1;
    int channels = -1;
    std::vector<uint8_t> img;
    ColorSpace color_space = LINEAR;

    Image(const std::string &file, const std::string &name, ColorSpace color_space = LINEAR);
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

    float ior = 1.5;
    float specular_transmission = 0;
    glm::vec2 pad = glm::vec2(0);

    int32_t color_tex_id = -1;
    glm::ivec3 tex_pad = glm::ivec3(-1);
};
