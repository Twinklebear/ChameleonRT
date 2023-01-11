#pragma once

#include <memory>
#include <string>
#include <vector>
#include "texture_channel_mask.h"
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
    Image(const uint8_t *buf,
          int width,
          int height,
          int channels,
          const std::string &name,
          ColorSpace color_space = LINEAR);
    Image() = default;
};

struct DisneyMaterial {
    glm::vec3 base_color = glm::vec3(0.9f);
    float metallic = 0.f;

    float specular = 0.f;
    float roughness = 1.f;
    float specular_tint = 0.f;
    float anisotropy = 0.f;

    float sheen = 0.f;
    float sheen_tint = 0.f;
    float clearcoat = 0.f;
    float clearcoat_gloss = 0.f;

    float ior = 1.5f;
    float specular_transmission = 0.f;
    glm::vec2 pad = glm::vec2(0.f);
};
