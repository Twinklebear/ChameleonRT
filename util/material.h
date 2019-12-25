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

    // TODO: change these to only store 16bit texture ids
    int32_t base_color_texture = -1;
    int32_t metallic_texture = -1;
    int32_t specular_texture = -1;
    int32_t roughness_texture = -1;
    int32_t specular_tint_texture = -1;
    int32_t anisotropy_texture = -1;
    int32_t sheen_texture = -1;
    int32_t sheen_tint_texture = -1;
    int32_t clearcoat_texture = -1;
    int32_t clearcoat_gloss_texture = -1;
    int32_t ior_texture = -1;
    int32_t specular_transmission_texture = -1;
    uint32_t texture_channel_mask = 0;
};
