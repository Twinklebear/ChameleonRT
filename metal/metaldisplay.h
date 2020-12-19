#pragma once

#include <SDL.h>
#include <SDL_syswm.h>
#include "display/display.h"
#include <glm/glm.hpp>

namespace metal {
struct Context;
struct Texture2D;
struct ShaderLibrary;
struct ComputePipeline;
}

struct MetalDisplayData;

struct MetalDisplay : Display {
    std::shared_ptr<metal::Context> context;
    std::shared_ptr<MetalDisplayData> data;
    std::shared_ptr<metal::Texture2D> upload_texture;
    std::shared_ptr<metal::ShaderLibrary> shader_library;
    std::shared_ptr<metal::ComputePipeline> pipeline;

    glm::uvec2 fb_dims;

    MetalDisplay(SDL_Window *window);

    ~MetalDisplay();

    std::string gpu_brand() override;

    std::string name() override;

    void resize(const int fb_width, const int fb_height) override;

    void new_frame() override;

    void display(const std::vector<uint32_t> &img) override;

    void display_native(std::shared_ptr<metal::Texture2D> &img);
};

