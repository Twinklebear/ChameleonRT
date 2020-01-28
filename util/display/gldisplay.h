#pragma once

#include <memory>
#include <SDL.h>
#include "display.h"
#include "gl_core_4_5.h"
#include "shader.h"
#include <glm/glm.hpp>

struct GLDisplay : Display {
    SDL_Window *window;
    SDL_GLContext gl_context;
    GLuint render_texture;
    GLuint vao;
    std::unique_ptr<Shader> display_render;
    glm::uvec2 fb_dims;

    GLDisplay(SDL_Window *window);

    ~GLDisplay() override;

    std::string gpu_brand() override;

    std::string name() override;

    void resize(const int fb_width, const int fb_height) override;

    void new_frame() override;

    void display(const std::vector<uint32_t> &img) override;

    void display_native(const GLuint img);
};
