#include "gldisplay.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include "imgui_impl_opengl3.h"
#include "imgui_impl_sdl.h"

const std::string fullscreen_quad_vs = R"(
#version 330 core

const vec4 pos[4] = vec4[4](
	vec4(-1, 1, 0.5, 1),
	vec4(-1, -1, 0.5, 1),
	vec4(1, 1, 0.5, 1),
	vec4(1, -1, 0.5, 1)
);

void main(void){
	gl_Position = pos[gl_VertexID];
}
)";

const std::string display_texture_fs = R"(
#version 330 core

uniform sampler2D img;

out vec4 color;

void main(void){ 
	ivec2 uv = ivec2(gl_FragCoord.x, textureSize(img, 0).y - gl_FragCoord.y);
	color = texelFetch(img, uv, 0);
})";

GLDisplay::GLDisplay(SDL_Window *win)
    : window(win), gl_context(SDL_GL_CreateContext(win)), render_texture(-1)
{
    SDL_GL_SetSwapInterval(0);
    SDL_GL_MakeCurrent(window, gl_context);

    if (!gladLoadGL()) {
        throw std::runtime_error("Failed to initialize OpenGL");
    }

    ImGui_ImplOpenGL3_Init("#version 330 core");

    display_render = std::make_unique<Shader>(fullscreen_quad_vs, display_texture_fs);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glDisable(GL_DEPTH_TEST);
}

GLDisplay::~GLDisplay()
{
    glDeleteVertexArrays(1, &vao);
    if (render_texture != -1) {
        glDeleteTextures(1, &render_texture);
    }
    ImGui_ImplOpenGL3_Shutdown();
    SDL_GL_DeleteContext(gl_context);
}

std::string GLDisplay::gpu_brand()
{
    return reinterpret_cast<const char *>(glGetString(GL_RENDERER));
}

std::string GLDisplay::name()
{
    return "OpenGL";
}

void GLDisplay::resize(const int fb_width, const int fb_height)
{
    fb_dims = glm::uvec2(fb_width, fb_height);
    if (render_texture != -1) {
        glDeleteTextures(1, &render_texture);
    }
    glGenTextures(1, &render_texture);
    glBindTexture(GL_TEXTURE_2D, render_texture);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA8,
                 fb_dims.x,
                 fb_dims.y,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void GLDisplay::new_frame()
{
    ImGui_ImplOpenGL3_NewFrame();
}

void GLDisplay::display(RenderBackend *renderer)
{
    auto *gl_native = dynamic_cast<GLNativeRenderer *>(renderer);
    if (gl_native) {
        display_native(gl_native->gl_display_texture);
    } else {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, render_texture);
        glTexSubImage2D(GL_TEXTURE_2D,
                        0,
                        0,
                        0,
                        fb_dims.x,
                        fb_dims.y,
                        GL_RGBA,
                        GL_UNSIGNED_BYTE,
                        renderer->img.data());

        display_native(render_texture);
    }
}

void GLDisplay::display_native(const GLuint img)
{
    glViewport(0, 0, fb_dims.x, fb_dims.y);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(display_render->program);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, img);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    SDL_GL_SwapWindow(window);
}
