#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>
#include <SDL.h>
#include "arcball_camera.h"
#include "gl_core_4_5.h"
#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_sdl.h"
#include "scene.h"
#include "shader.h"
#include "stb_image_write.h"
#include "tiny_obj_loader.h"
#include "util.h"

#if ENABLE_OSPRAY
#include "ospray/render_ospray.h"
#endif
#if ENABLE_OPTIX
#include "optix/render_optix.h"
#endif
#if ENABLE_EMBREE
#include "embree/render_embree.h"
#endif
#if ENABLE_DXR
#include "dxr/render_dxr.h"
#endif
#if ENABLE_VULKAN
#include "vulkan/render_vulkan.h"
#endif

const std::string USAGE =
    "Usage: <backend> <obj_file> [options]\n"
    "Backends:\n"
#if ENABLE_OSPRAY
    "\t-ospray    Render with OSPRay\n"
#endif
#if ENABLE_OPTIX
    "\t-optix     Render with OptiX\n"
#endif
#if ENABLE_EMBREE
    "\t-embree    Render with Embree\n"
#endif
#if ENABLE_DXR
    "\t-dxr       Render with DirectX Ray Tracing\n"
#endif
#if ENABLE_VULKAN
    "\t-vulkan    Render with Vulkan Ray Tracing\n"
#endif
    "Options:\n"
    "\t-eye <x> <y> <z>       Set the camera position\n"
    "\t-center <x> <y> <z>    Set the camera focus point\n"
    "\t-up <x> <y> <z>        Set the camera up vector\n"
    "\t-fov <fovy>            Specify the camera field of view (in degrees)\n"
    "\t-camera <n>            If the scene contains multiple cameras, specify which\n"
    "\t                       should be used. Defaults to the first camera\n"
    "\n";

const std::string fullscreen_quad_vs = R"(
#version 450 core

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
#version 450 core

layout(binding=0) uniform sampler2D img;

out vec4 color;

void main(void){ 
	ivec2 uv = ivec2(gl_FragCoord.xy);
	color = texelFetch(img, uv, 0);
})";

int win_width = 1280;
int win_height = 720;

void run_app(const std::vector<std::string> &args, SDL_Window *window);

glm::vec2 transform_mouse(glm::vec2 in)
{
    return glm::vec2(in.x * 2.f / win_width - 1.f, 1.f - 2.f * in.y / win_height);
}

int main(int argc, const char **argv)
{
    const std::vector<std::string> args(argv, argv + argc);
    auto fnd_help = std::find_if(args.begin(), args.end(), [](const std::string &a) {
        return a == "-h" || a == "--help";
    });

    if (argc < 3 || fnd_help != args.end()) {
        std::cout << USAGE;
        return 1;
    }

    if (SDL_Init(SDL_INIT_EVERYTHING) != 0) {
        std::cerr << "Failed to init SDL: " << SDL_GetError() << "\n";
        return -1;
    }

    const char *glsl_version = "#version 450 core";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);

    // Create window with graphics context
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

    SDL_Window *window = SDL_CreateWindow("ChameleonRT",
                                          SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED,
                                          win_width,
                                          win_height,
                                          SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    SDL_GL_SetSwapInterval(1);
    SDL_GL_MakeCurrent(window, gl_context);

    if (ogl_LoadFunctions() == ogl_LOAD_FAILED) {
        std::cerr << "Failed to initialize OpenGL\n";
        return 1;
    }

    // Setup Dear ImGui context
    ImGui::CreateContext();

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer bindings
    ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL3_Init(glsl_version);

    run_app(args, window);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}

void run_app(const std::vector<std::string> &args, SDL_Window *window)
{
    ImGuiIO &io = ImGui::GetIO();

    std::string scene_file;
    std::unique_ptr<RenderBackend> renderer = nullptr;
    bool got_camera_args = false;
    glm::vec3 eye(0, 0, 5);
    glm::vec3 center(0);
    glm::vec3 up(0, 1, 0);
    float fov_y = 65.f;
    size_t camera_id = 0;
    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "-eye") {
            eye.x = std::stof(args[++i]);
            eye.y = std::stof(args[++i]);
            eye.z = std::stof(args[++i]);
            got_camera_args = true;
        } else if (args[i] == "-center") {
            center.x = std::stof(args[++i]);
            center.y = std::stof(args[++i]);
            center.z = std::stof(args[++i]);
            got_camera_args = true;
        } else if (args[i] == "-up") {
            up.x = std::stof(args[++i]);
            up.y = std::stof(args[++i]);
            up.z = std::stof(args[++i]);
            got_camera_args = true;
        } else if (args[i] == "-fov") {
            fov_y = std::stof(args[++i]);
            got_camera_args = true;
        } else if (args[i] == "-camera") {
            camera_id = std::stol(args[++i]);
        }
#if ENABLE_OSPRAY
        else if (args[i] == "-ospray") {
            renderer = std::make_unique<RenderOSPRay>();
        }
#endif
#if ENABLE_OPTIX
        else if (args[i] == "-optix") {
            renderer = std::make_unique<RenderOptiX>();
        }
#endif
#if ENABLE_EMBREE
        else if (args[i] == "-embree") {
            renderer = std::make_unique<RenderEmbree>();
        }
#endif
#if ENABLE_DXR
        else if (args[i] == "-dxr") {
            renderer = std::make_unique<RenderDXR>();
        }
#endif
#if ENABLE_VULKAN
        else if (args[i] == "-vulkan") {
            renderer = std::make_unique<RenderVulkan>();
        }
#endif
        else {
            scene_file = args[i];
            canonicalize_path(scene_file);
        }
    }
    if (!renderer) {
        std::cout << "Error: No renderer backend or invalid backend name specified\n" << USAGE;
        std::exit(1);
    }
    if (scene_file.empty()) {
        std::cout << "Error: No model file specified\n" << USAGE;
        std::exit(1);
    }

    renderer->initialize(win_width, win_height);

    std::string scene_info;
    {
        Scene scene(scene_file);

        std::stringstream ss;
        ss << "Scene '" << scene_file << "':\n"
           << "# Unique Triangles: " << pretty_print_count(scene.unique_tris()) << "\n"
           << "# Total Triangles: " << pretty_print_count(scene.total_tris()) << "\n"
           << "# Geometries: " << scene.num_geometries() << "\n"
           << "# Meshes: " << scene.meshes.size() << "\n"
           << "# Instances: " << scene.instances.size() << "\n"
           << "# Materials: " << scene.materials.size() << "\n"
           << "# Textures: " << scene.textures.size() << "\n"
           << "# Lights: " << scene.lights.size() << "\n"
           << "# Cameras: " << scene.cameras.size();

        scene_info = ss.str();
        std::cout << scene_info << "\n";

        renderer->set_scene(scene);

        if (!got_camera_args && !scene.cameras.empty()) {
            eye = scene.cameras[camera_id].position;
            center = scene.cameras[camera_id].center;
            up = scene.cameras[camera_id].up;
            fov_y = scene.cameras[camera_id].fov_y;
        }
    }

    ArcballCamera camera(eye, center, up);
    Shader display_render(fullscreen_quad_vs, display_texture_fs);

    GLuint render_texture;
    glGenTextures(1, &render_texture);
    glBindTexture(GL_TEXTURE_2D, render_texture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, win_width, win_height);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    GLuint vao;
    glCreateVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glDisable(GL_DEPTH_TEST);

    const std::string rt_backend = renderer->name();
    const std::string cpu_brand = get_cpu_brand();
    const std::string gpu_brand = reinterpret_cast<const char *>(glGetString(GL_RENDERER));
    const std::string image_output = "chameleonrt.png";
    stbi_flip_vertically_on_write(true);

    size_t frame_id = 0;
    float render_time = 0.f;
    float rays_per_second = 0.f;
    glm::vec2 prev_mouse(-2.f);
    bool done = false;
    bool camera_changed = true;
    while (!done) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT) {
                done = true;
            }
            if (!io.WantCaptureKeyboard && event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_ESCAPE) {
                    done = true;
                } else if (event.key.keysym.sym == SDLK_p) {
                    auto eye = camera.eye();
                    auto center = camera.center();
                    auto up = camera.up();
                    std::cout << "-eye " << eye.x << " " << eye.y << " " << eye.z
                              << " -center " << center.x << " " << center.y << " " << center.z
                              << " -up " << up.x << " " << up.y << " " << up.z << " -fov "
                              << fov_y << "\n";
                } else if (event.key.keysym.sym == SDLK_s) {
                    std::cout << "Image saved to " << image_output << "\n";
                    stbi_write_png(image_output.c_str(),
                                   win_width,
                                   win_height,
                                   4,
                                   renderer->img.data(),
                                   4 * win_width);
                }
            }
            if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE &&
                event.window.windowID == SDL_GetWindowID(window)) {
                done = true;
            }
            if (!io.WantCaptureMouse) {
                if (event.type == SDL_MOUSEMOTION) {
                    const glm::vec2 cur_mouse =
                        transform_mouse(glm::vec2(event.motion.x, event.motion.y));
                    if (prev_mouse != glm::vec2(-2.f)) {
                        if (event.motion.state & SDL_BUTTON_LMASK) {
                            camera.rotate(prev_mouse, cur_mouse);
                            camera_changed = true;
                        } else if (event.motion.state & SDL_BUTTON_RMASK) {
                            camera.pan(cur_mouse - prev_mouse);
                            camera_changed = true;
                        }
                    }
                    prev_mouse = cur_mouse;
                } else if (event.type == SDL_MOUSEWHEEL) {
                    camera.zoom(event.wheel.y * 0.1);
                    camera_changed = true;
                }
            }
            if (event.type == SDL_WINDOWEVENT &&
                event.window.event == SDL_WINDOWEVENT_RESIZED) {
                frame_id = 0;
                win_width = event.window.data1;
                win_height = event.window.data2;
                io.DisplaySize.x = win_width;
                io.DisplaySize.y = win_height;
                renderer->initialize(win_width, win_height);

                glDeleteTextures(1, &render_texture);
                glGenTextures(1, &render_texture);
                // Setup the render textures for color and normals
                glBindTexture(GL_TEXTURE_2D, render_texture);
                glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, win_width, win_height);

                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            }
        }

        if (camera_changed) {
            frame_id = 0;
        }

        RenderStats stats =
            renderer->render(camera.eye(), camera.dir(), camera.up(), fov_y, camera_changed);
        ++frame_id;

        if (frame_id == 1) {
            render_time = stats.render_time;
            rays_per_second = stats.rays_per_second;
        } else {
            render_time += stats.render_time;
            rays_per_second += stats.rays_per_second;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame(window);
        ImGui::NewFrame();

        ImGui::Begin("Render Info");
        ImGui::Text("Render Time: %.3f ms/frame (%.1f FPS)",
                    render_time / frame_id,
                    1000.f / (render_time / frame_id));

        if (stats.rays_per_second > 0) {
            const std::string rays_per_sec = pretty_print_count(rays_per_second / frame_id);
            ImGui::Text("Rays per-second: %sRay/s", rays_per_sec.c_str());
        }

        ImGui::Text("Total Application Time: %.3f ms/frame (%.1f FPS)",
                    1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);
        ImGui::Text("RT Backend: %s", rt_backend.c_str());
        ImGui::Text("CPU: %s", cpu_brand.c_str());
        ImGui::Text("GPU: %s", gpu_brand.c_str());
        ImGui::Text("Accumulated Frames: %llu", frame_id);
        ImGui::Text("%s", scene_info.c_str());

        if (ImGui::Button("Save Image")) {
            std::cout << "Image saved to " << image_output << "\n";
            stbi_write_png(image_output.c_str(),
                           win_width,
                           win_height,
                           4,
                           renderer->img.data(),
                           4 * win_width);
        }

        ImGui::End();

        // Rendering
        ImGui::Render();
        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);

        glTexSubImage2D(GL_TEXTURE_2D,
                        0,
                        0,
                        0,
                        win_width,
                        win_height,
                        GL_RGBA,
                        GL_UNSIGNED_BYTE,
                        renderer->img.data());

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(display_render.program);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        SDL_GL_SwapWindow(window);

        camera_changed = false;
    }
}
