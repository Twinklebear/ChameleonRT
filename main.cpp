#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>
#include <SDL.h>
#include "arcball_camera.h"
#include "imgui.h"
#include "scene.h"
#include "stb_image_write.h"
#include "util.h"
#include "util/display/display.h"
#include "util/display/gldisplay.h"
#include "util/display/imgui_impl_sdl.h"
#include "util/render_plugin.h"

const std::string USAGE =
    "Usage: <backend> <mesh.obj/gltf/glb> [options]\n"
    "Render backend libraries should be named following (lib)crt_<backend>.(dll|so)\n"
    "Options:\n"
    "\t-eye <x> <y> <z>       Set the camera position\n"
    "\t-center <x> <y> <z>    Set the camera focus point\n"
    "\t-up <x> <y> <z>        Set the camera up vector\n"
    "\t-fov <fovy>            Specify the camera field of view (in degrees)\n"
    "\t-camera <n>            If the scene contains multiple cameras, specify which\n"
    "\t                       should be used. Defaults to the first camera\n"
    "\t-img <x> <y>           Specify the window dimensions. Defaults to 1280x720\n"
    "\n";

int win_width = 1280;
int win_height = 720;

void run_app(const std::vector<std::string> &args,
             SDL_Window *window,
             Display *display,
             RenderPlugin *render_plugin);

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

    std::unique_ptr<RenderPlugin> render_plugin =
        std::make_unique<RenderPlugin>("crt_" + args[1]);
    for (size_t i = 2; i < args.size(); ++i) {
        if (args[i] == "-img") {
            win_width = std::stoi(args[++i]);
            win_height = std::stoi(args[++i]);
            continue;
        }
    }

    const uint32_t window_flags = render_plugin->get_window_flags() | SDL_WINDOW_RESIZABLE;
    if (window_flags & SDL_WINDOW_OPENGL) {
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);

        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
        SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
    }

    SDL_Window *window = SDL_CreateWindow("ChameleonRT",
                                          SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED,
                                          win_width,
                                          win_height,
                                          window_flags);

    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplSDL2_Init(window);

    render_plugin->set_imgui_context(ImGui::GetCurrentContext());
    {
        std::unique_ptr<Display> display = render_plugin->make_display(window);
        run_app(args, window, display.get(), render_plugin.get());
    }

    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}

void run_app(const std::vector<std::string> &args,
             SDL_Window *window,
             Display *display,
             RenderPlugin *render_plugin)
{
    ImGuiIO &io = ImGui::GetIO();

    std::string scene_file;
    bool got_camera_args = false;
    glm::vec3 eye(0, 0, 5);
    glm::vec3 center(0);
    glm::vec3 up(0, 1, 0);
    float fov_y = 65.f;
    size_t camera_id = 0;
    std::string validation_img_prefix;
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
        } else if (args[i] == "-validation") {
            validation_img_prefix = args[++i];
        } else if (args[i] == "-img") {
            i += 2;
        } else if (args[i][0] != '-') {
            scene_file = args[i];
            canonicalize_path(scene_file);
        }
    }

    std::unique_ptr<RenderBackend> renderer = render_plugin->make_renderer(display);

    if (!renderer) {
        std::cout << "Error: No renderer backend or invalid backend name specified\n" << USAGE;
        std::exit(1);
    }
    if (scene_file.empty()) {
        std::cout << "Error: No model file specified\n" << USAGE;
        std::exit(1);
    }

    display->resize(win_width, win_height);
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
           << "# Parameterized Meshes: " << scene.parameterized_meshes.size() << "\n"
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

    const std::string rt_backend = renderer->name();
    const std::string cpu_brand = get_cpu_brand();
    const std::string gpu_brand = display->gpu_brand();
    const std::string image_output = "chameleonrt.png";
    const std::string display_frontend = display->name();

    size_t frame_id = 0;
    float render_time = 0.f;
    float rays_per_second = 0.f;
    glm::vec2 prev_mouse(-2.f);
    bool done = false;
    bool camera_changed = true;
    bool save_image = false;
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
                    save_image = true;
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

                display->resize(win_width, win_height);
                renderer->initialize(win_width, win_height);
            }
        }

        if (camera_changed) {
            frame_id = 0;
        }

        const bool need_readback = save_image || !validation_img_prefix.empty();
        RenderStats stats = renderer->render(
            camera.eye(), camera.dir(), camera.up(), fov_y, camera_changed, need_readback);

        ++frame_id;
        camera_changed = false;

        if (save_image) {
            save_image = false;
            std::cout << "Image saved to " << image_output << "\n";
            stbi_write_png(image_output.c_str(),
                           win_width,
                           win_height,
                           4,
                           renderer->img.data(),
                           4 * win_width);
        }
        if (!validation_img_prefix.empty()) {
            const std::string img_name = validation_img_prefix + render_plugin->get_name() +
                                         "-f" + std::to_string(frame_id) + ".png";
            stbi_write_png(img_name.c_str(),
                           win_width,
                           win_height,
                           4,
                           renderer->img.data(),
                           4 * win_width);
        }

        if (frame_id == 1) {
            render_time = stats.render_time;
            rays_per_second = stats.rays_per_second;
        } else {
            render_time += stats.render_time;
            rays_per_second += stats.rays_per_second;
        }

        display->new_frame();

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
        ImGui::Text("Display Frontend: %s", display_frontend.c_str());
        ImGui::Text("%s", scene_info.c_str());

        if (ImGui::Button("Save Image")) {
            save_image = true;
        }

        ImGui::End();
        ImGui::Render();

        display->display(renderer.get());
    }
}
