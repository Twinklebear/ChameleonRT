#include <iostream>
#include <algorithm>
#include <numeric>
#include <memory>
#include <vector>
#include <array>
#include <SDL.h>
#include "gl_core_4_5.h"
#include "tiny_obj_loader.h"
#include "imgui.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_opengl3.h"
#include "arcball_camera.h"
#include "shader.h"
#include "util.h"
#include "scene.h"

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

const std::string USAGE = "Usage: (backend) <obj_file> [camera]\n"
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

glm::vec2 transform_mouse(glm::vec2 in) {
	return glm::vec2(in.x * 2.f / win_width - 1.f, 1.f - 2.f * in.y / win_height);
}

int main(int argc, const char **argv) {
	if (argc < 3) {
		std::cout << USAGE;
		return 1;
	}

	if (SDL_Init(SDL_INIT_EVERYTHING) != 0) {
		std::cerr << "Failed to init SDL: " << SDL_GetError() << "\n";
		return -1;
	}

	const char* glsl_version = "#version 450 core";
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);

	// Create window with graphics context
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
	SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

	SDL_Window* window = SDL_CreateWindow("rtobj",
			SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, win_width, win_height,
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

	std::vector<std::string> args(argv, argv + argc);

	run_app(args, window);

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplSDL2_Shutdown();
	ImGui::DestroyContext();

	SDL_GL_DeleteContext(gl_context);
	SDL_DestroyWindow(window);
	SDL_Quit();

	return 0;
}

void run_app(const std::vector<std::string> &args, SDL_Window *window) {
	ImGuiIO& io = ImGui::GetIO();

	std::string scene_file;
	std::unique_ptr<RenderBackend> renderer = nullptr;
	glm::vec3 eye(0, 0, 5);
	glm::vec3 center(0);
	glm::vec3 up(0, 1, 0);
	for (size_t i = 1; i < args.size(); ++i) {
		if (args[i] == "-eye") {
			eye.x = std::stof(args[++i]);
			eye.y = std::stof(args[++i]);
			eye.z = std::stof(args[++i]);
		} else if (args[i] == "-center") {
			center.x = std::stof(args[++i]);
			center.y = std::stof(args[++i]);
			center.z = std::stof(args[++i]);
		} else if (args[i] == "-up") {
			up.x = std::stof(args[++i]);
			up.y = std::stof(args[++i]);
			up.z = std::stof(args[++i]);
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
		else {
			scene_file = args[i];
			canonicalize_path(scene_file);
		}
	}
	if (!renderer) {
		std::cout << "Error: No renderer backend or invalid backend name specified\n" << USAGE;
		std::exit(1);
	}

	renderer->initialize(win_width, win_height);

	size_t total_tris = 0;
	std::string num_tris;
	{
		Scene scene = Scene::load_obj(scene_file);
		std::cout << "Scene '" << scene_file << "' loaded:\n"
			<< "# Triangles: " << scene.total_tris() << "\n"
			<< "# Meshes: " << scene.meshes.size() << "\n"
			<< "# Materials: " << scene.materials.size() << "\n"
			<< "# Textures: " << scene.textures.size() << "\n"
			<< "# Lights: " << scene.lights.size() << "\n";

		total_tris = std::accumulate(scene.meshes.begin(), scene.meshes.end(), size_t(0),
				[](const size_t &s, const Mesh &m) { return s + m.indices.size(); });
		num_tris = pretty_print_count(total_tris);
		renderer->set_scene(scene);
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
	const std::string gpu_brand = reinterpret_cast<const char*>(glGetString(GL_RENDERER));

	size_t frame_id = 0;
	double avg_rays_per_sec = 0.f;
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
						<< " -up " << up.x << " " << up.y << " " << up.z << "\n";
				}
			}
			if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE
					&& event.window.windowID == SDL_GetWindowID(window)) {
				done = true;
			}
			if (!io.WantCaptureMouse) {
				if (event.type == SDL_MOUSEMOTION) {
					const glm::vec2 cur_mouse = transform_mouse(glm::vec2(event.motion.x, event.motion.y));
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
			if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_RESIZED) {
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

		const double rays_per_sec =
			renderer->render(camera.eye(), camera.dir(), camera.up(), 65.f, camera_changed);

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplSDL2_NewFrame(window);
		ImGui::NewFrame();

		ImGui::Begin("Debug Panel");
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
				1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		ImGui::Text("# Triangles: %s", num_tris.c_str());
		ImGui::Text("RT Backend: %s", rt_backend.c_str());
		ImGui::Text("CPU: %s", cpu_brand.c_str());
		ImGui::Text("GPU: %s", gpu_brand.c_str());
		ImGui::Text("Accumulated Frames: %llu", frame_id);

		// We don't instrument inside OSPRay so we don't show these statistics for it
		if (rays_per_sec > 0.0) {
			avg_rays_per_sec = avg_rays_per_sec + (rays_per_sec - avg_rays_per_sec) / (frame_id + 1);
			// TODO: I need to compute the stats properly now that we send secondary rays.
			// This will take some additional work since we basically need an additional buffer
			// in each renderer to track the # rays launched for each pixel
			//ImGui::Text("Avg. Rays/sec: %s/sec", pretty_print_count(avg_rays_per_sec).c_str());
		}

		ImGui::End();

		// Rendering
		ImGui::Render();
		glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);

		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, win_width, win_height, GL_RGBA,
				GL_UNSIGNED_BYTE, renderer->img.data());

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glUseProgram(display_render.program);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		SDL_GL_SwapWindow(window);

		++frame_id;
		camera_changed = false;
	}
}

