#include <iostream>
#include <chrono>
#include <glm/ext.hpp>
#include "render_vulkan.h"

RenderVulkan::RenderVulkan() {
}

RenderVulkan::~RenderVulkan() {
}

std::string RenderVulkan::name() {
	return "Vulkan Ray Tracing";
}

void RenderVulkan::initialize(const int fb_width, const int fb_height) {
	frame_id = 0;
	img.resize(fb_width * fb_height);
}

void RenderVulkan::set_scene(const Scene &scene) {
	frame_id = 0;
}

RenderStats RenderVulkan::render(const glm::vec3 &pos, const glm::vec3 &dir,
		const glm::vec3 &up, const float fovy, const bool camera_changed)
{
	using namespace std::chrono;
	RenderStats stats;

	if (camera_changed) {
		frame_id = 0;
	}

	++frame_id;
	return stats;
}

