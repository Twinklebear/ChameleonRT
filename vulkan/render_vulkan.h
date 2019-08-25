#pragma once

#include <vulkan/vulkan.h>
#include "vulkan_utils.h"
#include "render_backend.h"

struct RenderVulkan : RenderBackend {

	size_t frame_id = 0;

	RenderVulkan();
	virtual ~RenderVulkan();

	std::string name() override;
	void initialize(const int fb_width, const int fb_height) override;
	void set_scene(const Scene &scene) override;
	RenderStats render(const glm::vec3 &pos, const glm::vec3 &dir,
			const glm::vec3 &up, const float fovy, const bool camera_changed) override;

private:
	// TODO WILL: it's very similar to DXR
	/*
	void build_raytracing_pipeline();
	void build_shader_resource_heap();
	void build_shader_binding_table();
	void update_view_parameters(const glm::vec3 &pos, const glm::vec3 &dir,
		const glm::vec3 &up, const float fovy);
	void build_descriptor_heap();
	void sync_gpu();
	*/
};

