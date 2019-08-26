#pragma once

#include <memory>
#include <vulkan/vulkan.h>
#include "vulkan_utils.h"
#include "vulkanrt_utils.h"
#include "render_backend.h"

struct RenderVulkan : RenderBackend {
	vk::Device device;

	std::shared_ptr<vk::Buffer> shader_table, img_readback_buf;

	std::shared_ptr<vk::Texture2D> render_target;

	std::unique_ptr<vk::TriangleMesh> mesh;
	std::unique_ptr<vk::TopLevelBVH> scene;

	VkCommandPool command_pool = VK_NULL_HANDLE;
	VkCommandBuffer command_buffer = VK_NULL_HANDLE;

	VkPipeline rt_pipeline = VK_NULL_HANDLE;
	VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
	VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;

	VkDescriptorPool desc_pool = VK_NULL_HANDLE;
	VkDescriptorSet desc_set = VK_NULL_HANDLE;
	
	VkFence fence = VK_NULL_HANDLE;
	
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
	void build_raytracing_pipeline();
	void build_shader_descriptor_table();
	void build_shader_binding_table();
	void update_view_parameters(const glm::vec3 &pos, const glm::vec3 &dir,
		const glm::vec3 &up, const float fovy);
};

