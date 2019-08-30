#pragma once

#include <memory>
#include <unordered_map>
#include <vulkan/vulkan.h>
#include "render_backend.h"
#include "vulkan_utils.h"
#include "vulkanrt_utils.h"

struct RenderVulkan : RenderBackend {
    vk::Device device;

    std::shared_ptr<vk::Buffer> view_param_buf, img_readback_buf;

    std::shared_ptr<vk::Texture2D> render_target;

    std::vector<std::unique_ptr<vk::TriangleMesh>> meshes;
    std::unique_ptr<vk::TopLevelBVH> scene;

    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;

    vk::RTPipeline rt_pipeline;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout buffer_desc_layout = VK_NULL_HANDLE;

    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    // We need a set per varying size array of things we're sending
    VkDescriptorSet desc_set = VK_NULL_HANDLE;
    VkDescriptorSet index_desc_set = VK_NULL_HANDLE;
    VkDescriptorSet vert_desc_set = VK_NULL_HANDLE;
    VkDescriptorSet normals_desc_set = VK_NULL_HANDLE;
    VkDescriptorSet uv_desc_set = VK_NULL_HANDLE;

    vk::ShaderBindingTable shader_table;

    VkFence fence = VK_NULL_HANDLE;

    size_t frame_id = 0;

    RenderVulkan();
    virtual ~RenderVulkan();

    std::string name() override;
    void initialize(const int fb_width, const int fb_height) override;
    void set_scene(const Scene &scene) override;
    RenderStats render(const glm::vec3 &pos,
                       const glm::vec3 &dir,
                       const glm::vec3 &up,
                       const float fovy,
                       const bool camera_changed) override;

private:
    // TODO WILL: it's very similar to DXR
    void build_raytracing_pipeline();
    void build_shader_descriptor_table();
    void build_shader_binding_table();
    void update_view_parameters(const glm::vec3 &pos,
                                const glm::vec3 &dir,
                                const glm::vec3 &up,
                                const float fovy);
};
