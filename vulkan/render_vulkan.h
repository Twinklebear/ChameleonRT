#pragma once

#include <memory>
#include <unordered_map>
#include <vulkan/vulkan.h>
#include "render_backend.h"
#include "vulkan_utils.h"
#include "vulkanrt_utils.h"

struct RenderVulkan : RenderBackend {
    vkrt::Device device;

    std::shared_ptr<vkrt::Buffer> view_param_buf, img_readback_buf, mat_params, light_params;

    std::shared_ptr<vkrt::Texture2D> render_target, accum_buffer;

#ifdef REPORT_RAY_STATS
    std::shared_ptr<vkrt::Texture2D> ray_stats;
    std::shared_ptr<vkrt::Buffer> ray_stats_readback_buf;
#endif

    std::vector<std::unique_ptr<vkrt::TriangleMesh>> meshes;
    std::unique_ptr<vkrt::TopLevelBVH> scene;

    std::vector<uint32_t> material_ids;
    std::vector<std::shared_ptr<vkrt::Texture2D>> textures;
    VkSampler sampler = VK_NULL_HANDLE;

    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;

    VkCommandPool render_cmd_pool = VK_NULL_HANDLE;
    VkCommandBuffer render_cmd_buf = VK_NULL_HANDLE;
    VkCommandBuffer readback_cmd_buf = VK_NULL_HANDLE;

    vkrt::RTPipeline rt_pipeline;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout buffer_desc_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout textures_desc_layout = VK_NULL_HANDLE;

    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    // We need a set per varying size array of things we're sending
    VkDescriptorSet desc_set = VK_NULL_HANDLE;
    VkDescriptorSet index_desc_set = VK_NULL_HANDLE;
    VkDescriptorSet vert_desc_set = VK_NULL_HANDLE;
    VkDescriptorSet normals_desc_set = VK_NULL_HANDLE;
    VkDescriptorSet uv_desc_set = VK_NULL_HANDLE;
    VkDescriptorSet textures_desc_set = VK_NULL_HANDLE;

    vkrt::ShaderBindingTable shader_table;

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
    void build_raytracing_pipeline();

    void build_shader_descriptor_table();

    void build_shader_binding_table();

    void update_view_parameters(const glm::vec3 &pos,
                                const glm::vec3 &dir,
                                const glm::vec3 &up,
                                const float fovy);

    void record_command_buffers();
};
