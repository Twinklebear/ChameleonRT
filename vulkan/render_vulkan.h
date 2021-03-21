#pragma once

#include <memory>
#include <unordered_map>
#include <vulkan/vulkan.h>
#include "render_backend.h"
#include "vulkan_utils.h"
#include "vulkanrt_utils.h"

struct HitGroupParams {
    uint64_t vert_buf = 0;
    uint64_t idx_buf = 0;
    uint64_t normal_buf = 0;
    uint64_t uv_buf = 0;
    uint32_t num_normals = 0;
    uint32_t num_uvs = 0;
    uint32_t material_id = 0;
};

struct RenderVulkan : RenderBackend {
    std::shared_ptr<vkrt::Device> device;

    std::shared_ptr<vkrt::Buffer> view_param_buf, img_readback_buf, mat_params, light_params;

    std::shared_ptr<vkrt::Texture2D> render_target, accum_buffer;

#ifdef REPORT_RAY_STATS
    std::shared_ptr<vkrt::Texture2D> ray_stats;
    std::shared_ptr<vkrt::Buffer> ray_stats_readback_buf;
    std::vector<uint16_t> ray_counts;
#endif

    std::vector<std::unique_ptr<vkrt::TriangleMesh>> meshes;
    std::vector<ParameterizedMesh> parameterized_meshes;
    std::unique_ptr<vkrt::TopLevelBVH> scene_bvh;
    size_t total_geom = 0;

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
    VkDescriptorSetLayout textures_desc_layout = VK_NULL_HANDLE;

    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    // We need a set per varying size array of things we're sending
    VkDescriptorSet desc_set = VK_NULL_HANDLE;
    VkDescriptorSet textures_desc_set = VK_NULL_HANDLE;

    vkrt::ShaderBindingTable shader_table;

    VkFence fence = VK_NULL_HANDLE;

    size_t frame_id = 0;
    bool native_display = false;

    RenderVulkan(std::shared_ptr<vkrt::Device> device);

    RenderVulkan();

    virtual ~RenderVulkan();

    std::string name() override;

    void initialize(const int fb_width, const int fb_height) override;

    void set_scene(const Scene &scene) override;

    RenderStats render(const glm::vec3 &pos,
                       const glm::vec3 &dir,
                       const glm::vec3 &up,
                       const float fovy,
                       const bool camera_changed,
                       const bool readback_framebuffer) override;

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
