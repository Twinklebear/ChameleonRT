#pragma once

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl.h>
#include "dx12_utils.h"
#include "dxr_utils.h"
#include "render_backend.h"

struct RenderDXR : RenderBackend {
    Microsoft::WRL::ComPtr<ID3D12Device5> device;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> cmd_queue;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> cmd_allocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> cmd_list;

    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> render_cmd_allocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> render_cmd_list, readback_cmd_list;

    dxr::Buffer view_param_buf, img_readback_buf, instance_buf, material_param_buf, light_buf,
        ray_stats_readback_buf;

    dxr::Texture2D render_target, accum_buffer, ray_stats;
    std::vector<dxr::Texture2D> textures;

    std::vector<dxr::BottomLevelBVH> meshes;
    dxr::TopLevelBVH scene_bvh;

    dxr::RTPipeline rt_pipeline;
    dxr::DescriptorHeap raygen_desc_heap, raygen_sampler_heap;

    uint64_t fence_value = 1;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    HANDLE fence_evt;

    uint32_t frame_id = 0;
    bool native_display = false;

#ifdef REPORT_RAY_STATS
    std::vector<uint16_t> ray_counts;
#endif

    RenderDXR(Microsoft::WRL::ComPtr<ID3D12Device5> device, bool native_display);

    RenderDXR();

    virtual ~RenderDXR();

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
    void create_device_objects();

    void build_raytracing_pipeline();

    void build_shader_resource_heap();

    void build_shader_binding_table();

    void update_view_parameters(const glm::vec3 &pos,
                                const glm::vec3 &dir,
                                const glm::vec3 &up,
                                const float fovy);

    void build_descriptor_heap();

    void record_command_lists();

    void sync_gpu();
};
