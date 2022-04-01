#pragma once

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl.h>
#include "dx12_utils.h"
#include "dxdisplay.h"
#include "dxr_utils.h"
#include "render_backend.h"

struct RenderDXR : RenderBackend {
    Microsoft::WRL::ComPtr<IDXGIFactory2> factory;
    Microsoft::WRL::ComPtr<ID3D12Device5> device;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> cmd_queue;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> cmd_allocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> cmd_list;

    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> render_cmd_allocator,
        readback_cmd_allocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> render_cmd_list, readback_cmd_list;

    dxr::Buffer view_param_buf, img_readback_buf, instance_buf, material_param_buf, light_buf,
        ray_stats_readback_buf, view_param_device_buf;

    dxr::Texture2D render_target, accum_buffer, ray_stats;
    std::vector<dxr::Texture2D> textures;

    std::vector<dxr::BottomLevelBVH> meshes;
    dxr::TopLevelBVH scene_bvh;

    std::vector<ParameterizedMesh> parameterized_meshes;

    dxr::RTPipeline rt_pipeline;
    dxr::DescriptorHeap raygen_desc_heap, raygen_sampler_heap;

    uint64_t fence_value = 1;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    HANDLE fence_evt;

    uint32_t frame_id = 0;
    bool native_display = false;

    // Query pool to measure just dispatch rays perf
    Microsoft::WRL::ComPtr<ID3D12QueryHeap> timing_query_heap;
    dxr::Buffer query_resolve_buffer;

#ifdef DXR_AO
    float ao_distance = 1e20f;
#endif

#ifdef REPORT_RAY_STATS
    std::vector<uint16_t> ray_counts;
#endif

    RenderDXR(DXDisplay *display);

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

    RenderStats readback_render_stats(const bool readback_framebuffer) override;

    // TODO: Need a readback stats/results function to override here that returns the render
    // stats. Render should just submit the rendering work to be done, then here we'd do any
    // sync/readback kind of stuff

    // public:
    void create_device_objects(DXDisplay *display = nullptr);

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
