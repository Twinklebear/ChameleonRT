#pragma once

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl.h>
#include "dx12_utils.h"
#include "dxdisplay.h"
#include "dxr_utils.h"
#include "render_backend.h"

#define N_FRAMES_IN_FLIGHT 2

struct RenderDXR : RenderBackend {
    Microsoft::WRL::ComPtr<IDXGIFactory2> factory;
    Microsoft::WRL::ComPtr<ID3D12Device5> device;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> cmd_queue;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> cmd_allocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> cmd_list;

    // TODO: Need render cmd alloc and render cmd list per frame in flight
    std::array<Microsoft::WRL::ComPtr<ID3D12CommandAllocator>, N_FRAMES_IN_FLIGHT>
        render_cmd_allocators;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> render_cmd_list;

    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> readback_cmd_list;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> readback_cmd_allocator;

    dxr::Buffer img_readback_buf, instance_buf, material_param_buf, light_buf;
    // We need one readback ray stats buffer per frame in flight so they don't trample each
    // other's data
    std::array<dxr::Buffer, N_FRAMES_IN_FLIGHT> ray_stats_readback_bufs;

    // Do technically need 2 view param device bufs, but it's tough to swap them out because
    // they're in the SBT Hacky testing solution is to just not change the view params after
    // the first frame?
    dxr::Buffer view_param_upload_buf;
    dxr::Buffer view_param_device_buf;
    bool camera_params_dirty = false;
    bool readback_image = false;

    uint32_t active_set = 0;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> active_render_cmd_allocator;

    // TODO: I want to still output to the same accum buffer
    // So then I'd just have one frame in flight while I record the commands for the next one
    // I think that should be fine because the app design is really that the RT (GPU) work
    // will be the slower part. The NG1/etc. are just very cheap to run. So if we're building
    // the next frame on the CPU and can submit it right while the prev rendering is in flight
    // with a gpu side wait, that should be fine. Then for fast rendering we can still fill the
    // GPU with frames
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
    std::array<dxr::Buffer, N_FRAMES_IN_FLIGHT> query_resolve_buffers;

    std::array<uint32_t, N_FRAMES_IN_FLIGHT> frame_signal_vals;

    std::array<Microsoft::WRL::ComPtr<ID3D12Fence>, N_FRAMES_IN_FLIGHT> frame_fences;
    std::array<HANDLE, N_FRAMES_IN_FLIGHT> frame_events;

#if defined(DXR_AO) || defined(DXR_AO_TAILREC)
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

    RenderStats readback_render_stats() override;

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
