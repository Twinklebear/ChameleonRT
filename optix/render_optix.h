#pragma once

#include <optix.h>
#include "glad/glad.h"
#include "optix_utils.h"
#include "render_backend.h"

struct RenderOptiX : RenderBackend {
    CUcontext cuda_context;
    CUstream cuda_stream;

    OptixDeviceContext device;

    optix::Buffer framebuffer, accum_buffer, launch_params, mat_params, device_texture_list,
        light_params, ray_stats_buffer;

    std::vector<optix::TriangleMesh> meshes;
    std::vector<ParameterizedMesh> parameterized_meshes;
    std::vector<optix::Texture2D> textures;

    optix::TopLevelBVH scene_bvh;

    OptixPipeline pipeline;

    optix::ShaderTable shader_table;

    int width, height;
    uint32_t frame_id = 0;

    bool native_display = false;
    GLuint display_texture = -1;
    cudaGraphicsResource_t cu_display_texture;

#ifdef REPORT_RAY_STATS
    std::vector<uint16_t> ray_counts;
#endif

    RenderOptiX(bool native_display);
    ~RenderOptiX();

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
    void update_view_parameters(const glm::vec3 &pos,
                                const glm::vec3 &dir,
                                const glm::vec3 &up,
                                const float fovy);
    void sync_gpu();
};
