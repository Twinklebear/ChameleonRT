#pragma once

#include <CL/sycl.hpp>

#include <memory>
#include <utility>
#include <vector>
#include <embree4/rtcore.h>
#include "embree_utils.h"
#include "material.h"
#include "render_backend.h"
#include "sycl_utils.h"

struct RenderEmbree : RenderBackend {
    sycl::device sycl_device;
    sycl::context sycl_context;
    sycl::queue sycl_queue;

    RTCDevice device;
    glm::uvec2 fb_dims;

    // TODO: should take scene as shared ptr and keep ref to it,
    std::vector<ParameterizedMesh> parameterized_meshes;

    std::shared_ptr<embree::TopLevelBVH> scene_bvh;

    // std::vector<embree::MaterialParams, usm_shared_allocator<embree::MaterialParams>>
    //    material_params;

    // std::vector<QuadLight, usm_shared_allocator<QuadLight>> lights;

    // std::vector<embree::ISPCTexture2D, usm_shared_allocator<embree::ISPCTexture2D>>
    //    ispc_textures;
    // Texture data in device memory
    std::vector<std::shared_ptr<embree::Buffer>> texture_data;

    uint32_t frame_id = 0;

    // vec3f2
    std::shared_ptr<embree::Buffer> accum_buffer;
    // rgba8
    std::shared_ptr<embree::Buffer> framebuffer;
    std::shared_ptr<embree::Buffer> readback_framebuffer;

#ifdef REPORT_RAY_STATS
    std::shared_ptr<embree::Buffer> ray_stats;
    std::shared_ptr<embree::Buffer> readback_ray_stats;
#endif

    RenderEmbree();
    ~RenderEmbree();

    std::string name() override;
    void initialize(const int fb_width, const int fb_height) override;
    void set_scene(const Scene &scene) override;
    RenderStats render(const glm::vec3 &pos,
                       const glm::vec3 &dir,
                       const glm::vec3 &up,
                       const float fovy,
                       const bool camera_changed,
                       const bool readback_framebuffer) override;
};
