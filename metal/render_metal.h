#pragma once

#include <memory>
#include "render_backend.h"
#include "shader_types.h"

// We just declare the Metal API wrapper objects because we
// don't want to include the header here as it will pull in
// ObjC objects into this file which is included from plain C++ files
namespace metal {
struct Context;
struct ShaderLibrary;
struct ComputePipeline;
struct Heap;
struct Buffer;
struct Texture2D;
struct BottomLevelBVH;
struct TopLevelBVH;
}

struct RenderMetal : RenderBackend {
    std::shared_ptr<metal::Context> context;

    std::shared_ptr<metal::ShaderLibrary> shader_library;
    std::shared_ptr<metal::ComputePipeline> pipeline;

    std::shared_ptr<metal::Texture2D> render_target, accum_buffer;
#ifdef REPORT_RAY_STATS
    std::shared_ptr<metal::Texture2D> ray_stats;
    std::vector<uint16_t> ray_stats_readback;
#endif

    std::shared_ptr<metal::Heap> data_heap;
    std::shared_ptr<metal::Buffer> geometry_args_buffer;

    std::shared_ptr<metal::TopLevelBVH> bvh;
    std::vector<std::shared_ptr<metal::Buffer>> parameterized_mesh_material_ids;
    std::shared_ptr<metal::Buffer> instance_args_buffer;

    std::shared_ptr<metal::Buffer> material_buffer;
    std::vector<std::shared_ptr<metal::Texture2D>> textures;
    std::shared_ptr<metal::Buffer> texture_arg_buffer;

    std::shared_ptr<metal::Buffer> light_buffer;

    uint32_t frame_id = 0;
    bool native_display = false;

    RenderMetal(std::shared_ptr<metal::Context> context);

    RenderMetal();

    std::string name() override;

    void initialize(const int fb_width, const int fb_height) override;

    void set_scene(const Scene &scene) override;

    // Returns the rays per-second achieved, or -1 if this is not tracked
    RenderStats render(const glm::vec3 &pos,
                       const glm::vec3 &dir,
                       const glm::vec3 &up,
                       const float fovy,
                       const bool camera_changed,
                       const bool readback_framebuffer) override;

private:
    ViewParams compute_view_parameters(const glm::vec3 &pos,
                                       const glm::vec3 &dir,
                                       const glm::vec3 &up,
                                       const float fovy);

    void allocate_heap(const Scene &scene);

    std::vector<std::shared_ptr<metal::BottomLevelBVH>> build_meshes(const Scene &scene);

    void upload_textures(const std::vector<Image> &textures);
};

