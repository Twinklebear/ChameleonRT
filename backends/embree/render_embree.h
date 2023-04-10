#pragma once

#include <memory>
#include <utility>
#include <vector>
#include <embree4/rtcore.h>
#include "embree_utils.h"
#include "material.h"
#include "render_backend.h"

struct RenderEmbree : RenderBackend {
    RTCDevice device;
    glm::uvec2 fb_dims;

    // TODO: should take scene as shared ptr and keep ref to it,
    std::vector<ParameterizedMesh> parameterized_meshes;
    std::shared_ptr<embree::TopLevelBVH> scene_bvh;

    std::vector<embree::MaterialParams> material_params;
    std::vector<QuadLight> lights;
    std::vector<Image> textures;
    std::vector<embree::ISPCTexture2D> ispc_textures;

    uint32_t frame_id = 0;
    glm::uvec2 tile_size = glm::uvec2(64);
    std::vector<std::vector<float>> tiles;
    std::vector<std::vector<uint16_t>> ray_stats;
#ifdef REPORT_RAY_STATS
    std::vector<uint64_t> num_rays;
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
