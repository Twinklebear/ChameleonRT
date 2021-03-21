#include "render_embree.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <numeric>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#ifndef __aarch64__
#include <pmmintrin.h>
#include <xmmintrin.h>
#endif
#include <util.h>
#include "render_embree_ispc.h"
#include <glm/ext.hpp>

static std::unique_ptr<tbb::global_control> tbb_thread_config;

RenderEmbree::RenderEmbree()
{
#ifndef __aarch64__
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
    device = rtcNewDevice(nullptr);
}

RenderEmbree::~RenderEmbree()
{
    rtcReleaseDevice(device);
}

std::string RenderEmbree::name()
{
    return "Embree (w/ TBB & ISPC)";
}

void RenderEmbree::initialize(const int fb_width, const int fb_height)
{
    frame_id = 0;
    fb_dims = glm::ivec2(fb_width, fb_height);
    img.resize(fb_width * fb_height);

    const glm::uvec2 ntiles(fb_dims.x / tile_size.x + (fb_dims.x % tile_size.x != 0 ? 1 : 0),
                            fb_dims.y / tile_size.y + (fb_dims.y % tile_size.y != 0 ? 1 : 0));
    tiles.resize(ntiles.x * ntiles.y);
    ray_stats.resize(tiles.size());
    for (size_t i = 0; i < tiles.size(); ++i) {
        tiles[i].resize(tile_size.x * tile_size.y * 3, 0.f);
        ray_stats[i].resize(tile_size.x * tile_size.y, 0);
    }

#ifdef REPORT_RAY_STATS
    num_rays.resize(tiles.size(), 0);
#endif
}

void RenderEmbree::set_scene(const Scene &scene)
{
    frame_id = 0;

    std::vector<std::shared_ptr<embree::TriangleMesh>> meshes;
    for (const auto &mesh : scene.meshes) {
        std::vector<std::shared_ptr<embree::Geometry>> geometries;
        for (const auto &geom : mesh.geometries) {
            geometries.push_back(std::make_shared<embree::Geometry>(
                device, geom.vertices, geom.indices, geom.normals, geom.uvs));
        }

        meshes.push_back(std::make_shared<embree::TriangleMesh>(device, geometries));
    }

    paramerized_meshes = scene.parameterized_meshes;

    std::vector<std::shared_ptr<embree::Instance>> instances;
    for (const auto &inst : scene.instances) {
        const auto &pm = paramerized_meshes[inst.parameterized_mesh_id];
        instances.push_back(std::make_shared<embree::Instance>(
            device, meshes[pm.mesh_id], inst.transform, pm.material_ids));
    }

    scene_bvh = std::make_shared<embree::TopLevelBVH>(device, instances);

    textures = scene.textures;

    // Linearize any sRGB textures beforehand, since we don't have fancy sRGB texture
    // interpolation support in hardware
    tbb::parallel_for(size_t(0), textures.size(), [&](size_t i) {
        auto &img = textures[i];
        if (img.color_space == LINEAR) {
            return;
        }
        img.color_space = LINEAR;
        const int convert_channels = std::min(3, img.channels);
        tbb::parallel_for(size_t(0), size_t(img.width) * img.height, [&](size_t px) {
            for (int c = 0; c < convert_channels; ++c) {
                float x = img.img[px * img.channels + c] / 255.f;
                x = srgb_to_linear(x);
                img.img[px * img.channels + c] = glm::clamp(x * 255.f, 0.f, 255.f);
            }
        });
    });

    ispc_textures.reserve(textures.size());
    std::transform(textures.begin(),
                   textures.end(),
                   std::back_inserter(ispc_textures),
                   [](const Image &img) { return embree::ISPCTexture2D(img); });

    material_params.reserve(scene.materials.size());
    for (const auto &m : scene.materials) {
        embree::MaterialParams p;

        p.base_color = m.base_color;
        p.metallic = m.metallic;
        p.specular = m.specular;
        p.roughness = m.roughness;
        p.specular_tint = m.specular_tint;
        p.anisotropy = m.anisotropy;
        p.sheen = m.sheen;
        p.sheen_tint = m.sheen_tint;
        p.clearcoat = m.clearcoat;
        p.clearcoat_gloss = m.clearcoat_gloss;
        p.ior = m.ior;
        p.specular_transmission = m.specular_transmission;

        material_params.push_back(p);
    }

    lights = scene.lights;
}

RenderStats RenderEmbree::render(const glm::vec3 &pos,
                                 const glm::vec3 &dir,
                                 const glm::vec3 &up,
                                 const float fovy,
                                 const bool camera_changed,
                                 const bool readback_framebuffer)
{
    using namespace std::chrono;
    RenderStats stats;

    if (camera_changed) {
        frame_id = 0;
    }

    glm::vec2 img_plane_size;
    img_plane_size.y = 2.f * std::tan(glm::radians(0.5f * fovy));
    img_plane_size.x = img_plane_size.y * static_cast<float>(fb_dims.x) / fb_dims.y;

    embree::ViewParams view_params;
    view_params.pos = pos;
    view_params.dir_du = glm::normalize(glm::cross(dir, up)) * img_plane_size.x;
    view_params.dir_dv =
        -glm::normalize(glm::cross(view_params.dir_du, dir)) * img_plane_size.y;
    view_params.dir_top_left = dir - 0.5f * view_params.dir_du - 0.5f * view_params.dir_dv;
    view_params.frame_id = frame_id;

    embree::SceneContext ispc_scene;
    ispc_scene.scene = scene_bvh->handle;
    ispc_scene.instances = scene_bvh->ispc_instances.data();
    ispc_scene.materials = material_params.data();
    ispc_scene.textures = ispc_textures.data();
    ispc_scene.lights = lights.data();
    ispc_scene.num_lights = lights.size();

    // Round up the number of tiles we need to run in case the
    // framebuffer is not an even multiple of tile size
    const glm::uvec2 ntiles(fb_dims.x / tile_size.x + (fb_dims.x % tile_size.x != 0 ? 1 : 0),
                            fb_dims.y / tile_size.y + (fb_dims.y % tile_size.y != 0 ? 1 : 0));

    uint8_t *color = reinterpret_cast<uint8_t *>(img.data());

    auto start = high_resolution_clock::now();
    tbb::parallel_for(uint32_t(0), ntiles.x * ntiles.y, [&](uint32_t tile_id) {
        const glm::uvec2 tile = glm::uvec2(tile_id % ntiles.x, tile_id / ntiles.x);
        const glm::uvec2 tile_pos = tile * tile_size;
        const glm::uvec2 tile_end = glm::min(tile_pos + tile_size, fb_dims);
        const glm::uvec2 actual_tile_dims = tile_end - tile_pos;

        embree::Tile ispc_tile;
        ispc_tile.x = tile_pos.x;
        ispc_tile.y = tile_pos.y;
        ispc_tile.width = actual_tile_dims.x;
        ispc_tile.height = actual_tile_dims.y;
        ispc_tile.fb_width = fb_dims.x;
        ispc_tile.fb_height = fb_dims.y;
        ispc_tile.data = tiles[tile_id].data();
        ispc_tile.ray_stats = ray_stats[tile_id].data();

        ispc::trace_rays(&ispc_scene, &ispc_tile, &view_params);

        ispc::tile_to_uint8(&ispc_tile, color);
#ifdef REPORT_RAY_STATS
        num_rays[tile_id] = std::accumulate(
            ray_stats[tile_id].begin(),
            ray_stats[tile_id].end(),
            uint64_t(0),
            [](const uint64_t &total, const uint16_t &c) { return total + c; });
#endif
    });
    auto end = high_resolution_clock::now();
    stats.render_time = duration_cast<nanoseconds>(end - start).count() * 1.0e-6;

#ifdef REPORT_RAY_STATS
    const uint64_t total_rays = std::accumulate(num_rays.begin(), num_rays.end(), 0);
    stats.rays_per_second = total_rays / (stats.render_time * 1.0e-3);
#endif

    ++frame_id;

    return stats;
}
