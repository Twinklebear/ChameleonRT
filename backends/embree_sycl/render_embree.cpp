#include "render_embree.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <pmmintrin.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <util.h>
#include "embree_utils.h"
#include "sycl_utils.h"

#include "render_embree_kernel.inl"

RenderEmbree::RenderEmbree()
    : sycl_device(rtcSYCLDeviceSelector),
      sycl_context(sycl_device),
      sycl_queue(
          sycl_device,
          {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}),
      material_params(make_usm_device_read_only_allocator<embree::MaterialParams>(sycl_queue)),
      lights(make_usm_device_read_only_allocator<QuadLight>(sycl_queue)),
      ispc_textures(make_usm_device_read_only_allocator<embree::ISPCTexture2D>(sycl_queue))
{
    device = rtcNewSYCLDevice(sycl_context, nullptr);
}

RenderEmbree::~RenderEmbree()
{
    // Release the scene before shutting down Embree
    scene_bvh = nullptr;
    rtcReleaseDevice(device);
}

std::string RenderEmbree::name()
{
    return "Embree4 + SYCL on " + sycl_device.get_info<sycl::info::device::name>();
}

void RenderEmbree::initialize(const int fb_width, const int fb_height)
{
    frame_id = 0;
    fb_dims = glm::ivec2(fb_width, fb_height);
    img.resize(fb_width * fb_height);

    accum_buffer = std::make_shared<embree::Buffer>(
        fb_width * fb_height * 3 * sizeof(float), embree::MemorySpace::DEVICE, sycl_queue);

    framebuffer = std::make_shared<embree::Buffer>(
        fb_width * fb_height * 4, embree::MemorySpace::DEVICE, sycl_queue);
    readback_framebuffer = std::make_shared<embree::Buffer>(
        fb_width * fb_height * 4, embree::MemorySpace::HOST, sycl_queue);

#ifdef REPORT_RAY_STATS
    ray_stats = std::make_shared<embree::Buffer>(
        fb_width * fb_height * sizeof(uint16_t), embree::MemorySpace::DEVICE, sycl_queue);
    readback_ray_stats = std::make_shared<embree::Buffer>(
        fb_width * fb_height * sizeof(uint16_t), embree::MemorySpace::HOST, sycl_queue);
#endif
}

void RenderEmbree::set_scene(const Scene &scene)
{
    frame_id = 0;
    samples_per_pixel = scene.samples_per_pixel;

    std::vector<std::shared_ptr<embree::TriangleMesh>> meshes;
    for (const auto &mesh : scene.meshes) {
        std::vector<std::shared_ptr<embree::Geometry>> geometries;
        for (const auto &geom : mesh.geometries) {
            geometries.push_back(std::make_shared<embree::Geometry>(
                device, sycl_queue, geom.vertices, geom.indices, geom.normals, geom.uvs));
        }

        meshes.push_back(
            std::make_shared<embree::TriangleMesh>(device, sycl_queue, geometries));
    }

    parameterized_meshes = scene.parameterized_meshes;

    std::vector<std::shared_ptr<embree::Instance>> instances;
    for (const auto &inst : scene.instances) {
        const auto &pm = parameterized_meshes[inst.parameterized_mesh_id];
        instances.push_back(std::make_shared<embree::Instance>(
            device, sycl_queue, meshes[pm.mesh_id], inst.transform, pm.material_ids));
    }

    scene_bvh = std::make_shared<embree::TopLevelBVH>(device, sycl_queue, instances);

    // Make a copy of the texture data to linearize them here since we're not using HW
    // texturing yet due to some limitations in the SYCL APIs.
    auto textures = scene.textures;

    // Linearize any sRGB textures beforehand, since we don't have fancy sRGB texture
    // interpolation support in hardware (using software texturing here)
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
    for (size_t i = 0; i < textures.size(); ++i) {
        const auto &img = textures[i];

        // Upload the data to the GPU to swap the data ptr for a device memory
        auto gpu_data = std::make_shared<embree::Buffer>(
            img.img.size(), embree::MemorySpace::DEVICE, sycl_queue);
        gpu_data->upload(img.img, sycl_queue);
        texture_data.push_back(gpu_data);

        ispc_textures.emplace_back(img,
                                   reinterpret_cast<const uint8_t *>(gpu_data->device_ptr()));
    }
    sycl_queue.wait_and_throw();

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

    std::copy(scene.lights.begin(), scene.lights.end(), std::back_inserter(lights));
}

RenderStats RenderEmbree::render(const glm::vec3 &pos,
                                 const glm::vec3 &dir,
                                 const glm::vec3 &up,
                                 const float fovy,
                                 const bool camera_changed,
                                 const bool)
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
    view_params.samples_per_pixel = samples_per_pixel;

    embree::SceneContext ispc_scene;
    if (scene_bvh->instances.size() == 1) {
        ispc_scene.scene = scene_bvh->instances[0]->mesh->scene;
    } else {
        ispc_scene.scene = scene_bvh->handle;
    }
    ispc_scene.instances = scene_bvh->ispc_instances.data();
    ispc_scene.num_instances = scene_bvh->instances.size();
    ispc_scene.materials = material_params.data();
    ispc_scene.textures = ispc_textures.data();
    ispc_scene.lights = lights.data();
    ispc_scene.num_lights = lights.size();
    ispc_scene.fb_width = fb_dims.x;
    ispc_scene.fb_height = fb_dims.y;
    ispc_scene.framebuffer = reinterpret_cast<uint8_t *>(framebuffer->device_ptr());
    ispc_scene.accum_buffer = reinterpret_cast<float *>(accum_buffer->device_ptr());
#ifdef REPORT_RAY_STATS
    ispc_scene.ray_stats = reinterpret_cast<uint16_t *>(ray_stats->device_ptr());
#endif

    auto host_start = high_resolution_clock::now();
    auto event = sycl_queue.submit([=](sycl::handler &handler) {
        const sycl::range<2> dispatch_size(ispc_scene.fb_width, ispc_scene.fb_height);
        handler.parallel_for(dispatch_size, [=](sycl::item<2> task_idx) {
            kernel::trace_ray(ispc_scene, view_params, task_idx.get_id(0), task_idx.get_id(1));
        });
    });
    event.wait_and_throw();

    framebuffer->download(
        readback_framebuffer->host_ptr(), readback_framebuffer->size(), sycl_queue);
#ifdef REPORT_RAY_STATS
    ray_stats->download(
        readback_ray_stats->host_ptr(), readback_ray_stats->size(), sycl_queue);
#endif
    sycl_queue.wait_and_throw();
    auto host_end = high_resolution_clock::now();

    std::memcpy(img.data(), readback_framebuffer->host_ptr(), readback_framebuffer->size());

    // SYCL times are in nanoseconds
    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    stats.render_time = (end - start) * 1.0e-6;

#ifdef REPORT_RAY_STATS
    const uint16_t *rstats = reinterpret_cast<uint16_t *>(readback_ray_stats->host_ptr());
    const uint64_t total_rays =
        std::accumulate(rstats,
                        rstats + fb_dims.x * fb_dims.y,
                        uint64_t(0),
                        [](const uint64_t &total, const uint16_t &c) { return total + c; });

    stats.rays_per_second = total_rays / (stats.render_time * 1.0e-3);
#endif

    ++frame_id;

    return stats;
}
