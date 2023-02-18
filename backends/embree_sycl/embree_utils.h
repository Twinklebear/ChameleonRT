#pragma once

#include <CL/sycl.hpp>

#include <memory>
#include <utility>
#include <vector>
#include <embree4/rtcore.h>
#include "../../util/lights.h"
#include "material.h"
#include <glm/glm.hpp>

template <typename T>
using usm_shared_allocator = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

// To reduce padding of USM allocations we can use the usm::device_read_only property
// for data that will only be read on the device. This applies to all of our USM allocations
template <typename T>
inline usm_shared_allocator<T> make_usm_device_read_only_allocator(sycl::queue &queue)
{
    return usm_shared_allocator<T>(queue,
                                   {sycl::ext::oneapi::property::usm::device_read_only()});
}

template <typename T>
using usm_host_allocator = sycl::usm_allocator<T, sycl::usm::alloc::host>;

namespace embree {

struct Geometry {
    // vertex_buf is padded out by an extra vec3 to meet Embree's alignment requirements,
    // n_vertices is the real # of vertices, i.e. vertex_buf.size() - 1
    size_t n_vertices = 0;
    std::vector<glm::vec3, usm_shared_allocator<glm::vec3>> vertex_buf;
    std::vector<glm::uvec3, usm_shared_allocator<glm::uvec3>> index_buf;
    std::vector<glm::vec3, usm_shared_allocator<glm::vec3>> normal_buf;
    std::vector<glm::vec2, usm_shared_allocator<glm::vec2>> uv_buf;

    RTCGeometry geom = 0;

    Geometry() = default;

    Geometry(RTCDevice &device,
             sycl::queue &sycl_queue,
             const std::vector<glm::vec3> &verts,
             const std::vector<glm::uvec3> &indices,
             const std::vector<glm::vec3> &normals,
             const std::vector<glm::vec2> &uvs);

    ~Geometry();

    Geometry(const Geometry &) = delete;
    Geometry &operator=(const Geometry &) = delete;
};

struct ISPCGeometry {
    const glm::vec3 *vertex_buf = nullptr;
    const glm::uvec3 *index_buf = nullptr;
    const glm::vec3 *normal_buf = nullptr;
    const glm::vec2 *uv_buf = nullptr;

    ISPCGeometry() = default;
    ISPCGeometry(const Geometry &geom);
};

struct TriangleMesh {
    RTCScene scene = 0;

    std::vector<std::shared_ptr<Geometry>> geometries;
    std::vector<ISPCGeometry, usm_shared_allocator<ISPCGeometry>> ispc_geometries;

    TriangleMesh() = default;

    TriangleMesh(RTCDevice &device,
                 sycl::queue &sycl_queue,
                 std::vector<std::shared_ptr<Geometry>> &geometries);

    ~TriangleMesh();

    TriangleMesh(const TriangleMesh &) = delete;
    TriangleMesh &operator=(const TriangleMesh &) = delete;
};

struct Instance {
    RTCGeometry handle = 0;
    std::shared_ptr<TriangleMesh> mesh = nullptr;
    glm::mat4 object_to_world, world_to_object;
    std::vector<uint32_t, usm_shared_allocator<uint32_t>> material_ids;

    Instance() = default;

    Instance(RTCDevice &device,
             sycl::queue &sycl_queue,
             std::shared_ptr<TriangleMesh> &mesh,
             const glm::mat4 &object_to_world,
             const std::vector<uint32_t> &material_ids);

    ~Instance();

    Instance(const Instance &) = delete;
    Instance &operator=(const Instance &) = delete;
};

struct ISPCInstance {
    const ISPCGeometry *geometries = nullptr;
    // TODO need to store matrix in usm/device mem
    float object_to_world[16];
    float world_to_object[16];
    const uint32_t *material_ids = nullptr;

    ISPCInstance() = default;
    ISPCInstance(const Instance &instance);
};

struct TopLevelBVH {
    RTCScene handle = 0;
    std::vector<std::shared_ptr<Instance>> instances;
    std::vector<ISPCInstance, usm_shared_allocator<ISPCInstance>> ispc_instances;

    TopLevelBVH() = default;
    TopLevelBVH(RTCDevice &device,
                sycl::queue &sycl_queue,
                const std::vector<std::shared_ptr<Instance>> &instances);
    ~TopLevelBVH();

    TopLevelBVH(const TopLevelBVH &) = delete;
    TopLevelBVH &operator=(const TopLevelBVH &) = delete;
};

struct ISPCTexture2D {
    int width = -1;
    int height = -1;
    int channels = -1;
    const uint8_t *data = nullptr;

    ISPCTexture2D(const Image &img, const uint8_t *gpu_data);
    ISPCTexture2D() = default;
};

struct MaterialParams {
    glm::vec3 base_color = glm::vec3(0.9f);
    float metallic = 0.f;

    float specular = 0.f;
    float roughness = 1.f;
    float specular_tint = 0.f;
    float anisotropy = 0.f;

    float sheen = 0.f;
    float sheen_tint = 0.f;
    float clearcoat = 0.f;
    float clearcoat_gloss = 0.f;

    float ior = 1.5f;
    float specular_transmission = 0.f;
};

struct ViewParams {
    glm::vec3 pos, dir_du, dir_dv, dir_top_left;
    uint32_t frame_id;
    uint32_t samples_per_pixel;
};

struct SceneContext {
    RTCScene scene;
    ISPCInstance *instances;
    MaterialParams *materials;
    QuadLight *lights;
    ISPCTexture2D *textures;
    uint32_t num_lights;
    uint32_t num_instances;
    uint32_t fb_width, fb_height;

    float *accum_buffer;
    uint8_t *framebuffer;
    uint16_t *ray_stats;
};

}
