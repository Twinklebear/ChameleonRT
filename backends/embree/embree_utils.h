#pragma once

#include <memory>
#include <utility>
#include <vector>
#include <embree4/rtcore.h>
#include "lights.h"
#include "material.h"
#include <glm/glm.hpp>

namespace embree {

struct Geometry {
    // vertex_buf is padded out by an extra vec3 for Embree's alignment requirements
    // n_vertices = the real # of vertices, ie vertex_buf.size() - 1
    size_t n_vertices = 0;
    std::vector<glm::vec3> vertex_buf;
    std::vector<glm::uvec3> index_buf;
    std::vector<glm::vec3> normal_buf;
    std::vector<glm::vec2> uv_buf;

    RTCGeometry geom = 0;

    Geometry() = default;

    Geometry(RTCDevice &device,
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

class TriangleMesh {
    RTCScene scene = 0;

public:
    std::vector<std::shared_ptr<Geometry>> geometries;
    std::vector<ISPCGeometry> ispc_geometries;

    TriangleMesh() = default;

    TriangleMesh(RTCDevice &device, std::vector<std::shared_ptr<Geometry>> &geometries);

    ~TriangleMesh();

    TriangleMesh(const TriangleMesh &) = delete;
    TriangleMesh &operator=(const TriangleMesh &) = delete;

    RTCScene handle();
};

struct Instance {
    RTCGeometry handle = 0;
    std::shared_ptr<TriangleMesh> mesh = nullptr;
    glm::mat4 object_to_world, world_to_object;
    std::vector<uint32_t> material_ids;

    Instance() = default;

    Instance(RTCDevice &device,
             std::shared_ptr<TriangleMesh> &mesh,
             const glm::mat4 &object_to_world,
             const std::vector<uint32_t> &material_ids);

    ~Instance();

    Instance(const Instance &) = delete;
    Instance &operator=(const Instance &) = delete;
};

struct ISPCInstance {
    const ISPCGeometry *geometries = nullptr;
    const float *object_to_world = nullptr;
    const float *world_to_object = nullptr;
    const uint32_t *material_ids = nullptr;

    ISPCInstance() = default;
    ISPCInstance(const Instance &instance);
};

struct TopLevelBVH {
    RTCScene handle = 0;
    std::vector<std::shared_ptr<Instance>> instances;
    std::vector<ISPCInstance> ispc_instances;

    TopLevelBVH() = default;
    TopLevelBVH(RTCDevice &device, const std::vector<std::shared_ptr<Instance>> &instances);
    ~TopLevelBVH();

    TopLevelBVH(const TopLevelBVH &) = delete;
    TopLevelBVH &operator=(const TopLevelBVH &) = delete;
};

struct ISPCTexture2D {
    int width = -1;
    int height = -1;
    int channels = -1;
    const uint8_t *data = nullptr;

    ISPCTexture2D(const Image &img);
    ISPCTexture2D() = default;
};

struct MaterialParams {
    glm::vec3 base_color = glm::vec3(0.9f);
    float metallic = 0;

    float specular = 0;
    float roughness = 1;
    float specular_tint = 0;
    float anisotropy = 0;

    float sheen = 0;
    float sheen_tint = 0;
    float clearcoat = 0;
    float clearcoat_gloss = 0;

    float ior = 1.5;
    float specular_transmission = 0;
};

struct ViewParams {
    glm::vec3 pos, dir_du, dir_dv, dir_top_left;
    uint32_t frame_id;
};

struct SceneContext {
    RTCScene scene;
    ISPCInstance *instances;
    MaterialParams *materials;
    QuadLight *lights;
    ISPCTexture2D *textures;
    uint32_t num_lights;
    uint32_t samples_per_pixel;
};

struct Tile {
    uint32_t x, y;
    uint32_t width, height;
    uint32_t fb_width, fb_height;
    float *data;
    uint16_t *ray_stats;
};

}
