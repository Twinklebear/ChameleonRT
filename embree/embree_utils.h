#pragma once

#include <memory>
#include <utility>
#include <vector>
#include <embree3/rtcore.h>
#include "lights.h"
#include "material.h"
#include <glm/glm.hpp>

namespace embree {

struct Geometry {
    std::vector<glm::vec4> vertex_buf;
    std::vector<glm::uvec3> index_buf;
    std::vector<glm::vec3> normal_buf;
    std::vector<glm::vec2> uv_buf;
    uint32_t material_id = 0;

    RTCBuffer vbuf = 0;
    RTCBuffer ibuf = 0;

    RTCGeometry geom = 0;

    Geometry() = default;

    Geometry(RTCDevice &device,
             const std::vector<glm::vec3> &verts,
             const std::vector<glm::uvec3> &indices,
             const std::vector<glm::vec3> &normals,
             const std::vector<glm::vec2> &uvs,
             uint32_t material_id);

    ~Geometry();

    Geometry(const Geometry &) = delete;
    Geometry &operator=(const Geometry &) = delete;
};

struct ISPCGeometry {
    const glm::vec4 *vertex_buf = nullptr;
    const glm::uvec3 *index_buf = nullptr;
    const glm::vec3 *normal_buf = nullptr;
    const glm::vec2 *uv_buf = nullptr;
    uint32_t material_id = 0;

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

    Instance() = default;

    Instance(RTCDevice &device,
             std::shared_ptr<TriangleMesh> &mesh,
             const glm::mat4 &object_to_world = glm::mat4(1.f));

    ~Instance();

    Instance(const Instance &) = delete;
    Instance &operator=(const Instance &) = delete;
};

struct ISPCInstance {
    const ISPCGeometry *geometries = nullptr;
    const float *object_to_world = nullptr;
    const float *world_to_object = nullptr;

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

    ISPCTexture2D *color_texture = nullptr;
};

struct RaySoA {
    std::vector<float> org_x;
    std::vector<float> org_y;
    std::vector<float> org_z;
    std::vector<float> tnear;

    std::vector<float> dir_x;
    std::vector<float> dir_y;
    std::vector<float> dir_z;
    std::vector<float> time;

    std::vector<float> tfar;

    std::vector<unsigned int> mask;
    std::vector<unsigned int> id;
    std::vector<unsigned int> flags;

    RaySoA() = default;
    RaySoA(const size_t nrays);

    void resize(const size_t nrays);
};

struct HitSoA {
    std::vector<float> ng_x;
    std::vector<float> ng_y;
    std::vector<float> ng_z;

    std::vector<float> u;
    std::vector<float> v;

    std::vector<unsigned int> prim_id;
    std::vector<unsigned int> geom_id;
    std::vector<unsigned int> inst_id;

    HitSoA() = default;
    HitSoA(const size_t nrays);

    void resize(const size_t nrays);
};

RTCRayHitNp make_ray_hit_soa(RaySoA &rays, HitSoA &hits);

struct ViewParams {
    glm::vec3 pos, dir_du, dir_dv, dir_top_left;
    uint32_t frame_id;
};

struct SceneContext {
    RTCScene scene;
    ISPCInstance *instances;
    MaterialParams *materials;
    QuadLight *lights;
    uint32_t num_lights;
};

struct Tile {
    uint32_t x, y;
    uint32_t width, height;
    uint32_t fb_width, fb_height;
    float *data;
    uint16_t *ray_stats;
};

}
