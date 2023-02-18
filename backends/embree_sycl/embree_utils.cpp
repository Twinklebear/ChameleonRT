#include "embree_utils.h"
#include <algorithm>
#include <iterator>
#include <limits>
// Doesn't interact well w/ DPCPP or C++17?
// #include <glm/ext.hpp>

namespace embree {

Geometry::Geometry(RTCDevice &device,
                   sycl::queue &sycl_queue,
                   const std::vector<glm::vec3> &verts,
                   const std::vector<glm::uvec3> &indices,
                   const std::vector<glm::vec3> &normals,
                   const std::vector<glm::vec2> &uvs)
    : n_vertices(verts.size()),
      vertex_buf(verts.begin(),
                 verts.end(),
                 make_usm_device_read_only_allocator<glm::vec3>(sycl_queue)),
      index_buf(indices.begin(),
                indices.end(),
                make_usm_device_read_only_allocator<glm::uvec3>(sycl_queue)),
      normal_buf(normals.begin(),
                 normals.end(),
                 make_usm_device_read_only_allocator<glm::vec3>(sycl_queue)),
      uv_buf(
          uvs.begin(), uvs.end(), make_usm_device_read_only_allocator<glm::vec2>(sycl_queue)),
      geom(rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE))
{
    // Pad the vertex_buf out to align it
    vertex_buf.push_back(glm::vec3(0.f));

    rtcSetSharedGeometryBuffer(geom,
                               RTC_BUFFER_TYPE_VERTEX,
                               0,
                               RTC_FORMAT_FLOAT3,
                               vertex_buf.data(),
                               0,
                               sizeof(glm::vec3),
                               n_vertices);
    rtcSetSharedGeometryBuffer(geom,
                               RTC_BUFFER_TYPE_INDEX,
                               0,
                               RTC_FORMAT_UINT3,
                               index_buf.data(),
                               0,
                               sizeof(glm::uvec3),
                               index_buf.size());
    rtcCommitGeometry(geom);
}

Geometry::~Geometry()
{
    if (geom) {
        rtcReleaseGeometry(geom);
    }
}

ISPCGeometry::ISPCGeometry(const Geometry &geom)
    : vertex_buf(geom.vertex_buf.data()), index_buf(geom.index_buf.data())
{
    if (!geom.normal_buf.empty()) {
        normal_buf = geom.normal_buf.data();
    }

    if (!geom.uv_buf.empty()) {
        uv_buf = geom.uv_buf.data();
    }
}

TriangleMesh::TriangleMesh(RTCDevice &device,
                           sycl::queue &sycl_queue,
                           std::vector<std::shared_ptr<Geometry>> &geoms)
    : scene(rtcNewScene(device)),
      geometries(geoms),
      ispc_geometries(make_usm_device_read_only_allocator<ISPCGeometry>(sycl_queue))
{
    ispc_geometries.reserve(geometries.size());
    std::transform(geometries.begin(),
                   geometries.end(),
                   std::back_inserter(ispc_geometries),
                   [](const std::shared_ptr<Geometry> &g) { return ISPCGeometry(*g); });

    for (auto &g : geometries) {
        rtcAttachGeometry(scene, g->geom);
    }
    rtcCommitScene(scene);
}

TriangleMesh::~TriangleMesh()
{
    if (scene) {
        rtcReleaseScene(scene);
    }
}

Instance::Instance(RTCDevice &device,
                   sycl::queue &sycl_queue,
                   std::shared_ptr<TriangleMesh> &mesh,
                   const glm::mat4 &xfm,
                   const std::vector<uint32_t> &material_ids)
    : handle(rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE)),
      mesh(mesh),
      object_to_world(xfm),
      world_to_object(glm::inverse(object_to_world)),
      material_ids(material_ids.begin(),
                   material_ids.end(),
                   make_usm_device_read_only_allocator<uint32_t>(sycl_queue))
{
    rtcSetGeometryInstancedScene(handle, mesh->scene);
    rtcSetGeometryTransform(handle, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, &object_to_world);
    rtcCommitGeometry(handle);
}

Instance::~Instance()
{
    if (handle) {
        rtcReleaseGeometry(handle);
    }
}

ISPCInstance::ISPCInstance(const Instance &instance)
    : geometries(instance.mesh->ispc_geometries.data()),
      material_ids(instance.material_ids.data())
{
    std::memcpy(object_to_world, &instance.object_to_world, sizeof(glm::mat4));
    std::memcpy(world_to_object, &instance.world_to_object, sizeof(glm::mat4));
}

TopLevelBVH::TopLevelBVH(RTCDevice &device,
                         sycl::queue &sycl_queue,
                         const std::vector<std::shared_ptr<Instance>> &inst)
    : handle(rtcNewScene(device)),
      instances(inst),
      ispc_instances(make_usm_device_read_only_allocator<ISPCInstance>(sycl_queue))
{
    for (const auto &i : instances) {
        rtcAttachGeometry(handle, i->handle);
        ispc_instances.push_back(*i);
    }
    rtcCommitScene(handle);
}

TopLevelBVH::~TopLevelBVH()
{
    if (handle) {
        rtcReleaseScene(handle);
    }
}

ISPCTexture2D::ISPCTexture2D(const Image &img, const uint8_t *gpu_data)
    : width(img.width), height(img.height), channels(img.channels), data(gpu_data)
{
}
}
