#include "embree_utils.h"
#include <algorithm>
#include <iterator>
#include <limits>
#include <glm/ext.hpp>

namespace embree {

Geometry::Geometry(RTCDevice &device,
                   const std::vector<glm::vec3> &verts,
                   const std::vector<glm::uvec3> &indices,
                   const std::vector<glm::vec3> &normals,
                   const std::vector<glm::vec2> &uvs)
    : index_buf(indices),
      normal_buf(normals),
      uv_buf(uvs),
      geom(rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE))
{
    vertex_buf.reserve(verts.size());
    std::transform(
        verts.begin(), verts.end(), std::back_inserter(vertex_buf), [](const glm::vec3 &v) {
            return glm::vec4(v, 0.f);
        });

    vbuf =
        rtcNewSharedBuffer(device, vertex_buf.data(), vertex_buf.size() * sizeof(glm::vec4));
    ibuf = rtcNewSharedBuffer(device, index_buf.data(), index_buf.size() * sizeof(glm::uvec3));

    rtcSetGeometryBuffer(geom,
                         RTC_BUFFER_TYPE_VERTEX,
                         0,
                         RTC_FORMAT_FLOAT3,
                         vbuf,
                         0,
                         sizeof(glm::vec4),
                         vertex_buf.size());
    rtcSetGeometryBuffer(geom,
                         RTC_BUFFER_TYPE_INDEX,
                         0,
                         RTC_FORMAT_UINT3,
                         ibuf,
                         0,
                         sizeof(glm::uvec3),
                         index_buf.size());

    rtcUpdateGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0);
    rtcUpdateGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0);
    rtcCommitGeometry(geom);
}

Geometry::~Geometry()
{
    if (geom) {
        rtcReleaseGeometry(geom);
        rtcReleaseBuffer(vbuf);
        rtcReleaseBuffer(ibuf);
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

TriangleMesh::TriangleMesh(RTCDevice &device, std::vector<std::shared_ptr<Geometry>> &geoms)
    : scene(rtcNewScene(device)), geometries(geoms)
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

RTCScene TriangleMesh::handle()
{
    return scene;
}

Instance::Instance(RTCDevice &device,
                   std::shared_ptr<TriangleMesh> &mesh,
                   const glm::mat4 &xfm,
                   const std::vector<uint32_t> &material_ids)
    : handle(rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE)),
      mesh(mesh),
      object_to_world(xfm),
      world_to_object(glm::inverse(object_to_world)),
      material_ids(material_ids)
{
    rtcSetGeometryInstancedScene(handle, mesh->handle());
    rtcSetGeometryTransform(
        handle, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, glm::value_ptr(object_to_world));
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
      object_to_world(glm::value_ptr(instance.object_to_world)),
      world_to_object(glm::value_ptr(instance.world_to_object)),
      material_ids(instance.material_ids.data())
{
}

TopLevelBVH::TopLevelBVH(RTCDevice &device, const std::vector<std::shared_ptr<Instance>> &inst)
    : handle(rtcNewScene(device)), instances(inst)
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

ISPCTexture2D::ISPCTexture2D(const Image &img)
    : width(img.width), height(img.height), channels(img.channels), data(img.img.data())
{
}
}
