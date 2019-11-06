#include "embree_utils.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <util.h>
#include <glm/ext.hpp>

namespace embree {

Geometry::Geometry(RTCDevice &device,
                   const std::vector<glm::vec3> &verts,
                   const std::vector<glm::uvec3> &indices,
                   const std::vector<glm::vec3> &normals,
                   const std::vector<glm::vec2> &uvs,
                   uint32_t material_id)
    : index_buf(indices),
      normal_buf(normals),
      uv_buf(uvs),
      material_id(material_id),
      geom(rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE))
{
    vertex_buf.reserve(verts.size());
    std::transform(
        verts.begin(), verts.end(), std::back_inserter(vertex_buf), [](const glm::vec3 &v) {
            return glm::vec4(v, 0.f);
        });

    vbuf = rtcNewSharedBuffer(device, vertex_buf.data(), vertex_buf.size() * sizeof(glm::vec4));
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
    : vertex_buf(geom.vertex_buf.data()),
      index_buf(geom.index_buf.data()),
      material_id(geom.material_id)
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

Instance::Instance(RTCDevice &device, std::shared_ptr<TriangleMesh> &mesh, const glm::mat4 &xfm)
    : handle(rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE)),
      mesh(mesh),
      object_to_world(xfm),
      world_to_object(glm::inverse(object_to_world))
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
      world_to_object(glm::value_ptr(instance.world_to_object))
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

ShaderRecord::ShaderRecord(const std::string &name, uint64_t program_handle, size_t param_size)
    : name(name), program_handle(program_handle), param_size(param_size)
{
}

ShaderTable::ShaderTable(const ShaderRecord &raygen_record,
                         const std::vector<ShaderRecord> &miss_records,
                         const std::vector<ShaderRecord> &hitgroup_records)
{
    const size_t raygen_entry_size =
        align_to(raygen_record.param_size + EMBREE_SBT_HEADER_SIZE, EMBREE_SBT_ALIGNMENT);

    size_t miss_entry_size = 0;
    for (const auto &m : miss_records) {
        miss_entry_size = std::max(
            miss_entry_size, align_to(m.param_size + EMBREE_SBT_HEADER_SIZE, EMBREE_SBT_ALIGNMENT));
    }

    size_t hitgroup_entry_size = 0;
    for (const auto &h : hitgroup_records) {
        hitgroup_entry_size =
            std::max(hitgroup_entry_size,
                     align_to(h.param_size + EMBREE_SBT_HEADER_SIZE, EMBREE_SBT_ALIGNMENT));
    }

    const size_t sbt_size = raygen_entry_size + miss_records.size() * miss_entry_size +
                            hitgroup_records.size() * hitgroup_entry_size;

    shader_table.resize(sbt_size, 0);

    ispc_table.raygen = &shader_table[0];

    ispc_table.miss_shaders = &shader_table[raygen_entry_size];
    ispc_table.miss_stride = miss_entry_size;

    ispc_table.hit_groups =
        &shader_table[raygen_entry_size + ispc_table.miss_stride * miss_records.size()];
    ispc_table.hit_group_stride = hitgroup_entry_size;

    size_t offset = 0;
    record_offsets[raygen_record.name] = offset;
    std::memcpy(&shader_table[offset], &raygen_record.program_handle, EMBREE_SBT_HEADER_SIZE);
    offset += raygen_entry_size;

    for (const auto &m : miss_records) {
        record_offsets[m.name] = offset;
        std::memcpy(&shader_table[offset], &m.program_handle, EMBREE_SBT_HEADER_SIZE);
        offset += miss_entry_size;
    }

    for (const auto &h : hitgroup_records) {
        record_offsets[h.name] = offset;
        std::memcpy(&shader_table[offset], &h.program_handle, EMBREE_SBT_HEADER_SIZE);
        offset += hitgroup_entry_size;
    }
}

uint8_t *ShaderTable::get_shader_record(const std::string &shader)
{
    return &shader_table[record_offsets[shader]];
}

ISPCShaderTable &ShaderTable::table()
{
    return ispc_table;
}

ShaderTableBuilder &ShaderTableBuilder::set_raygen(const std::string &name,
                                                   uint64_t program,
                                                   size_t param_size)
{
    raygen_record = ShaderRecord(name, program, param_size);
    return *this;
}

ShaderTableBuilder &ShaderTableBuilder::add_miss(const std::string &name,
                                                 uint64_t program,
                                                 size_t param_size)
{
    miss_records.emplace_back(name, program, param_size);
    return *this;
}

ShaderTableBuilder &ShaderTableBuilder::add_hitgroup(const std::string &name,
                                                     uint64_t program,
                                                     size_t param_size)
{
    hitgroup_records.emplace_back(name, program, param_size);
    return *this;
}

ShaderTable ShaderTableBuilder::build()
{
    return ShaderTable(raygen_record, miss_records, hitgroup_records);
}

}
