#include "vulkanrt_utils.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include "util.h"
#include <glm/glm.hpp>

namespace vkrt {

Geometry::Geometry(std::shared_ptr<Buffer> verts,
                   std::shared_ptr<Buffer> indices,
                   std::shared_ptr<Buffer> normal_buf,
                   std::shared_ptr<Buffer> uv_buf,
                   uint32_t geom_flags)
    : vertex_buf(verts), index_buf(indices), normal_buf(normal_buf), uv_buf(uv_buf)
{
    geom_desc.sType = VK_STRUCTURE_TYPE_GEOMETRY_NV;
    geom_desc.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_NV;
    geom_desc.geometry.triangles.sType = VK_STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV;
    geom_desc.geometry.triangles.vertexData = vertex_buf->handle();
    geom_desc.geometry.triangles.vertexOffset = 0;
    geom_desc.geometry.triangles.vertexCount = vertex_buf->size() / sizeof(glm::vec3);
    geom_desc.geometry.triangles.vertexStride = sizeof(glm::vec3);
    geom_desc.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;

    geom_desc.geometry.triangles.indexData = index_buf->handle();
    geom_desc.geometry.triangles.indexOffset = 0;
    geom_desc.geometry.triangles.indexCount = index_buf->size() / sizeof(uint32_t);
    geom_desc.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    geom_desc.geometry.triangles.transformData = VK_NULL_HANDLE;
    geom_desc.geometry.triangles.transformOffset = 0;
    geom_desc.flags = geom_flags;

    // Must be set even if not used
    geom_desc.geometry.aabbs = {};
    geom_desc.geometry.aabbs.sType = {VK_STRUCTURE_TYPE_GEOMETRY_AABB_NV};
}

TriangleMesh::TriangleMesh(Device &dev, std::vector<Geometry> geoms, uint32_t build_flags)
    : device(&dev),
      build_flags((VkBuildAccelerationStructureFlagBitsNV)build_flags),
      geometries(geoms)

{
    std::transform(geometries.begin(),
                   geometries.end(),
                   std::back_inserter(geom_descs),
                   [](const Geometry &g) { return g.geom_desc; });

    accel_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
    accel_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV;
    accel_info.flags = build_flags;
    accel_info.instanceCount = 0;
    accel_info.geometryCount = geom_descs.size();
    accel_info.pGeometries = geom_descs.data();

    VkAccelerationStructureCreateInfoNV create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
    create_info.info = accel_info;
    CHECK_VULKAN(
        vkCreateAccelerationStructure(device->logical_device(), &create_info, nullptr, &bvh));
}

TriangleMesh::~TriangleMesh()
{
    if (bvh != VK_NULL_HANDLE) {
        vkDestroyAccelerationStructure(device->logical_device(), bvh, nullptr);
        vkFreeMemory(device->logical_device(), bvh_mem, nullptr);
    }
}

void TriangleMesh::enqueue_build(VkCommandBuffer &cmd_buf)
{
    // Determine how much memory the acceleration structure will need
    VkAccelerationStructureMemoryRequirementsInfoNV mem_info = {};
    mem_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
    mem_info.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV;
    mem_info.accelerationStructure = bvh;

    VkMemoryRequirements2 mem_reqs = {};
    vkGetAccelerationStructureMemoryRequirements(
        device->logical_device(), &mem_info, &mem_reqs);
    // Allocate space for the build output
    bvh_mem = device->alloc(mem_reqs.memoryRequirements.size,
                            mem_reqs.memoryRequirements.memoryTypeBits,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Allocate scratch space for the build
    mem_info.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV;
    vkGetAccelerationStructureMemoryRequirements(
        device->logical_device(), &mem_info, &mem_reqs);
    scratch = Buffer::device(
        *device, mem_reqs.memoryRequirements.size, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Bind the build output mem to the BVH
    VkBindAccelerationStructureMemoryInfoNV bind_mem_info = {};
    bind_mem_info.sType = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
    bind_mem_info.accelerationStructure = bvh;
    bind_mem_info.memory = bvh_mem;
    CHECK_VULKAN(
        vkBindAccelerationStructureMemory(device->logical_device(), 1, &bind_mem_info));

    // Enqueue the build commands into the command buffer
    vkCmdBuildAccelerationStructure(cmd_buf,
                                    &accel_info,
                                    VK_NULL_HANDLE,
                                    0,
                                    false,
                                    bvh,
                                    VK_NULL_HANDLE,
                                    scratch->handle(),
                                    0);

    // Memory barrier to have subsequent commands wait on build completion
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV |
                            VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV |
                            VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;

    vkCmdPipelineBarrier(cmd_buf,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
                         0,
                         1,
                         &barrier,
                         0,
                         nullptr,
                         0,
                         nullptr);

    // Read the compacted size if we're compacting
    if (build_flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_NV) {
        VkQueryPoolCreateInfo pool_ci = {};
        pool_ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        pool_ci.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_NV;
        pool_ci.queryCount = 1;
        CHECK_VULKAN(
            vkCreateQueryPool(device->logical_device(), &pool_ci, nullptr, &query_pool));

        vkCmdResetQueryPool(cmd_buf, query_pool, 0, 1);
        vkCmdBeginQuery(cmd_buf, query_pool, 0, 0);
        vkCmdWriteAccelerationStructuresProperties(
            cmd_buf,
            1,
            &bvh,
            VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_NV,
            query_pool,
            0);
        vkCmdEndQuery(cmd_buf, query_pool, 0);
    }
}

void TriangleMesh::enqueue_compaction(VkCommandBuffer &cmd_buf)
{
    if (!(build_flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_NV)) {
        return;
    }
    uint64_t compacted_size = 0;
    CHECK_VULKAN(vkGetQueryPoolResults(device->logical_device(),
                                       query_pool,
                                       0,
                                       1,
                                       sizeof(uint64_t),
                                       &compacted_size,
                                       sizeof(uint64_t),
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));

    // Same memory type requirements as the original structure, just less space needed
    VkAccelerationStructureMemoryRequirementsInfoNV mem_info = {};
    mem_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
    mem_info.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV;
    mem_info.accelerationStructure = bvh;

    VkMemoryRequirements2 mem_reqs = {};
    vkGetAccelerationStructureMemoryRequirements(
        device->logical_device(), &mem_info, &mem_reqs);
    compacted_size = align_to(compacted_size, mem_reqs.memoryRequirements.alignment);

    compacted_mem = device->alloc(compacted_size,
                                  mem_reqs.memoryRequirements.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkAccelerationStructureCreateInfoNV create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
    create_info.info = accel_info;
    CHECK_VULKAN(vkCreateAccelerationStructure(
        device->logical_device(), &create_info, nullptr, &compacted_bvh));

    // Bind the compacted mem to the compacted BVH handle
    VkBindAccelerationStructureMemoryInfoNV bind_mem_info = {};
    bind_mem_info.sType = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
    bind_mem_info.accelerationStructure = compacted_bvh;
    bind_mem_info.memory = compacted_mem;
    CHECK_VULKAN(
        vkBindAccelerationStructureMemory(device->logical_device(), 1, &bind_mem_info));

    vkCmdCopyAccelerationStructure(
        cmd_buf, compacted_bvh, bvh, VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_NV);
}

void TriangleMesh::finalize()
{
    scratch = nullptr;

    // Compaction is done, so swap the old handle with the compacted one and free the old
    // output memory
    if (build_flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_NV) {
        vkDestroyQueryPool(device->logical_device(), query_pool, nullptr);
        query_pool = VK_NULL_HANDLE;

        vkDestroyAccelerationStructure(device->logical_device(), bvh, nullptr);
        vkFreeMemory(device->logical_device(), bvh_mem, nullptr);

        bvh = compacted_bvh;
        bvh_mem = compacted_mem;

        compacted_bvh = VK_NULL_HANDLE;
        compacted_mem = VK_NULL_HANDLE;
    }

    CHECK_VULKAN(vkGetAccelerationStructureHandle(
        device->logical_device(), bvh, sizeof(uint64_t), &handle));
}

TopLevelBVH::TopLevelBVH(Device &dev,
                         std::shared_ptr<Buffer> &inst_buf,
                         const std::vector<Instance> &instances,
                         uint32_t build_flags)
    : device(&dev),
      build_flags((VkBuildAccelerationStructureFlagBitsNV)build_flags),
      instance_buf(inst_buf),
      instances(instances)
{
    accel_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
    accel_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV;
    accel_info.instanceCount = instance_buf->size() / sizeof(GeometryInstance);
    accel_info.geometryCount = 0;

    VkAccelerationStructureCreateInfoNV create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
    create_info.info = accel_info;
    CHECK_VULKAN(
        vkCreateAccelerationStructure(device->logical_device(), &create_info, nullptr, &bvh));
}

TopLevelBVH::~TopLevelBVH()
{
    if (bvh != VK_NULL_HANDLE) {
        vkDestroyAccelerationStructure(device->logical_device(), bvh, nullptr);
        vkFreeMemory(device->logical_device(), bvh_mem, nullptr);
    }
}

void TopLevelBVH::enqueue_build(VkCommandBuffer &cmd_buf)
{
    // Determine how much memory the acceleration structure will need
    VkAccelerationStructureMemoryRequirementsInfoNV mem_info = {};
    mem_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
    mem_info.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV;
    mem_info.accelerationStructure = bvh;

    VkMemoryRequirements2 mem_reqs = {};
    vkGetAccelerationStructureMemoryRequirements(
        device->logical_device(), &mem_info, &mem_reqs);
    // Allocate space for the build output
    bvh_mem = device->alloc(mem_reqs.memoryRequirements.size,
                            mem_reqs.memoryRequirements.memoryTypeBits,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Determine how much additional memory we need for the build
    mem_info.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV;
    vkGetAccelerationStructureMemoryRequirements(
        device->logical_device(), &mem_info, &mem_reqs);
    scratch = Buffer::device(
        *device, mem_reqs.memoryRequirements.size, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Bind the build output mem to the BVH
    VkBindAccelerationStructureMemoryInfoNV bind_mem_info = {};
    bind_mem_info.sType = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
    bind_mem_info.accelerationStructure = bvh;
    bind_mem_info.memory = bvh_mem;
    CHECK_VULKAN(
        vkBindAccelerationStructureMemory(device->logical_device(), 1, &bind_mem_info));

    vkCmdBuildAccelerationStructure(cmd_buf,
                                    &accel_info,
                                    instance_buf->handle(),
                                    0,
                                    false,
                                    bvh,
                                    VK_NULL_HANDLE,
                                    scratch->handle(),
                                    0);

    // Enqueue a barrier on the build
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV |
                            VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV |
                            VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;

    vkCmdPipelineBarrier(cmd_buf,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
                         0,
                         1,
                         &barrier,
                         0,
                         nullptr,
                         0,
                         nullptr);
}

void TopLevelBVH::finalize()
{
    scratch = nullptr;
    CHECK_VULKAN(vkGetAccelerationStructureHandle(
        device->logical_device(), bvh, sizeof(uint64_t), &handle));
}

size_t TopLevelBVH::num_instances() const
{
    return accel_info.instanceCount;
}

const uint8_t *RTPipeline::shader_ident(const std::string &name) const
{
    auto fnd = shader_ident_offsets.find(name);
    if (fnd == shader_ident_offsets.end()) {
        throw std::runtime_error("Shader identifier " + name + " not found!");
    }
    return &shader_identifiers[fnd->second];
}

size_t RTPipeline::shader_ident_size() const
{
    return ident_size;
}

VkPipeline RTPipeline::handle()
{
    return pipeline;
}

ShaderGroup::ShaderGroup(const std::string &name,
                         const std::shared_ptr<ShaderModule> &shader_module,
                         const std::string &entry_point,
                         VkShaderStageFlagBits stage,
                         VkRayTracingShaderGroupTypeNV group)
    : shader_module(shader_module),
      stage(stage),
      group(group),
      name(name),
      entry_point(entry_point)
{
}

RTPipelineBuilder &RTPipelineBuilder::set_raygen(const std::string &name,
                                                 const std::shared_ptr<ShaderModule> &shader,
                                                 const std::string &entry_point)
{
    shaders.emplace_back(name,
                         shader,
                         entry_point,
                         VK_SHADER_STAGE_RAYGEN_BIT_NV,
                         VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV);
    return *this;
}

RTPipelineBuilder &RTPipelineBuilder::add_miss(const std::string &name,
                                               const std::shared_ptr<ShaderModule> &shader,
                                               const std::string &entry_point)
{
    shaders.emplace_back(name,
                         shader,
                         entry_point,
                         VK_SHADER_STAGE_MISS_BIT_NV,
                         VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV);
    return *this;
}

RTPipelineBuilder &RTPipelineBuilder::add_hitgroup(const std::string &name,
                                                   const std::shared_ptr<ShaderModule> &shader,
                                                   const std::string &entry_point)
{
    shaders.emplace_back(name,
                         shader,
                         entry_point,
                         VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV,
                         VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_NV);
    return *this;
}

RTPipelineBuilder &RTPipelineBuilder::set_layout(VkPipelineLayout l)
{
    layout = l;
    return *this;
}

RTPipelineBuilder &RTPipelineBuilder::set_recursion_depth(uint32_t depth)
{
    recursion_depth = depth;
    return *this;
}

RTPipeline RTPipelineBuilder::build(Device &device)
{
    std::vector<VkPipelineShaderStageCreateInfo> shader_info;
    std::vector<VkRayTracingShaderGroupCreateInfoNV> group_info;

    RTPipeline pipeline;

    pipeline.ident_size = device.raytracing_properties().shaderGroupHandleSize;
    for (const auto &sg : shaders) {
        VkPipelineShaderStageCreateInfo ss_ci = {};
        ss_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ss_ci.stage = sg.stage;
        ss_ci.module = sg.shader_module->module;
        ss_ci.pName = sg.entry_point.c_str();

        VkRayTracingShaderGroupCreateInfoNV g_ci = {};
        g_ci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV;
        g_ci.type = sg.group;
        if (sg.stage == VK_SHADER_STAGE_RAYGEN_BIT_NV ||
            sg.stage == VK_SHADER_STAGE_MISS_BIT_NV) {
            g_ci.generalShader = shader_info.size();
            g_ci.closestHitShader = VK_SHADER_UNUSED_NV;
            g_ci.anyHitShader = VK_SHADER_UNUSED_NV;
            g_ci.intersectionShader = VK_SHADER_UNUSED_NV;
        } else if (sg.stage == VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV) {
            g_ci.generalShader = VK_SHADER_UNUSED_NV;
            g_ci.closestHitShader = shader_info.size();
            g_ci.anyHitShader = VK_SHADER_UNUSED_NV;
            g_ci.intersectionShader = VK_SHADER_UNUSED_NV;
        } else {
            throw std::runtime_error("Unhandled shader stage!");
        }

        pipeline.shader_ident_offsets[sg.name] = shader_info.size() * pipeline.ident_size;

        shader_info.push_back(ss_ci);
        group_info.push_back(g_ci);
    }

    VkRayTracingPipelineCreateInfoNV pipeline_create_info = {};
    pipeline_create_info.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_NV;
    pipeline_create_info.stageCount = shader_info.size();
    pipeline_create_info.pStages = shader_info.data();
    pipeline_create_info.groupCount = group_info.size();
    pipeline_create_info.pGroups = group_info.data();
    pipeline_create_info.maxRecursionDepth = recursion_depth;
    pipeline_create_info.layout = layout;
    CHECK_VULKAN(vkCreateRayTracingPipelines(device.logical_device(),
                                             VK_NULL_HANDLE,
                                             1,
                                             &pipeline_create_info,
                                             nullptr,
                                             &pipeline.pipeline));

    pipeline.shader_identifiers.resize(shader_info.size() * pipeline.ident_size, 0);
    CHECK_VULKAN(vkGetRayTracingShaderGroupHandles(device.logical_device(),
                                                   pipeline.pipeline,
                                                   0,
                                                   shader_info.size(),
                                                   pipeline.shader_identifiers.size(),
                                                   pipeline.shader_identifiers.data()));

    return pipeline;
}

ShaderRecord::ShaderRecord(const std::string &name,
                           const std::string &shader_name,
                           size_t param_size)
    : name(name), shader_name(shader_name), param_size(param_size)
{
}

void ShaderBindingTable::map_sbt()
{
    assert(!sbt_mapping);
    sbt_mapping = reinterpret_cast<uint8_t *>(upload_sbt->map());
}

uint8_t *ShaderBindingTable::sbt_params(const std::string &name)
{
    assert(sbt_mapping);
    auto fnd = sbt_param_offsets.find(name);
    if (fnd == sbt_param_offsets.end()) {
        throw std::runtime_error("Failed to find SBT entry for group " + name);
    }
    return sbt_mapping + fnd->second;
}

void ShaderBindingTable::unmap_sbt()
{
    upload_sbt->unmap();
    sbt_mapping = nullptr;
}

SBTBuilder::SBTBuilder(const RTPipeline *pipeline) : pipeline(pipeline) {}

SBTBuilder &SBTBuilder::set_raygen(const ShaderRecord &sr)
{
    raygen = sr;
    return *this;
}

SBTBuilder &SBTBuilder::add_miss(const ShaderRecord &sr)
{
    miss_records.push_back(sr);
    return *this;
}

SBTBuilder &SBTBuilder::add_hitgroup(const ShaderRecord &sr)
{
    hitgroups.push_back(sr);
    return *this;
}

ShaderBindingTable SBTBuilder::build(Device &device)
{
    ShaderBindingTable sbt;
    sbt.raygen_stride =
        device.raytracing_properties().shaderGroupHandleSize + raygen.param_size;
    sbt.miss_start =
        align_to(sbt.raygen_stride, device.raytracing_properties().shaderGroupBaseAlignment);

    for (const auto &m : miss_records) {
        sbt.miss_stride =
            std::max(sbt.miss_stride,
                     device.raytracing_properties().shaderGroupHandleSize + m.param_size);
    }

    sbt.hitgroup_start = align_to(sbt.miss_start + sbt.miss_stride + miss_records.size(),
                                  device.raytracing_properties().shaderGroupBaseAlignment);
    for (const auto &h : hitgroups) {
        sbt.hitgroup_stride =
            std::max(sbt.hitgroup_stride,
                     device.raytracing_properties().shaderGroupHandleSize + h.param_size);
    }

    const size_t sbt_size =
        align_to(sbt.hitgroup_start + sbt.hitgroup_stride * hitgroups.size(),
                 device.raytracing_properties().shaderGroupBaseAlignment);
    sbt.upload_sbt = Buffer::host(device, sbt_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    sbt.sbt =
        Buffer::host(device,
                     sbt_size,
                     VK_BUFFER_USAGE_RAY_TRACING_BIT_NV | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    sbt.map_sbt();

    const size_t ident_size = device.raytracing_properties().shaderGroupHandleSize;
    size_t offset = 0;
    // Copy the shader identifier and record where to write the parameters
    std::memcpy(sbt.sbt_mapping, pipeline->shader_ident(raygen.shader_name), ident_size);
    sbt.sbt_param_offsets[raygen.name] = ident_size;

    offset = sbt.miss_start;
    for (const auto &m : miss_records) {
        // Copy the shader identifier and record where to write the parameters
        std::memcpy(
            sbt.sbt_mapping + offset, pipeline->shader_ident(m.shader_name), ident_size);
        sbt.sbt_param_offsets[m.name] = offset + ident_size;
        offset += sbt.miss_stride;
    }

    offset = sbt.hitgroup_start;
    for (const auto &hg : hitgroups) {
        // Copy the shader identifier and record where to write the parameters
        std::memcpy(
            sbt.sbt_mapping + offset, pipeline->shader_ident(hg.shader_name), ident_size);
        sbt.sbt_param_offsets[hg.name] = offset + ident_size;
        offset += sbt.hitgroup_stride;
    }

    sbt.unmap_sbt();

    return sbt;
}
}
