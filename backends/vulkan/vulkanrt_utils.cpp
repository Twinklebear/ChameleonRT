#include "vulkanrt_utils.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <numeric>
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
    geom_desc.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom_desc.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geom_desc.flags = geom_flags;

    geom_desc.geometry.triangles.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geom_desc.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    geom_desc.geometry.triangles.vertexData.deviceAddress = vertex_buf->device_address();
    geom_desc.geometry.triangles.vertexStride = sizeof(glm::vec3);
    geom_desc.geometry.triangles.maxVertex = num_vertices();

    geom_desc.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    geom_desc.geometry.triangles.indexData.deviceAddress = index_buf->device_address();

    geom_desc.geometry.triangles.transformData.deviceAddress = NULL;
}

uint32_t Geometry::num_vertices() const
{
    return vertex_buf->size() / sizeof(glm::vec3);
}

uint32_t Geometry::num_triangles() const
{
    return index_buf->size() / sizeof(glm::uvec3);
}

TriangleMesh::TriangleMesh(Device &dev, std::vector<Geometry> geoms, uint32_t build_flags)
    : device(&dev),
      build_flags((VkBuildAccelerationStructureFlagBitsKHR)build_flags),
      geometries(geoms)

{
    std::transform(geometries.begin(),
                   geometries.end(),
                   std::back_inserter(geom_descs),
                   [](const Geometry &g) { return g.geom_desc; });
}

TriangleMesh::~TriangleMesh()
{
    if (bvh != VK_NULL_HANDLE) {
        DestroyAccelerationStructureKHR(device->logical_device(), bvh, nullptr);
    }
}

void TriangleMesh::enqueue_build(VkCommandBuffer &cmd_buf)
{
    // Determine how much memory the acceleration structure will need
    VkAccelerationStructureBuildGeometryInfoKHR build_info = {};
    build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build_info.flags = build_flags;
    build_info.geometryCount = geom_descs.size();
    build_info.pGeometries = geom_descs.data();

    std::vector<uint32_t> primitive_counts;
    std::transform(geometries.begin(),
                   geometries.end(),
                   std::back_inserter(primitive_counts),
                   [](const Geometry &g) { return g.num_triangles(); });

    VkAccelerationStructureBuildSizesInfoKHR build_size_info = {};
    build_size_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    GetAccelerationStructureBuildSizesKHR(device->logical_device(),
                                          VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                          &build_info,
                                          primitive_counts.data(),
                                          &build_size_info);

    bvh_buf = Buffer::device(*device,
                             build_size_info.accelerationStructureSize,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);

    scratch_buf = Buffer::device(
        *device,
        build_size_info.buildScratchSize,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    // Create the acceleration structure
    VkAccelerationStructureCreateInfoKHR as_create_info = {};
    as_create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    as_create_info.buffer = bvh_buf->handle();
    as_create_info.size = build_size_info.accelerationStructureSize;
    as_create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    CHECK_VULKAN(CreateAccelerationStructureKHR(
        device->logical_device(), &as_create_info, nullptr, &bvh));

    // Enqueue the acceleration structure build
    VkAccelerationStructureBuildGeometryInfoKHR accel_build_info = {};
    accel_build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    accel_build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    accel_build_info.flags = build_flags;
    accel_build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    accel_build_info.dstAccelerationStructure = bvh;
    accel_build_info.geometryCount = geom_descs.size();
    accel_build_info.pGeometries = geom_descs.data();
    accel_build_info.scratchData.deviceAddress = scratch_buf->device_address();

    std::vector<VkAccelerationStructureBuildRangeInfoKHR> build_offset_info;
    std::transform(geometries.begin(),
                   geometries.end(),
                   std::back_inserter(build_offset_info),
                   [](const Geometry &g) {
                       VkAccelerationStructureBuildRangeInfoKHR offset = {};
                       offset.primitiveCount = g.num_triangles();
                       offset.primitiveOffset = 0;
                       offset.firstVertex = 0;
                       offset.transformOffset = 0;
                       return offset;
                   });

    VkAccelerationStructureBuildRangeInfoKHR *build_offset_info_ptr = build_offset_info.data();
    // Enqueue the build commands into the command buffer
    CmdBuildAccelerationStructuresKHR(cmd_buf, 1, &accel_build_info, &build_offset_info_ptr);

    // Memory barrier to have subsequent commands wait on build completion
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                            VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                            VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    vkCmdPipelineBarrier(cmd_buf,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0,
                         1,
                         &barrier,
                         0,
                         nullptr,
                         0,
                         nullptr);

    // Read the compacted size if we're compacting
    if (build_flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR) {
        VkQueryPoolCreateInfo pool_ci = {};
        pool_ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        pool_ci.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
        pool_ci.queryCount = 1;
        CHECK_VULKAN(
            vkCreateQueryPool(device->logical_device(), &pool_ci, nullptr, &query_pool));

        vkCmdResetQueryPool(cmd_buf, query_pool, 0, 1);
        vkCmdBeginQuery(cmd_buf, query_pool, 0, 0);
        CmdWriteAccelerationStructuresPropertiesKHR(
            cmd_buf,
            1,
            &bvh,
            VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
            query_pool,
            0);
        vkCmdEndQuery(cmd_buf, query_pool, 0);
    }
}

void TriangleMesh::enqueue_compaction(VkCommandBuffer &cmd_buf)
{
    if (!(build_flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR)) {
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

    compacted_buf = Buffer::device(*device,
                                   compacted_size,
                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                       VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);

    // Create the compacted acceleration structure
    VkAccelerationStructureCreateInfoKHR as_create_info = {};
    as_create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    as_create_info.buffer = compacted_buf->handle();
    as_create_info.size = compacted_size;
    as_create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    CHECK_VULKAN(CreateAccelerationStructureKHR(
        device->logical_device(), &as_create_info, nullptr, &compacted_bvh));

    VkCopyAccelerationStructureInfoKHR copy_info = {};
    copy_info.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR;
    copy_info.src = bvh;
    copy_info.dst = compacted_bvh;
    copy_info.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
    CmdCopyAccelerationStructureKHR(cmd_buf, &copy_info);
}

void TriangleMesh::finalize()
{
    scratch_buf = nullptr;

    // Compaction is done, so swap the old handle with the compacted one and free the old
    // output memory
    if (build_flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR) {
        vkDestroyQueryPool(device->logical_device(), query_pool, nullptr);
        query_pool = VK_NULL_HANDLE;

        DestroyAccelerationStructureKHR(device->logical_device(), bvh, nullptr);

        bvh = compacted_bvh;
        bvh_buf = compacted_buf;

        compacted_bvh = VK_NULL_HANDLE;
    }
    VkAccelerationStructureDeviceAddressInfoKHR addr_info = {};
    addr_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addr_info.accelerationStructure = bvh;
    handle = GetAccelerationStructureDeviceAddressKHR(device->logical_device(), &addr_info);
}

TopLevelBVH::TopLevelBVH(Device &dev,
                         std::shared_ptr<Buffer> &inst_buf,
                         const std::vector<Instance> &instances,
                         uint32_t build_flags)
    : device(&dev),
      build_flags((VkBuildAccelerationStructureFlagBitsKHR)build_flags),
      instance_buf(inst_buf),
      instances(instances)
{
}

TopLevelBVH::~TopLevelBVH()
{
    if (bvh != VK_NULL_HANDLE) {
        DestroyAccelerationStructureKHR(device->logical_device(), bvh, nullptr);
    }
}

void TopLevelBVH::enqueue_build(VkCommandBuffer &cmd_buf)
{
    VkAccelerationStructureGeometryKHR instance_desc = {};
    instance_desc.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    instance_desc.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    instance_desc.flags = 0;
    instance_desc.geometry.instances.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instance_desc.geometry.instances.arrayOfPointers = false;
    instance_desc.geometry.instances.data.deviceAddress = instance_buf->device_address();

    // Determine how much memory the acceleration structure will need
    VkAccelerationStructureBuildGeometryInfoKHR build_info = {};
    build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    build_info.flags = build_flags;
    build_info.geometryCount = 1;
    build_info.pGeometries = &instance_desc;

    const uint32_t instance_count = instances.size();
    VkAccelerationStructureBuildSizesInfoKHR build_size_info = {};
    build_size_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    GetAccelerationStructureBuildSizesKHR(device->logical_device(),
                                          VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                          &build_info,
                                          &instance_count,
                                          &build_size_info);

    bvh_buf = Buffer::device(*device,
                             build_size_info.accelerationStructureSize,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);

    scratch_buf = Buffer::device(
        *device,
        build_size_info.buildScratchSize,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    // Create the acceleration structure
    VkAccelerationStructureCreateInfoKHR as_create_info = {};
    as_create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    as_create_info.buffer = bvh_buf->handle();
    as_create_info.offset = 0;
    as_create_info.size = build_size_info.accelerationStructureSize;
    as_create_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    CHECK_VULKAN(CreateAccelerationStructureKHR(
        device->logical_device(), &as_create_info, nullptr, &bvh));

    // Enqueue the acceleration structure build
    VkAccelerationStructureBuildGeometryInfoKHR accel_build_info = {};
    accel_build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    accel_build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    accel_build_info.flags = build_flags;
    accel_build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    accel_build_info.dstAccelerationStructure = bvh;
    accel_build_info.geometryCount = 1;
    accel_build_info.pGeometries = &instance_desc;
    accel_build_info.scratchData.deviceAddress = scratch_buf->device_address();

    VkAccelerationStructureBuildRangeInfoKHR build_offset_info = {};
    build_offset_info.primitiveCount = instances.size();
    build_offset_info.primitiveOffset = 0;
    build_offset_info.firstVertex = 0;
    build_offset_info.transformOffset = 0;

    VkAccelerationStructureBuildRangeInfoKHR *build_offset_info_ptr = &build_offset_info;
    // Enqueue the build commands into the command buffer
    CmdBuildAccelerationStructuresKHR(cmd_buf, 1, &accel_build_info, &build_offset_info_ptr);

    // Enqueue a barrier on the build
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                            VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                            VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    vkCmdPipelineBarrier(cmd_buf,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
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
    scratch_buf = nullptr;
    VkAccelerationStructureDeviceAddressInfoKHR addr_info = {};
    addr_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addr_info.accelerationStructure = bvh;
    handle = GetAccelerationStructureDeviceAddressKHR(device->logical_device(), &addr_info);
}

size_t TopLevelBVH::num_instances() const
{
    return instances.size();
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
                         VkRayTracingShaderGroupTypeKHR group)
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
                         VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                         VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR);
    return *this;
}

RTPipelineBuilder &RTPipelineBuilder::add_miss(const std::string &name,
                                               const std::shared_ptr<ShaderModule> &shader,
                                               const std::string &entry_point)
{
    shaders.emplace_back(name,
                         shader,
                         entry_point,
                         VK_SHADER_STAGE_MISS_BIT_KHR,
                         VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR);
    return *this;
}

RTPipelineBuilder &RTPipelineBuilder::add_hitgroup(const std::string &name,
                                                   const std::shared_ptr<ShaderModule> &shader,
                                                   const std::string &entry_point)
{
    shaders.emplace_back(name,
                         shader,
                         entry_point,
                         VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                         VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR);
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
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> group_info;

    RTPipeline pipeline;

    pipeline.ident_size = device.raytracing_pipeline_properties().shaderGroupHandleSize;
    for (const auto &sg : shaders) {
        VkPipelineShaderStageCreateInfo ss_ci = {};
        ss_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ss_ci.stage = sg.stage;
        ss_ci.module = sg.shader_module->module;
        ss_ci.pName = sg.entry_point.c_str();

        VkRayTracingShaderGroupCreateInfoKHR g_ci = {};
        g_ci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        g_ci.type = sg.group;
        if (sg.stage == VK_SHADER_STAGE_RAYGEN_BIT_KHR ||
            sg.stage == VK_SHADER_STAGE_MISS_BIT_KHR) {
            g_ci.generalShader = shader_info.size();
            g_ci.closestHitShader = VK_SHADER_UNUSED_KHR;
            g_ci.anyHitShader = VK_SHADER_UNUSED_KHR;
            g_ci.intersectionShader = VK_SHADER_UNUSED_KHR;
        } else if (sg.stage == VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR) {
            g_ci.generalShader = VK_SHADER_UNUSED_KHR;
            g_ci.closestHitShader = shader_info.size();
            g_ci.anyHitShader = VK_SHADER_UNUSED_KHR;
            g_ci.intersectionShader = VK_SHADER_UNUSED_KHR;
        } else {
            throw std::runtime_error("Unhandled shader stage!");
        }

        pipeline.shader_ident_offsets[sg.name] = shader_info.size() * pipeline.ident_size;

        shader_info.push_back(ss_ci);
        group_info.push_back(g_ci);
    }

    VkRayTracingPipelineCreateInfoKHR pipeline_create_info = {};
    pipeline_create_info.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    pipeline_create_info.stageCount = shader_info.size();
    pipeline_create_info.pStages = shader_info.data();
    pipeline_create_info.groupCount = group_info.size();
    pipeline_create_info.pGroups = group_info.data();
    pipeline_create_info.maxPipelineRayRecursionDepth = recursion_depth;
    pipeline_create_info.layout = layout;
    CHECK_VULKAN(CreateRayTracingPipelinesKHR(device.logical_device(),
                                              VK_NULL_HANDLE,
                                              VK_NULL_HANDLE,
                                              1,
                                              &pipeline_create_info,
                                              nullptr,
                                              &pipeline.pipeline));

    pipeline.shader_identifiers.resize(shader_info.size() * pipeline.ident_size, 0);
    CHECK_VULKAN(GetRayTracingShaderGroupHandlesKHR(device.logical_device(),
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
    const uint32_t group_handle_size =
        device.raytracing_pipeline_properties().shaderGroupHandleSize;
    const uint32_t group_alignment =
        device.raytracing_pipeline_properties().shaderGroupBaseAlignment;

    ShaderBindingTable sbt;
    sbt.raygen.stride = align_to(group_handle_size + raygen.param_size, group_handle_size);
    sbt.raygen.size = sbt.raygen.stride;

    const uint32_t miss_offset = align_to(sbt.raygen.size, group_alignment);

    sbt.miss.stride = 0;
    for (const auto &m : miss_records) {
        sbt.miss.stride = std::max(
            sbt.miss.stride, align_to(group_handle_size + m.param_size, group_handle_size));
    }
    sbt.miss.size = sbt.miss.stride * miss_records.size();

    const uint32_t hitgroup_offset = align_to(miss_offset + sbt.miss.size, group_alignment);
    sbt.hitgroup.stride = 0;
    for (const auto &h : hitgroups) {
        sbt.hitgroup.stride =
            std::max(sbt.hitgroup.stride,
                     align_to(group_handle_size + h.param_size, group_handle_size));
    }
    sbt.hitgroup.size = sbt.hitgroup.stride * hitgroups.size();

    const size_t sbt_size = align_to(hitgroup_offset + sbt.hitgroup.size, group_alignment);
    sbt.upload_sbt = Buffer::host(device, sbt_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    sbt.sbt = Buffer::device(device,
                             sbt_size,
                             VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    sbt.raygen.deviceAddress = sbt.sbt->device_address();
    sbt.miss.deviceAddress = sbt.sbt->device_address() + miss_offset;
    sbt.hitgroup.deviceAddress = sbt.sbt->device_address() + hitgroup_offset;

    sbt.map_sbt();

    size_t offset = 0;
    // Copy the shader identifier and record where to write the parameters
    std::memcpy(
        sbt.sbt_mapping, pipeline->shader_ident(raygen.shader_name), group_handle_size);
    sbt.sbt_param_offsets[raygen.name] = group_handle_size;

    offset = miss_offset;
    for (const auto &m : miss_records) {
        // Copy the shader identifier and record where to write the parameters
        std::memcpy(sbt.sbt_mapping + offset,
                    pipeline->shader_ident(m.shader_name),
                    group_handle_size);
        sbt.sbt_param_offsets[m.name] = offset + group_handle_size;
        offset += sbt.miss.stride;
    }

    offset = hitgroup_offset;
    for (const auto &hg : hitgroups) {
        // Copy the shader identifier and record where to write the parameters
        std::memcpy(sbt.sbt_mapping + offset,
                    pipeline->shader_ident(hg.shader_name),
                    group_handle_size);
        sbt.sbt_param_offsets[hg.name] = offset + group_handle_size;
        offset += sbt.hitgroup.stride;
    }

    sbt.unmap_sbt();

    return sbt;
}
}
