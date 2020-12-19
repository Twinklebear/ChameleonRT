#pragma once

#include <memory>
#include <unordered_map>
#include "material.h"
#include "mesh.h"
#include "vulkan_utils.h"
#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

namespace vkrt {

struct Geometry {
    std::shared_ptr<Buffer> vertex_buf, index_buf, normal_buf, uv_buf;
    VkAccelerationStructureGeometryKHR geom_desc = {};

    Geometry() = default;

    Geometry(std::shared_ptr<Buffer> vertex_buf,
             std::shared_ptr<Buffer> index_buf,
             std::shared_ptr<Buffer> normal_buf,
             std::shared_ptr<Buffer> uv_buf,
             uint32_t geom_flags = VK_GEOMETRY_OPAQUE_BIT_KHR);

    uint32_t num_vertices() const;

    uint32_t num_triangles() const;
};

class TriangleMesh {
    Device *device = nullptr;
    std::vector<VkAccelerationStructureGeometryKHR> geom_descs;

    VkBuildAccelerationStructureFlagBitsKHR build_flags =
        (VkBuildAccelerationStructureFlagBitsKHR)0;

    std::shared_ptr<Buffer> bvh_buf, scratch_buf, compacted_buf;
    VkQueryPool query_pool = VK_NULL_HANDLE;
    VkAccelerationStructureKHR compacted_bvh = VK_NULL_HANDLE;

public:
    std::vector<Geometry> geometries;
    VkAccelerationStructureKHR bvh = VK_NULL_HANDLE;
    uint64_t handle = 0;

    // TODO: Allow other vertex and index formats? Right now this
    // assumes vec3f verts and uint3 indices
    TriangleMesh(
        Device &dev,
        std::vector<Geometry> geometries,
        uint32_t build_flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                               VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR);

    TriangleMesh() = default;
    ~TriangleMesh();

    // todo move operators

    TriangleMesh(const TriangleMesh &) = delete;
    TriangleMesh &operator=(const TriangleMesh &) = delete;

    /* After calling build the commands are placed in the command list
     * with a barrier to wait on the completion of the build
     */
    void enqueue_build(VkCommandBuffer &cmd_buf);

    /* Enqueue the BVH compaction copy if the BVH was built with compaction enabled.
     * The BVH build must have been enqueued and completed so that the post build info is
     * available
     */
    void enqueue_compaction(VkCommandBuffer &cmd_buf);

    /* Finalize the BVH build structures to release any scratch space.
     * Must call after enqueue compaction if performing compaction, otherwise
     * this can be called after the work from enqueue build has been finished
     */
    void finalize();
};

class TopLevelBVH {
    Device *device = nullptr;
    VkBuildAccelerationStructureFlagBitsKHR build_flags =
        (VkBuildAccelerationStructureFlagBitsKHR)0;

    std::shared_ptr<Buffer> bvh_buf, scratch_buf;

public:
    std::shared_ptr<Buffer> instance_buf;
    std::vector<Instance> instances;
    VkAccelerationStructureKHR bvh = VK_NULL_HANDLE;
    uint64_t handle = 0;

    // TODO: Re-check on compacting the top-level BVH in DXR, it seems to be do-able
    // in OptiX, maybe DXR and Vulkan too?
    TopLevelBVH(
        Device &dev,
        std::shared_ptr<Buffer> &instance_buf,
        const std::vector<Instance> &instances,
        uint32_t build_flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

    TopLevelBVH() = default;
    ~TopLevelBVH();

    // todo move operators

    TopLevelBVH(const TopLevelBVH &) = delete;
    TopLevelBVH &operator=(const TopLevelBVH &) = delete;

    /* After calling build the commands are placed in the command list, with a
     * UAV barrier to wait on the completion of the build before other commands are
     * run, but does not submit the command list.
     */
    void enqueue_build(VkCommandBuffer &cmd_buf);

    // Free the BVH build scratch space
    void finalize();

    size_t num_instances() const;
};

class RTPipeline {
    std::vector<uint8_t> shader_identifiers;
    std::unordered_map<std::string, size_t> shader_ident_offsets;
    size_t ident_size = 0;
    VkPipeline pipeline = VK_NULL_HANDLE;

    friend class RTPipelineBuilder;

public:
    const uint8_t *shader_ident(const std::string &name) const;

    size_t shader_ident_size() const;

    VkPipeline handle();
};

struct ShaderGroup {
    const std::shared_ptr<ShaderModule> shader_module;
    VkShaderStageFlagBits stage;
    VkRayTracingShaderGroupTypeKHR group;
    std::string name;
    std::string entry_point;

    ShaderGroup(const std::string &name,
                const std::shared_ptr<ShaderModule> &shader_module,
                const std::string &entry_point,
                VkShaderStageFlagBits stage,
                VkRayTracingShaderGroupTypeKHR group);
};

class RTPipelineBuilder {
    std::vector<ShaderGroup> shaders;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    uint32_t recursion_depth = 1;

public:
    RTPipelineBuilder &set_raygen(const std::string &name,
                                  const std::shared_ptr<ShaderModule> &shader,
                                  const std::string &entry_point = "main");

    RTPipelineBuilder &add_miss(const std::string &name,
                                const std::shared_ptr<ShaderModule> &shader,
                                const std::string &entry_point = "main");

    RTPipelineBuilder &add_hitgroup(const std::string &name,
                                    const std::shared_ptr<ShaderModule> &shader,
                                    const std::string &entry_point = "main");

    RTPipelineBuilder &set_layout(VkPipelineLayout layout);

    RTPipelineBuilder &set_recursion_depth(uint32_t depth);

    RTPipeline build(Device &device);
};

struct ShaderRecord {
    std::string name;
    std::string shader_name;
    size_t param_size = 0;

    ShaderRecord() = default;
    ShaderRecord(const std::string &name, const std::string &shader_name, size_t param_size);
};

struct ShaderBindingTable {
    std::unordered_map<std::string, size_t> sbt_param_offsets;
    std::shared_ptr<Buffer> upload_sbt;
    uint8_t *sbt_mapping = nullptr;

    std::shared_ptr<Buffer> sbt;
    VkStridedDeviceAddressRegionKHR raygen = {};
    VkStridedDeviceAddressRegionKHR miss = {};
    VkStridedDeviceAddressRegionKHR hitgroup = {};

    void map_sbt();

    uint8_t *sbt_params(const std::string &name);

    void unmap_sbt();
};

class SBTBuilder {
    const RTPipeline *pipeline = nullptr;
    ShaderRecord raygen;
    std::vector<ShaderRecord> miss_records;
    std::vector<ShaderRecord> hitgroups;

public:
    SBTBuilder(const RTPipeline *pipeline);

    SBTBuilder &set_raygen(const ShaderRecord &sr);

    SBTBuilder &add_miss(const ShaderRecord &sr);

    // TODO: Maybe similar to DXR where we take the per-ray type hit groups? Or should I change
    // the DXR one to work more like this? How would the shader indexing work out easiest if I
    // start mixing multiple geometries into a bottom level BVH?
    SBTBuilder &add_hitgroup(const ShaderRecord &sr);

    ShaderBindingTable build(Device &device);
};

}
