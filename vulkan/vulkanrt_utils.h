#pragma once

#include <memory>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>
#include "vulkan_utils.h"

namespace vk {

// See https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/vkspec.html#acceleration-structure-instance
struct GeometryInstance {
	float transform[12];
	uint32_t instance_custom_index : 24;
	uint32_t mask : 8;
	uint32_t instance_offset : 24;
	uint32_t flags : 8;
	uint64_t acceleration_structure_handle;
};

class TriangleMesh {
	Device *device = nullptr;
	VkGeometryNV geom_desc = {};
	VkBuildAccelerationStructureFlagBitsNV build_flags = (VkBuildAccelerationStructureFlagBitsNV)0;
	VkAccelerationStructureInfoNV accel_info = {};

	VkDeviceMemory bvh_mem, compacted_mem;
	std::shared_ptr<Buffer> scratch;
	VkQueryPool query_pool = VK_NULL_HANDLE;
	VkAccelerationStructureNV compacted_bvh = VK_NULL_HANDLE;

public:
	std::shared_ptr<Buffer> vertex_buf, index_buf, normal_buf, uv_buf;
	VkAccelerationStructureNV bvh = VK_NULL_HANDLE;
	uint64_t handle = 0;

	// TODO: Allow other vertex and index formats? Right now this
	// assumes vec3f verts and uint3 indices
	TriangleMesh(Device &dev, std::shared_ptr<Buffer> &vertex_buf, std::shared_ptr<Buffer> &index_buf,
			std::shared_ptr<Buffer> &normal_buf, std::shared_ptr<Buffer> &uv_buf,
		uint32_t geom_flags = VK_GEOMETRY_OPAQUE_BIT_NV,
		uint32_t build_flags =
			VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_NV
			| VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_NV);

	TriangleMesh() = default;
	~TriangleMesh();

	// todo move operators

	TriangleMesh(const TriangleMesh&) = delete;
	TriangleMesh& operator=(const TriangleMesh&) = delete;

	/* After calling build the commands are placed in the command list
	 * with a barrier to wait on the completion of the build
	 */
	void enqueue_build(VkCommandBuffer &cmd_buf);

	/* Enqueue the BVH compaction copy if the BVH was built with compaction enabled.
	 * The BVH build must have been enqueued and completed so that the post build info is available
	 * TODO: query compacted size via vkCmdWriteAccelerationStructuresPropertiesNV
	 */
	void enqueue_compaction(VkCommandBuffer &cmd_buf);

	/* Finalize the BVH build structures to release any scratch space.
	 * Must call after enqueue compaction if performing compaction, otherwise
	 * this can be called after the work from enqueue build has been finished
	 */
	void finalize();

	size_t num_tris() const;
};

class TopLevelBVH {
	Device *device = nullptr;
	size_t n_instances = 0;
	VkBuildAccelerationStructureFlagBitsNV build_flags = (VkBuildAccelerationStructureFlagBitsNV)0;
	VkAccelerationStructureInfoNV accel_info = {};

	VkDeviceMemory bvh_mem;
	std::shared_ptr<Buffer> scratch;

public:
	std::shared_ptr<Buffer> instance_buf;
	VkAccelerationStructureNV bvh = VK_NULL_HANDLE;
	uint64_t handle = 0;

	// TODO: Re-check on compacting the top-level BVH in DXR, it seems to be do-able
	// in OptiX, maybe DXR and Vulkan too?
	TopLevelBVH(Buffer instance_buf, size_t num_instances,
		uint32_t build_flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_NV);

	TopLevelBVH() = default;
	~TopLevelBVH();

	// todo move operators

	TopLevelBVH(const TopLevelBVH&) = delete;
	TopLevelBVH& operator=(const TopLevelBVH&) = delete;

	/* After calling build the commands are placed in the command list, with a
	 * UAV barrier to wait on the completion of the build before other commands are
	 * run, but does not submit the command list.
	 */
	void enqueue_build(VkCommandBuffer &cmd_buf);

	// Free the BVH build scratch space
	void finalize();

	size_t num_instances() const;
};


}

