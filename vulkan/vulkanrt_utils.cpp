#include <iostream>
#include <glm/glm.hpp>
#include "util.h"
#include "vulkanrt_utils.h"

namespace vk {

TriangleMesh::TriangleMesh(Device &dev, std::shared_ptr<Buffer> &verts, std::shared_ptr<Buffer> &indices,
		std::shared_ptr<Buffer> &normals, std::shared_ptr<Buffer> &uvs,
		uint32_t geom_flags, uint32_t build_flags)
	: device(&dev), vertex_buf(verts), index_buf(indices), normal_buf(normals), uv_buf(uvs),
	  build_flags((VkBuildAccelerationStructureFlagBitsNV)build_flags)
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
	geom_desc.geometry.aabbs.sType = { VK_STRUCTURE_TYPE_GEOMETRY_AABB_NV };

	accel_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
	accel_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV;
	accel_info.flags = build_flags;
	accel_info.instanceCount = 0;
	accel_info.geometryCount = 1;
	accel_info.pGeometries = &geom_desc;

	VkAccelerationStructureCreateInfoNV create_info = {};
	create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
	create_info.info = accel_info;
	CHECK_VULKAN(vkCreateAccelerationStructure(device->logical_device(), &create_info, nullptr, &bvh));
}


TriangleMesh::~TriangleMesh() {
	if (bvh != VK_NULL_HANDLE) {
		vkDestroyAccelerationStructure(device->logical_device(), bvh, nullptr);
		vkFreeMemory(device->logical_device(), bvh_mem, nullptr);
	}
}

void TriangleMesh::enqueue_build(VkCommandBuffer &cmd_buf) {
	// Determine how much memory the acceleration structure will need
	VkAccelerationStructureMemoryRequirementsInfoNV mem_info = {};
	mem_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
	mem_info.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV;
	mem_info.accelerationStructure = bvh;

	VkMemoryRequirements2 mem_reqs = {};
	vkGetAccelerationStructureMemoryRequirements(device->logical_device(), &mem_info, &mem_reqs);
	// TODO WILL: For a single triangle it requests 64k output and 64k scratch? It seems like a lot.
	std::cout << "BLAS will need " << pretty_print_count(mem_reqs.memoryRequirements.size) << "b output space\n";
	// Allocate space for the build output
	bvh_mem = device->alloc(mem_reqs.memoryRequirements.size, mem_reqs.memoryRequirements.memoryTypeBits,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	// Allocate scratch space for the build
	mem_info.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV;
	vkGetAccelerationStructureMemoryRequirements(device->logical_device(), &mem_info, &mem_reqs);
	// TODO WILL: For a single triangle it requests 64k output and 64k scratch? It seems like a lot.
	std::cout << "BLAS will need " << pretty_print_count(mem_reqs.memoryRequirements.size) << "b scratch space\n";
	scratch = Buffer::device(*device, mem_reqs.memoryRequirements.size, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	// Bind the build output mem to the BVH
	VkBindAccelerationStructureMemoryInfoNV bind_mem_info = {};
	bind_mem_info.sType = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
	bind_mem_info.accelerationStructure = bvh;
	bind_mem_info.memory = bvh_mem;
	CHECK_VULKAN(vkBindAccelerationStructureMemory(device->logical_device(), 1, &bind_mem_info));

	// Enqueue the build commands into the command buffer
	vkCmdBuildAccelerationStructure(cmd_buf, &accel_info, VK_NULL_HANDLE, 0, false, bvh, VK_NULL_HANDLE,
			scratch->handle(), 0);

	// Memory barrier to have subsequent commands wait on build completion
	VkMemoryBarrier barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	barrier.srcAccessMask =
		VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;
	barrier.dstAccessMask =
		VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;

	vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
		VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV, 0,
		1, &barrier, 0, nullptr, 0, nullptr);

	// Read the compacted size if we're compacting
	if (build_flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_NV) {
		VkQueryPoolCreateInfo pool_ci = {};
		pool_ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
		pool_ci.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_NV;
		pool_ci.queryCount = 1;
		CHECK_VULKAN(vkCreateQueryPool(device->logical_device(), &pool_ci, nullptr, &query_pool));

		vkCmdResetQueryPool(cmd_buf, query_pool, 0, 1);
		vkCmdBeginQuery(cmd_buf, query_pool, 0, 0);
		vkCmdWriteAccelerationStructuresProperties(cmd_buf, 1, &bvh,
				VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_NV, query_pool, 0);
		vkCmdEndQuery(cmd_buf, query_pool, 0);
	}
}

void TriangleMesh::enqueue_compaction(VkCommandBuffer &cmd_buf) {
	uint64_t compacted_size = 0;
	CHECK_VULKAN(vkGetQueryPoolResults(device->logical_device(), query_pool, 0, 1,
		sizeof(uint64_t), &compacted_size, sizeof(uint64_t),
		VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
	std::cout << "BLAS will compact to " << compacted_size << "\n";

	// Same memory type requirements as the original structure, just less space needed
	VkAccelerationStructureMemoryRequirementsInfoNV mem_info = {};
	mem_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
	mem_info.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV;
	mem_info.accelerationStructure = bvh;

	VkMemoryRequirements2 mem_reqs = {};
	vkGetAccelerationStructureMemoryRequirements(device->logical_device(), &mem_info, &mem_reqs);
	compacted_mem = device->alloc(compacted_size, mem_reqs.memoryRequirements.memoryTypeBits,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	VkAccelerationStructureCreateInfoNV create_info = {};
	create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
	create_info.info = accel_info;
	CHECK_VULKAN(vkCreateAccelerationStructure(device->logical_device(), &create_info, nullptr, &compacted_bvh));

	// Bind the compacted mem to the compacted BVH handle
	VkBindAccelerationStructureMemoryInfoNV bind_mem_info = {};
	bind_mem_info.sType = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
	bind_mem_info.accelerationStructure = compacted_bvh;
	bind_mem_info.memory = compacted_mem;
	CHECK_VULKAN(vkBindAccelerationStructureMemory(device->logical_device(), 1, &bind_mem_info));

	vkCmdCopyAccelerationStructure(cmd_buf, compacted_bvh, bvh, VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_NV);
}

void TriangleMesh::finalize() {
	scratch = nullptr;

	// Compaction is done, so swap the old handle with the compacted one and free the old output memory
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

	CHECK_VULKAN(vkGetAccelerationStructureHandle(device->logical_device(), bvh,
				sizeof(uint64_t), &handle));
}

size_t TriangleMesh::num_tris() const {
	return geom_desc.geometry.triangles.indexCount / 3;
}

TopLevelBVH::TopLevelBVH(Device &dev, std::shared_ptr<Buffer> &inst_buf, uint32_t build_flags)
	: device(&dev), build_flags((VkBuildAccelerationStructureFlagBitsNV)build_flags), instance_buf(inst_buf)

{
	accel_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
	accel_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV;
	accel_info.instanceCount = instance_buf->size() / sizeof(GeometryInstance);
	accel_info.geometryCount = 0;

	VkAccelerationStructureCreateInfoNV create_info = {};
	create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
	create_info.info = accel_info;
	CHECK_VULKAN(vkCreateAccelerationStructure(device->logical_device(), &create_info, nullptr, &bvh));
}

TopLevelBVH::~TopLevelBVH() {
	if (bvh != VK_NULL_HANDLE) {
		vkDestroyAccelerationStructure(device->logical_device(), bvh, nullptr);
		vkFreeMemory(device->logical_device(), bvh_mem, nullptr);
	}
}

void TopLevelBVH::enqueue_build(VkCommandBuffer &cmd_buf) {
	// Determine how much memory the acceleration structure will need
	VkAccelerationStructureMemoryRequirementsInfoNV mem_info = {};
	mem_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
	mem_info.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV;
	mem_info.accelerationStructure = bvh;

	VkMemoryRequirements2 mem_reqs = {};
	vkGetAccelerationStructureMemoryRequirements(device->logical_device(), &mem_info, &mem_reqs);
	// TODO WILL: For a single triangle it requests 64k output and 64k scratch? It seems like a lot.
	std::cout << "TLAS will need " << mem_reqs.memoryRequirements.size << "b output space\n";
	// Allocate space for the build output
	bvh_mem = device->alloc(mem_reqs.memoryRequirements.size, mem_reqs.memoryRequirements.memoryTypeBits,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	// Determine how much additional memory we need for the build
	mem_info.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV;
	vkGetAccelerationStructureMemoryRequirements(device->logical_device(), &mem_info, &mem_reqs);
	std::cout << "TLAS will need " << mem_reqs.memoryRequirements.size << "b scratch space\n";
	scratch = Buffer::device(*device, mem_reqs.memoryRequirements.size, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	// Bind the build output mem to the BVH
	VkBindAccelerationStructureMemoryInfoNV bind_mem_info = {};
	bind_mem_info.sType = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
	bind_mem_info.accelerationStructure = bvh;
	bind_mem_info.memory = bvh_mem;
	CHECK_VULKAN(vkBindAccelerationStructureMemory(device->logical_device(), 1, &bind_mem_info));

	vkCmdBuildAccelerationStructure(cmd_buf, &accel_info, instance_buf->handle(), 0, false,
			bvh, VK_NULL_HANDLE, scratch->handle(), 0);

	// Enqueue a barrier on the build
	VkMemoryBarrier barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	barrier.srcAccessMask =
		VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;
	barrier.dstAccessMask =
		VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV;

	vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
		VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV, 0,
		1, &barrier, 0, nullptr, 0, nullptr);
}

void TopLevelBVH::finalize() {
	scratch = nullptr;
	CHECK_VULKAN(vkGetAccelerationStructureHandle(device->logical_device(), bvh,
				sizeof(uint64_t), &handle));
}

size_t TopLevelBVH::num_instances() const {
	return accel_info.instanceCount;
}

}

