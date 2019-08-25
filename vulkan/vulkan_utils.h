#pragma once

#include <memory>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#define CHECK_VULKAN(FN) \
	{ \
		VkResult r = FN; \
		if (r != VK_SUCCESS) {\
			std::cout << #FN << " failed\n" << std::flush; \
			throw std::runtime_error(#FN " failed!");  \
		} \
	}


extern PFN_vkCreateAccelerationStructureNV vkCreateAccelerationStructure;
extern PFN_vkDestroyAccelerationStructureNV vkDestroyAccelerationStructure;
extern PFN_vkBindAccelerationStructureMemoryNV vkBindAccelerationStructureMemory;
extern PFN_vkGetAccelerationStructureHandleNV vkGetAccelerationStructureHandle;
extern PFN_vkGetAccelerationStructureMemoryRequirementsNV vkGetAccelerationStructureMemoryRequirements;
extern PFN_vkCmdBuildAccelerationStructureNV vkCmdBuildAccelerationStructure;
extern PFN_vkCreateRayTracingPipelinesNV vkCreateRayTracingPipelines;
extern PFN_vkGetRayTracingShaderGroupHandlesNV vkGetRayTracingShaderGroupHandles;
extern PFN_vkCmdTraceRaysNV vkCmdTraceRays;

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

class Device {
	VkInstance instance = VK_NULL_HANDLE;
	VkPhysicalDevice physical_device = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;
	VkQueue queue = VK_NULL_HANDLE;

	uint32_t graphics_queue_index = -1;

	VkPhysicalDeviceMemoryProperties mem_props = {};
	VkPhysicalDeviceRayTracingPropertiesNV rt_props = {};

public:
	Device();
	~Device();

	Device(Device &&d);
	Device& operator=(Device &&d);

	Device(const Device &) = delete;
	Device& operator=(const Device &) = delete;

	VkDevice logical_device();

	VkQueue graphics_queue();

	VkCommandPool make_command_pool(VkCommandPoolCreateFlagBits flags);

	uint32_t memory_type_index(uint32_t type_filter, VkMemoryPropertyFlags props) const;

	const VkPhysicalDeviceMemoryProperties& memory_properties() const;
	const VkPhysicalDeviceRayTracingPropertiesNV& raytracing_properties() const;

private:
	void make_instance();
	void select_physical_device();
	void make_logical_device();
};

class Buffer {
	size_t buf_size = 0;
	VkBuffer buf = VK_NULL_HANDLE;
	VkDeviceMemory mem = VK_NULL_HANDLE;
	Device *vkdevice = nullptr;
	bool host_visible = false;

	static VkBufferCreateInfo create_info(size_t nbytes, VkBufferUsageFlags usage);

	static VkMemoryAllocateInfo alloc_info(Device &device, const VkBuffer &buf,
			VkMemoryPropertyFlags mem_props);

	static std::shared_ptr<Buffer> make_buffer(Device &device, size_t nbytes, VkBufferUsageFlags usage,
			VkMemoryPropertyFlags mem_props);

public:
	Buffer() = default;
	~Buffer();
	Buffer(Buffer &&b);
	Buffer& operator=(Buffer &&b);

	Buffer(const Buffer &) = delete;
	Buffer& operator=(const Buffer &) = delete;

	static std::shared_ptr<Buffer> host(Device &device, size_t nbytes, VkBufferUsageFlags usage);
	static std::shared_ptr<Buffer> device(Device &device, size_t nbytes, VkBufferUsageFlags usage);

	// Map the entire range of the buffer
	void* map();
	// Map a subset of the buffer starting at offset of some size
	void* map(size_t offset, size_t size);

	void unmap();

	size_t size() const;
};

/*
class Texture2D {
	glm::uvec2 timds = glm::uvec2(0);
};

*/
}

