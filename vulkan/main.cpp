#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <array>
#include <SDL.h>
#include <SDL_syswm.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>
#include "spirv_shaders_embedded_spv.h"

#define CHECK_VULKAN(FN) \
	{ \
		VkResult r = FN; \
		if (r != VK_SUCCESS) {\
			std::cout << #FN << " failed\n" << std::flush; \
			throw std::runtime_error(#FN " failed!");  \
		} \
	}

const std::array<float, 9> triangle_verts = {
	0.0, -0.5, 0.0,
	0.5, 0.5, 0.0,
	-0.5, 0.5, 0.0
};

const std::array<uint32_t, 3> triangle_indices = { 0, 1, 2 };

// See https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/vkspec.html#acceleration-structure-instance
struct VkGeometryInstanceNV {
	float transform[12];
	uint32_t instance_custom_index : 24;
	uint32_t mask : 8;
	uint32_t instance_offset : 24;
	uint32_t flags : 8;
	uint64_t acceleration_structure_handle;
};

PFN_vkCreateAccelerationStructureNV vkCreateAccelerationStructure;
PFN_vkDestroyAccelerationStructureNV vkDestroyAccelerationStructure;
PFN_vkBindAccelerationStructureMemoryNV vkBindAccelerationStructureMemory;
PFN_vkGetAccelerationStructureHandleNV vkGetAccelerationStructureHandle;
PFN_vkGetAccelerationStructureMemoryRequirementsNV vkGetAccelerationStructureMemoryRequirements;
PFN_vkCmdBuildAccelerationStructureNV vkCmdBuildAccelerationStructure;
PFN_vkCreateRayTracingPipelinesNV vkCreateRayTracingPipelines;
PFN_vkGetRayTracingShaderGroupHandlesNV vkGetRayTracingShaderGroupHandles;
PFN_vkCmdTraceRaysNV vkCmdTraceRays;

uint32_t get_memory_type_index(uint32_t type_filter, VkMemoryPropertyFlags props, const VkPhysicalDeviceMemoryProperties &mem_props) {
	for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
		if (type_filter & (1 << i) && (mem_props.memoryTypes[i].propertyFlags & props) == props) {
			return i;
		}
	}
	throw std::runtime_error("failed to find appropriate memory");
}

void setup_vulkan_rtx_fcns(VkDevice &device) {
	vkCreateAccelerationStructure = reinterpret_cast<PFN_vkCreateAccelerationStructureNV>(vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureNV"));
	vkDestroyAccelerationStructure = reinterpret_cast<PFN_vkDestroyAccelerationStructureNV>(vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureNV"));
	vkBindAccelerationStructureMemory = reinterpret_cast<PFN_vkBindAccelerationStructureMemoryNV>(vkGetDeviceProcAddr(device, "vkBindAccelerationStructureMemoryNV"));
	vkGetAccelerationStructureHandle = reinterpret_cast<PFN_vkGetAccelerationStructureHandleNV>(vkGetDeviceProcAddr(device, "vkGetAccelerationStructureHandleNV"));
	vkGetAccelerationStructureMemoryRequirements = reinterpret_cast<PFN_vkGetAccelerationStructureMemoryRequirementsNV>(vkGetDeviceProcAddr(device, "vkGetAccelerationStructureMemoryRequirementsNV"));
	vkCmdBuildAccelerationStructure = reinterpret_cast<PFN_vkCmdBuildAccelerationStructureNV>(vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructureNV"));
	vkCreateRayTracingPipelines = reinterpret_cast<PFN_vkCreateRayTracingPipelinesNV>(vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesNV"));
	vkGetRayTracingShaderGroupHandles = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesNV>(vkGetDeviceProcAddr(device, "vkGetRayTracingShaderGroupHandlesNV"));
	vkCmdTraceRays = reinterpret_cast<PFN_vkCmdTraceRaysNV>(vkGetDeviceProcAddr(device, "vkCmdTraceRaysNV"));
}

int win_width = 1280;
int win_height = 720;

int main(int argc, const char** argv) {
	if (SDL_Init(SDL_INIT_EVERYTHING) != 0) {
		std::cerr << "Failed to init SDL: " << SDL_GetError() << "\n";
		return -1;
	}

	SDL_Window* window = SDL_CreateWindow("SDL2 + Vulkan",
		SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, win_width, win_height, 0);

	{
		uint32_t extension_count = 0;
		vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
		std::cout << "num extensions: " << extension_count << "\n";
		std::vector<VkExtensionProperties> extensions(extension_count, VkExtensionProperties{});
		vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extensions.data());
		std::cout << "Available extensions:\n";
		for (const auto& e : extensions) {
			std::cout << e.extensionName << "\n";
		}
	}

	const std::array<const char*, 1> validation_layers = {
		"VK_LAYER_KHRONOS_validation"
	};

	// Make the Vulkan Instance
	VkInstance vk_instance = VK_NULL_HANDLE;
	{
		VkApplicationInfo app_info = {};
		app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		app_info.pApplicationName = "SDL2 + Vulkan";
		app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		app_info.pEngineName = "None";
		app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		app_info.apiVersion = VK_API_VERSION_1_1;

		const std::array<const char*, 2> extension_names = {
			VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
		};

		VkInstanceCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		create_info.pApplicationInfo = &app_info;
		create_info.enabledExtensionCount = extension_names.size();
		create_info.ppEnabledExtensionNames = extension_names.data();
		create_info.enabledLayerCount = validation_layers.size();
		create_info.ppEnabledLayerNames = validation_layers.data();

		CHECK_VULKAN(vkCreateInstance(&create_info, nullptr, &vk_instance));
	}

	VkSurfaceKHR vk_surface = VK_NULL_HANDLE;
	{
		SDL_SysWMinfo wm_info;
		SDL_VERSION(&wm_info.version);
		SDL_GetWindowWMInfo(window, &wm_info);

		VkWin32SurfaceCreateInfoKHR create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
		create_info.hwnd = wm_info.info.win.window;
		create_info.hinstance = wm_info.info.win.hinstance;
		CHECK_VULKAN(vkCreateWin32SurfaceKHR(vk_instance, &create_info, nullptr, &vk_surface));
	}

	VkPhysicalDevice vk_physical_device = VK_NULL_HANDLE;
	{
		uint32_t device_count = 0;
		vkEnumeratePhysicalDevices(vk_instance, &device_count, nullptr);
		std::cout << "Found " << device_count << " devices\n";
		std::vector<VkPhysicalDevice> devices(device_count, VkPhysicalDevice{});
		vkEnumeratePhysicalDevices(vk_instance, &device_count, devices.data());

		const bool has_discrete_gpu = std::find_if(devices.begin(), devices.end(),
			[](const VkPhysicalDevice& d) {
				VkPhysicalDeviceProperties properties;
				vkGetPhysicalDeviceProperties(d, &properties);
				return properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
			}) != devices.end();

			for (const auto& d : devices) {
				VkPhysicalDeviceProperties properties;
				VkPhysicalDeviceFeatures features;
				vkGetPhysicalDeviceProperties(d, &properties);
				vkGetPhysicalDeviceFeatures(d, &features);
				std::cout << properties.deviceName << "\n";

				// Check for RTX support
				uint32_t extension_count = 0;
				vkEnumerateDeviceExtensionProperties(d, nullptr, &extension_count, nullptr);
				std::cout << "num extensions: " << extension_count << "\n";
				std::vector<VkExtensionProperties> extensions(extension_count, VkExtensionProperties{});
				vkEnumerateDeviceExtensionProperties(d, nullptr, &extension_count, extensions.data());
				std::cout << "Device available extensions:\n";
				for (const auto& e : extensions) {
					std::cout << e.extensionName << "\n";
				}

				if (has_discrete_gpu && properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
					vk_physical_device = d;
					break;
				}
				else if (!has_discrete_gpu && properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
					vk_physical_device = d;
					break;
				}
			}
	}

	VkDevice vk_device = VK_NULL_HANDLE;
	VkQueue vk_queue = VK_NULL_HANDLE;
	uint32_t graphics_queue_index = -1;
	{
		uint32_t num_queue_families = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(vk_physical_device, &num_queue_families, nullptr);
		std::vector<VkQueueFamilyProperties> family_props(num_queue_families, VkQueueFamilyProperties{});
		vkGetPhysicalDeviceQueueFamilyProperties(vk_physical_device, &num_queue_families, family_props.data());
		for (uint32_t i = 0; i < num_queue_families; ++i) {
			// We want present and graphics on the same queue (kind of assume this will be supported on any discrete GPU)
			VkBool32 present_support = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(vk_physical_device, i, vk_surface, &present_support);
			if (present_support && (family_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
				graphics_queue_index = i;
			}
		}
		std::cout << "Graphics queue is " << graphics_queue_index << "\n";
		const float queue_priority = 1.f;

		VkDeviceQueueCreateInfo queue_create_info = {};
		queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queue_create_info.queueFamilyIndex = graphics_queue_index;
		queue_create_info.queueCount = 1;
		queue_create_info.pQueuePriorities = &queue_priority;

		VkPhysicalDeviceFeatures device_features = {};

		const std::array<const char*, 3> device_extensions = {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_NV_RAY_TRACING_EXTENSION_NAME,
			VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME
		};

		VkDeviceCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		create_info.queueCreateInfoCount = 1;
		create_info.pQueueCreateInfos = &queue_create_info;
		create_info.enabledLayerCount = validation_layers.size();
		create_info.ppEnabledLayerNames = validation_layers.data();
		create_info.enabledExtensionCount = device_extensions.size();
		create_info.ppEnabledExtensionNames = device_extensions.data();
		create_info.pEnabledFeatures = &device_features;
		CHECK_VULKAN(vkCreateDevice(vk_physical_device, &create_info, nullptr, &vk_device));
		setup_vulkan_rtx_fcns(vk_device);

		vkGetDeviceQueue(vk_device, graphics_queue_index, 0, &vk_queue);
	}

	VkPhysicalDeviceMemoryProperties mem_props = {};
	vkGetPhysicalDeviceMemoryProperties(vk_physical_device, &mem_props);

	// Query info about raytracing capabilities, shader header size, etc.
	VkPhysicalDeviceRayTracingPropertiesNV raytracing_props = {};
	{
		raytracing_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PROPERTIES_NV;
		VkPhysicalDeviceProperties2 props = {};
		props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
		props.pNext = &raytracing_props;
		props.properties = {};
		vkGetPhysicalDeviceProperties2(vk_physical_device, &props);

		std::cout << "Raytracing props:\n"
			<< "max recursion depth: " << raytracing_props.maxRecursionDepth
			<< "\nSBT handle size: " << raytracing_props.shaderGroupHandleSize
			<< "\nShader group base align: " << raytracing_props.shaderGroupBaseAlignment << "\n";
	}

	// Setup swapchain, assume a real GPU so don't bother querying the capabilities, just get what we want
	VkExtent2D swapchain_extent = {};
	swapchain_extent.width = win_width;
	swapchain_extent.height = win_height;
	const VkFormat swapchain_img_format = VK_FORMAT_B8G8R8A8_UNORM;

	VkSwapchainKHR vk_swapchain = VK_NULL_HANDLE;
	std::vector<VkImage> swapchain_images;
	std::vector<VkImageView> swapchain_image_views;
	{
		VkSwapchainCreateInfoKHR create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		create_info.surface = vk_surface;
		create_info.minImageCount = 2;
		create_info.imageFormat = swapchain_img_format;
		create_info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
		create_info.imageExtent = swapchain_extent;
		create_info.imageArrayLayers = 1;
		create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		// We only have 1 queue
		create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		create_info.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
		create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		create_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
		create_info.clipped = true;
		create_info.oldSwapchain = VK_NULL_HANDLE;
		CHECK_VULKAN(vkCreateSwapchainKHR(vk_device, &create_info, nullptr, &vk_swapchain));

		// Get the swap chain images
		uint32_t num_swapchain_imgs = 0;
		vkGetSwapchainImagesKHR(vk_device, vk_swapchain, &num_swapchain_imgs, nullptr);
		swapchain_images.resize(num_swapchain_imgs);
		vkGetSwapchainImagesKHR(vk_device, vk_swapchain, &num_swapchain_imgs, swapchain_images.data());

		for (const auto& img : swapchain_images) {
			VkImageViewCreateInfo view_create_info = {};
			view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			view_create_info.image = img;
			view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
			view_create_info.format = swapchain_img_format;

			view_create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			view_create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			view_create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			view_create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

			view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			view_create_info.subresourceRange.baseMipLevel = 0;
			view_create_info.subresourceRange.levelCount = 1;
			view_create_info.subresourceRange.baseArrayLayer = 0;
			view_create_info.subresourceRange.layerCount = 1;

			VkImageView img_view;
			CHECK_VULKAN(vkCreateImageView(vk_device, &view_create_info, nullptr, &img_view));
			swapchain_image_views.push_back(img_view);
		}
	}

	// Setup the command pool
	VkCommandPool vk_command_pool;
	{
		VkCommandPoolCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		create_info.queueFamilyIndex = graphics_queue_index;
		CHECK_VULKAN(vkCreateCommandPool(vk_device, &create_info, nullptr, &vk_command_pool));
	}

	std::array<VkCommandBuffer, 2> command_buffers;
	{
		VkCommandBufferAllocateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		info.commandPool = vk_command_pool;
		info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		info.commandBufferCount = 2;
		CHECK_VULKAN(vkAllocateCommandBuffers(vk_device, &info, command_buffers.data()));
	}

	// Use the first commanf buffer for running some general stuff
	VkCommandBuffer command_buffer = command_buffers[0];

	// Upload vertex data to the GPU by staging in host memory, then copying to GPU memory
	VkBuffer vertex_buffer = VK_NULL_HANDLE;
	VkDeviceMemory vertex_mem = VK_NULL_HANDLE;
	{
		VkBufferCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		info.size = sizeof(float) * triangle_verts.size();
		info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VkBuffer upload_buffer = VK_NULL_HANDLE;
		info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		CHECK_VULKAN(vkCreateBuffer(vk_device, &info, nullptr, &upload_buffer));

		info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		CHECK_VULKAN(vkCreateBuffer(vk_device, &info, nullptr, &vertex_buffer));

		VkMemoryRequirements mem_reqs = {};
		vkGetBufferMemoryRequirements(vk_device, vertex_buffer, &mem_reqs);

		// Allocate the upload staging buffer
		VkMemoryAllocateInfo alloc_info = {};
		alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		alloc_info.allocationSize = mem_reqs.size;
		alloc_info.memoryTypeIndex = get_memory_type_index(mem_reqs.memoryTypeBits,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, mem_props);
		VkDeviceMemory upload_mem = VK_NULL_HANDLE;
		CHECK_VULKAN(vkAllocateMemory(vk_device, &alloc_info, nullptr, &upload_mem));
		vkBindBufferMemory(vk_device, upload_buffer, upload_mem, 0);

		alloc_info.memoryTypeIndex = get_memory_type_index(mem_reqs.memoryTypeBits,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mem_props);
		CHECK_VULKAN(vkAllocateMemory(vk_device, &alloc_info, nullptr, &vertex_mem));
		vkBindBufferMemory(vk_device, vertex_buffer, vertex_mem, 0);

		// Now map the upload heap buffer and copy our data in
		float* upload_mapping = nullptr;
		vkMapMemory(vk_device, upload_mem, 0, info.size, 0, reinterpret_cast<void**>(&upload_mapping));
		std::memcpy(upload_mapping, triangle_verts.data(), info.size);
		vkUnmapMemory(vk_device, upload_mem);

		VkCommandBufferBeginInfo begin_info = {};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

		VkBufferCopy copy_cmd = {};
		copy_cmd.size = info.size;
		vkCmdCopyBuffer(command_buffer, upload_buffer, vertex_buffer, 1, &copy_cmd);

		CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

		// Submit the copy to run
		VkSubmitInfo submit_info = {};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffer;
		CHECK_VULKAN(vkQueueSubmit(vk_queue, 1, &submit_info, VK_NULL_HANDLE));
		vkQueueWaitIdle(vk_queue);

		// We didn't make the buffers individually reset-able, but we're just using it as temp
		// one to do this upload so clear the pool to reset
		vkResetCommandPool(vk_device, vk_command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);

		vkDestroyBuffer(vk_device, upload_buffer, nullptr);
		vkFreeMemory(vk_device, upload_mem, nullptr);
	}

	VkBuffer index_buffer = VK_NULL_HANDLE;
	VkDeviceMemory index_mem = VK_NULL_HANDLE;
	{
		VkBufferCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		info.size = sizeof(uint32_t) * triangle_indices.size();
		info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VkBuffer upload_buffer = VK_NULL_HANDLE;
		info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		CHECK_VULKAN(vkCreateBuffer(vk_device, &info, nullptr, &upload_buffer));

		info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		CHECK_VULKAN(vkCreateBuffer(vk_device, &info, nullptr, &index_buffer));

		VkMemoryRequirements mem_reqs = {};
		vkGetBufferMemoryRequirements(vk_device, index_buffer, &mem_reqs);

		// Allocate the upload staging buffer
		VkMemoryAllocateInfo alloc_info = {};
		alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		alloc_info.allocationSize = mem_reqs.size;
		alloc_info.memoryTypeIndex = get_memory_type_index(mem_reqs.memoryTypeBits,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, mem_props);
		VkDeviceMemory upload_mem = VK_NULL_HANDLE;
		CHECK_VULKAN(vkAllocateMemory(vk_device, &alloc_info, nullptr, &upload_mem));
		vkBindBufferMemory(vk_device, upload_buffer, upload_mem, 0);

		alloc_info.memoryTypeIndex = get_memory_type_index(mem_reqs.memoryTypeBits,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mem_props);
		CHECK_VULKAN(vkAllocateMemory(vk_device, &alloc_info, nullptr, &index_mem));
		vkBindBufferMemory(vk_device, index_buffer, index_mem, 0);

		// Now map the upload heap buffer and copy our data in
		uint32_t* upload_mapping = nullptr;
		vkMapMemory(vk_device, upload_mem, 0, info.size, 0, reinterpret_cast<void**>(&upload_mapping));
		std::memcpy(upload_mapping, triangle_indices.data(), info.size);
		vkUnmapMemory(vk_device, upload_mem);

		VkCommandBufferBeginInfo begin_info = {};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

		VkBufferCopy copy_cmd = {};
		copy_cmd.size = info.size;
		vkCmdCopyBuffer(command_buffer, upload_buffer, index_buffer, 1, &copy_cmd);

		CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

		// Submit the copy to run
		VkSubmitInfo submit_info = {};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffer;
		CHECK_VULKAN(vkQueueSubmit(vk_queue, 1, &submit_info, VK_NULL_HANDLE));
		vkQueueWaitIdle(vk_queue);

		// We didn't make the buffers individually reset-able, but we're just using it as temp
		// one to do this upload so clear the pool to reset
		vkResetCommandPool(vk_device, vk_command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);

		vkDestroyBuffer(vk_device, upload_buffer, nullptr);
		vkFreeMemory(vk_device, upload_mem, nullptr);
	}

	// Build the bottom level acceleration structure
	VkAccelerationStructureNV blas = VK_NULL_HANDLE;
	VkDeviceMemory blas_mem = VK_NULL_HANDLE;
	uint64_t blas_handle = 0;
	{
		VkGeometryNV geom_desc = {};
		geom_desc.sType = VK_STRUCTURE_TYPE_GEOMETRY_NV;
		geom_desc.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_NV;
		geom_desc.geometry.triangles.sType = VK_STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV;
		geom_desc.geometry.triangles.vertexData = vertex_buffer;
		geom_desc.geometry.triangles.vertexOffset = 0;
		geom_desc.geometry.triangles.vertexCount = triangle_verts.size() / 3;
		geom_desc.geometry.triangles.vertexStride = 3 * sizeof(float);
		geom_desc.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
		geom_desc.geometry.triangles.indexData = index_buffer;
		geom_desc.geometry.triangles.indexOffset = 0;
		geom_desc.geometry.triangles.indexCount = triangle_indices.size();
		geom_desc.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
		geom_desc.geometry.triangles.transformData = VK_NULL_HANDLE;
		geom_desc.geometry.triangles.transformOffset = 0;
		geom_desc.flags = VK_GEOMETRY_OPAQUE_BIT_NV;
		// Must be set even if not used
		geom_desc.geometry.aabbs = {};
		geom_desc.geometry.aabbs.sType = { VK_STRUCTURE_TYPE_GEOMETRY_AABB_NV };

		VkAccelerationStructureInfoNV accel_info = {};
		accel_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
		accel_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV;
		accel_info.instanceCount = 0;
		accel_info.geometryCount = 1;
		accel_info.pGeometries = &geom_desc;

		VkAccelerationStructureCreateInfoNV create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
		create_info.info = accel_info;
		CHECK_VULKAN(vkCreateAccelerationStructure(vk_device, &create_info, nullptr, &blas));

		// Determine how much memory the acceleration structure will need
		VkAccelerationStructureMemoryRequirementsInfoNV mem_info = {};
		mem_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
		mem_info.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV;
		mem_info.accelerationStructure = blas;

		VkMemoryRequirements2 mem_reqs = {};
		vkGetAccelerationStructureMemoryRequirements(vk_device, &mem_info, &mem_reqs);
		// TODO WILL: For a single triangle it requests 64k output and 64k scratch? It seems like a lot.
		std::cout << "BLAS will need " << mem_reqs.memoryRequirements.size << "b output space\n";

		VkMemoryAllocateInfo alloc_info = {};
		alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		alloc_info.allocationSize = mem_reqs.memoryRequirements.size;
		alloc_info.memoryTypeIndex = get_memory_type_index(mem_reqs.memoryRequirements.memoryTypeBits,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mem_props);
		CHECK_VULKAN(vkAllocateMemory(vk_device, &alloc_info, nullptr, &blas_mem));

		// Determine how much additional memory we need for the build
		mem_info.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV;
		vkGetAccelerationStructureMemoryRequirements(vk_device, &mem_info, &mem_reqs);
		std::cout << "BLAS will need " << mem_reqs.memoryRequirements.size << "b scratch space\n";

		alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		alloc_info.allocationSize = mem_reqs.memoryRequirements.size;
		alloc_info.memoryTypeIndex = get_memory_type_index(mem_reqs.memoryRequirements.memoryTypeBits,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mem_props);
		VkDeviceMemory scratch_mem = VK_NULL_HANDLE;
		CHECK_VULKAN(vkAllocateMemory(vk_device, &alloc_info, nullptr, &scratch_mem));

		// Associate the scratch mem with a buffer
		VkBufferCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		info.size = mem_reqs.memoryRequirements.size;
		info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VkBuffer scratch_buffer = VK_NULL_HANDLE;
		info.usage = VK_BUFFER_USAGE_RAY_TRACING_BIT_NV;
		CHECK_VULKAN(vkCreateBuffer(vk_device, &info, nullptr, &scratch_buffer));
		vkBindBufferMemory(vk_device, scratch_buffer, scratch_mem, 0);

		VkBindAccelerationStructureMemoryInfoNV bind_mem_info = {};
		bind_mem_info.sType = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
		bind_mem_info.accelerationStructure = blas;
		bind_mem_info.memory = blas_mem;
		CHECK_VULKAN(vkBindAccelerationStructureMemory(vk_device, 1, &bind_mem_info));

		VkCommandBufferBeginInfo begin_info = {};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

		vkCmdBuildAccelerationStructure(command_buffer, &accel_info, VK_NULL_HANDLE, 0, false, blas, VK_NULL_HANDLE, scratch_buffer, 0);

		CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

		// Submit the copy to run
		VkSubmitInfo submit_info = {};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffer;
		CHECK_VULKAN(vkQueueSubmit(vk_queue, 1, &submit_info, VK_NULL_HANDLE));
		vkQueueWaitIdle(vk_queue);

		CHECK_VULKAN(vkGetAccelerationStructureHandle(vk_device, blas, sizeof(uint64_t), &blas_handle));

		// We didn't make the buffers individually reset-able, but we're just using it as temp
		// one to do this upload so clear the pool to reset
		vkResetCommandPool(vk_device, vk_command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);

		// TODO LATER compaction via vkCmdCopyAccelerationStructureNV

		vkDestroyBuffer(vk_device, scratch_buffer, nullptr);
		vkFreeMemory(vk_device, scratch_mem, nullptr);
	}

	// Write the instance data
	VkBuffer instance_buffer = VK_NULL_HANDLE;
	VkDeviceMemory instance_mem = VK_NULL_HANDLE;
	{
		VkBufferCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		info.size = sizeof(VkGeometryInstanceNV);
		info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VkBuffer upload_buffer = VK_NULL_HANDLE;
		info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		CHECK_VULKAN(vkCreateBuffer(vk_device, &info, nullptr, &upload_buffer));

		info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		CHECK_VULKAN(vkCreateBuffer(vk_device, &info, nullptr, &instance_buffer));

		VkMemoryRequirements mem_reqs = {};
		vkGetBufferMemoryRequirements(vk_device, instance_buffer, &mem_reqs);

		// Allocate the upload staging buffer
		VkMemoryAllocateInfo alloc_info = {};
		alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		alloc_info.allocationSize = mem_reqs.size;
		alloc_info.memoryTypeIndex = get_memory_type_index(mem_reqs.memoryTypeBits,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, mem_props);
		VkDeviceMemory upload_mem = VK_NULL_HANDLE;
		CHECK_VULKAN(vkAllocateMemory(vk_device, &alloc_info, nullptr, &upload_mem));
		vkBindBufferMemory(vk_device, upload_buffer, upload_mem, 0);

		alloc_info.memoryTypeIndex = get_memory_type_index(mem_reqs.memoryTypeBits,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mem_props);
		CHECK_VULKAN(vkAllocateMemory(vk_device, &alloc_info, nullptr, &instance_mem));
		vkBindBufferMemory(vk_device, instance_buffer, instance_mem, 0);

		// Now map the upload heap buffer and copy our data in
		VkGeometryInstanceNV* upload_mapping = nullptr;
		vkMapMemory(vk_device, upload_mem, 0, info.size, 0, reinterpret_cast<void**>(&upload_mapping));
		std::memset(upload_mapping, 0, sizeof(VkGeometryInstanceNV));

		// Transform is 4x3 row-major
		// a b c
		// e f g
		// h i j
		// k l m
		// or do they mean 3x4 row-major?
		// a b c d
		// e f g h
		// i j k l
		// Because the identity seems to be set properly this way....
		upload_mapping->transform[0] = 1.f;
		upload_mapping->transform[5] = 1.f;
		upload_mapping->transform[10] = 1.f;
		
		upload_mapping->instance_custom_index = 0;
		upload_mapping->mask = 0xff;
		upload_mapping->instance_offset = 0;
		upload_mapping->flags = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_NV | VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV;
		upload_mapping->acceleration_structure_handle = blas_handle;

		vkUnmapMemory(vk_device, upload_mem);

		VkCommandBufferBeginInfo begin_info = {};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

		VkBufferCopy copy_cmd = {};
		copy_cmd.size = info.size;
		vkCmdCopyBuffer(command_buffer, upload_buffer, instance_buffer, 1, &copy_cmd);

		CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

		// Submit the copy to run
		VkSubmitInfo submit_info = {};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffer;
		CHECK_VULKAN(vkQueueSubmit(vk_queue, 1, &submit_info, VK_NULL_HANDLE));
		vkQueueWaitIdle(vk_queue);

		// We didn't make the buffers individually reset-able, but we're just using it as temp
		// one to do this upload so clear the pool to reset
		vkResetCommandPool(vk_device, vk_command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);

		vkDestroyBuffer(vk_device, upload_buffer, nullptr);
		vkFreeMemory(vk_device, upload_mem, nullptr);
	}

	// Build the top level acceleration structure
	VkAccelerationStructureNV tlas = VK_NULL_HANDLE;
	VkDeviceMemory tlas_mem = VK_NULL_HANDLE;
	uint64_t tlas_handle = 0;
	{
		VkAccelerationStructureInfoNV accel_info = {};
		accel_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
		accel_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV;
		accel_info.instanceCount = 1;
		accel_info.geometryCount = 0;

		VkAccelerationStructureCreateInfoNV create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
		create_info.info = accel_info;
		CHECK_VULKAN(vkCreateAccelerationStructure(vk_device, &create_info, nullptr, &tlas));

		// Determine how much memory the acceleration structure will need
		VkAccelerationStructureMemoryRequirementsInfoNV mem_info = {};
		mem_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
		mem_info.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV;
		mem_info.accelerationStructure = tlas;

		VkMemoryRequirements2 mem_reqs = {};
		vkGetAccelerationStructureMemoryRequirements(vk_device, &mem_info, &mem_reqs);
		// TODO WILL: For a single triangle it requests 64k output and 64k scratch? It seems like a lot.
		std::cout << "TLAS will need " << mem_reqs.memoryRequirements.size << "b output space\n";

		VkMemoryAllocateInfo alloc_info = {};
		alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		alloc_info.allocationSize = mem_reqs.memoryRequirements.size;
		alloc_info.memoryTypeIndex = get_memory_type_index(mem_reqs.memoryRequirements.memoryTypeBits,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mem_props);
		CHECK_VULKAN(vkAllocateMemory(vk_device, &alloc_info, nullptr, &tlas_mem));

		// Determine how much additional memory we need for the build
		mem_info.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV;
		vkGetAccelerationStructureMemoryRequirements(vk_device, &mem_info, &mem_reqs);
		std::cout << "TLAS will need " << mem_reqs.memoryRequirements.size << "b scratch space\n";

		alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		alloc_info.allocationSize = mem_reqs.memoryRequirements.size;
		alloc_info.memoryTypeIndex = get_memory_type_index(mem_reqs.memoryRequirements.memoryTypeBits,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mem_props);
		VkDeviceMemory scratch_mem = VK_NULL_HANDLE;
		CHECK_VULKAN(vkAllocateMemory(vk_device, &alloc_info, nullptr, &scratch_mem));

		// Associate the scratch mem with a buffer
		VkBufferCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		info.size = mem_reqs.memoryRequirements.size;
		info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VkBuffer scratch_buffer = VK_NULL_HANDLE;
		info.usage = VK_BUFFER_USAGE_RAY_TRACING_BIT_NV;
		CHECK_VULKAN(vkCreateBuffer(vk_device, &info, nullptr, &scratch_buffer));
		vkBindBufferMemory(vk_device, scratch_buffer, scratch_mem, 0);

		VkBindAccelerationStructureMemoryInfoNV bind_mem_info = {};
		bind_mem_info.sType = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
		bind_mem_info.accelerationStructure = tlas;
		bind_mem_info.memory = tlas_mem;
		CHECK_VULKAN(vkBindAccelerationStructureMemory(vk_device, 1, &bind_mem_info));

		VkCommandBufferBeginInfo begin_info = {};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

		vkCmdBuildAccelerationStructure(command_buffer, &accel_info, instance_buffer, 0, false, tlas, VK_NULL_HANDLE, scratch_buffer, 0);

		CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

		// Submit the copy to run
		VkSubmitInfo submit_info = {};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffer;
		CHECK_VULKAN(vkQueueSubmit(vk_queue, 1, &submit_info, VK_NULL_HANDLE));
		vkQueueWaitIdle(vk_queue);

		CHECK_VULKAN(vkGetAccelerationStructureHandle(vk_device, tlas, sizeof(uint64_t), &tlas_handle));

		// We didn't make the buffers individually reset-able, but we're just using it as temp
		// one to do this upload so clear the pool to reset
		vkResetCommandPool(vk_device, vk_command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);

		// TODO LATER compaction via vkCmdCopyAccelerationStructureNV

		vkDestroyBuffer(vk_device, scratch_buffer, nullptr);
		vkFreeMemory(vk_device, scratch_mem, nullptr);
	}

	// Setup the output image
	VkImage rt_output = VK_NULL_HANDLE;
	VkImageView rt_output_view = VK_NULL_HANDLE;
	{
		VkImageCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		create_info.imageType = VK_IMAGE_TYPE_2D;
		create_info.format = swapchain_img_format;
		create_info.extent.width = win_width;
		create_info.extent.height = win_height;
		create_info.extent.depth = 1;
		create_info.mipLevels = 1;
		create_info.arrayLayers = 1;
		create_info.samples = VK_SAMPLE_COUNT_1_BIT;
		create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
		create_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
		create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		CHECK_VULKAN(vkCreateImage(vk_device, &create_info, nullptr, &rt_output));

		VkMemoryRequirements mem_reqs = {};
		vkGetImageMemoryRequirements(vk_device, rt_output, &mem_reqs);

		// Allocate the upload staging buffer
		VkMemoryAllocateInfo alloc_info = {};
		alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		alloc_info.allocationSize = mem_reqs.size;
		alloc_info.memoryTypeIndex = get_memory_type_index(mem_reqs.memoryTypeBits,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mem_props);
		VkDeviceMemory img_mem = VK_NULL_HANDLE;
		CHECK_VULKAN(vkAllocateMemory(vk_device, &alloc_info, nullptr, &img_mem));
		CHECK_VULKAN(vkBindImageMemory(vk_device, rt_output, img_mem, 0));

		VkImageViewCreateInfo view_create_info = {};
		view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		view_create_info.image = rt_output;
		view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
		view_create_info.format = swapchain_img_format;

		view_create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		view_create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		view_create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		view_create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

		view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		view_create_info.subresourceRange.baseMipLevel = 0;
		view_create_info.subresourceRange.levelCount = 1;
		view_create_info.subresourceRange.baseArrayLayer = 0;
		view_create_info.subresourceRange.layerCount = 1;

		CHECK_VULKAN(vkCreateImageView(vk_device, &view_create_info, nullptr, &rt_output_view));

		// Transition the image over to general layout so we can write to it in the raygen program
		VkCommandBufferBeginInfo begin_info = {};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

		VkImageMemoryBarrier img_mem_barrier = {};
		img_mem_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		img_mem_barrier.image = rt_output;
		img_mem_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		img_mem_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		img_mem_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		img_mem_barrier.subresourceRange.baseMipLevel = 0;
		img_mem_barrier.subresourceRange.levelCount = 1;
		img_mem_barrier.subresourceRange.baseArrayLayer = 0;
		img_mem_barrier.subresourceRange.layerCount = 1;

		vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			0, 0, nullptr, 0, nullptr, 1, &img_mem_barrier);

		CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

		// Submit the copy to run
		VkSubmitInfo submit_info = {};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffer;
		CHECK_VULKAN(vkQueueSubmit(vk_queue, 1, &submit_info, VK_NULL_HANDLE));
		vkQueueWaitIdle(vk_queue);

		// We didn't make the buffers individually reset-able, but we're just using it as temp
		// one to do this upload so clear the pool to reset
		vkResetCommandPool(vk_device, vk_command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
	}

	// Desriptors for current test pipeline
	// 0: top level AS
	// 1: output image view
	// SBT Layout:
	// 0: raygen
	// 1: miss
	// 2: hitgroup
	// Build the ray tracing pipeline
	VkPipeline rt_pipeline = VK_NULL_HANDLE;
	VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
	VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
	{
		VkDescriptorSetLayoutBinding tlas_binding = {};
		tlas_binding.binding = 0;
		tlas_binding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV;
		tlas_binding.descriptorCount = 1;
		tlas_binding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

		VkDescriptorSetLayoutBinding fb_binding = {};
		fb_binding.binding = 1;
		fb_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		fb_binding.descriptorCount = 1;
		fb_binding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

		const std::array<VkDescriptorSetLayoutBinding, 2> desc_set = {
			tlas_binding, fb_binding
		};

		VkDescriptorSetLayoutCreateInfo desc_create_info = {};
		desc_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		desc_create_info.bindingCount = desc_set.size();
		desc_create_info.pBindings = desc_set.data();

		CHECK_VULKAN(vkCreateDescriptorSetLayout(vk_device, &desc_create_info, nullptr, &desc_layout));

		VkPipelineLayoutCreateInfo pipeline_create_info = {};
		pipeline_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipeline_create_info.setLayoutCount = 1;
		pipeline_create_info.pSetLayouts = &desc_layout;

		CHECK_VULKAN(vkCreatePipelineLayout(vk_device, &pipeline_create_info, nullptr, &pipeline_layout));

		// Load the shader modules for our pipeline
		VkShaderModuleCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		create_info.codeSize = sizeof(raygen_spv);
		create_info.pCode = raygen_spv;
		VkShaderModule raygen_shader_module = VK_NULL_HANDLE;
		CHECK_VULKAN(vkCreateShaderModule(vk_device, &create_info, nullptr, &raygen_shader_module));

		create_info.codeSize = sizeof(miss_spv);
		create_info.pCode = miss_spv;
		VkShaderModule miss_shader_module = VK_NULL_HANDLE;
		CHECK_VULKAN(vkCreateShaderModule(vk_device, &create_info, nullptr, &miss_shader_module));

		create_info.codeSize = sizeof(hit_spv);
		create_info.pCode = hit_spv;
		VkShaderModule closest_hit_shader_module = VK_NULL_HANDLE;
		CHECK_VULKAN(vkCreateShaderModule(vk_device, &create_info, nullptr, &closest_hit_shader_module));

		std::array<VkPipelineShaderStageCreateInfo, 3> shader_create_info = { VkPipelineShaderStageCreateInfo{} };
		for (auto& ci : shader_create_info) {
			ci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			ci.pName = "main";
		}
		shader_create_info[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_NV;
		shader_create_info[0].module = raygen_shader_module;

		shader_create_info[1].stage = VK_SHADER_STAGE_MISS_BIT_NV;
		shader_create_info[1].module = miss_shader_module;

		shader_create_info[2].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;
		shader_create_info[2].module = closest_hit_shader_module;

		std::array<VkRayTracingShaderGroupCreateInfoNV, 3> rt_shader_groups = { VkRayTracingShaderGroupCreateInfoNV{} };
		for (auto& g : rt_shader_groups) {
			g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV;
			g.generalShader = VK_SHADER_UNUSED_NV;
			g.closestHitShader = VK_SHADER_UNUSED_NV;
			g.anyHitShader = VK_SHADER_UNUSED_NV;
			g.intersectionShader = VK_SHADER_UNUSED_NV;
		}

		// Raygen group [0]
		rt_shader_groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV;
		rt_shader_groups[0].generalShader = 0;

		// Miss group [1]
		rt_shader_groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV;
		rt_shader_groups[1].generalShader = 1;

		// Hit group [2]
		rt_shader_groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_NV;
		rt_shader_groups[2].closestHitShader = 2;

		VkRayTracingPipelineCreateInfoNV rt_pipeline_create_info = {};
		rt_pipeline_create_info.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_NV;
		rt_pipeline_create_info.stageCount = shader_create_info.size();
		rt_pipeline_create_info.pStages = shader_create_info.data();
		rt_pipeline_create_info.groupCount = rt_shader_groups.size();
		rt_pipeline_create_info.pGroups = rt_shader_groups.data();
		rt_pipeline_create_info.maxRecursionDepth = 1;
		rt_pipeline_create_info.layout = pipeline_layout;
		CHECK_VULKAN(vkCreateRayTracingPipelines(vk_device, VK_NULL_HANDLE, 1, &rt_pipeline_create_info, nullptr, &rt_pipeline));
	}

	// Build the SBT
	// TODO: Layer for perf this should also be uploaded to the device
	VkBuffer sbt_buffer = VK_NULL_HANDLE;
	VkDeviceMemory sbt_mem = VK_NULL_HANDLE;
	{
		VkBufferCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		info.size = 3 * raytracing_props.shaderGroupHandleSize;
		info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		info.usage = VK_BUFFER_USAGE_RAY_TRACING_BIT_NV;

		std::cout << "SBT size: " << info.size << "\n";

		CHECK_VULKAN(vkCreateBuffer(vk_device, &info, nullptr, &sbt_buffer));

		VkMemoryRequirements mem_reqs = {};
		vkGetBufferMemoryRequirements(vk_device, sbt_buffer, &mem_reqs);

		VkMemoryAllocateInfo alloc_info = {};
		alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		alloc_info.allocationSize = mem_reqs.size;
		alloc_info.memoryTypeIndex = get_memory_type_index(mem_reqs.memoryTypeBits,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, mem_props);
		CHECK_VULKAN(vkAllocateMemory(vk_device, &alloc_info, nullptr, &sbt_mem));
		vkBindBufferMemory(vk_device, sbt_buffer, sbt_mem, 0);

		uint8_t* sbt_mapping = nullptr;
		vkMapMemory(vk_device, sbt_mem, 0, info.size, 0, reinterpret_cast<void**>(&sbt_mapping));

		// Get the shader identifiers
		// Note: for now this is the same size as the SBT, but this will not always be the case
		std::vector<uint8_t> shader_identifiers(3 * raytracing_props.shaderGroupHandleSize, 0);
		CHECK_VULKAN(vkGetRayTracingShaderGroupHandles(vk_device, rt_pipeline, 0, 3,
			shader_identifiers.size(), shader_identifiers.data()));

		// Copy raygen handle
		std::memcpy(sbt_mapping, shader_identifiers.data(), raytracing_props.shaderGroupHandleSize);
		// Copy miss handle
		std::memcpy(sbt_mapping + raytracing_props.shaderGroupHandleSize,
			shader_identifiers.data() + raytracing_props.shaderGroupHandleSize,
			raytracing_props.shaderGroupHandleSize);
		// Copy hitgroup handle
		std::memcpy(sbt_mapping + 2 * raytracing_props.shaderGroupHandleSize,
			shader_identifiers.data() + 2 * raytracing_props.shaderGroupHandleSize,
			raytracing_props.shaderGroupHandleSize);

		vkUnmapMemory(vk_device, sbt_mem);
	}

	// Build the descriptor set
	VkDescriptorPool desc_pool = VK_NULL_HANDLE;
	VkDescriptorSet desc_set = VK_NULL_HANDLE;
	{
		const std::array<VkDescriptorPoolSize, 2> pool_sizes = {
			VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV, 1 },
			VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 }
		};
		VkDescriptorPoolCreateInfo pool_create_info = {};
		pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		pool_create_info.maxSets = 1;
		pool_create_info.poolSizeCount = pool_sizes.size();
		pool_create_info.pPoolSizes = pool_sizes.data();
		CHECK_VULKAN(vkCreateDescriptorPool(vk_device, &pool_create_info, nullptr, &desc_pool));

		VkDescriptorSetAllocateInfo alloc_info = {};
		alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		alloc_info.descriptorPool = desc_pool;
		alloc_info.descriptorSetCount = 1;
		alloc_info.pSetLayouts = &desc_layout;
		CHECK_VULKAN(vkAllocateDescriptorSets(vk_device, &alloc_info, &desc_set));

		VkWriteDescriptorSetAccelerationStructureNV write_tlas_info = {};
		write_tlas_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_NV;
		write_tlas_info.accelerationStructureCount = 1;
		write_tlas_info.pAccelerationStructures = &tlas;

		VkWriteDescriptorSet write_tlas = {};
		write_tlas.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write_tlas.pNext = &write_tlas_info;
		write_tlas.dstSet = desc_set;
		write_tlas.dstBinding = 0;
		write_tlas.descriptorCount = 1;
		write_tlas.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV;

		VkDescriptorImageInfo img_desc = {};
		img_desc.imageView = rt_output_view;
		img_desc.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		VkWriteDescriptorSet write_img = {};
		write_img.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write_img.dstSet = desc_set;
		write_img.dstBinding = 1;
		write_img.descriptorCount = 1;
		write_img.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		write_img.pImageInfo = &img_desc;

		const std::array<VkWriteDescriptorSet, 2> write_descs = { write_tlas, write_img };
		vkUpdateDescriptorSets(vk_device, write_descs.size(), write_descs.data(), 0, nullptr);
	}

	// Finally, record the rendering commands
	{
		for (size_t i = 0; i < command_buffers.size(); ++i) {
			VkCommandBufferBeginInfo begin_info = {};
			begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			CHECK_VULKAN(vkBeginCommandBuffer(command_buffers[i], &begin_info));

			vkCmdBindPipeline(command_buffers[i], VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, rt_pipeline);
			vkCmdBindDescriptorSets(command_buffers[i], VK_PIPELINE_BIND_POINT_RAY_TRACING_NV,
				pipeline_layout, 0, 1, &desc_set, 0, nullptr);

			vkCmdTraceRays(command_buffers[i],
				sbt_buffer, 0,
				sbt_buffer, raytracing_props.shaderGroupHandleSize, raytracing_props.shaderGroupHandleSize,
				sbt_buffer, raytracing_props.shaderGroupHandleSize * 2, raytracing_props.shaderGroupHandleSize,
				VK_NULL_HANDLE, 0, 0, win_width, win_height, 1);

			// Transition swapchain image to copy dst and render output to copy source
			// TODO: Are these barriers necessary? From the docs on vkCmdCopyImage it sounds like
			// the src and dst layouts are already ok for the copy
			{
				std::array<VkImageMemoryBarrier, 2> barriers;
				for (auto &b : barriers) {
					b = VkImageMemoryBarrier{};
					b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
					b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
					b.subresourceRange.baseMipLevel = 0;
					b.subresourceRange.levelCount = 1;
					b.subresourceRange.baseArrayLayer = 0;
					b.subresourceRange.layerCount = 1;
				}
				barriers[0].image = swapchain_images[i];
				barriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
				barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
				barriers[1].image = rt_output;
				barriers[1].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
				barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	
				vkCmdPipelineBarrier(command_buffers[i], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
					0, 0, nullptr, 0, nullptr, barriers.size(), barriers.data());
			}

			VkImageSubresourceLayers copy_subresource = {};
			copy_subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			copy_subresource.mipLevel = 0;
			copy_subresource.baseArrayLayer = 0;
			copy_subresource.layerCount = 1;

			VkImageCopy img_copy = {};
			img_copy.srcSubresource = copy_subresource;
			img_copy.srcOffset.x = 0;
			img_copy.srcOffset.y = 0;
			img_copy.srcOffset.z = 0;
			
			img_copy.dstSubresource = copy_subresource;
			img_copy.dstOffset.x = 0;
			img_copy.dstOffset.y = 0;
			img_copy.dstOffset.z = 0;

			img_copy.extent.width = win_width;
			img_copy.extent.height = win_height;
			img_copy.extent.depth = 1;

			vkCmdCopyImage(command_buffers[i], rt_output, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				swapchain_images[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1, &img_copy);

			// Transition the images back to their original layouts
			{
				std::array<VkImageMemoryBarrier, 2> barriers;
				for (auto& b : barriers) {
					b = VkImageMemoryBarrier{};
					b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
					b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
					b.subresourceRange.baseMipLevel = 0;
					b.subresourceRange.levelCount = 1;
					b.subresourceRange.baseArrayLayer = 0;
					b.subresourceRange.layerCount = 1;
				}
				barriers[0].image = swapchain_images[i];
				barriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
				barriers[0].newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
				barriers[1].image = rt_output;
				barriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
				barriers[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;

				vkCmdPipelineBarrier(command_buffers[i], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
					0, 0, nullptr, 0, nullptr, barriers.size(), barriers.data());
			}

			CHECK_VULKAN(vkEndCommandBuffer(command_buffers[i]));
		}
	}

	VkSemaphore img_avail_semaphore = VK_NULL_HANDLE;
	VkSemaphore render_finished_semaphore = VK_NULL_HANDLE;
	{
		VkSemaphoreCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		CHECK_VULKAN(vkCreateSemaphore(vk_device, &info, nullptr, &img_avail_semaphore));
		CHECK_VULKAN(vkCreateSemaphore(vk_device, &info, nullptr, &render_finished_semaphore));
	}

	// We use a fence to wait for the rendering work to finish
	VkFence vk_fence = VK_NULL_HANDLE;
	{
		VkFenceCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		CHECK_VULKAN(vkCreateFence(vk_device, &info, nullptr, &vk_fence));
	}

	std::cout << "Running loop\n";
	bool done = false;
	while (!done) {
		SDL_Event event;
		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_QUIT) {
				done = true;
			}
			if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE) {
				done = true;
			}
			if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE
					&& event.window.windowID == SDL_GetWindowID(window)) {
				done = true;
			}
		}

		// Get an image from the swap chain
		uint32_t img_index = 0;
		CHECK_VULKAN(vkAcquireNextImageKHR(vk_device, vk_swapchain, std::numeric_limits<uint64_t>::max(),
			img_avail_semaphore, VK_NULL_HANDLE, &img_index));

		// We need to wait for the image before we can run the commands to draw to it, and signal
		// the render finished one when we're done
		const std::array<VkSemaphore, 1> wait_semaphores = { img_avail_semaphore };
		const std::array<VkSemaphore, 1> signal_semaphores = { render_finished_semaphore };
		const std::array<VkPipelineStageFlags, 1> wait_stages = { VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT };

		CHECK_VULKAN(vkResetFences(vk_device, 1, &vk_fence));

		VkSubmitInfo submit_info = {};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.waitSemaphoreCount = wait_semaphores.size();
		submit_info.pWaitSemaphores = wait_semaphores.data();
		submit_info.pWaitDstStageMask = wait_stages.data();
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffers[img_index];
		submit_info.signalSemaphoreCount = signal_semaphores.size();
		submit_info.pSignalSemaphores = signal_semaphores.data();
		CHECK_VULKAN(vkQueueSubmit(vk_queue, 1, &submit_info, vk_fence));

	
		// Finally, present the updated image in the swap chain
		std::array<VkSwapchainKHR, 1> present_chain = { vk_swapchain };
		VkPresentInfoKHR present_info = {};
		present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		present_info.waitSemaphoreCount = signal_semaphores.size();
		present_info.pWaitSemaphores = signal_semaphores.data();
		present_info.swapchainCount = present_chain.size();
		present_info.pSwapchains = present_chain.data();
		present_info.pImageIndices = &img_index;
		CHECK_VULKAN(vkQueuePresentKHR(vk_queue, &present_info));

		// Wait for the frame to finish
		CHECK_VULKAN(vkWaitForFences(vk_device, 1, &vk_fence, true, std::numeric_limits<uint64_t>::max()));
	}

	vkDestroySemaphore(vk_device, img_avail_semaphore, nullptr);
	vkDestroySemaphore(vk_device, render_finished_semaphore, nullptr);
	vkDestroyFence(vk_device, vk_fence, nullptr);
	vkDestroyCommandPool(vk_device, vk_command_pool, nullptr);
	vkDestroySwapchainKHR(vk_device, vk_swapchain, nullptr);
	
	for (auto &v : swapchain_image_views) {
		vkDestroyImageView(vk_device, v, nullptr);
	}	
	vkDestroySurfaceKHR(vk_instance, vk_surface, nullptr);
	vkDestroyDevice(vk_device, nullptr);
	vkDestroyInstance(vk_instance, nullptr);

	SDL_DestroyWindow(window);
	SDL_Quit();

	return 0;
}

