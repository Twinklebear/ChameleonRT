#include <iostream>
#include <array>
#include <chrono>
#include <glm/ext.hpp>
#include "util.h"
#include "render_vulkan.h"
#include "spv_shaders_embedded_spv.h"

RenderVulkan::RenderVulkan() {
	command_pool = device.make_command_pool();
	{
		VkCommandBufferAllocateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		info.commandPool = command_pool;
		info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		info.commandBufferCount = 1;
		CHECK_VULKAN(vkAllocateCommandBuffers(device.logical_device(), &info, &command_buffer));
	}

	{
		VkFenceCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		CHECK_VULKAN(vkCreateFence(device.logical_device(), &info, nullptr, &fence));
	}

	view_param_buf = vk::Buffer::host(device, 4 * sizeof(glm::vec4) + sizeof(uint32_t),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

RenderVulkan::~RenderVulkan() {
	vkDestroyFence(device.logical_device(), fence, nullptr);
	vkDestroyCommandPool(device.logical_device(), command_pool, nullptr);
	vkDestroyPipeline(device.logical_device(), rt_pipeline, nullptr);
	vkDestroyDescriptorPool(device.logical_device(), desc_pool, nullptr);
}

std::string RenderVulkan::name() {
	return "Vulkan Ray Tracing";
}

void RenderVulkan::initialize(const int fb_width, const int fb_height) {
	frame_id = 0;
	img.resize(fb_width * fb_height);

	render_target = vk::Texture2D::device(device, glm::uvec2(fb_width, fb_height),
		VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT);

	img_readback_buf = vk::Buffer::host(device, img.size() * render_target->pixel_size(),
		VK_BUFFER_USAGE_TRANSFER_DST_BIT);

	// Change image to the general layout
	{
		VkCommandBufferBeginInfo begin_info = {};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

		VkImageMemoryBarrier img_mem_barrier = {};
		img_mem_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		img_mem_barrier.image = render_target->image_handle();
		img_mem_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		img_mem_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		img_mem_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		img_mem_barrier.subresourceRange.baseMipLevel = 0;
		img_mem_barrier.subresourceRange.levelCount = 1;
		img_mem_barrier.subresourceRange.baseArrayLayer = 0;
		img_mem_barrier.subresourceRange.layerCount = 1;

		vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			0, 0, nullptr, 0, nullptr, 1, &img_mem_barrier);

		CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

		// Submit the copy to run
		VkSubmitInfo submit_info = {};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffer;
		CHECK_VULKAN(vkQueueSubmit(device.graphics_queue(), 1, &submit_info, VK_NULL_HANDLE));
		vkQueueWaitIdle(device.graphics_queue());

		// We didn't make the buffers individually reset-able, but we're just using it as temp
		// one to do this upload so clear the pool to reset
		vkResetCommandPool(device.logical_device(), command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
	}
}

void RenderVulkan::set_scene(const Scene &scene_data) {
	frame_id = 0;

	auto &scene_mesh = scene_data.meshes[0];

	// Upload triangle vertices to the device
	auto upload_verts = vk::Buffer::host(device, scene_mesh.vertices.size() * sizeof(glm::vec3),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
	{
		void *map = upload_verts->map();
		std::memcpy(map, scene_mesh.vertices.data(), upload_verts->size());
		upload_verts->unmap();
	}

	auto upload_indices = vk::Buffer::host(device, scene_mesh.indices.size() * sizeof(glm::uvec3),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
	{
		void *map = upload_indices->map();
		std::memcpy(map, scene_mesh.indices.data(), upload_indices->size());
		upload_indices->unmap();
	}

	// Note: eventually the data will be passed to the hit program likely as a shader storage buffer
	auto vertex_buf = vk::Buffer::device(device, upload_verts->size(),
		VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	auto index_buf = vk::Buffer::device(device, upload_indices->size(),
		VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	// Execute the upload to the device
	{
		VkCommandBufferBeginInfo begin_info = {};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

		VkBufferCopy copy_cmd = {};
		copy_cmd.size = upload_verts->size();
		vkCmdCopyBuffer(command_buffer, upload_verts->handle(), vertex_buf->handle(), 1, &copy_cmd);

		copy_cmd.size = upload_indices->size();
		vkCmdCopyBuffer(command_buffer, upload_indices->handle(), index_buf->handle(), 1, &copy_cmd);

		CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

		VkSubmitInfo submit_info = {};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffer;
		CHECK_VULKAN(vkQueueSubmit(device.graphics_queue(), 1, &submit_info, VK_NULL_HANDLE));
		vkQueueWaitIdle(device.graphics_queue());

		vkResetCommandPool(device.logical_device(), command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
	}

	std::shared_ptr<vk::Buffer> normals_buf = nullptr;
	std::shared_ptr<vk::Buffer> uv_buf = nullptr;

	// Build the bottom level acceleration structure
	// No compaction for now (does Vulkan have some min space requirement for the BVH?)
	mesh = std::make_unique<vk::TriangleMesh>(device, vertex_buf, index_buf, normals_buf, uv_buf);
	// Build the BVH
	{
		// TODO: some convenience utils for this
		VkCommandBufferBeginInfo begin_info = {};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

		mesh->enqueue_build(command_buffer);

		CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

		VkSubmitInfo submit_info = {};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffer;
		CHECK_VULKAN(vkQueueSubmit(device.graphics_queue(), 1, &submit_info, VK_NULL_HANDLE));
		vkQueueWaitIdle(device.graphics_queue());

		vkResetCommandPool(device.logical_device(), command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
	}

	// Compact the BVH
	{
		VkCommandBufferBeginInfo begin_info = {};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

		mesh->enqueue_compaction(command_buffer);

		CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

		VkSubmitInfo submit_info = {};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffer;
		CHECK_VULKAN(vkQueueSubmit(device.graphics_queue(), 1, &submit_info, VK_NULL_HANDLE));
		vkQueueWaitIdle(device.graphics_queue());

		vkResetCommandPool(device.logical_device(), command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
	}

	mesh->finalize();

	// Setup the instance buffer
	auto upload_instances = vk::Buffer::host(device, sizeof(vk::GeometryInstance), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
	{
		vk::GeometryInstance *map = reinterpret_cast<vk::GeometryInstance*>(upload_instances->map());
		map->transform[0] = 1.f;
		map->transform[5] = 1.f;
		map->transform[10] = 1.f;

		map->instance_custom_index = 0;
		map->mask = 0xff;
		map->instance_offset = 0;
		map->flags = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_NV | VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV;
		map->acceleration_structure_handle = mesh->handle;

		upload_instances->unmap();
	}

	auto instance_buf = vk::Buffer::device(device, upload_instances->size(), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
	// Upload the instance data to the device
	{
		VkCommandBufferBeginInfo begin_info = {};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

		VkBufferCopy copy_cmd = {};
		copy_cmd.size = upload_instances->size();
		vkCmdCopyBuffer(command_buffer, upload_instances->handle(), instance_buf->handle(), 1, &copy_cmd);

		CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

		VkSubmitInfo submit_info = {};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffer;
		CHECK_VULKAN(vkQueueSubmit(device.graphics_queue(), 1, &submit_info, VK_NULL_HANDLE));
		vkQueueWaitIdle(device.graphics_queue());

		vkResetCommandPool(device.logical_device(), command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
	}

	// Build the top level BVH
	scene = std::make_unique<vk::TopLevelBVH>(device, instance_buf);
	{
		// TODO: some convenience utils for this
		VkCommandBufferBeginInfo begin_info = {};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

		scene->enqueue_build(command_buffer);

		CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

		VkSubmitInfo submit_info = {};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffer;
		CHECK_VULKAN(vkQueueSubmit(device.graphics_queue(), 1, &submit_info, VK_NULL_HANDLE));
		vkQueueWaitIdle(device.graphics_queue());

		vkResetCommandPool(device.logical_device(), command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
	}
	scene->finalize();

	build_raytracing_pipeline();
	build_shader_descriptor_table();
	build_shader_binding_table();
}

RenderStats RenderVulkan::render(const glm::vec3 &pos, const glm::vec3 &dir,
		const glm::vec3 &up, const float fovy, const bool camera_changed)
{
	using namespace std::chrono;
	RenderStats stats;

	if (camera_changed) {
		frame_id = 0;
		update_view_parameters(pos, dir, up, fovy);
	}

	// TODO: Save the commands (and do this in the DXR backend too)
	// instead of re-recording each frame
	VkCommandBufferBeginInfo begin_info = {};
	begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

	vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, rt_pipeline);
	vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV,
		pipeline_layout, 0, 1, &desc_set, 0, nullptr);

	const size_t shader_record_size = device.raytracing_properties().shaderGroupHandleSize;
	// Testing sending some params through the SBT
	const size_t hit_record_size = align_to(shader_record_size + sizeof(float),
		device.raytracing_properties().shaderGroupBaseAlignment);
	vkCmdTraceRays(command_buffer,
		shader_table->handle(), 0,
		shader_table->handle(), shader_record_size, shader_record_size,
		shader_table->handle(), 2 * shader_record_size, hit_record_size,
		VK_NULL_HANDLE, 0, 0, render_target->dims().x, render_target->dims().y, 1);

	// Barrier for rendering to finish
	// TODO: Later when I want to time the rendering separately from the image readback
	// we'll want for the render commands to finish, then do the read so this barrier
	// won't be needed
	vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
		VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
		0, 0, nullptr, 0, nullptr, 0, nullptr);

	VkImageSubresourceLayers copy_subresource = {};
	copy_subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	copy_subresource.mipLevel = 0;
	copy_subresource.baseArrayLayer = 0;
	copy_subresource.layerCount = 1;

	VkBufferImageCopy img_copy = {};
	img_copy.bufferOffset = 0;
	// Buffer is tightly packed
	img_copy.bufferRowLength = 0;
	img_copy.bufferImageHeight = 0;
	img_copy.imageSubresource = copy_subresource;
	img_copy.imageOffset.x = 0;
	img_copy.imageOffset.y = 0;
	img_copy.imageOffset.z = 0;
	img_copy.imageExtent.width = render_target->dims().x;
	img_copy.imageExtent.height = render_target->dims().y;
	img_copy.imageExtent.depth = 1;

	vkCmdCopyImageToBuffer(command_buffer, render_target->image_handle(), VK_IMAGE_LAYOUT_GENERAL,
		img_readback_buf->handle(), 1, &img_copy);

	CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

	// Now submit the commands
	CHECK_VULKAN(vkResetFences(device.logical_device(), 1, &fence));

	VkSubmitInfo submit_info = {};
	submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers = &command_buffer;
	CHECK_VULKAN(vkQueueSubmit(device.graphics_queue(), 1, &submit_info, fence));

	CHECK_VULKAN(vkWaitForFences(device.logical_device(), 1, &fence, true,
		std::numeric_limits<uint64_t>::max()));

	vkResetCommandPool(device.logical_device(), command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);

	std::memcpy(img.data(), img_readback_buf->map(), img_readback_buf->size());
	img_readback_buf->unmap();

	++frame_id;
	return stats;
}

void RenderVulkan::build_raytracing_pipeline() {
	VkDescriptorSetLayoutBinding scene_binding = {};
	scene_binding.binding = 0;
	scene_binding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV;
	scene_binding.descriptorCount = 1;
	scene_binding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

	VkDescriptorSetLayoutBinding fb_binding = {};
	fb_binding.binding = 1;
	fb_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	fb_binding.descriptorCount = 1;
	fb_binding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

	VkDescriptorSetLayoutBinding view_param_binding = {};
	view_param_binding.binding = 2;
	view_param_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	view_param_binding.descriptorCount = 1;
	view_param_binding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

	VkDescriptorSetLayoutBinding index_data_binding = {};
	index_data_binding.binding = 3;
	index_data_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	index_data_binding.descriptorCount = 1;
	index_data_binding.stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;

	VkDescriptorSetLayoutBinding vert_data_binding = {};
	vert_data_binding.binding = 4;
	vert_data_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	vert_data_binding.descriptorCount = 1;
	vert_data_binding.stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;

	const std::vector<VkDescriptorSetLayoutBinding> desc_set = {
		scene_binding, fb_binding, view_param_binding, index_data_binding, vert_data_binding
	};

	VkDescriptorSetLayoutCreateInfo desc_create_info = {};
	desc_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	desc_create_info.bindingCount = desc_set.size();
	desc_create_info.pBindings = desc_set.data();

	CHECK_VULKAN(vkCreateDescriptorSetLayout(device.logical_device(), &desc_create_info,
		nullptr, &desc_layout));

	VkPipelineLayoutCreateInfo pipeline_create_info = {};
	pipeline_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipeline_create_info.setLayoutCount = 1;
	pipeline_create_info.pSetLayouts = &desc_layout;

	CHECK_VULKAN(vkCreatePipelineLayout(device.logical_device(), &pipeline_create_info,
		nullptr, &pipeline_layout));

	// Load the shader modules for our pipeline
	vk::ShaderModule raygen_shader(device, raygen_spv, sizeof(raygen_spv));
	vk::ShaderModule miss_shader(device, miss_spv, sizeof(miss_spv));
	vk::ShaderModule closest_hit_shader(device, hit_spv, sizeof(hit_spv));

	std::array<VkPipelineShaderStageCreateInfo, 3> shader_create_info = { VkPipelineShaderStageCreateInfo{} };
	for (auto& ci : shader_create_info) {
		ci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		ci.pName = "main";
	}
	shader_create_info[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_NV;
	shader_create_info[0].module = raygen_shader.module;

	shader_create_info[1].stage = VK_SHADER_STAGE_MISS_BIT_NV;
	shader_create_info[1].module = miss_shader.module;

	shader_create_info[2].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;
	shader_create_info[2].module = closest_hit_shader.module;

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
	CHECK_VULKAN(vkCreateRayTracingPipelines(device.logical_device(), VK_NULL_HANDLE,
		1, &rt_pipeline_create_info, nullptr, &rt_pipeline));
}

void RenderVulkan::build_shader_descriptor_table() {
	const std::vector<VkDescriptorPoolSize> pool_sizes = {
			VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV, 1 },
			VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
			VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
			VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 }

	};
	VkDescriptorPoolCreateInfo pool_create_info = {};
	pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_create_info.maxSets = 1;
	pool_create_info.poolSizeCount = pool_sizes.size();
	pool_create_info.pPoolSizes = pool_sizes.data();
	CHECK_VULKAN(vkCreateDescriptorPool(device.logical_device(), &pool_create_info, nullptr, &desc_pool));

	VkDescriptorSetAllocateInfo alloc_info = {};
	alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	alloc_info.descriptorPool = desc_pool;
	alloc_info.descriptorSetCount = 1;
	alloc_info.pSetLayouts = &desc_layout;
	CHECK_VULKAN(vkAllocateDescriptorSets(device.logical_device(), &alloc_info, &desc_set));

	VkWriteDescriptorSetAccelerationStructureNV write_tlas_info = {};
	write_tlas_info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_NV;
	write_tlas_info.accelerationStructureCount = 1;
	write_tlas_info.pAccelerationStructures = &scene->bvh;

	VkWriteDescriptorSet write_tlas = {};
	write_tlas.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write_tlas.pNext = &write_tlas_info;
	write_tlas.dstSet = desc_set;
	write_tlas.dstBinding = 0;
	write_tlas.descriptorCount = 1;
	write_tlas.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV;

	VkDescriptorImageInfo img_desc = {};
	img_desc.imageView = render_target->view_handle();
	img_desc.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkWriteDescriptorSet write_img = {};
	write_img.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write_img.dstSet = desc_set;
	write_img.dstBinding = 1;
	write_img.descriptorCount = 1;
	write_img.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	write_img.pImageInfo = &img_desc;

	VkDescriptorBufferInfo buf_desc = {};
	buf_desc.buffer = view_param_buf->handle();
	buf_desc.offset = 0;
	buf_desc.range = view_param_buf->size();

	VkWriteDescriptorSet write_buf = {};
	write_buf.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write_buf.dstSet = desc_set;
	write_buf.dstBinding = 2;
	write_buf.descriptorCount = 1;
	write_buf.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	write_buf.pBufferInfo = &buf_desc;

	VkDescriptorBufferInfo index_buf_desc = {};
	index_buf_desc.buffer = mesh->index_buf->handle();
	index_buf_desc.offset = 0;
	index_buf_desc.range = mesh->index_buf->size();

	VkWriteDescriptorSet write_index_buf = {};
	write_index_buf.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write_index_buf.dstSet = desc_set;
	write_index_buf.dstBinding = 3;
	write_index_buf.descriptorCount = 1;
	write_index_buf.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	write_index_buf.pBufferInfo = &index_buf_desc;

	VkDescriptorBufferInfo vert_buf_desc = {};
	vert_buf_desc.buffer = mesh->vertex_buf->handle();
	vert_buf_desc.offset = 0;
	vert_buf_desc.range = mesh->vertex_buf->size();

	VkWriteDescriptorSet write_vert_buf = {};
	write_vert_buf.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write_vert_buf.dstSet = desc_set;
	write_vert_buf.dstBinding = 4;
	write_vert_buf.descriptorCount = 1;
	write_vert_buf.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	write_vert_buf.pBufferInfo = &vert_buf_desc;

	const std::vector<VkWriteDescriptorSet> write_descs = {
		write_tlas, write_img, write_buf, write_index_buf, write_vert_buf
	};
	vkUpdateDescriptorSets(device.logical_device(), write_descs.size(), write_descs.data(), 0, nullptr);
}

void RenderVulkan::build_shader_binding_table() {
	const size_t shader_ident_size = device.raytracing_properties().shaderGroupHandleSize;
	const size_t shader_record_size = device.raytracing_properties().shaderGroupHandleSize;
	// Testing sending some params through the SBT
	const size_t hit_record_size = align_to(shader_record_size + sizeof(float),
		device.raytracing_properties().shaderGroupBaseAlignment);
	const size_t sbt_size = 2 * shader_record_size + hit_record_size;

	std::cout << "SBT size: " << sbt_size << "\n";
	auto upload_sbt = vk::Buffer::host(device, sbt_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

	uint8_t *sbt_mapping = reinterpret_cast<uint8_t*>(upload_sbt->map());
	// Get the shader identifiers
	// Note: for now this is the same size as the SBT, but this will not always be the case
	std::vector<uint8_t> shader_identifiers(3 * shader_ident_size, 0);
	CHECK_VULKAN(vkGetRayTracingShaderGroupHandles(device.logical_device(), rt_pipeline, 0, 3,
		shader_identifiers.size(), shader_identifiers.data()));

	for (size_t i = 0; i < 3; ++i) {
		std::memcpy(sbt_mapping + i * shader_ident_size,
			shader_identifiers.data() + i * shader_ident_size,
			shader_ident_size);
	}
	// Write the test params to the hit group
	float test_value = 0.5;
	std::memcpy(sbt_mapping + 3 * shader_ident_size, &test_value, sizeof(float));
	upload_sbt->unmap();

	// Upload the SBT to the GPU
	shader_table = vk::Buffer::device(device, upload_sbt->size(),
		VK_BUFFER_USAGE_RAY_TRACING_BIT_NV | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

	{
		VkCommandBufferBeginInfo begin_info = {};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

		VkBufferCopy copy_cmd = {};
		copy_cmd.size = upload_sbt->size();
		vkCmdCopyBuffer(command_buffer, upload_sbt->handle(), shader_table->handle(), 1, &copy_cmd);

		CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

		VkSubmitInfo submit_info = {};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffer;
		CHECK_VULKAN(vkQueueSubmit(device.graphics_queue(), 1, &submit_info, VK_NULL_HANDLE));
		vkQueueWaitIdle(device.graphics_queue());

		vkResetCommandPool(device.logical_device(), command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
	}
}

void RenderVulkan::update_view_parameters(const glm::vec3 &pos, const glm::vec3 &dir,
	const glm::vec3 &up, const float fovy)
{
	glm::vec2 img_plane_size;
	img_plane_size.y = 2.f * std::tan(glm::radians(0.5f * fovy));
	img_plane_size.x = img_plane_size.y
		* static_cast<float>(render_target->dims().x) / render_target->dims().y;

	const glm::vec3 dir_du = glm::normalize(glm::cross(dir, up)) * img_plane_size.x;
	const glm::vec3 dir_dv = glm::normalize(glm::cross(dir_du, dir)) * img_plane_size.y;
	const glm::vec3 dir_top_left = dir - 0.5f * dir_du - 0.5f * dir_dv;

	uint8_t* buf = static_cast<uint8_t*>(view_param_buf->map());
	{
		glm::vec4* vecs = reinterpret_cast<glm::vec4*>(buf);
		vecs[0] = glm::vec4(pos, 0.f);
		vecs[1] = glm::vec4(dir_du, 0.f);
		vecs[2] = glm::vec4(dir_dv, 0.f);
		vecs[3] = glm::vec4(dir_top_left, 0.f);
	}
	{
		uint32_t* fid = reinterpret_cast<uint32_t*>(buf + 4 * sizeof(glm::vec4));
		*fid = frame_id;
	}
	view_param_buf->unmap();
}
