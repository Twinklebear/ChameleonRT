#include "render_vulkan.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include "spv_shaders_embedded_spv.h"
#include "util.h"
#include <glm/ext.hpp>

RenderVulkan::RenderVulkan(std::shared_ptr<vkrt::Device> dev)
    : device(dev), native_display(true)
{
    command_pool = device->make_command_pool(VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
    {
        VkCommandBufferAllocateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        info.commandPool = command_pool;
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount = 1;
        CHECK_VULKAN(
            vkAllocateCommandBuffers(device->logical_device(), &info, &command_buffer));
    }

    render_cmd_pool = device->make_command_pool();
    {
        VkCommandBufferAllocateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        info.commandPool = render_cmd_pool;
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount = 1;
        CHECK_VULKAN(
            vkAllocateCommandBuffers(device->logical_device(), &info, &render_cmd_buf));
        CHECK_VULKAN(
            vkAllocateCommandBuffers(device->logical_device(), &info, &readback_cmd_buf));
    }

    {
        VkFenceCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        CHECK_VULKAN(vkCreateFence(device->logical_device(), &info, nullptr, &fence));
    }

    view_param_buf = vkrt::Buffer::host(*device,
                                        4 * sizeof(glm::vec4) + sizeof(uint32_t),
                                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

RenderVulkan::RenderVulkan() : RenderVulkan(std::make_shared<vkrt::Device>())
{
    native_display = false;
}

RenderVulkan::~RenderVulkan()
{
    vkDestroySampler(device->logical_device(), sampler, nullptr);
    vkDestroyCommandPool(device->logical_device(), command_pool, nullptr);
    vkDestroyCommandPool(device->logical_device(), render_cmd_pool, nullptr);
    vkDestroyPipelineLayout(device->logical_device(), pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device->logical_device(), desc_layout, nullptr);
    vkDestroyDescriptorSetLayout(device->logical_device(), textures_desc_layout, nullptr);
    vkDestroyDescriptorPool(device->logical_device(), desc_pool, nullptr);
    vkDestroyFence(device->logical_device(), fence, nullptr);
    vkDestroyPipeline(device->logical_device(), rt_pipeline.handle(), nullptr);
}

std::string RenderVulkan::name()
{
    return "Vulkan Ray Tracing";
}

void RenderVulkan::initialize(const int fb_width, const int fb_height)
{
    frame_id = 0;
    img.resize(fb_width * fb_height);

    render_target =
        vkrt::Texture2D::device(*device,
                                glm::uvec2(fb_width, fb_height),
                                VK_FORMAT_R8G8B8A8_UNORM,
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT);

    accum_buffer = vkrt::Texture2D::device(*device,
                                           glm::uvec2(fb_width, fb_height),
                                           VK_FORMAT_R32G32B32A32_SFLOAT,
                                           VK_IMAGE_USAGE_STORAGE_BIT);

    img_readback_buf = vkrt::Buffer::host(
        *device, img.size() * render_target->pixel_size(), VK_BUFFER_USAGE_TRANSFER_DST_BIT);

#ifdef REPORT_RAY_STATS
    ray_stats =
        vkrt::Texture2D::device(*device,
                                glm::uvec2(fb_width, fb_height),
                                VK_FORMAT_R16_UINT,
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
    ray_stats_readback_buf = vkrt::Buffer::host(
        *device, img.size() * ray_stats->pixel_size(), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    ray_counts.resize(img.size(), 0);
#endif

    // Change image and accum buffer to the general layout
    {
        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

#ifndef REPORT_RAY_STATS
        std::array<VkImageMemoryBarrier, 2> barriers = {};
#else
        std::array<VkImageMemoryBarrier, 3> barriers = {};
#endif
        for (auto &b : barriers) {
            b = VkImageMemoryBarrier{};
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            b.subresourceRange.baseMipLevel = 0;
            b.subresourceRange.levelCount = 1;
            b.subresourceRange.baseArrayLayer = 0;
            b.subresourceRange.layerCount = 1;
        }
        barriers[0].image = render_target->image_handle();
        barriers[1].image = accum_buffer->image_handle();
#ifdef REPORT_RAY_STATS
        barriers[2].image = ray_stats->image_handle();
#endif

        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             barriers.size(),
                             barriers.data());

        CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

        // Submit the copy to run
        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        CHECK_VULKAN(vkQueueSubmit(device->graphics_queue(), 1, &submit_info, VK_NULL_HANDLE));
        CHECK_VULKAN(vkQueueWaitIdle(device->graphics_queue()));

        // We didn't make the buffers individually reset-able, but we're just using it as temp
        // one to do this upload so clear the pool to reset
        vkResetCommandPool(device->logical_device(),
                           command_pool,
                           VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
    }

    // If we've loaded the scene and are rendering (i.e., the window was just resized while
    // running) update the descriptor sets and re-record the rendering commands
    if (desc_set != VK_NULL_HANDLE) {
        vkrt::DescriptorSetUpdater()
            .write_storage_image(desc_set, 1, render_target)
            .write_storage_image(desc_set, 2, accum_buffer)
#ifdef REPORT_RAY_STATS
            .write_storage_image(desc_set, 6, ray_stats)
#endif
            .update(*device);

        record_command_buffers();
    }
}

void RenderVulkan::set_scene(const Scene &scene)
{
    frame_id = 0;

    // TODO: We can actually run all these uploads and BVH builds in parallel
    // using multiple command lists, as long as the BVH builds don't need so
    // much build + scratch that we run out of GPU memory.
    // Some helpers for managing the temp upload heap buf allocation and queuing of
    // the commands would help to make it easier to write the parallel load version
    for (const auto &mesh : scene.meshes) {
        std::vector<vkrt::Geometry> geometries;
        for (const auto &geom : mesh.geometries) {
            // Upload triangle vertices to the device
            auto upload_verts = vkrt::Buffer::host(*device,
                                                   geom.vertices.size() * sizeof(glm::vec3),
                                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            {
                void *map = upload_verts->map();
                std::memcpy(map, geom.vertices.data(), upload_verts->size());
                upload_verts->unmap();
            }

            auto upload_indices = vkrt::Buffer::host(*device,
                                                     geom.indices.size() * sizeof(glm::uvec3),
                                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            {
                void *map = upload_indices->map();
                std::memcpy(map, geom.indices.data(), upload_indices->size());
                upload_indices->unmap();
            }

            std::shared_ptr<vkrt::Buffer> upload_normals = nullptr;
            std::shared_ptr<vkrt::Buffer> normal_buf = nullptr;
            if (!geom.normals.empty()) {
                upload_normals = vkrt::Buffer::host(*device,
                                                    geom.normals.size() * sizeof(glm::vec3),
                                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
                normal_buf = vkrt::Buffer::device(
                    *device,
                    upload_normals->size(),
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

                void *map = upload_normals->map();
                std::memcpy(map, geom.normals.data(), upload_normals->size());
                upload_normals->unmap();
            }

            std::shared_ptr<vkrt::Buffer> upload_uvs = nullptr;
            std::shared_ptr<vkrt::Buffer> uv_buf = nullptr;
            if (!geom.uvs.empty()) {
                upload_uvs = vkrt::Buffer::host(*device,
                                                geom.uvs.size() * sizeof(glm::vec2),
                                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
                uv_buf = vkrt::Buffer::device(*device,
                                              upload_uvs->size(),
                                              VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

                void *map = upload_uvs->map();
                std::memcpy(map, geom.uvs.data(), upload_uvs->size());
                upload_uvs->unmap();
            }

            auto vertex_buf = vkrt::Buffer::device(
                *device,
                upload_verts->size(),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

            auto index_buf = vkrt::Buffer::device(
                *device,
                upload_indices->size(),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

            // Execute the upload to the device
            {
                VkCommandBufferBeginInfo begin_info = {};
                begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

                VkBufferCopy copy_cmd = {};
                copy_cmd.size = upload_verts->size();
                vkCmdCopyBuffer(command_buffer,
                                upload_verts->handle(),
                                vertex_buf->handle(),
                                1,
                                &copy_cmd);

                copy_cmd.size = upload_indices->size();
                vkCmdCopyBuffer(command_buffer,
                                upload_indices->handle(),
                                index_buf->handle(),
                                1,
                                &copy_cmd);

                if (upload_normals) {
                    copy_cmd.size = upload_normals->size();
                    vkCmdCopyBuffer(command_buffer,
                                    upload_normals->handle(),
                                    normal_buf->handle(),
                                    1,
                                    &copy_cmd);
                }

                if (upload_uvs) {
                    copy_cmd.size = upload_uvs->size();
                    vkCmdCopyBuffer(
                        command_buffer, upload_uvs->handle(), uv_buf->handle(), 1, &copy_cmd);
                }

                CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

                VkSubmitInfo submit_info = {};
                submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submit_info.commandBufferCount = 1;
                submit_info.pCommandBuffers = &command_buffer;
                CHECK_VULKAN(
                    vkQueueSubmit(device->graphics_queue(), 1, &submit_info, VK_NULL_HANDLE));
                CHECK_VULKAN(vkQueueWaitIdle(device->graphics_queue()));

                vkResetCommandPool(device->logical_device(),
                                   command_pool,
                                   VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
            }

            geometries.emplace_back(vertex_buf, index_buf, normal_buf, uv_buf);
            ++total_geom;
        }

        // Build the bottom level acceleration structure
        auto bvh = std::make_unique<vkrt::TriangleMesh>(*device, geometries);
        {
            // TODO: some convenience utils for this
            VkCommandBufferBeginInfo begin_info = {};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

            bvh->enqueue_build(command_buffer);

            CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

            VkSubmitInfo submit_info = {};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &command_buffer;
            CHECK_VULKAN(
                vkQueueSubmit(device->graphics_queue(), 1, &submit_info, VK_NULL_HANDLE));
            CHECK_VULKAN(vkQueueWaitIdle(device->graphics_queue()));

            vkResetCommandPool(device->logical_device(),
                               command_pool,
                               VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
        }

        // Compact the BVH
        {
            VkCommandBufferBeginInfo begin_info = {};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

            bvh->enqueue_compaction(command_buffer);

            CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

            VkSubmitInfo submit_info = {};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &command_buffer;
            CHECK_VULKAN(
                vkQueueSubmit(device->graphics_queue(), 1, &submit_info, VK_NULL_HANDLE));
            CHECK_VULKAN(vkQueueWaitIdle(device->graphics_queue()));

            vkResetCommandPool(device->logical_device(),
                               command_pool,
                               VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
        }
        bvh->finalize();
        meshes.emplace_back(std::move(bvh));
    }

    parameterized_meshes = scene.parameterized_meshes;
    std::vector<uint32_t> parameterized_mesh_sbt_offsets;
    {
        // Compute the offsets each parameterized mesh will be written too in the SBT,
        // these are then the instance SBT offsets shared by each instance
        uint32_t offset = 0;
        for (const auto &pm : parameterized_meshes) {
            parameterized_mesh_sbt_offsets.push_back(offset);
            offset += meshes[pm.mesh_id]->geometries.size();
        }
    }

    std::shared_ptr<vkrt::Buffer> instance_buf;
    {
        // Setup the instance buffer
        auto upload_instances = vkrt::Buffer::host(
            *device,
            scene.instances.size() * sizeof(VkAccelerationStructureInstanceKHR),
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        VkAccelerationStructureInstanceKHR *map =
            reinterpret_cast<VkAccelerationStructureInstanceKHR *>(upload_instances->map());

        for (size_t i = 0; i < scene.instances.size(); ++i) {
            const auto &inst = scene.instances[i];
            std::memset(&map[i], 0, sizeof(VkAccelerationStructureInstanceKHR));
            map[i].instanceCustomIndex = i;
            map[i].instanceShaderBindingTableRecordOffset =
                parameterized_mesh_sbt_offsets[inst.parameterized_mesh_id];
            map[i].flags = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
            map[i].accelerationStructureReference =
                meshes[parameterized_meshes[inst.parameterized_mesh_id].mesh_id]->handle;
            map[i].mask = 0xff;

            // Note: 4x3 row major
            const glm::mat4 m = glm::transpose(inst.transform);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 4; ++c) {
                    map[i].transform.matrix[r][c] = m[r][c];
                }
            }
        }
        upload_instances->unmap();

        instance_buf = vkrt::Buffer::device(
            *device,
            upload_instances->size(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        // Upload the instance data to the device
        {
            VkCommandBufferBeginInfo begin_info = {};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

            VkBufferCopy copy_cmd = {};
            copy_cmd.size = upload_instances->size();
            vkCmdCopyBuffer(command_buffer,
                            upload_instances->handle(),
                            instance_buf->handle(),
                            1,
                            &copy_cmd);

            CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

            VkSubmitInfo submit_info = {};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &command_buffer;
            CHECK_VULKAN(
                vkQueueSubmit(device->graphics_queue(), 1, &submit_info, VK_NULL_HANDLE));
            CHECK_VULKAN(vkQueueWaitIdle(device->graphics_queue()));

            vkResetCommandPool(device->logical_device(),
                               command_pool,
                               VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
        }
    }

    // Build the top level BVH
    scene_bvh = std::make_unique<vkrt::TopLevelBVH>(*device, instance_buf, scene.instances);
    {
        // TODO: some convenience utils for this
        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

        scene_bvh->enqueue_build(command_buffer);

        CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        CHECK_VULKAN(vkQueueSubmit(device->graphics_queue(), 1, &submit_info, VK_NULL_HANDLE));
        CHECK_VULKAN(vkQueueWaitIdle(device->graphics_queue()));

        vkResetCommandPool(device->logical_device(),
                           command_pool,
                           VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
    }
    scene_bvh->finalize();

    mat_params = vkrt::Buffer::device(
        *device,
        scene.materials.size() * sizeof(DisneyMaterial),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    {
        auto upload_mat_params =
            vkrt::Buffer::host(*device, mat_params->size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        void *map = upload_mat_params->map();
        std::memcpy(map, scene.materials.data(), upload_mat_params->size());
        upload_mat_params->unmap();

        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

        VkBufferCopy copy_cmd = {};
        copy_cmd.size = upload_mat_params->size();
        vkCmdCopyBuffer(
            command_buffer, upload_mat_params->handle(), mat_params->handle(), 1, &copy_cmd);

        CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        CHECK_VULKAN(vkQueueSubmit(device->graphics_queue(), 1, &submit_info, VK_NULL_HANDLE));
        CHECK_VULKAN(vkQueueWaitIdle(device->graphics_queue()));

        vkResetCommandPool(device->logical_device(),
                           command_pool,
                           VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
    }

    // Upload the scene textures
    for (const auto &t : scene.textures) {
        auto format =
            t.color_space == SRGB ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
        auto tex = vkrt::Texture2D::device(
            *device,
            glm::uvec2(t.width, t.height),
            format,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

        auto upload_buf = vkrt::Buffer::host(
            *device, tex->pixel_size() * t.width * t.height, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        void *map = upload_buf->map();
        std::memcpy(map, t.img.data(), upload_buf->size());
        upload_buf->unmap();

        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

        // Transition image to the general layout
        VkImageMemoryBarrier img_mem_barrier = {};
        img_mem_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        img_mem_barrier.image = tex->image_handle();
        img_mem_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        img_mem_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        img_mem_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        img_mem_barrier.subresourceRange.baseMipLevel = 0;
        img_mem_barrier.subresourceRange.levelCount = 1;
        img_mem_barrier.subresourceRange.baseArrayLayer = 0;
        img_mem_barrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &img_mem_barrier);

        VkImageSubresourceLayers copy_subresource = {};
        copy_subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_subresource.mipLevel = 0;
        copy_subresource.baseArrayLayer = 0;
        copy_subresource.layerCount = 1;

        VkBufferImageCopy img_copy = {};
        img_copy.bufferOffset = 0;
        img_copy.bufferRowLength = 0;
        img_copy.bufferImageHeight = 0;
        img_copy.imageSubresource = copy_subresource;
        img_copy.imageOffset.x = 0;
        img_copy.imageOffset.y = 0;
        img_copy.imageOffset.z = 0;
        img_copy.imageExtent.width = t.width;
        img_copy.imageExtent.height = t.height;
        img_copy.imageExtent.depth = 1;

        vkCmdCopyBufferToImage(command_buffer,
                               upload_buf->handle(),
                               tex->image_handle(),
                               VK_IMAGE_LAYOUT_GENERAL,
                               1,
                               &img_copy);

        // Transition image to shader read optimal layout
        img_mem_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        img_mem_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &img_mem_barrier);

        CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        CHECK_VULKAN(vkQueueSubmit(device->graphics_queue(), 1, &submit_info, VK_NULL_HANDLE));
        CHECK_VULKAN(vkQueueWaitIdle(device->graphics_queue()));

        vkResetCommandPool(device->logical_device(),
                           command_pool,
                           VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);

        textures.push_back(tex);
    }

    {
        VkSamplerCreateInfo sampler_info = {};
        sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler_info.magFilter = VK_FILTER_LINEAR;
        sampler_info.minFilter = VK_FILTER_LINEAR;
        sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.minLod = 0;
        sampler_info.maxLod = 0;
        CHECK_VULKAN(
            vkCreateSampler(device->logical_device(), &sampler_info, nullptr, &sampler));
    }

    light_params = vkrt::Buffer::device(
        *device,
        sizeof(QuadLight) * scene.lights.size(),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    {
        auto upload_light_params = vkrt::Buffer::host(
            *device, light_params->size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        void *map = upload_light_params->map();
        std::memcpy(map, scene.lights.data(), upload_light_params->size());
        upload_light_params->unmap();

        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

        VkBufferCopy copy_cmd = {};
        copy_cmd.size = upload_light_params->size();
        vkCmdCopyBuffer(command_buffer,
                        upload_light_params->handle(),
                        light_params->handle(),
                        1,
                        &copy_cmd);

        CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        CHECK_VULKAN(vkQueueSubmit(device->graphics_queue(), 1, &submit_info, VK_NULL_HANDLE));
        CHECK_VULKAN(vkQueueWaitIdle(device->graphics_queue()));

        vkResetCommandPool(device->logical_device(),
                           command_pool,
                           VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
    }

    build_raytracing_pipeline();
    build_shader_descriptor_table();
    build_shader_binding_table();
    record_command_buffers();
}

RenderStats RenderVulkan::render(const glm::vec3 &pos,
                                 const glm::vec3 &dir,
                                 const glm::vec3 &up,
                                 const float fovy,
                                 const bool camera_changed,
                                 const bool readback_framebuffer)
{
    using namespace std::chrono;
    RenderStats stats;

    if (camera_changed) {
        frame_id = 0;
    }

    update_view_parameters(pos, dir, up, fovy);

    CHECK_VULKAN(vkResetFences(device->logical_device(), 1, &fence));

    auto start = high_resolution_clock::now();
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &render_cmd_buf;
    CHECK_VULKAN(vkQueueSubmit(device->graphics_queue(), 1, &submit_info, fence));

#ifdef REPORT_RAY_STATS
    const bool need_readback = true;
#else
    const bool need_readback = !native_display || readback_framebuffer;
#endif

    // Queue the readback copy to start once rendering is done
    if (need_readback) {
        submit_info.pCommandBuffers = &readback_cmd_buf;
        CHECK_VULKAN(vkQueueSubmit(device->graphics_queue(), 1, &submit_info, VK_NULL_HANDLE));
    }

    // Wait for just the rendering commands to complete to only time the ray tracing time
    CHECK_VULKAN(vkWaitForFences(
        device->logical_device(), 1, &fence, true, std::numeric_limits<uint64_t>::max()));
    auto end = high_resolution_clock::now();
    stats.render_time = duration_cast<nanoseconds>(end - start).count() * 1.0e-6;

    // Now wait for the device to finish the readback copy as well
    if (need_readback) {
        CHECK_VULKAN(vkQueueWaitIdle(device->graphics_queue()));
        std::memcpy(img.data(), img_readback_buf->map(), img_readback_buf->size());
        img_readback_buf->unmap();
    }

#ifdef REPORT_RAY_STATS
    std::memcpy(ray_counts.data(),
                ray_stats_readback_buf->map(),
                ray_counts.size() * sizeof(uint16_t));
    ray_stats_readback_buf->unmap();

    const uint64_t total_rays =
        std::accumulate(ray_counts.begin(),
                        ray_counts.end(),
                        uint64_t(0),
                        [](const uint64_t &total, const uint16_t &c) { return total + c; });
    stats.rays_per_second = total_rays / (stats.render_time * 1.0e-3);
#endif

    ++frame_id;
    return stats;
}

void RenderVulkan::build_raytracing_pipeline()
{
    // Maybe have the builder auto compute the binding numbers? May be annoying if tweaking
    // layouts or something though
    desc_layout =
        vkrt::DescriptorSetLayoutBuilder()
            .add_binding(0,
                         1,
                         VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                         VK_SHADER_STAGE_RAYGEN_BIT_KHR)
            .add_binding(
                1, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR)
            .add_binding(
                2, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR)
            .add_binding(
                3, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR)
            .add_binding(
                4, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR)
            .add_binding(
                5, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR)
#ifdef REPORT_RAY_STATS
            .add_binding(
                6, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR)
#endif
            .build(*device);

    const size_t total_geom =
        std::accumulate(meshes.begin(),
                        meshes.end(),
                        0,
                        [](size_t n, const std::unique_ptr<vkrt::TriangleMesh> &t) {
                            return n + t->geometries.size();
                        });

    textures_desc_layout =
        vkrt::DescriptorSetLayoutBuilder()
            .add_binding(0,
                         std::max(textures.size(), size_t(1)),
                         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                         VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                         VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT)
            .build(*device);

    std::vector<VkDescriptorSetLayout> descriptor_layouts = {desc_layout,
                                                             textures_desc_layout};

    VkPipelineLayoutCreateInfo pipeline_create_info = {};
    pipeline_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_create_info.setLayoutCount = descriptor_layouts.size();
    pipeline_create_info.pSetLayouts = descriptor_layouts.data();

    CHECK_VULKAN(vkCreatePipelineLayout(
        device->logical_device(), &pipeline_create_info, nullptr, &pipeline_layout));

    // Load the shader modules for our pipeline and build the pipeline
    auto raygen_shader =
        std::make_shared<vkrt::ShaderModule>(*device, raygen_spv, sizeof(raygen_spv));

    auto miss_shader =
        std::make_shared<vkrt::ShaderModule>(*device, miss_spv, sizeof(miss_spv));
    auto occlusion_miss_shader = std::make_shared<vkrt::ShaderModule>(
        *device, occlusion_miss_spv, sizeof(occlusion_miss_spv));

    auto closest_hit_shader =
        std::make_shared<vkrt::ShaderModule>(*device, hit_spv, sizeof(hit_spv));

    rt_pipeline = vkrt::RTPipelineBuilder()
                      .set_raygen("raygen", raygen_shader)
                      .add_miss("miss", miss_shader)
                      .add_miss("occlusion_miss", occlusion_miss_shader)
                      .add_hitgroup("closest_hit", closest_hit_shader)
                      .set_recursion_depth(1)
                      .set_layout(pipeline_layout)
                      .build(*device);
}

void RenderVulkan::build_shader_descriptor_table()
{
    const std::vector<VkDescriptorPoolSize> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                             std::max(uint32_t(textures.size()), uint32_t(1))}};

    VkDescriptorPoolCreateInfo pool_create_info = {};
    pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_create_info.maxSets = 6;
    pool_create_info.poolSizeCount = pool_sizes.size();
    pool_create_info.pPoolSizes = pool_sizes.data();
    CHECK_VULKAN(vkCreateDescriptorPool(
        device->logical_device(), &pool_create_info, nullptr, &desc_pool));

    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = desc_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &desc_layout;
    CHECK_VULKAN(vkAllocateDescriptorSets(device->logical_device(), &alloc_info, &desc_set));

    alloc_info.pSetLayouts = &textures_desc_layout;
    CHECK_VULKAN(
        vkAllocateDescriptorSets(device->logical_device(), &alloc_info, &textures_desc_set));

    std::vector<vkrt::CombinedImageSampler> combined_samplers;
    for (const auto &t : textures) {
        combined_samplers.emplace_back(t, sampler);
    }

    auto updater = vkrt::DescriptorSetUpdater()
                       .write_acceleration_structure(desc_set, 0, scene_bvh)
                       .write_storage_image(desc_set, 1, render_target)
                       .write_storage_image(desc_set, 2, accum_buffer)
                       .write_ubo(desc_set, 3, view_param_buf)
                       .write_ssbo(desc_set, 4, mat_params)
                       .write_ssbo(desc_set, 5, light_params);
#ifdef REPORT_RAY_STATS
    updater.write_storage_image(desc_set, 6, ray_stats);
#endif

    if (!combined_samplers.empty()) {
        updater.write_combined_sampler_array(textures_desc_set, 0, combined_samplers);
    }
    updater.update(*device);
}

void RenderVulkan::build_shader_binding_table()
{
    vkrt::SBTBuilder sbt_builder(&rt_pipeline);
    sbt_builder.set_raygen(vkrt::ShaderRecord("raygen", "raygen", sizeof(uint32_t)))
        .add_miss(vkrt::ShaderRecord("miss", "miss", 0))
        .add_miss(vkrt::ShaderRecord("occlusion_miss", "occlusion_miss", 0));

    for (size_t i = 0; i < parameterized_meshes.size(); ++i) {
        const auto &pm = parameterized_meshes[i];
        for (size_t j = 0; j < meshes[pm.mesh_id]->geometries.size(); ++j) {
            const std::string hg_name =
                "HitGroup_param_mesh" + std::to_string(i) + "_geom" + std::to_string(j);
            sbt_builder.add_hitgroup(
                vkrt::ShaderRecord(hg_name, "closest_hit", sizeof(HitGroupParams)));
        }
    }

    shader_table = sbt_builder.build(*device);

    shader_table.map_sbt();

    // Raygen shader gets the number of lights through the SBT
    {
        uint32_t *params = reinterpret_cast<uint32_t *>(shader_table.sbt_params("raygen"));
        *params = light_params->size() / sizeof(QuadLight);
    }

    for (size_t i = 0; i < parameterized_meshes.size(); ++i) {
        const auto &pm = parameterized_meshes[i];
        for (size_t j = 0; j < meshes[pm.mesh_id]->geometries.size(); ++j) {
            auto &geom = meshes[pm.mesh_id]->geometries[j];
            const std::string hg_name =
                "HitGroup_param_mesh" + std::to_string(i) + "_geom" + std::to_string(j);

            HitGroupParams *params =
                reinterpret_cast<HitGroupParams *>(shader_table.sbt_params(hg_name));

            params->vert_buf = meshes[pm.mesh_id]->geometries[j].vertex_buf->device_address();
            params->idx_buf = meshes[pm.mesh_id]->geometries[j].index_buf->device_address();

            if (meshes[pm.mesh_id]->geometries[j].normal_buf) {
                params->normal_buf =
                    meshes[pm.mesh_id]->geometries[j].normal_buf->device_address();
                params->num_normals = 1;
            } else {
                params->num_normals = 0;
            }

            if (meshes[pm.mesh_id]->geometries[j].uv_buf) {
                params->uv_buf = meshes[pm.mesh_id]->geometries[j].uv_buf->device_address();
                params->num_uvs = 1;
            } else {
                params->num_uvs = 0;
            }
            params->material_id = pm.material_ids[j];
        }
    }

    {
        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

        VkBufferCopy copy_cmd = {};
        copy_cmd.size = shader_table.upload_sbt->size();
        vkCmdCopyBuffer(command_buffer,
                        shader_table.upload_sbt->handle(),
                        shader_table.sbt->handle(),
                        1,
                        &copy_cmd);

        CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        CHECK_VULKAN(vkQueueSubmit(device->graphics_queue(), 1, &submit_info, VK_NULL_HANDLE));
        CHECK_VULKAN(vkQueueWaitIdle(device->graphics_queue()));

        vkResetCommandPool(device->logical_device(),
                           command_pool,
                           VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
    }
}

void RenderVulkan::update_view_parameters(const glm::vec3 &pos,
                                          const glm::vec3 &dir,
                                          const glm::vec3 &up,
                                          const float fovy)
{
    glm::vec2 img_plane_size;
    img_plane_size.y = 2.f * std::tan(glm::radians(0.5f * fovy));
    img_plane_size.x = img_plane_size.y * static_cast<float>(render_target->dims().x) /
                       render_target->dims().y;

    const glm::vec3 dir_du = glm::normalize(glm::cross(dir, up)) * img_plane_size.x;
    const glm::vec3 dir_dv = -glm::normalize(glm::cross(dir_du, dir)) * img_plane_size.y;
    const glm::vec3 dir_top_left = dir - 0.5f * dir_du - 0.5f * dir_dv;

    uint8_t *buf = static_cast<uint8_t *>(view_param_buf->map());
    {
        glm::vec4 *vecs = reinterpret_cast<glm::vec4 *>(buf);
        vecs[0] = glm::vec4(pos, 0.f);
        vecs[1] = glm::vec4(dir_du, 0.f);
        vecs[2] = glm::vec4(dir_dv, 0.f);
        vecs[3] = glm::vec4(dir_top_left, 0.f);
    }
    {
        uint32_t *fid = reinterpret_cast<uint32_t *>(buf + 4 * sizeof(glm::vec4));
        *fid = frame_id;
    }
    view_param_buf->unmap();
}

void RenderVulkan::record_command_buffers()
{
    vkResetCommandPool(device->logical_device(),
                       render_cmd_pool,
                       VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    CHECK_VULKAN(vkBeginCommandBuffer(render_cmd_buf, &begin_info));

    vkCmdBindPipeline(
        render_cmd_buf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rt_pipeline.handle());

    const std::vector<VkDescriptorSet> descriptor_sets = {desc_set, textures_desc_set};

    vkCmdBindDescriptorSets(render_cmd_buf,
                            VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            pipeline_layout,
                            0,
                            descriptor_sets.size(),
                            descriptor_sets.data(),
                            0,
                            nullptr);

    VkStridedDeviceAddressRegionKHR callable_table = {};
    callable_table.deviceAddress = VK_NULL_HANDLE;

    vkrt::CmdTraceRaysKHR(render_cmd_buf,
                          &shader_table.raygen,
                          &shader_table.miss,
                          &shader_table.hitgroup,
                          &callable_table,
                          render_target->dims().x,
                          render_target->dims().y,
                          1);

    // Queue a barrier for rendering to finish
    vkCmdPipelineBarrier(render_cmd_buf,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                         0,
                         0,
                         nullptr,
                         0,
                         nullptr,
                         0,
                         nullptr);

    CHECK_VULKAN(vkEndCommandBuffer(render_cmd_buf));

    CHECK_VULKAN(vkBeginCommandBuffer(readback_cmd_buf, &begin_info));

    VkImageSubresourceLayers copy_subresource = {};
    copy_subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_subresource.mipLevel = 0;
    copy_subresource.baseArrayLayer = 0;
    copy_subresource.layerCount = 1;

    VkBufferImageCopy img_copy = {};
    img_copy.bufferOffset = 0;
    img_copy.bufferRowLength = 0;
    img_copy.bufferImageHeight = 0;
    img_copy.imageSubresource = copy_subresource;
    img_copy.imageOffset.x = 0;
    img_copy.imageOffset.y = 0;
    img_copy.imageOffset.z = 0;
    img_copy.imageExtent.width = render_target->dims().x;
    img_copy.imageExtent.height = render_target->dims().y;
    img_copy.imageExtent.depth = 1;

    vkCmdCopyImageToBuffer(readback_cmd_buf,
                           render_target->image_handle(),
                           VK_IMAGE_LAYOUT_GENERAL,
                           img_readback_buf->handle(),
                           1,
                           &img_copy);

#ifdef REPORT_RAY_STATS
    img_copy.bufferOffset = 0;
    img_copy.bufferRowLength = 0;
    img_copy.bufferImageHeight = 0;
    img_copy.imageSubresource = copy_subresource;
    img_copy.imageOffset.x = 0;
    img_copy.imageOffset.y = 0;
    img_copy.imageOffset.z = 0;
    img_copy.imageExtent.width = render_target->dims().x;
    img_copy.imageExtent.height = render_target->dims().y;
    img_copy.imageExtent.depth = 1;

    vkCmdCopyImageToBuffer(readback_cmd_buf,
                           ray_stats->image_handle(),
                           VK_IMAGE_LAYOUT_GENERAL,
                           ray_stats_readback_buf->handle(),
                           1,
                           &img_copy);
#endif

    CHECK_VULKAN(vkEndCommandBuffer(readback_cmd_buf));
}
