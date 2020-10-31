#include "vkdisplay.h"
#include <algorithm>
#include <cstring>
#include <iterator>
#include <SDL_vulkan.h>
#include <vulkan/vulkan.h>
#include "display/imgui_impl_sdl.h"
#include "imgui_impl_vulkan.h"
#include "vulkan_utils.h"

#if !SDL_VERSION_ATLEAST(2, 0, 8)
#error "SDL 2.0.8 or higher is required for the Vulkan display frontend"
#endif

const static std::vector<std::string> logical_device_extensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

std::vector<std::string> get_instance_extensions(SDL_Window *window)
{
    uint32_t sdl_extension_count = 0;
    if (!SDL_Vulkan_GetInstanceExtensions(window, &sdl_extension_count, nullptr)) {
        throw std::runtime_error("Failed to get SDL vulkan extension count");
    }

    std::vector<const char *> sdl_extensions(sdl_extension_count, nullptr);
    if (!SDL_Vulkan_GetInstanceExtensions(
            window, &sdl_extension_count, sdl_extensions.data())) {
        throw std::runtime_error("Failed to get SDL vulkan extension list");
    }

    std::vector<std::string> instance_extensions;
    std::transform(sdl_extensions.begin(),
                   sdl_extensions.end(),
                   std::back_inserter(instance_extensions),
                   [](const char *str) { return std::string(str); });
    return instance_extensions;
}

VKDisplay::VKDisplay(SDL_Window *window)
    : device(std::make_shared<vkrt::Device>(get_instance_extensions(window),
                                            logical_device_extensions))
{
    SDL_version ver;
    SDL_GetVersion(&ver);
    if (ver.major == 2 && ver.minor == 0 && ver.patch < 8) {
        std::cout << "SDL 2.0.8 or higher is required for the Vulkan display frontend\n";
        throw std::runtime_error(
            "SDL 2.0.8 or higher is required for the Vulkan display frontend");
    }

    if (!SDL_Vulkan_CreateSurface(window, device->instance(), &surface)) {
        throw std::runtime_error("Failed to create Vulkan surface using SDL");
    }

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
    {
        VkSemaphoreCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        CHECK_VULKAN(
            vkCreateSemaphore(device->logical_device(), &info, nullptr, &img_avail_semaphore));
        CHECK_VULKAN(vkCreateSemaphore(
            device->logical_device(), &info, nullptr, &present_ready_semaphore));
    }
    {
        VkFenceCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        CHECK_VULKAN(vkCreateFence(device->logical_device(), &info, nullptr, &fence));
    }

    VkBool32 present_supported = false;
    CHECK_VULKAN(vkGetPhysicalDeviceSurfaceSupportKHR(
        device->physical_device(), device->queue_index(), surface, &present_supported));
    if (!present_supported) {
        throw std::runtime_error("Present is not supported on the graphics queue!?");
    }

    // Setup ImGui render pass
    {
        VkAttachmentDescription attachment = {};
        attachment.format = VK_FORMAT_B8G8R8A8_UNORM;
        attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        // TODO: I think I want load op load to preserve the ray traced image I copied onto the
        // back buffer
        attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentReference color_attachment = {};
        color_attachment.attachment = 0;
        color_attachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color_attachment;
        VkSubpassDependency dependency = {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        info.attachmentCount = 1;
        info.pAttachments = &attachment;
        info.subpassCount = 1;
        info.pSubpasses = &subpass;
        info.dependencyCount = 1;
        info.pDependencies = &dependency;
        CHECK_VULKAN(
            vkCreateRenderPass(device->logical_device(), &info, nullptr, &imgui_render_pass));
    }

    {
        VkDescriptorPoolSize pool_size = {};
        pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        pool_size.descriptorCount = 1;

        VkDescriptorPoolCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        info.maxSets = 1;
        info.poolSizeCount = 1;
        info.pPoolSizes = &pool_size;
        CHECK_VULKAN(vkCreateDescriptorPool(
            device->logical_device(), &info, nullptr, &imgui_desc_pool));
    }

    ImGui_ImplSDL2_InitForVulkan(window);

    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = device->instance();
    init_info.PhysicalDevice = device->physical_device();
    init_info.Device = device->logical_device();
    init_info.QueueFamily = device->queue_index();
    init_info.Queue = device->graphics_queue();
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.DescriptorPool = imgui_desc_pool;
    init_info.Allocator = nullptr;
    init_info.MinImageCount = 2;
    init_info.ImageCount = 2;  // can I assume 2 from the swap chain?
    init_info.CheckVkResultFn = [](const VkResult err) { CHECK_VULKAN(err); };
    ImGui_ImplVulkan_Init(&init_info, imgui_render_pass);

    // Upload ImGui's font texture
    vkResetCommandPool(
        device->logical_device(), command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));
    ImGui_ImplVulkan_CreateFontsTexture(command_buffer);

    CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

    const std::array<VkPipelineStageFlags, 1> wait_stages = {
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};

    CHECK_VULKAN(vkResetFences(device->logical_device(), 1, &fence));

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.waitSemaphoreCount = 0;
    submit_info.pWaitDstStageMask = wait_stages.data();
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    submit_info.signalSemaphoreCount = 0;
    CHECK_VULKAN(vkQueueSubmit(device->graphics_queue(), 1, &submit_info, fence));
    CHECK_VULKAN(vkWaitForFences(
        device->logical_device(), 1, &fence, true, std::numeric_limits<uint64_t>::max()));

    ImGui_ImplVulkan_DestroyFontUploadObjects();
}

VKDisplay::~VKDisplay()
{
    ImGui_ImplVulkan_Shutdown();
    for (auto &v : swap_chain_image_views) {
        vkDestroyImageView(device->logical_device(), v, nullptr);
    }
    for (auto &f : framebuffers) {
        vkDestroyFramebuffer(device->logical_device(), f, nullptr);
    }
    vkDestroyDescriptorPool(device->logical_device(), imgui_desc_pool, nullptr);
    vkDestroyRenderPass(device->logical_device(), imgui_render_pass, nullptr);
    vkDestroySemaphore(device->logical_device(), img_avail_semaphore, nullptr);
    vkDestroySemaphore(device->logical_device(), present_ready_semaphore, nullptr);
    vkDestroyFence(device->logical_device(), fence, nullptr);
    vkDestroyCommandPool(device->logical_device(), command_pool, nullptr);
    vkDestroySwapchainKHR(device->logical_device(), swap_chain, nullptr);
    vkDestroySurfaceKHR(device->instance(), surface, nullptr);
}

std::string VKDisplay::gpu_brand()
{
    VkPhysicalDeviceProperties properties = {};
    vkGetPhysicalDeviceProperties(device->physical_device(), &properties);
    return properties.deviceName;
}

std::string VKDisplay::name()
{
    return "Vulkan";
}

void VKDisplay::resize(const int fb_width, const int fb_height)
{
    if (swap_chain != VK_NULL_HANDLE) {
        for (auto &v : swap_chain_image_views) {
            vkDestroyImageView(device->logical_device(), v, nullptr);
        }
        for (auto &f : framebuffers) {
            vkDestroyFramebuffer(device->logical_device(), f, nullptr);
        }
        vkDestroySwapchainKHR(device->logical_device(), swap_chain, nullptr);
    }

    fb_dims = glm::uvec2(fb_width, fb_height);
    upload_texture = vkrt::Texture2D::device(
        *device,
        fb_dims,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    upload_buffer = vkrt::Buffer::host(
        *device, sizeof(uint32_t) * fb_dims.x * fb_dims.y, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

    VkExtent2D swapchain_extent = {};
    swapchain_extent.width = fb_dims.x;
    swapchain_extent.height = fb_dims.y;

    VkSwapchainCreateInfoKHR create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = surface;
    create_info.minImageCount = 2;
    create_info.imageFormat = VK_FORMAT_B8G8R8A8_UNORM;
    create_info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    create_info.imageExtent = swapchain_extent;
    create_info.imageArrayLayers = 1;
    create_info.imageUsage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create_info.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    create_info.clipped = true;
    create_info.oldSwapchain = VK_NULL_HANDLE;
    CHECK_VULKAN(
        vkCreateSwapchainKHR(device->logical_device(), &create_info, nullptr, &swap_chain));

    // Get the swap chain images
    uint32_t num_swapchain_imgs = 0;
    vkGetSwapchainImagesKHR(
        device->logical_device(), swap_chain, &num_swapchain_imgs, nullptr);
    swap_chain_images.resize(num_swapchain_imgs);
    vkGetSwapchainImagesKHR(
        device->logical_device(), swap_chain, &num_swapchain_imgs, swap_chain_images.data());

    swap_chain_image_views.resize(num_swapchain_imgs);
    // Make image views and framebuffers for the imgui render pass
    for (size_t i = 0; i < swap_chain_images.size(); ++i) {
        VkImageViewCreateInfo view_create_info = {};
        view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_create_info.image = swap_chain_images[i];
        view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_create_info.format = VK_FORMAT_B8G8R8A8_UNORM;

        view_create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

        view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_create_info.subresourceRange.baseMipLevel = 0;
        view_create_info.subresourceRange.levelCount = 1;
        view_create_info.subresourceRange.baseArrayLayer = 0;
        view_create_info.subresourceRange.layerCount = 1;

        CHECK_VULKAN(vkCreateImageView(
            device->logical_device(), &view_create_info, nullptr, &swap_chain_image_views[i]));
    }

    framebuffers.resize(num_swapchain_imgs);
    for (size_t i = 0; i < swap_chain_image_views.size(); ++i) {
        VkFramebufferCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        info.renderPass = imgui_render_pass;
        info.attachmentCount = 1;
        info.pAttachments = &swap_chain_image_views[i];
        info.width = fb_dims.x;
        info.height = fb_dims.y;
        info.layers = 1;

        CHECK_VULKAN(
            vkCreateFramebuffer(device->logical_device(), &info, nullptr, &framebuffers[i]));
    }

    ImGui_ImplVulkan_SetMinImageCount(num_swapchain_imgs);
}

void VKDisplay::new_frame()
{
    ImGui_ImplVulkan_NewFrame();
}

void VKDisplay::display(const std::vector<uint32_t> &img)
{
    std::memcpy(upload_buffer->map(), img.data(), upload_buffer->size());
    upload_buffer->unmap();

    vkResetCommandPool(
        device->logical_device(), command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

    VkImageMemoryBarrier img_mem_barrier = {};
    img_mem_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    img_mem_barrier.image = upload_texture->image_handle();
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
    img_copy.imageExtent.width = fb_dims.x;
    img_copy.imageExtent.height = fb_dims.y;
    img_copy.imageExtent.depth = 1;

    vkCmdCopyBufferToImage(command_buffer,
                           upload_buffer->handle(),
                           upload_texture->image_handle(),
                           VK_IMAGE_LAYOUT_GENERAL,
                           1,
                           &img_copy);

    CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    CHECK_VULKAN(vkQueueSubmit(device->graphics_queue(), 1, &submit_info, VK_NULL_HANDLE));
    vkQueueWaitIdle(device->graphics_queue());

    display_native(upload_texture);
}

void VKDisplay::display_native(std::shared_ptr<vkrt::Texture2D> &img)
{
    uint32_t back_buffer_idx = 0;
    CHECK_VULKAN(vkAcquireNextImageKHR(device->logical_device(),
                                       swap_chain,
                                       std::numeric_limits<uint64_t>::max(),
                                       img_avail_semaphore,
                                       VK_NULL_HANDLE,
                                       &back_buffer_idx));

    vkResetCommandPool(
        device->logical_device(), command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

    // Transition image to the transfer dest for the blit layout
    VkImageMemoryBarrier img_mem_barrier = {};
    img_mem_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    img_mem_barrier.image = swap_chain_images[back_buffer_idx];
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

    VkImageBlit blit = {};
    blit.srcSubresource = copy_subresource;
    blit.srcOffsets[0].x = 0;
    blit.srcOffsets[0].y = 0;
    blit.srcOffsets[0].z = 0;
    blit.srcOffsets[1].x = fb_dims.x;
    blit.srcOffsets[1].y = fb_dims.y;
    blit.srcOffsets[1].z = 1;
    blit.dstSubresource = copy_subresource;
    blit.dstOffsets[0].x = 0;
    blit.dstOffsets[0].y = 0;
    blit.dstOffsets[0].z = 0;
    blit.dstOffsets[1].x = fb_dims.x;
    blit.dstOffsets[1].y = fb_dims.y;
    blit.dstOffsets[1].z = 1;

    vkCmdBlitImage(command_buffer,
                   img->image_handle(),
                   VK_IMAGE_LAYOUT_GENERAL,
                   swap_chain_images[back_buffer_idx],
                   VK_IMAGE_LAYOUT_GENERAL,
                   1,
                   &blit,
                   VK_FILTER_NEAREST);

    VkRenderPassBeginInfo render_pass_info = {};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = imgui_render_pass;
    render_pass_info.framebuffer = framebuffers[back_buffer_idx];
    render_pass_info.renderArea.extent.width = fb_dims.x;
    render_pass_info.renderArea.extent.height = fb_dims.y;
    render_pass_info.clearValueCount = 0;
    vkCmdBeginRenderPass(command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), command_buffer);
    vkCmdEndRenderPass(command_buffer);

    CHECK_VULKAN(vkEndCommandBuffer(command_buffer));

    const std::array<VkPipelineStageFlags, 1> wait_stages = {
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};

    CHECK_VULKAN(vkResetFences(device->logical_device(), 1, &fence));

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = &img_avail_semaphore;
    submit_info.pWaitDstStageMask = wait_stages.data();
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &present_ready_semaphore;
    CHECK_VULKAN(vkQueueSubmit(device->graphics_queue(), 1, &submit_info, fence));

    // Finally, present the updated image in the swap chain
    VkPresentInfoKHR present_info = {};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &present_ready_semaphore;
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &swap_chain;
    present_info.pImageIndices = &back_buffer_idx;
    auto err = vkQueuePresentKHR(device->graphics_queue(), &present_info);
    switch (err) {
    case VK_SUCCESS:
    case VK_SUBOPTIMAL_KHR:
        // On Linux it seems we get the error failing to present before we get the window
        // resized event from SDL to update the swap chain, so filter out these errors
    case VK_ERROR_OUT_OF_DATE_KHR:
        break;
    default:
        // Other errors are actual problems
        CHECK_VULKAN(err);
        break;
    }

    // Wait for the present to finish
    CHECK_VULKAN(vkWaitForFences(
        device->logical_device(), 1, &fence, true, std::numeric_limits<uint64_t>::max()));
}
