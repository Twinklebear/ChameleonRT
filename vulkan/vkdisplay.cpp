#include "vkdisplay.h"
#include <SDL_syswm.h>
// TODO need to include and do setup for different OS
#include <vulkan/vulkan_win32.h>
#include "display/imgui_impl_sdl.h"
#include "imgui_impl_vulkan.h"

const static std::vector<std::string> instance_extensions = {
    VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};

const static std::vector<std::string> logical_device_extensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

VKDisplay::VKDisplay(SDL_Window *window)
    : device(instance_extensions, logical_device_extensions)
{
    {
        SDL_SysWMinfo wm_info;
        SDL_VERSION(&wm_info.version);
        SDL_GetWindowWMInfo(window, &wm_info);

        VkWin32SurfaceCreateInfoKHR create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        create_info.hwnd = wm_info.info.win.window;
        create_info.hinstance = wm_info.info.win.hinstance;
        CHECK_VULKAN(
            vkCreateWin32SurfaceKHR(device.instance(), &create_info, nullptr, &surface));
    }

    command_pool = device.make_command_pool(VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
    {
        VkCommandBufferAllocateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        info.commandPool = command_pool;
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount = 1;
        CHECK_VULKAN(
            vkAllocateCommandBuffers(device.logical_device(), &info, &command_buffer));
    }
    {
        VkSemaphoreCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        CHECK_VULKAN(
            vkCreateSemaphore(device.logical_device(), &info, nullptr, &img_avail_semaphore));
        CHECK_VULKAN(vkCreateSemaphore(
            device.logical_device(), &info, nullptr, &present_ready_semaphore));
    }
    {
        VkFenceCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        CHECK_VULKAN(vkCreateFence(device.logical_device(), &info, nullptr, &fence));
    }

    VkBool32 present_supported = false;
    CHECK_VULKAN(vkGetPhysicalDeviceSurfaceSupportKHR(
        device.physical_device(), device.queue_index(), surface, &present_supported));
    if (!present_supported) {
        throw std::runtime_error("Present is not supported on the graphics queue!?");
    }

    // TODO Setup ImGui
}

VKDisplay::~VKDisplay()
{
    vkDestroySemaphore(device.logical_device(), img_avail_semaphore, nullptr);
    vkDestroySemaphore(device.logical_device(), present_ready_semaphore, nullptr);
    vkDestroyFence(device.logical_device(), fence, nullptr);
    vkDestroyCommandPool(device.logical_device(), command_pool, nullptr);
    vkDestroySwapchainKHR(device.logical_device(), swap_chain, nullptr);
    vkDestroySurfaceKHR(device.instance(), surface, nullptr);
}

std::string VKDisplay::gpu_brand()
{
    return "TODO";
}

void VKDisplay::resize(const int fb_width, const int fb_height)
{
    if (swap_chain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device.logical_device(), swap_chain, nullptr);
    }

    fb_dims = glm::uvec2(fb_width, fb_height);
    /*
    {
        uint32_t num_formats = 0;
        CHECK_VULKAN(vkGetPhysicalDeviceSurfaceFormatsKHR(
            device.physical_device(), surface, &num_formats, nullptr));
        std::cout << "num formats: " << num_formats << "\n";
        std::vector<VkSurfaceFormatKHR> formats;
        formats.resize(num_formats);
        CHECK_VULKAN(vkGetPhysicalDeviceSurfaceFormatsKHR(
            device.physical_device(), surface, &num_formats, formats.data()));

        for (const auto &fmt : formats) {
            std::cout << "Format: " << fmt.format << ", color space: " << fmt.colorSpace
                      << "\n";
        }
    }
    */
    VkExtent2D swapchain_extent = {};
    swapchain_extent.width = fb_dims.x;
    swapchain_extent.height = fb_dims.y;

    VkSwapchainCreateInfoKHR create_info = {0};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = surface;
    create_info.minImageCount = 2;
    // TODO: The RTX 2070 on Windows says it only supports BGRA through Vulkan, but it seems to
    // work just fine with an RGBA image format. So can I just ignore this?
    create_info.imageFormat = VK_FORMAT_R8G8B8A8_UNORM;
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
        vkCreateSwapchainKHR(device.logical_device(), &create_info, nullptr, &swap_chain));

    // Get the swap chain images
    uint32_t num_swapchain_imgs = 0;
    vkGetSwapchainImagesKHR(device.logical_device(), swap_chain, &num_swapchain_imgs, nullptr);
    swap_chain_images.resize(num_swapchain_imgs);
    vkGetSwapchainImagesKHR(
        device.logical_device(), swap_chain, &num_swapchain_imgs, swap_chain_images.data());

    upload_texture = vkrt::Buffer::host(
        device, sizeof(uint32_t) * fb_dims.x * fb_dims.y, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
}

void VKDisplay::new_frame()
{
    // TODO imgui new frame
}

void VKDisplay::display(const std::vector<uint32_t> &img)
{
    uint32_t back_buffer_idx = 0;
    CHECK_VULKAN(vkAcquireNextImageKHR(device.logical_device(),
                                       swap_chain,
                                       std::numeric_limits<uint64_t>::max(),
                                       img_avail_semaphore,
                                       VK_NULL_HANDLE,
                                       &back_buffer_idx));
    uint32_t *upload = reinterpret_cast<uint32_t *>(upload_texture->map());
    for (size_t i = 0; i < fb_dims.y; ++i) {
        std::memcpy(upload + i * fb_dims.x,
                    img.data() + (fb_dims.y - i - 1) * fb_dims.x,
                    fb_dims.x * sizeof(uint32_t));
    }
    upload_texture->unmap();

    vkResetCommandPool(
        device.logical_device(), command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VULKAN(vkBeginCommandBuffer(command_buffer, &begin_info));

    // Transition image to the general layout
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
                           upload_texture->handle(),
                           swap_chain_images[back_buffer_idx],
                           VK_IMAGE_LAYOUT_GENERAL,
                           1,
                           &img_copy);

    // Transition image to shader read optimal layout
    img_mem_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    img_mem_barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
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

    const std::array<VkPipelineStageFlags, 1> wait_stages = {
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};

    CHECK_VULKAN(vkResetFences(device.logical_device(), 1, &fence));

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = &img_avail_semaphore;
    submit_info.pWaitDstStageMask = wait_stages.data();
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &present_ready_semaphore;
    CHECK_VULKAN(vkQueueSubmit(device.graphics_queue(), 1, &submit_info, fence));

    // Finally, present the updated image in the swap chain
    VkPresentInfoKHR present_info = {};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &present_ready_semaphore;
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &swap_chain;
    present_info.pImageIndices = &back_buffer_idx;
    CHECK_VULKAN(vkQueuePresentKHR(device.graphics_queue(), &present_info));

    // Wait for the present to finish
    CHECK_VULKAN(vkWaitForFences(
        device.logical_device(), 1, &fence, true, std::numeric_limits<uint64_t>::max()));
}
