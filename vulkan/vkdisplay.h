#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>
#include <SDL.h>
#include "display/display.h"
#include "vulkan_utils.h"
#include <glm/glm.hpp>

struct VKDisplay : Display {
    std::shared_ptr<vkrt::Device> device;
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;

    glm::uvec2 fb_dims;
    VkSwapchainKHR swap_chain = VK_NULL_HANDLE;
    std::vector<VkImage> swap_chain_images;
    std::vector<VkImageView> swap_chain_image_views;
    std::vector<VkFramebuffer> framebuffers;
    std::shared_ptr<vkrt::Buffer> upload_buffer;
    std::shared_ptr<vkrt::Texture2D> upload_texture;

    VkRenderPass imgui_render_pass = VK_NULL_HANDLE;
    VkDescriptorPool imgui_desc_pool = VK_NULL_HANDLE;

    VkSemaphore img_avail_semaphore = VK_NULL_HANDLE;
    VkSemaphore present_ready_semaphore = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    VKDisplay(SDL_Window *window);

    ~VKDisplay();

    std::string gpu_brand() override;

    std::string name() override;

    void resize(const int fb_width, const int fb_height) override;

    void new_frame() override;

    void display(const std::vector<uint32_t> &img) override;

    void display_native(std::shared_ptr<vkrt::Texture2D> &img);
};
