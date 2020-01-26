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
    vkrt::Device device;
    VkSurfaceKHR surface;

    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;

    glm::uvec2 fb_dims;
    VkSwapchainKHR swap_chain = VK_NULL_HANDLE;
    std::vector<VkImage> swap_chain_images;
    std::shared_ptr<vkrt::Buffer> upload_texture;

    VkSemaphore img_avail_semaphore, present_ready_semaphore;
    VkFence fence;

    VKDisplay(SDL_Window *window);

    ~VKDisplay();

    std::string gpu_brand() override;

    void resize(const int fb_width, const int fb_height) override;

    void new_frame() override;

    void display(const std::vector<uint32_t> &img) override;
};
