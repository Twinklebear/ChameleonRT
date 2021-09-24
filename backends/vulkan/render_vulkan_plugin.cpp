#include <SDL.h>
#include "imgui.h"
#include "render_plugin.h"
#include "render_vulkan.h"
#include "vkdisplay.h"

uint32_t get_sdl_window_flags()
{
    return SDL_WINDOW_VULKAN;
}

void set_imgui_context(ImGuiContext *context)
{
    ImGui::SetCurrentContext(context);
}

std::unique_ptr<Display> make_display(SDL_Window *window)
{
    return std::make_unique<VKDisplay>(window);
}

std::unique_ptr<RenderBackend> make_renderer(Display *display)
{
    auto *vk_display = dynamic_cast<VKDisplay *>(display);
    if (vk_display) {
        return std::make_unique<RenderVulkan>(vk_display->device);
    }
    return std::make_unique<RenderVulkan>();
}

POPULATE_PLUGIN_FUNCTIONS(get_sdl_window_flags, set_imgui_context, make_display, make_renderer)

