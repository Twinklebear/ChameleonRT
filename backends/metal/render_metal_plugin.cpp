#include <SDL.h>
#include "imgui.h"
#include "metaldisplay.h"
#include "render_metal.h"
#include "render_plugin.h"

uint32_t get_sdl_window_flags()
{
    return 0;
}

void set_imgui_context(ImGuiContext *context)
{
    ImGui::SetCurrentContext(context);
}

std::unique_ptr<Display> make_display(SDL_Window *window)
{
    return std::make_unique<MetalDisplay>(window);
}

std::unique_ptr<RenderBackend> make_renderer(Display *display)
{
    auto *metal_display = dynamic_cast<MetalDisplay *>(display);
    if (metal_display) {
        return std::make_unique<RenderMetal>(metal_display->context);
    }
    return std::make_unique<RenderMetal>();
}

POPULATE_PLUGIN_FUNCTIONS(get_sdl_window_flags, set_imgui_context, make_display, make_renderer)
