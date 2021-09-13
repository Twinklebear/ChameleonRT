#include <SDL.h>
#include "dxdisplay.h"
#include "imgui.h"
#include "render_dxr.h"
#include "render_plugin.h"

uint32_t get_sdl_window_flags()
{
    return 0;
}

void set_imgui_context(ImGuiContext *context)
{
    std::cout << "context = " << context << "\n";
    ImGui::SetCurrentContext(context);
}

std::unique_ptr<Display> make_display(SDL_Window *window)
{
    return std::make_unique<DXDisplay>(window);
}

std::unique_ptr<RenderBackend> make_renderer(Display *display)
{
    auto *dx_display = dynamic_cast<DXDisplay *>(display);
    if (dx_display) {
        return std::make_unique<RenderDXR>(dx_display->device);
    }
    return std::make_unique<RenderDXR>();
}

POPULATE_PLUGIN_FUNCTIONS(get_sdl_window_flags, set_imgui_context, make_display, make_renderer)

