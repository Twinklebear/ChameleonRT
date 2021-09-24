#include <SDL.h>
#include "display/gldisplay.h"
#include "imgui.h"
#include "render_embree.h"
#include "render_plugin.h"

uint32_t get_sdl_window_flags()
{
    return SDL_WINDOW_OPENGL;
}

void set_imgui_context(ImGuiContext *context)
{
    ImGui::SetCurrentContext(context);
}

std::unique_ptr<Display> make_display(SDL_Window *window)
{
    return std::make_unique<GLDisplay>(window);
}

std::unique_ptr<RenderBackend> make_renderer(Display *)
{
    return std::make_unique<RenderEmbree>();
}

POPULATE_PLUGIN_FUNCTIONS(get_sdl_window_flags, set_imgui_context, make_display, make_renderer)
