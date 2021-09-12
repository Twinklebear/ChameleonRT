#pragma once

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <string>
#include <SDL.h>
#include "display/display.h"
#include "imgui.h"
#include "render_backend.h"

/* Plugins need to provide implementations of
 * - PopulateFunctionTableFn via the POPULATE_PLUGIN_FUNCTIONS macro
 * which will populate the functions
 * - GetSDLWindowFlagsFn
 * - SetImGuiContextFn
 * - MakeDisplayFn
 * - MakeRendererFn
 */
struct RenderPluginFunctionTable {
    // Callback functions provided by each plugin
    using GetSDLWindowFlagsFn = uint32_t (*)();

    using SetImGuiContextFn = void (*)(ImGuiContext *context);

    using MakeDisplayFn = std::unique_ptr<Display> (*)(SDL_Window *window);

    using MakeRendererFn = std::unique_ptr<RenderBackend> (*)(Display *display);

    // The callbacks for the loaded plugin
    GetSDLWindowFlagsFn get_window_flags = nullptr;

    SetImGuiContextFn set_imgui_context = nullptr;

    MakeDisplayFn make_display = nullptr;

    MakeRendererFn make_renderer = nullptr;
};

#ifdef _WIN32
#define POPULATE_PLUGIN_FUNCTIONS(                                   \
    GET_WINDOW_FLAGS, SET_IMGUI_CTX, MAKE_DISPLAY, MAKE_RENDERER)    \
    extern "C" __declspec(dllexport) void populate_plugin_functions( \
        RenderPluginFunctionTable *fn_table)                         \
    {                                                                \
        fn_table->get_window_flags = GET_WINDOW_FLAGS;               \
        fn_table->set_imgui_context = SET_IMGUI_CTX;                 \
        fn_table->make_display = MAKE_DISPLAY;                       \
        fn_table->make_renderer = MAKE_RENDERER;                     \
    }
#else
#define POPULATE_PLUGIN_FUNCTIONS(                                                 \
    GET_WINDOW_FLAGS, SET_IMGUI_CTX, MAKE_DISPLAY, MAKE_RENDERER)                  \
    extern "C" void populate_plugin_functions(RenderPluginFunctionTable *fn_table) \
    {                                                                              \
        fn_table->get_window_flags = GET_WINDOW_FLAGS;                             \
        fn_table->set_imgui_context = SET_IMGUI_CTX;                               \
        fn_table->make_display = MAKE_DISPLAY;                                     \
        fn_table->make_renderer = MAKE_RENDERER;                                   \
    }
#endif

struct RenderPlugin {
private:
#ifdef _WIN32
    HMODULE plugin = 0;
#else
    void *plugin = nullptr;
#endif

    std::string name;

    using PopulateFunctionTableFn = void (*)(RenderPluginFunctionTable *);

    RenderPluginFunctionTable function_table;

public:
    RenderPlugin(const std::string &plugin_name);

    ~RenderPlugin();

    uint32_t get_window_flags() const;

    void set_imgui_context(ImGuiContext *context);

    std::unique_ptr<Display> make_display(SDL_Window *window) const;

    std::unique_ptr<RenderBackend> make_renderer(Display *display) const;

    const std::string &get_name() const;

private:
    // Load a function pointer from the shared library. Note that T
    // must be a pointer to function type.
    template <typename T>
    T get_fn(const std::string &fcn_name)
    {
#ifdef _WIN32
        FARPROC fn = GetProcAddress(plugin, fcn_name.c_str());
#else
        void *fn = dlsym(plugin, fcn_name.c_str());
#endif
        if (fn == NULL) {
            throw std::runtime_error("Function " + fcn_name + " is not in the plugin");
            return nullptr;
        } else {
            return reinterpret_cast<T>(fn);
        }
    }
};
