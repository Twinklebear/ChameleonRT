#include "render_plugin.h"
#include <stdexcept>

#ifdef _WIN32
#define PLUGIN_PREFIX
#define PLUGIN_SUFFIX ".dll"
#else
#define PLUGIN_PREFIX "lib"
#define PLUGIN_SUFFIX ".so"
#endif

RenderPlugin::RenderPlugin(const std::string &plugin_name) : name(plugin_name)
{
    const std::string plugin_file_name = PLUGIN_PREFIX + plugin_name + PLUGIN_SUFFIX;
    std::string error_msg;
#ifdef _WIN32
    plugin = LoadLibrary(plugin_file_name.c_str());
    if (!plugin) {
        auto err = GetLastError();
        LPTSTR msg_buf;
        FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                          FORMAT_MESSAGE_IGNORE_INSERTS,
                      NULL,
                      err,
                      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                      (LPTSTR)&msg_buf,
                      0,
                      NULL);
        error_msg = msg_buf;
        LocalFree(msg_buf);
    }
#else
    plugin = dlopen(plugin_file_name.c_str(), RTLD_LAZY);
    if (!plugin) {
        error_msg = dlerror();
    }
#endif

    if (!plugin) {
        throw std::runtime_error("Failed to load plugin '" + plugin_file_name +
                                 "' due to: " + error_msg);
    }

    auto populate_fcn_table = get_fn<PopulateFunctionTableFn>("populate_plugin_functions");
    if (!populate_fcn_table) {
        throw std::runtime_error("Plugin " + plugin_file_name +
                                 " is the POPULATE_PLUGIN_FUNCTIONS macro");
    }
    populate_fcn_table(&function_table);
}

RenderPlugin::~RenderPlugin()
{
#ifdef _WIN32
    FreeLibrary(plugin);
#else
    dlclose(plugin);
#endif
}

uint32_t RenderPlugin::get_window_flags() const
{
    return function_table.get_window_flags();
}

void RenderPlugin::set_imgui_context(ImGuiContext *context)
{
    function_table.set_imgui_context(context);
}

std::unique_ptr<Display> RenderPlugin::make_display(SDL_Window *window) const
{
    return function_table.make_display(window);
}

std::unique_ptr<RenderBackend> RenderPlugin::make_renderer(Display *display) const
{
    return function_table.make_renderer(display);
}

const std::string &RenderPlugin::get_name() const
{
    return name;
}
