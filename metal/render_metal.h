#pragma once

#include <memory>
#include "render_backend.h"
#include "shader_types.h"

// Need to be a bit careful here to not pull in Obj-C headers
// in to render_metal.h which is included in plain C++ code
// RenderMetalData contains all the actual Obj-C objects/etc.
// we need for interacting with the Metal API
struct RenderMetalData;

struct RenderMetal : RenderBackend {
    std::shared_ptr<RenderMetalData> metal;

    glm::uvec2 fb_dims;
    uint32_t frame_id = 0;
    bool native_display = false;

    RenderMetal();

    virtual ~RenderMetal();

    std::string name() override;

    void initialize(const int fb_width, const int fb_height) override;

    void set_scene(const Scene &scene) override;

    // Returns the rays per-second achieved, or -1 if this is not tracked
    RenderStats render(const glm::vec3 &pos,
                       const glm::vec3 &dir,
                       const glm::vec3 &up,
                       const float fovy,
                       const bool camera_changed,
                       const bool readback_framebuffer) override;

private:
    ViewParams compute_view_parameters(const glm::vec3 &pos,
                                       const glm::vec3 &dir,
                                       const glm::vec3 &up,
                                       const float fovy);
};

